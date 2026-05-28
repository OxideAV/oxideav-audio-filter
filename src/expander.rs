//! Downward expander — the proportional inverse of [`Compressor`].
//!
//! [`Compressor`]: crate::compressor::Compressor
//!
//! Where a compressor reduces dynamic range by attenuating signal
//! **above** a threshold, an expander increases dynamic range by
//! attenuating signal **below** a threshold. The ratio is greater than
//! `1.0` in both cases, but the slope is applied on the opposite side
//! of the threshold: every `1 dB` the input falls below the threshold
//! produces `R dB` of output attenuation.
//!
//! # Static curve
//!
//! Below the threshold the gain reduction (in dB) is
//!
//! ```text
//! under     = max(0, threshold_db - env_db)        // 0 when above threshold
//! gr_db     = -(R - 1) · under                     // ≤ 0
//! ```
//!
//! With a soft knee of width `W` centred on the threshold the segment
//! `[-W/2, +W/2]` around the threshold uses a quadratic blend so the
//! curve has continuous slope. Defining `over = env_db - threshold_db`
//! (positive *above* threshold, signed),
//!
//! ```text
//! gr_db = 0                                               if over ≥ +W/2
//!       = -(R - 1) · (W/2 - over)² / (2W)                 if -W/2 < over < +W/2
//!       = -(R - 1) · |over|                               if over ≤ -W/2
//! ```
//!
//! At `over = -W/2` the upper-knee value is `0` and the lower-knee
//! value is `0` (both segments meet at the same point); at `over =
//! +W/2` both segments give `-(R - 1) · W/2 / 2 = 0`. Differentiating
//! shows the slope matches at both knee boundaries.
//!
//! # Limit cases
//!
//! * `ratio = 1.0` → identity (`gr_db ≡ 0` everywhere). Used as a
//!   "bypass" sentinel.
//! * `ratio = ∞` → hard downward gate: any signal below the
//!   `(threshold_db - W/2)` lower-knee edge is multiplied by `0`,
//!   producing absolute silence. This is the limit of `-(R - 1) · under
//!   = -∞ dB → gain 0`. The clamp `gr_db ≥ -200 dB` in
//!   [`static_gain_reduction_db`] approximates this safely without
//!   overflowing `10^(gr_db/20)`.
//!
//! # Difference from [`NoiseGate`]
//!
//! [`NoiseGate`]: crate::noise_gate::NoiseGate
//!
//! A noise gate is a *binary* device — its output gain ramps from
//! `0.0` (closed) to `1.0` (open), with hold + attack + release timing
//! controlling when the binary transitions happen. The expander is
//! *proportional*: a `-50 dBFS` signal with a `-40 dBFS` threshold and
//! `2:1` ratio gets `(50 - 40) · (2 - 1) = 10 dB` attenuation; a `-60
//! dBFS` signal under the same settings gets `20 dB` attenuation. As
//! input falls further the attenuation grows linearly in dB (giving a
//! gentle fade-out into silence) rather than slamming closed at a
//! discrete threshold crossing.
//!
//! In broadcast / recording chains a gentle `1.5:1` or `2:1` expander
//! is preferred over a hard gate for dialog and acoustic instruments,
//! since the gradual roll-off masks room noise without the "pumping"
//! artefacts of a slammed gate. The hard-gate limit (`ratio = ∞`) is
//! still available for cases where binary action is wanted.
//!
//! # Detector
//!
//! Identical one-pole envelope follower as [`Compressor`]: peak across
//! channels with separate attack / release time constants
//!
//! ```text
//! α = exp(-1 / (τ · fs))
//! drive       = max(|x_0|, |x_1|, …)
//! if drive > env: env ← α_atk · env + (1 - α_atk) · drive
//! else:           env ← α_rel · env + (1 - α_rel) · drive
//! ```
//!
//! Shared (peak-linked) across channels so the stereo image does not
//! shift when only one side dips below threshold. Channels are
//! otherwise scaled by the same per-sample gain.
//!
//! # Streaming
//!
//! Stream-rate-agnostic: detector coefficients are re-derived against
//! the live [`AudioStreamParams::sample_rate`] on every `process`
//! call. The envelope state is preserved across calls so block
//! boundaries are seamless.
//!
//! # Provenance
//!
//! Algorithm derived from first principles by mirroring the static
//! curve in [`crate::compressor`] across the threshold and reflecting
//! the slope sign. No external library source consulted; the
//! detector + static-curve algebra is standard dynamic-range-processor
//! material that is already in use elsewhere in this crate.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Downward expander.
///
/// Below the threshold each `1 dB` of additional under-shoot produces
/// `R dB` of output attenuation (where `R = ratio - 1`). Above the
/// threshold the signal passes through unchanged. A soft knee can
/// smooth the transition.
#[derive(Debug, Clone)]
pub struct Expander {
    threshold_db: f32,
    /// Expansion ratio (≥ 1). `1.0` is identity; `f32::INFINITY` is a
    /// hard downward gate.
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    /// Soft-knee width in dB centred on `threshold_db`. `0.0` → hard
    /// knee.
    knee_db: f32,
    makeup_gain_db: f32,
    state: Option<ExpanderState>,
}

#[derive(Debug, Clone)]
struct ExpanderState {
    sample_rate: u32,
    /// `exp(-1 / (τ · fs))` for attack.
    alpha_atk: f32,
    /// `exp(-1 / (τ · fs))` for release.
    alpha_rel: f32,
    /// Linear envelope follower, shared across channels.
    env: f32,
}

impl Expander {
    /// Build a new downward expander.
    ///
    /// * `threshold_db` — drive in dBFS where expansion begins
    ///   (signal *below* this gets attenuated).
    /// * `ratio` — input/output dB slope on the *under-shoot* side.
    ///   Must be ≥ 1. `1.0` is identity (bypass). `f32::INFINITY`
    ///   collapses to a hard downward gate.
    /// * `attack_ms`, `release_ms` — one-pole follower time constants
    ///   (same convention as [`crate::compressor::Compressor`]).
    /// * `knee_db` — soft-knee width in dB (`0` → hard knee).
    /// * `makeup_gain_db` — post-expansion linear gain in dB.
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
        makeup_gain_db: f32,
    ) -> Self {
        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_ms: attack_ms.max(0.0),
            release_ms: release_ms.max(0.0),
            knee_db: knee_db.max(0.0),
            makeup_gain_db,
            state: None,
        }
    }

    /// Convenience constructor for a downward expander with a hard
    /// knee and unity make-up gain. Sensible defaults for general
    /// noise-floor management.
    pub fn downward(threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self::new(threshold_db, ratio, attack_ms, release_ms, 0.0, 0.0)
    }

    /// Convenience constructor for the brick-wall limit case: any
    /// signal below `threshold_db` is multiplied by `0`. Equivalent
    /// to `Expander::new(threshold, f32::INFINITY, attack, release,
    /// 0.0, 0.0)`.
    pub fn gate(threshold_db: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self::new(threshold_db, f32::INFINITY, attack_ms, release_ms, 0.0, 0.0)
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if needs_rebuild {
            self.state = Some(ExpanderState {
                sample_rate,
                alpha_atk: time_constant_alpha(self.attack_ms, sample_rate),
                alpha_rel: time_constant_alpha(self.release_ms, sample_rate),
                env: 0.0,
            });
        }
    }

    fn process_block(&mut self, channels: &mut [Vec<f32>]) {
        let n_chan = channels.len();
        if n_chan == 0 {
            return;
        }
        let n_samples = channels[0].len();
        let makeup_lin = 10.0f32.powf(self.makeup_gain_db / 20.0);
        let threshold_db = self.threshold_db;
        let knee = self.knee_db;
        let ratio = self.ratio;
        let state = self.state.as_mut().expect("ensure_state ran");

        for s in 0..n_samples {
            // Peak across channels.
            let mut drive = 0.0f32;
            for ch in channels.iter().take(n_chan) {
                let v = ch[s].abs();
                if v > drive {
                    drive = v;
                }
            }

            // One-pole follower — attack on rising, release on falling.
            if drive > state.env {
                state.env = state.alpha_atk * state.env + (1.0 - state.alpha_atk) * drive;
            } else {
                state.env = state.alpha_rel * state.env + (1.0 - state.alpha_rel) * drive;
            }

            // dB drive of envelope (floor ≈ -200 dBFS).
            let env_db = 20.0 * state.env.max(1.0e-10).log10();
            let gr_db = static_gain_reduction_db(env_db, threshold_db, knee, ratio);
            let gain = 10.0f32.powf(gr_db / 20.0) * makeup_lin;

            for ch in channels.iter_mut().take(n_chan) {
                ch[s] *= gain;
            }
        }
    }
}

/// Static downward-expansion curve: dB gain reduction (≤ 0) for a
/// given drive in dB.
///
/// * `over = drive_db - threshold_db` (signed; positive above the
///   threshold, negative below).
/// * `slope = -(ratio - 1)` (≤ 0; the per-dB attenuation under the
///   threshold).
/// * `knee` is the soft-knee width in dB centred on the threshold;
///   `0` is a hard knee.
///
/// `ratio` of `1.0` returns `0.0` (identity); `ratio = ∞` returns a
/// floor of `-200 dB` for any input under the (hard or lower-knee)
/// threshold, which is approximately silence after the `10^(gr/20)`
/// linear conversion.
fn static_gain_reduction_db(drive_db: f32, threshold_db: f32, knee: f32, ratio: f32) -> f32 {
    if ratio <= 1.0 {
        return 0.0;
    }
    let over = drive_db - threshold_db;

    // Hard-gate limit case (ratio = ∞): no smooth slope. Anything
    // below the lower-knee edge collapses to silence; the in-knee
    // segment fades quadratically from 0 to -200 dB over a half-knee
    // width to keep continuity.
    if ratio.is_infinite() {
        let half = knee * 0.5;
        if over >= half {
            return 0.0;
        }
        if knee > 0.0 && over > -half {
            // Quadratic blend from 0 (at +half) to -200 dB (at -half).
            let t = (half - over) / knee; // 0 at +half, 1 at -half
            return -200.0 * t * t;
        }
        return -200.0;
    }

    // Finite ratio: per-dB attenuation below threshold.
    let slope = -(ratio - 1.0); // ≤ 0

    if knee > 0.0 && over > -knee * 0.5 && over < knee * 0.5 {
        // Smooth-knee quadratic blend joining (over=+W/2, 0) to
        // (over=-W/2, slope · -W/2 = +(ratio-1)·W/2). Parameterise
        // by `t = (W/2 - over)/W ∈ (0, 1)` so the segment value
        // grows quadratically with `t`.
        let t = (knee * 0.5 - over) / knee; // 0 at +W/2, 1 at -W/2
        slope * (knee * 0.5) * t * t
    } else if over <= -knee * 0.5 {
        slope * (-over)
    } else {
        0.0
    }
}

/// `α = exp(-1 / (τ · fs))`. Returns 0 if `τ ≤ 0` (instantaneous).
fn time_constant_alpha(time_ms: f32, sample_rate: u32) -> f32 {
    let tau_s = (time_ms / 1000.0).max(0.0);
    if tau_s <= 1.0e-9 || sample_rate == 0 {
        0.0
    } else {
        let n = tau_s * sample_rate as f32;
        (-1.0 / n).exp()
    }
}

impl AudioFilter for Expander {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate);
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.process_block(&mut channels);
        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db(linear: f32) -> f32 {
        20.0 * linear.max(1.0e-12).log10()
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    fn sine(amp: f32, freq_hz: f32, fs: u32, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let w = 2.0 * std::f32::consts::PI * freq_hz / fs as f32;
        for i in 0..n {
            out.push(amp * (i as f32 * w).sin());
        }
        out
    }

    #[test]
    fn above_threshold_passes_through() {
        let fs = 48_000u32;
        // Threshold = -40 dBFS, signal = -10 dBFS → way above.
        let mut exp = Expander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 0.0);
        exp.ensure_state(fs);
        let amp = 10.0f32.powf(-10.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        exp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][4_096..]);
        let in_rms = rms(&x[4_096..]);
        let gain_db = db(tail_rms) - db(in_rms);
        assert!(
            gain_db.abs() < 0.5,
            "above-threshold gain = {} dB (expected ≈ 0)",
            gain_db
        );
    }

    #[test]
    fn two_to_one_at_10db_under_yields_10db_reduction() {
        // 2:1 expander: slope = -(2-1) = -1 dB per dB under threshold.
        // 10 dB under-shoot → 10 dB attenuation.
        let fs = 48_000u32;
        let mut exp = Expander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 0.0);
        exp.ensure_state(fs);
        // -50 dBFS sine, threshold = -40 dBFS → 10 dB under.
        let amp = 10.0f32.powf(-50.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        exp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][24_576..]);
        let in_rms = rms(&x[24_576..]);
        let gain_db = db(tail_rms) - db(in_rms);
        // Expected: under = 10 dB, gr = -1 · 10 = -10 dB.
        // Peak detector ≠ RMS detector, allow ~1.5 dB slack.
        assert!(
            (gain_db + 10.0).abs() < 1.5,
            "steady-state expander reduction = {} dB (expected ≈ -10)",
            gain_db
        );
    }

    #[test]
    fn three_to_one_at_20db_under_yields_40db_reduction() {
        // 3:1 expander: slope = -(3-1) = -2 dB per dB under threshold.
        // 20 dB under-shoot → 40 dB attenuation.
        let fs = 48_000u32;
        let mut exp = Expander::new(-30.0, 3.0, 5.0, 50.0, 0.0, 0.0);
        exp.ensure_state(fs);
        let amp = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS → 20 dB under
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        exp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][24_576..]);
        let in_rms = rms(&x[24_576..]);
        let gain_db = db(tail_rms) - db(in_rms);
        // Peak detector vs RMS measurement gives ~2.5 dB sine slack at
        // this depth (peak of a sine is √2× RMS = 3 dB, which the
        // detector smooths via the one-pole follower toward
        // somewhere between peak and RMS depending on the time
        // constants).
        assert!(
            (gain_db + 40.0).abs() < 3.0,
            "3:1 expander reduction = {} dB (expected ≈ -40)",
            gain_db
        );
    }

    #[test]
    fn ratio_one_is_identity() {
        // ratio = 1 → bypass.
        let fs = 48_000u32;
        let mut exp = Expander::new(-20.0, 1.0, 5.0, 50.0, 0.0, 0.0);
        exp.ensure_state(fs);
        let amp = 10.0f32.powf(-60.0 / 20.0); // well below threshold
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        exp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][4_096..]);
        let in_rms = rms(&x[4_096..]);
        let gain_db = db(tail_rms) - db(in_rms);
        assert!(
            gain_db.abs() < 0.1,
            "ratio=1 should be identity, got {} dB",
            gain_db
        );
    }

    #[test]
    fn infinite_ratio_silences_below_threshold() {
        // Hard downward gate: anything well below threshold → silence.
        let fs = 48_000u32;
        let mut exp = Expander::gate(-30.0, 1.0, 10.0);
        exp.ensure_state(fs);
        let amp = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS, 20 dB under
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        exp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][4_096..]);
        // Should be silenced (or extremely close to it).
        assert!(
            tail_rms < 1.0e-6,
            "hard-gate tail RMS = {} (expected ~0)",
            tail_rms
        );
    }

    #[test]
    fn soft_knee_continuity() {
        // At the lower knee edge the gain reduction must match the
        // hard-knee curve; in the middle of the knee the curve must
        // be strictly between the unity-gain segment and the slope
        // segment.
        let knee = 10.0f32;
        let ratio = 4.0f32;
        let threshold = -40.0f32;
        let upper = static_gain_reduction_db(threshold + knee * 0.5, threshold, knee, ratio);
        let middle = static_gain_reduction_db(threshold, threshold, knee, ratio);
        let lower_edge = static_gain_reduction_db(threshold - knee * 0.5, threshold, knee, ratio);
        let well_under = static_gain_reduction_db(threshold - 2.0 * knee, threshold, knee, ratio);

        assert!(
            (upper - 0.0).abs() < 1.0e-4,
            "upper-knee edge should be ~0 dB gr, got {}",
            upper
        );
        // At threshold (centre of knee): blend value = slope·(W/2)·(1/2)² = slope·W/8.
        let expected_middle = -(ratio - 1.0) * knee / 8.0;
        assert!(
            (middle - expected_middle).abs() < 1.0e-4,
            "middle-knee gr {} expected {}",
            middle,
            expected_middle
        );
        // At the lower knee edge: blend value = slope·(W/2)·1² = slope·W/2.
        let expected_lower = -(ratio - 1.0) * knee * 0.5;
        assert!(
            (lower_edge - expected_lower).abs() < 1.0e-4,
            "lower-knee edge gr {} expected {}",
            lower_edge,
            expected_lower
        );
        // Well-below: hard-slope formula. under = 2W, slope = -(R-1).
        let expected_well = -(ratio - 1.0) * 2.0 * knee;
        assert!(
            (well_under - expected_well).abs() < 1.0e-4,
            "well-under gr {} expected {}",
            well_under,
            expected_well
        );
        // Monotone non-increasing as drive falls.
        assert!(upper >= middle && middle >= lower_edge && lower_edge >= well_under);
    }

    #[test]
    fn stereo_channels_use_linked_detector() {
        // Drive only the left channel; right channel should see the
        // same per-sample gain since the detector is peak-linked.
        let fs = 48_000u32;
        let mut exp = Expander::new(-30.0, 4.0, 1.0, 10.0, 0.0, 0.0);
        exp.ensure_state(fs);
        let amp_l = 10.0f32.powf(-10.0 / 20.0); // -10 dBFS on L
        let amp_r = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS on R
        let n = 8_192usize;
        let l = sine(amp_l, 1_000.0, fs, n);
        let r = sine(amp_r, 1_000.0, fs, n);
        let mut channels = vec![l.clone(), r.clone()];
        exp.process_block(&mut channels);
        // Loud L keeps the detector above threshold, so R should not
        // be attenuated by the expander.
        let r_tail = rms(&channels[1][4_096..]);
        let r_in = rms(&r[4_096..]);
        let g_db = db(r_tail) - db(r_in);
        assert!(
            g_db.abs() < 1.0,
            "linked-detector should leave quiet R unattenuated when L is loud; got {} dB",
            g_db
        );
    }

    #[test]
    fn makeup_gain_adds_post_curve() {
        // Above-threshold signal with 6 dB make-up → +6 dB output.
        let fs = 48_000u32;
        let mut exp = Expander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 6.0);
        exp.ensure_state(fs);
        let amp = 10.0f32.powf(-10.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        exp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][4_096..]);
        let in_rms = rms(&x[4_096..]);
        let gain_db = db(tail_rms) - db(in_rms);
        assert!(
            (gain_db - 6.0).abs() < 0.5,
            "expected +6 dB make-up gain, got {} dB",
            gain_db
        );
    }

    #[test]
    fn parameter_clamping() {
        // ratio < 1, negative times, negative knee all get clamped.
        let exp = Expander::new(-30.0, 0.5, -10.0, -10.0, -3.0, 0.0);
        assert!(exp.ratio >= 1.0);
        assert!(exp.attack_ms >= 0.0);
        assert!(exp.release_ms >= 0.0);
        assert!(exp.knee_db >= 0.0);
    }

    #[test]
    fn rate_invariance() {
        // The static-curve calculation is rate-independent (only the
        // detector coefficients change with fs). Compute the steady-
        // state attenuation at two sample rates and compare.
        let attenuation_at = |fs: u32| {
            let mut exp = Expander::new(-30.0, 4.0, 1.0, 5.0, 0.0, 0.0);
            exp.ensure_state(fs);
            // Long enough block at each rate for the envelope to settle.
            let amp = 10.0f32.powf(-50.0 / 20.0);
            let n = (fs as usize) / 4; // 250 ms
            let x = sine(amp, 1_000.0, fs, n);
            let mut ch = vec![x.clone()];
            exp.process_block(&mut ch);
            let tail = &ch[0][n - (n / 8)..];
            let r_in = rms(&x[n - (n / 8)..]);
            let r_out = rms(tail);
            db(r_out) - db(r_in)
        };
        let a44 = attenuation_at(44_100);
        let a96 = attenuation_at(96_000);
        // Both should land near the closed-form -60 dB (20 dB under,
        // 4:1 → slope -3 → -60 dB). Allow a few dB envelope settling
        // slack but the two rates must agree closely.
        assert!(
            (a44 - a96).abs() < 1.5,
            "rate-invariance: 44.1 kHz {} dB vs 96 kHz {} dB",
            a44,
            a96
        );
    }

    #[test]
    fn streaming_continuity_across_split_calls() {
        // Processing N samples in one call vs N/2 + N/2 should give
        // (near-)identical results because the envelope state crosses
        // the call boundary.
        let fs = 48_000u32;
        let amp = 10.0f32.powf(-50.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);

        let mut exp_a = Expander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 0.0);
        exp_a.ensure_state(fs);
        let mut block_a = vec![x.clone()];
        exp_a.process_block(&mut block_a);

        let mut exp_b = Expander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 0.0);
        exp_b.ensure_state(fs);
        let mut first = vec![x[..4_096].to_vec()];
        exp_b.process_block(&mut first);
        let mut second = vec![x[4_096..].to_vec()];
        exp_b.process_block(&mut second);
        let mut concat = first[0].clone();
        concat.extend_from_slice(&second[0]);

        // Sample-by-sample identity (the envelope is the same scalar
        // either way; no block-boundary smoothing artefacts).
        for i in 0..block_a[0].len() {
            assert!(
                (block_a[0][i] - concat[i]).abs() < 1.0e-6,
                "split-call drift at sample {}: {} vs {}",
                i,
                block_a[0][i],
                concat[i]
            );
        }
    }
}
