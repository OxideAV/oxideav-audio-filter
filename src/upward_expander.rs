//! Upward expander — boosts loud signal **above** a threshold even
//! louder, *widening* the dynamic range from the top.
//!
//! This completes the four-quadrant taxonomy of dynamic-range
//! processors. Distinguishing (a) which side of the threshold is acted
//! on and (b) whether the action *narrows* (moves the level toward the
//! threshold) or *widens* (moves it away) the dynamic range:
//!
//! | side of threshold | narrow (toward threshold)       | widen (away from threshold) |
//! | ----------------- | ------------------------------- | --------------------------- |
//! | **above**         | downward **compression**        | **upward expansion** (this) |
//! | **below**         | upward **compression**          | downward **expansion**      |
//!
//! The other three quadrants already ship:
//! [`Compressor`](crate::compressor::Compressor) (downward compression —
//! attenuates loud signal above threshold),
//! [`UpwardCompressor`](crate::upward_compressor::UpwardCompressor)
//! (upward compression — boosts quiet signal below threshold), and
//! [`Expander`](crate::expander::Expander) (downward expansion —
//! attenuates quiet signal below threshold). Upward expansion is the
//! remaining corner: it *increases* the level of signal that already sits
//! **above** the threshold, so loud passages get even louder while quiet
//! signal below the threshold is left untouched. The dynamic range grows
//! from the top — the opposite action to downward compression, applied
//! on the same (above-threshold) side. It restores or exaggerates
//! macro-dynamics (e.g. re-opening a transient an over-compressed master
//! has flattened, or accentuating the swell of a crescendo).
//!
//! # Static curve
//!
//! Let `env_db` be the detector's level in dBFS and define the
//! *over-shoot*
//!
//! ```text
//! over = max(0, env_db - threshold_db)         // 0 at/below threshold, >0 above
//! ```
//!
//! The boost applied above the threshold is
//!
//! ```text
//! boost_db = min(range_db, (R - 1) · over)     // ≥ 0
//! ```
//!
//! * `R` (`ratio`) is the expansion ratio (≥ 1). The slope `(R - 1)` is
//!   the extra dB of output for each dB the input is above the
//!   threshold: at `R = 2` a sample `D` dB over the threshold ends up
//!   `2D` dB over it; at `R = 1` nothing changes → identity.
//! * `range_db` (≥ 0) caps the maximum boost so the loudest peaks are not
//!   amplified without bound (and never driven past full-scale by an
//!   unbounded slope). With a finite range the boost saturates at
//!   `range_db` once the over-shoot exceeds `range_db / (R - 1)`.
//!
//! Below the threshold (`over = 0`) the gain is unity, so quiet signal
//! passes through unchanged — the defining property that distinguishes
//! upward expansion from a flat make-up boost.
//!
//! With a soft knee of width `W` centred on the threshold, the segment
//! `under ∈ (-W/2, +W/2)` (where `under = threshold_db - env_db`) uses a
//! quadratic blend so the boost has continuous slope across the knee:
//!
//! ```text
//! boost_db = 0                                         if under ≥ +W/2  (well below threshold)
//!          = (R - 1) · (W/2 - under)² / (2W)           if -W/2 < under < +W/2
//!          = min(range_db, (R - 1) · (-under))         if under ≤ -W/2  (well above threshold)
//! ```
//!
//! This is the above-threshold mirror of the below-threshold knee in
//! [`UpwardCompressor`](crate::upward_compressor::UpwardCompressor): the
//! roles of "under-shoot" and "over-shoot" are swapped. The range cap is
//! applied last, after the knee blend, so the smooth transition near the
//! threshold is never clipped (it only matters far above, where the curve
//! has already left the knee).
//!
//! # Detector
//!
//! Same one-pole peak-linked envelope follower as
//! [`Compressor`](crate::compressor::Compressor) /
//! [`Expander`](crate::expander::Expander) /
//! [`UpwardCompressor`](crate::upward_compressor::UpwardCompressor):
//! peak across channels with separate attack / release time constants
//!
//! ```text
//! α = exp(-1 / (τ · fs))
//! drive = max(|x_0|, |x_1|, …)
//! env  ← α · env + (1 - α) · drive          (α = α_atk if drive > env else α_rel)
//! ```
//!
//! For an upward expander "attack" is the speed with which the boost
//! engages as the signal *rises* above the threshold and "release" is
//! the speed with which it backs off as the signal falls again. The
//! follower itself is symmetric (it just tracks the level); the static
//! curve maps the tracked level to a boost.
//!
//! # Difference from neighbours
//!
//! * [`Compressor`](crate::compressor::Compressor) — acts *above* the
//!   threshold like this filter but *reduces* level (narrowing range);
//!   upward expansion *boosts* (widening). They are sign-flipped on the
//!   same side.
//! * [`Expander`](crate::expander::Expander) — also *widens* the range
//!   like this filter but acts *below* the threshold by attenuating;
//!   upward expansion acts *above* by boosting. They are the two
//!   range-widening quadrants.
//! * [`UpwardCompressor`](crate::upward_compressor::UpwardCompressor) —
//!   boosts like this filter but *below* the threshold (narrowing the
//!   range from the bottom); upward expansion boosts *above* the
//!   threshold (widening from the top).
//! * [`TransientDesigner`](crate::transient_designer::TransientDesigner)
//!   — shapes attack / sustain via two envelope speeds; upward expansion
//!   is a static-curve level-dependent boost, not an envelope-difference
//!   shaper.
//!
//! # Streaming
//!
//! Stream-rate-agnostic: detector coefficients are re-derived against
//! the live [`AudioStreamParams::sample_rate`] on every `process`
//! call; the envelope state is preserved across calls so block
//! boundaries are seamless.
//!
//! # Provenance
//!
//! The four-quadrant taxonomy of dynamic-range processing (the two
//! compression directions and the two expansion directions, "upward
//! expansion makes the louder sounds above the threshold even louder")
//! is described in `docs/audio/filter/`'s dynamic-range reference. The
//! static-curve algebra mirrors
//! [`crate::upward_compressor`] / [`crate::compressor`] /
//! [`crate::expander`] (one-pole follower, soft-knee quadratic blend)
//! reflected to the boost-above-threshold quadrant; no external
//! implementation was consulted.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Upward expander.
///
/// Boosts signal **above** `threshold_db` by `(ratio - 1)` of each dB of
/// over-shoot (capped at `range_db`), leaving signal below the threshold
/// unchanged. The overall dynamic range widens from the top while quiet
/// passages are preserved.
#[derive(Debug, Clone)]
pub struct UpwardExpander {
    threshold_db: f32,
    /// Expansion ratio (≥ 1). `1.0` is identity; larger values boost the
    /// over-shoot more steeply.
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    /// Soft-knee width in dB centred on `threshold_db`. `0.0` → hard
    /// knee.
    knee_db: f32,
    /// Maximum boost in dB (≥ 0). Bounds amplification of the loudest
    /// peaks so an unbounded slope cannot drive them past full-scale.
    range_db: f32,
    state: Option<UpwardExpanderState>,
}

#[derive(Debug, Clone)]
struct UpwardExpanderState {
    sample_rate: u32,
    alpha_atk: f32,
    alpha_rel: f32,
    /// Linear envelope follower, shared across channels.
    env: f32,
}

impl UpwardExpander {
    /// Build a new upward expander.
    ///
    /// * `threshold_db` — dBFS above which the boost is applied.
    /// * `ratio` — expansion ratio (≥ 1). `1.0` is identity (bypass);
    ///   larger values widen the range from the top more aggressively.
    /// * `attack_ms`, `release_ms` — one-pole follower time constants
    ///   (same convention as [`crate::compressor::Compressor`]).
    /// * `knee_db` — soft-knee width in dB (`0` → hard knee).
    /// * `range_db` — maximum boost in dB (≥ 0); caps peak amplification.
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
        range_db: f32,
    ) -> Self {
        Self {
            threshold_db,
            ratio: ratio.max(1.0),
            attack_ms: attack_ms.max(0.0),
            release_ms: release_ms.max(0.0),
            knee_db: knee_db.max(0.0),
            range_db: range_db.max(0.0),
            state: None,
        }
    }

    /// Convenience constructor: hard knee, default `range_db` of `12 dB`
    /// (a sensible cap that re-opens flattened transients without driving
    /// the loudest peaks far past their original level).
    pub fn upward(threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self::new(threshold_db, ratio, attack_ms, release_ms, 0.0, 12.0)
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if needs_rebuild {
            self.state = Some(UpwardExpanderState {
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
        let threshold_db = self.threshold_db;
        let knee = self.knee_db;
        let ratio = self.ratio;
        let range = self.range_db;
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

            let env_db = 20.0 * state.env.max(1.0e-10).log10();
            let boost_db = static_boost_db(env_db, threshold_db, knee, ratio, range);
            let gain = 10.0f32.powf(boost_db / 20.0);

            for ch in channels.iter_mut().take(n_chan) {
                ch[s] *= gain;
            }
        }
    }
}

/// Static upward-expansion curve: dB boost (≥ 0) for a given drive in
/// dB.
///
/// * `over = drive_db - threshold_db` (signed; positive above the
///   threshold, negative below).
/// * `slope = ratio - 1` (≥ 0; the extra dB of output per dB of
///   over-shoot).
/// * `knee` is the soft-knee width in dB centred on the threshold; `0`
///   is a hard knee.
/// * `range` caps the boost (applied after the knee blend).
///
/// `ratio` of `1.0` returns `0.0` (identity).
fn static_boost_db(drive_db: f32, threshold_db: f32, knee: f32, ratio: f32, range: f32) -> f32 {
    if ratio <= 1.0 || range <= 0.0 {
        return 0.0;
    }
    let slope = ratio - 1.0;
    let over = drive_db - threshold_db;
    let half = knee * 0.5;

    let raw = if over <= -half {
        // At or below the lower knee edge: no boost.
        0.0
    } else if knee > 0.0 && over < half {
        // Soft-knee quadratic blend joining (over=-W/2, 0) to
        // (over=+W/2, slope·W/2). Parameterise by
        // `t = (over + W/2)/W ∈ (0, 1)`; the boost grows quadratically
        // with `t`, matching the slope `slope` at the upper edge.
        let t = (over + half) / knee; // 0 at -W/2, 1 at +W/2
        slope * half * t * t
    } else {
        // Above the (hard or upper-knee) threshold: linear in over-shoot.
        slope * over
    };

    raw.min(range)
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

impl AudioFilter for UpwardExpander {
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
    fn below_threshold_passes_through() {
        // Threshold = -10 dBFS, signal = -40 dBFS → below → unity.
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-10.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        ue.ensure_state(fs);
        let amp = 10.0f32.powf(-40.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        ue.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][4_096..])) - db(rms(&x[4_096..]));
        assert!(
            gain_db.abs() < 0.5,
            "below-threshold gain = {} dB (expected ≈ 0)",
            gain_db
        );
    }

    #[test]
    fn two_to_one_at_10db_over_yields_10db_boost() {
        // 2:1 → slope = 1. 10 dB over → +10 dB boost (within range cap).
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        ue.ensure_state(fs);
        let amp = 10.0f32.powf(-30.0 / 20.0); // -30 dBFS → 10 dB over
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        ue.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][24_576..])) - db(rms(&x[24_576..]));
        assert!(
            (gain_db - 10.0).abs() < 1.5,
            "2:1 upward-expand boost = {} dB (expected ≈ +10)",
            gain_db
        );
    }

    #[test]
    fn ratio_one_point_five_at_10db_over_yields_5db_boost() {
        // 1.5:1 → slope = 0.5. 10 dB over → +5 dB boost.
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-40.0, 1.5, 5.0, 50.0, 0.0, 24.0);
        ue.ensure_state(fs);
        let amp = 10.0f32.powf(-30.0 / 20.0); // 10 dB over
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        ue.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][24_576..])) - db(rms(&x[24_576..]));
        assert!(
            (gain_db - 5.0).abs() < 1.5,
            "1.5:1 upward-expand boost = {} dB (expected ≈ +5)",
            gain_db
        );
    }

    #[test]
    fn range_caps_the_boost() {
        // Big over-shoot with a steep ratio would boost a lot; range = 6 dB
        // caps it.
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-60.0, 4.0, 5.0, 50.0, 0.0, 6.0);
        ue.ensure_state(fs);
        let amp = 10.0f32.powf(-20.0 / 20.0); // 40 dB over (uncapped → +120 dB)
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        ue.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][24_576..])) - db(rms(&x[24_576..]));
        assert!(
            (gain_db - 6.0).abs() < 1.0,
            "range-capped boost = {} dB (expected ≈ +6)",
            gain_db
        );
    }

    #[test]
    fn ratio_one_is_identity() {
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-40.0, 1.0, 5.0, 50.0, 0.0, 24.0);
        ue.ensure_state(fs);
        let amp = 10.0f32.powf(-10.0 / 20.0); // well above threshold
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        ue.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][4_096..])) - db(rms(&x[4_096..]));
        assert!(gain_db.abs() < 0.1, "ratio=1 not identity: {} dB", gain_db);
    }

    #[test]
    fn zero_range_is_identity() {
        // range_db = 0 → no boost ever, even far above threshold.
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-40.0, 4.0, 5.0, 50.0, 0.0, 0.0);
        ue.ensure_state(fs);
        let amp = 10.0f32.powf(-10.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        ue.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][4_096..])) - db(rms(&x[4_096..]));
        assert!(gain_db.abs() < 0.1, "range=0 not identity: {} dB", gain_db);
    }

    #[test]
    fn soft_knee_continuity_and_monotonicity() {
        // Lower-knee value must be ~0; the knee centre sits strictly
        // between the unity segment and the slope segment; boost grows
        // monotonically as drive rises.
        let knee = 10.0f32;
        let ratio = 4.0f32;
        let threshold = -40.0f32;
        let range = 100.0f32; // effectively uncapped
        let slope = ratio - 1.0;

        let lower = static_boost_db(threshold - knee * 0.5, threshold, knee, ratio, range);
        let middle = static_boost_db(threshold, threshold, knee, ratio, range);
        let upper_edge = static_boost_db(threshold + knee * 0.5, threshold, knee, ratio, range);
        let well_over = static_boost_db(threshold + 2.0 * knee, threshold, knee, ratio, range);

        assert!(
            lower.abs() < 1.0e-4,
            "lower-knee edge should be ~0 dB boost, got {}",
            lower
        );
        // At threshold (centre of knee): blend = slope·(W/2)·(1/2)² = slope·W/8.
        let expected_middle = slope * knee / 8.0;
        assert!(
            (middle - expected_middle).abs() < 1.0e-4,
            "middle-knee boost {} expected {}",
            middle,
            expected_middle
        );
        // At upper knee edge: blend = slope·(W/2)·1² = slope·W/2.
        let expected_upper = slope * knee * 0.5;
        assert!(
            (upper_edge - expected_upper).abs() < 1.0e-4,
            "upper-knee edge boost {} expected {}",
            upper_edge,
            expected_upper
        );
        // Well-over: hard slope. over = 2W → boost = slope·2W.
        let expected_well = slope * 2.0 * knee;
        assert!(
            (well_over - expected_well).abs() < 1.0e-4,
            "well-over boost {} expected {}",
            well_over,
            expected_well
        );
        // Monotone non-decreasing as drive rises.
        assert!(lower <= middle && middle <= upper_edge && upper_edge <= well_over);
    }

    #[test]
    fn knee_slope_matches_at_upper_edge() {
        // The quadratic blend must meet the linear above-threshold slope
        // with matching first derivative at over = +W/2 (C¹ continuity).
        let knee = 12.0f32;
        let ratio = 3.0f32;
        let threshold = -30.0f32;
        let range = 1000.0f32;
        let eps = 1.0e-2f32;
        // Just inside the knee vs just outside; finite-difference slopes.
        let inside_lo =
            static_boost_db(threshold + knee * 0.5 - eps, threshold, knee, ratio, range);
        let inside_hi = static_boost_db(threshold + knee * 0.5, threshold, knee, ratio, range);
        let outside_lo = static_boost_db(threshold + knee * 0.5, threshold, knee, ratio, range);
        let outside_hi =
            static_boost_db(threshold + knee * 0.5 + eps, threshold, knee, ratio, range);
        let slope_in = (inside_hi - inside_lo) / eps;
        let slope_out = (outside_hi - outside_lo) / eps;
        assert!(
            (slope_in - slope_out).abs() < 0.05,
            "knee slope discontinuity at upper edge: {} vs {}",
            slope_in,
            slope_out
        );
    }

    #[test]
    fn stereo_channels_use_linked_detector() {
        // Quiet L keeps the linked detector — but a loud R drives it
        // above threshold, so the quiet L is ALSO boosted (peaks across
        // channels link the boost, preserving the stereo image).
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-30.0, 4.0, 1.0, 10.0, 0.0, 24.0);
        ue.ensure_state(fs);
        let amp_l = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS on L (below threshold)
        let amp_r = 10.0f32.powf(-10.0 / 20.0); // -10 dBFS on R (above)
        let n = 8_192usize;
        let l = sine(amp_l, 1_000.0, fs, n);
        let r = sine(amp_r, 1_000.0, fs, n);
        let mut channels = vec![l.clone(), r.clone()];
        ue.process_block(&mut channels);
        let g_l = db(rms(&channels[0][4_096..])) - db(rms(&l[4_096..]));
        // Linked peak detector tracks R (loud) → both channels boosted.
        assert!(
            g_l > 3.0,
            "linked-detector should boost quiet L when R is loud; got {} dB",
            g_l
        );
    }

    #[test]
    fn parameter_clamping() {
        let ue = UpwardExpander::new(-30.0, 0.5, -10.0, -10.0, -3.0, -5.0);
        assert!(ue.ratio >= 1.0);
        assert!(ue.attack_ms >= 0.0);
        assert!(ue.release_ms >= 0.0);
        assert!(ue.knee_db >= 0.0);
        assert!(ue.range_db >= 0.0);
    }

    #[test]
    fn rate_invariance() {
        // Static curve is rate-independent; only detector coefficients
        // change with fs. Steady-state boost must agree across rates.
        let boost_at = |fs: u32| {
            let mut ue = UpwardExpander::new(-40.0, 3.0, 1.0, 5.0, 0.0, 100.0);
            ue.ensure_state(fs);
            let amp = 10.0f32.powf(-30.0 / 20.0); // 10 dB over
            let n = (fs as usize) / 4; // 250 ms
            let x = sine(amp, 1_000.0, fs, n);
            let mut ch = vec![x.clone()];
            ue.process_block(&mut ch);
            let tail = &ch[0][n - (n / 8)..];
            db(rms(tail)) - db(rms(&x[n - (n / 8)..]))
        };
        let a44 = boost_at(44_100);
        let a96 = boost_at(96_000);
        assert!(
            (a44 - a96).abs() < 1.5,
            "rate-invariance: 44.1 kHz {} dB vs 96 kHz {} dB",
            a44,
            a96
        );
    }

    #[test]
    fn streaming_continuity_across_split_calls() {
        // One N-sample call vs two N/2 calls must be sample-identical
        // because the envelope state crosses the call boundary.
        let fs = 48_000u32;
        let amp = 10.0f32.powf(-30.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);

        let mut ue_a = UpwardExpander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        ue_a.ensure_state(fs);
        let mut block_a = vec![x.clone()];
        ue_a.process_block(&mut block_a);

        let mut ue_b = UpwardExpander::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        ue_b.ensure_state(fs);
        let mut first = vec![x[..4_096].to_vec()];
        ue_b.process_block(&mut first);
        let mut second = vec![x[4_096..].to_vec()];
        ue_b.process_block(&mut second);
        let mut concat = first[0].clone();
        concat.extend_from_slice(&second[0]);

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

    #[test]
    fn widens_dynamic_range_from_above() {
        // The headline behaviour: a two-level programme (loud burst then
        // quiet tail) should have its loud/quiet *difference* increased,
        // with the quiet part essentially untouched.
        let fs = 48_000u32;
        let mut ue = UpwardExpander::new(-30.0, 3.0, 2.0, 20.0, 0.0, 24.0);
        ue.ensure_state(fs);
        let loud = sine(10.0f32.powf(-6.0 / 20.0), 1_000.0, fs, 16_384); // -6 dBFS (above)
        let quiet = sine(10.0f32.powf(-50.0 / 20.0), 1_000.0, fs, 16_384); // -50 dBFS (below)
        let mut prog = loud.clone();
        prog.extend_from_slice(&quiet);
        let mut channels = vec![prog.clone()];
        ue.process_block(&mut channels);

        let in_loud = db(rms(&prog[8_192..16_384]));
        let in_quiet = db(rms(&prog[24_576..]));
        let out_loud = db(rms(&channels[0][8_192..16_384]));
        let out_quiet = db(rms(&channels[0][24_576..]));

        let in_range = in_loud - in_quiet;
        let out_range = out_loud - out_quiet;
        assert!(
            (out_quiet - in_quiet).abs() < 0.5,
            "quiet part should pass through; in {} out {}",
            in_quiet,
            out_quiet
        );
        assert!(
            out_range > in_range + 5.0,
            "dynamic range should widen: in {} dB out {} dB",
            in_range,
            out_range
        );
    }
}
