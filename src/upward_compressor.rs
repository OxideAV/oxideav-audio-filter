//! Upward compressor — boosts quiet signal **below** a threshold.
//!
//! The dynamic-range-processor family has four quadrants, distinguished
//! by (a) which side of the threshold is acted on and (b) whether the
//! action *narrows* or *widens* the dynamic range:
//!
//! | side of threshold | reduce range (toward threshold) | widen range (away) |
//! | ----------------- | ------------------------------- | ------------------ |
//! | **above**         | downward **compression**        | upward expansion   |
//! | **below**         | **upward compression** (this)   | downward expansion |
//!
//! This crate already ships downward compression
//! ([`Compressor`](crate::compressor::Compressor) — attenuates loud
//! signal above threshold) and downward expansion
//! ([`Expander`](crate::expander::Expander) — attenuates quiet signal
//! below threshold). The remaining quadrant routinely reached for in
//! mastering is **upward compression**: it *raises* the level of quiet
//! passages that sit below the threshold while leaving the loud peaks
//! above it untouched, so the overall dynamic range shrinks from the
//! bottom rather than from the top. Where downward compression turns
//! the loud parts down (and the engineer then makes up the lost level),
//! upward compression turns the quiet parts up — preserving transient
//! peaks and perceived "punch" while lifting the noise-floor-adjacent
//! detail into audibility.
//!
//! # Static curve
//!
//! Let `env_db` be the detector's level in dBFS and define the
//! *under-shoot*
//!
//! ```text
//! under = max(0, threshold_db - env_db)        // 0 at/above threshold, >0 below
//! ```
//!
//! The makeup boost applied below the threshold is
//!
//! ```text
//! boost_db = min(range_db, (1 - 1/R) · under)  // ≥ 0
//! ```
//!
//! * `R` (`ratio`) is the compression ratio (≥ 1). The factor
//!   `(1 - 1/R)` is the fraction of each dB of under-shoot that is
//!   recovered: at `R = 2` half the under-shoot is restored, at
//!   `R = ∞` the full under-shoot is restored (a sample `D` dB below
//!   the threshold is lifted *to* the threshold). `R = 1` recovers
//!   nothing → identity.
//! * `range_db` (≥ 0) caps the maximum boost so the noise floor is not
//!   amplified without bound as the input approaches silence. With a
//!   finite range the boost saturates at `range_db` once the under-shoot
//!   exceeds `range_db / (1 - 1/R)`.
//!
//! Above the threshold (`under = 0`) the gain is unity, so peaks pass
//! through unchanged — the defining property that distinguishes upward
//! compression from a simple make-up-gain'd downward compressor.
//!
//! With a soft knee of width `W` centred on the threshold, the segment
//! `over ∈ (-W/2, +W/2)` (where `over = env_db - threshold_db`) uses a
//! quadratic blend so the boost has continuous slope across the knee:
//!
//! ```text
//! boost_db = 0                                          if over ≥ +W/2
//!          = (1 - 1/R) · (W/2 - over)² / (2W)           if -W/2 < over < +W/2
//!          = min(range_db, (1 - 1/R) · (-over))         if over ≤ -W/2
//! ```
//!
//! The range cap is applied last, after the knee blend, so the smooth
//! transition near the threshold is never clipped (it only matters far
//! below, where the curve has already left the knee).
//!
//! # Detector
//!
//! Same one-pole peak-linked envelope follower as
//! [`Compressor`](crate::compressor::Compressor) /
//! [`Expander`](crate::expander::Expander): peak across channels with
//! separate attack / release time constants
//!
//! ```text
//! α = exp(-1 / (τ · fs))
//! drive = max(|x_0|, |x_1|, …)
//! env  ← α · env + (1 - α) · drive          (α = α_atk if drive > env else α_rel)
//! ```
//!
//! For an upward compressor "attack" is the speed with which the boost
//! engages as the signal *falls* below the threshold and "release" is
//! the speed with which it backs off as the signal rises again. The
//! follower itself is symmetric (it just tracks the level); the static
//! curve maps the tracked level to a boost.
//!
//! # Difference from neighbours
//!
//! * [`Compressor`](crate::compressor::Compressor) — acts *above* the
//!   threshold, *reduces* level. Make-up gain there is a flat boost on
//!   *all* signal; here the boost is level-dependent and acts only
//!   *below* the threshold.
//! * [`Expander`](crate::expander::Expander) — also acts below the
//!   threshold but *attenuates* (widening range); this *boosts*
//!   (narrowing range). They are sign-flipped on the same side.
//! * [`GainNormalizer`](crate::gain_normalizer::GainNormalizer) — a slow
//!   programme-level AGC that drives the *whole* signal toward a target;
//!   upward compression is per-sample and leaves peaks alone.
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
//! The two-type (downward / upward) taxonomy of dynamic-range
//! compression is described in `docs/audio/filter/`'s dynamic-range
//! reference. The static-curve algebra mirrors
//! [`crate::compressor`] / [`crate::expander`] (one-pole follower,
//! soft-knee quadratic blend) reflected to the boost-below-threshold
//! quadrant; no external implementation was consulted.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Upward compressor.
///
/// Boosts signal **below** `threshold_db` by `(1 - 1/ratio)` of each dB
/// of under-shoot (capped at `range_db`), leaving signal above the
/// threshold unchanged. The overall dynamic range narrows from the
/// bottom while transient peaks are preserved.
#[derive(Debug, Clone)]
pub struct UpwardCompressor {
    threshold_db: f32,
    /// Compression ratio (≥ 1). `1.0` is identity; larger values
    /// recover more of the under-shoot. `f32::INFINITY` lifts quiet
    /// signal all the way up to the threshold (subject to `range_db`).
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    /// Soft-knee width in dB centred on `threshold_db`. `0.0` → hard
    /// knee.
    knee_db: f32,
    /// Maximum boost in dB (≥ 0). Bounds amplification of the noise
    /// floor as the input approaches silence.
    range_db: f32,
    state: Option<UpwardState>,
}

#[derive(Debug, Clone)]
struct UpwardState {
    sample_rate: u32,
    alpha_atk: f32,
    alpha_rel: f32,
    /// Linear envelope follower, shared across channels.
    env: f32,
}

impl UpwardCompressor {
    /// Build a new upward compressor.
    ///
    /// * `threshold_db` — dBFS below which the boost is applied.
    /// * `ratio` — compression ratio (≥ 1). `1.0` is identity (bypass);
    ///   `f32::INFINITY` lifts quiet signal to the threshold (subject to
    ///   `range_db`).
    /// * `attack_ms`, `release_ms` — one-pole follower time constants
    ///   (same convention as [`crate::compressor::Compressor`]).
    /// * `knee_db` — soft-knee width in dB (`0` → hard knee).
    /// * `range_db` — maximum boost in dB (≥ 0); caps noise-floor
    ///   amplification.
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

    /// Convenience constructor: hard knee, default `range_db` of `24 dB`
    /// (a sensible cap that keeps the bottom-most −90 dBFS detail from
    /// being lifted by more than four bits of perceived level).
    pub fn upward(threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self::new(threshold_db, ratio, attack_ms, release_ms, 0.0, 24.0)
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if needs_rebuild {
            self.state = Some(UpwardState {
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

/// Static upward-compression curve: dB boost (≥ 0) for a given drive
/// in dB.
///
/// * `over = drive_db - threshold_db` (signed; positive above the
///   threshold, negative below).
/// * `recover = 1 - 1/ratio` (∈ [0, 1]; the fraction of each dB of
///   under-shoot that is recovered).
/// * `knee` is the soft-knee width in dB centred on the threshold;
///   `0` is a hard knee.
/// * `range` caps the boost (applied after the knee blend).
///
/// `ratio` of `1.0` returns `0.0` (identity); `ratio = ∞` returns the
/// full under-shoot (lift to threshold), still capped at `range`.
fn static_boost_db(drive_db: f32, threshold_db: f32, knee: f32, ratio: f32, range: f32) -> f32 {
    if ratio <= 1.0 || range <= 0.0 {
        return 0.0;
    }
    // recover = 1 - 1/R; ratio = ∞ → recover = 1.
    let recover = if ratio.is_infinite() {
        1.0
    } else {
        1.0 - 1.0 / ratio
    };
    let over = drive_db - threshold_db;
    let half = knee * 0.5;

    let raw = if over >= half {
        // At or above the upper knee edge: no boost.
        0.0
    } else if knee > 0.0 && over > -half {
        // Soft-knee quadratic blend joining (over=+W/2, 0) to
        // (over=-W/2, recover·W/2). Parameterise by
        // `t = (W/2 - over)/W ∈ (0, 1)`; the boost grows quadratically
        // with `t`, matching the slope `recover` at the lower edge.
        let t = (half - over) / knee; // 0 at +W/2, 1 at -W/2
        recover * half * t * t
    } else {
        // Below the (hard or lower-knee) threshold: linear in under-shoot.
        recover * (-over)
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

impl AudioFilter for UpwardCompressor {
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
        // Threshold = -40 dBFS, signal = -10 dBFS → above → unity.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-10.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][4_096..])) - db(rms(&x[4_096..]));
        assert!(
            gain_db.abs() < 0.5,
            "above-threshold gain = {} dB (expected ≈ 0)",
            gain_db
        );
    }

    #[test]
    fn two_to_one_at_10db_under_yields_5db_boost() {
        // 2:1 → recover = 1 - 1/2 = 0.5. 10 dB under → +5 dB boost.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS → 10 dB under
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][24_576..])) - db(rms(&x[24_576..]));
        // Peak detector vs RMS measurement → allow slack.
        assert!(
            (gain_db - 5.0).abs() < 1.5,
            "2:1 upward boost = {} dB (expected ≈ +5)",
            gain_db
        );
    }

    #[test]
    fn four_to_one_at_20db_under_yields_15db_boost() {
        // 4:1 → recover = 0.75. 20 dB under → +15 dB boost.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-30.0, 4.0, 5.0, 50.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS → 20 dB under
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][24_576..])) - db(rms(&x[24_576..]));
        assert!(
            (gain_db - 15.0).abs() < 2.5,
            "4:1 upward boost = {} dB (expected ≈ +15)",
            gain_db
        );
    }

    #[test]
    fn infinite_ratio_lifts_to_threshold() {
        // ratio = ∞ → recover = 1 → -50 dBFS at a -40 threshold gets
        // +10 dB (lifted up to the threshold), within range cap.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-40.0, f32::INFINITY, 5.0, 50.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-50.0 / 20.0); // 10 dB under
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][24_576..])) - db(rms(&x[24_576..]));
        assert!(
            (gain_db - 10.0).abs() < 1.5,
            "∞:1 upward boost = {} dB (expected ≈ +10, lift to threshold)",
            gain_db
        );
    }

    #[test]
    fn range_caps_the_boost() {
        // Deep under-shoot with ∞ ratio would boost a lot; range = 6 dB
        // caps it.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-20.0, f32::INFINITY, 5.0, 50.0, 0.0, 6.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-60.0 / 20.0); // 40 dB under (uncapped → +40 dB)
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
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
        let mut uc = UpwardCompressor::new(-20.0, 1.0, 5.0, 50.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-60.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][4_096..])) - db(rms(&x[4_096..]));
        assert!(gain_db.abs() < 0.1, "ratio=1 not identity: {} dB", gain_db);
    }

    #[test]
    fn zero_range_is_identity() {
        // range_db = 0 → no boost ever, even far below threshold.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-20.0, f32::INFINITY, 5.0, 50.0, 0.0, 0.0);
        uc.ensure_state(fs);
        let amp = 10.0f32.powf(-60.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        uc.process_block(&mut channels);
        let gain_db = db(rms(&channels[0][4_096..])) - db(rms(&x[4_096..]));
        assert!(gain_db.abs() < 0.1, "range=0 not identity: {} dB", gain_db);
    }

    #[test]
    fn soft_knee_continuity_and_monotonicity() {
        // Hard-knee value at the lower edge must equal the soft-knee
        // value there; the knee centre must be strictly between the
        // unity segment and the slope segment; boost grows monotonically
        // as drive falls.
        let knee = 10.0f32;
        let ratio = 4.0f32;
        let threshold = -40.0f32;
        let range = 100.0f32; // effectively uncapped
        let recover = 1.0 - 1.0 / ratio;

        let upper = static_boost_db(threshold + knee * 0.5, threshold, knee, ratio, range);
        let middle = static_boost_db(threshold, threshold, knee, ratio, range);
        let lower_edge = static_boost_db(threshold - knee * 0.5, threshold, knee, ratio, range);
        let well_under = static_boost_db(threshold - 2.0 * knee, threshold, knee, ratio, range);

        assert!(
            upper.abs() < 1.0e-4,
            "upper-knee edge should be ~0 dB boost, got {}",
            upper
        );
        // At threshold (centre of knee): blend = recover·(W/2)·(1/2)² = recover·W/8.
        let expected_middle = recover * knee / 8.0;
        assert!(
            (middle - expected_middle).abs() < 1.0e-4,
            "middle-knee boost {} expected {}",
            middle,
            expected_middle
        );
        // At lower knee edge: blend = recover·(W/2)·1² = recover·W/2.
        let expected_lower = recover * knee * 0.5;
        assert!(
            (lower_edge - expected_lower).abs() < 1.0e-4,
            "lower-knee edge boost {} expected {}",
            lower_edge,
            expected_lower
        );
        // Well-under: hard slope. under = 2W → boost = recover·2W.
        let expected_well = recover * 2.0 * knee;
        assert!(
            (well_under - expected_well).abs() < 1.0e-4,
            "well-under boost {} expected {}",
            well_under,
            expected_well
        );
        // Monotone non-decreasing as drive falls.
        assert!(upper <= middle && middle <= lower_edge && lower_edge <= well_under);
    }

    #[test]
    fn stereo_channels_use_linked_detector() {
        // Loud L keeps the linked detector above threshold, so quiet R
        // is NOT boosted (peaks preserve the stereo image).
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-30.0, f32::INFINITY, 1.0, 10.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let amp_l = 10.0f32.powf(-10.0 / 20.0); // -10 dBFS on L (above threshold)
        let amp_r = 10.0f32.powf(-50.0 / 20.0); // -50 dBFS on R (below)
        let n = 8_192usize;
        let l = sine(amp_l, 1_000.0, fs, n);
        let r = sine(amp_r, 1_000.0, fs, n);
        let mut channels = vec![l.clone(), r.clone()];
        uc.process_block(&mut channels);
        let g_db = db(rms(&channels[1][4_096..])) - db(rms(&r[4_096..]));
        assert!(
            g_db.abs() < 1.0,
            "linked-detector should leave R unboosted when L is loud; got {} dB",
            g_db
        );
    }

    #[test]
    fn parameter_clamping() {
        let uc = UpwardCompressor::new(-30.0, 0.5, -10.0, -10.0, -3.0, -5.0);
        assert!(uc.ratio >= 1.0);
        assert!(uc.attack_ms >= 0.0);
        assert!(uc.release_ms >= 0.0);
        assert!(uc.knee_db >= 0.0);
        assert!(uc.range_db >= 0.0);
    }

    #[test]
    fn rate_invariance() {
        // Static curve is rate-independent; only detector coefficients
        // change with fs. Steady-state boost must agree across rates.
        let boost_at = |fs: u32| {
            let mut uc = UpwardCompressor::new(-30.0, 4.0, 1.0, 5.0, 0.0, 100.0);
            uc.ensure_state(fs);
            let amp = 10.0f32.powf(-50.0 / 20.0); // 20 dB under
            let n = (fs as usize) / 4; // 250 ms
            let x = sine(amp, 1_000.0, fs, n);
            let mut ch = vec![x.clone()];
            uc.process_block(&mut ch);
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
        let amp = 10.0f32.powf(-50.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);

        let mut uc_a = UpwardCompressor::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        uc_a.ensure_state(fs);
        let mut block_a = vec![x.clone()];
        uc_a.process_block(&mut block_a);

        let mut uc_b = UpwardCompressor::new(-40.0, 2.0, 5.0, 50.0, 0.0, 24.0);
        uc_b.ensure_state(fs);
        let mut first = vec![x[..4_096].to_vec()];
        uc_b.process_block(&mut first);
        let mut second = vec![x[4_096..].to_vec()];
        uc_b.process_block(&mut second);
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
    fn narrows_dynamic_range_from_below() {
        // The headline behaviour: a two-level programme (loud burst then
        // quiet tail) should have its loud/quiet *difference* reduced,
        // with the loud part essentially untouched.
        let fs = 48_000u32;
        let mut uc = UpwardCompressor::new(-30.0, 4.0, 2.0, 20.0, 0.0, 24.0);
        uc.ensure_state(fs);
        let loud = sine(10.0f32.powf(-6.0 / 20.0), 1_000.0, fs, 16_384); // -6 dBFS
        let quiet = sine(10.0f32.powf(-50.0 / 20.0), 1_000.0, fs, 16_384); // -50 dBFS
        let mut prog = loud.clone();
        prog.extend_from_slice(&quiet);
        let mut channels = vec![prog.clone()];
        uc.process_block(&mut channels);

        let in_loud = db(rms(&prog[8_192..16_384]));
        let in_quiet = db(rms(&prog[24_576..]));
        let out_loud = db(rms(&channels[0][8_192..16_384]));
        let out_quiet = db(rms(&channels[0][24_576..]));

        let in_range = in_loud - in_quiet;
        let out_range = out_loud - out_quiet;
        assert!(
            (out_loud - in_loud).abs() < 0.5,
            "loud part should pass through; in {} out {}",
            in_loud,
            out_loud
        );
        assert!(
            out_range < in_range - 5.0,
            "dynamic range should narrow: in {} dB out {} dB",
            in_range,
            out_range
        );
    }
}
