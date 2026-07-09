//! Parallel ("New York" / "Motown") compression.
//!
//! Inserting a compressor in a *parallel* signal path instead of in
//! series is known as **parallel compression** (see the "Parallel
//! compression" section of `docs/audio/filter/`'s dynamic-range
//! compression reference). The dry input is split: one copy passes
//! through unchanged, the other is sent through a compressor; the two
//! are then summed.
//!
//! The reference describes the effect as "a form of upward compression
//! that facilitates dynamic control without significant audible side
//! effects": combining a linear signal with a compressor and reducing
//! the output gain of the compression chain "results in low-level
//! detail enhancement without any peak reduction; the compressor
//! significantly adds to the combined gain at low levels only." The
//! aggressive variant ("New York compression / Motown compression")
//! chooses a high ratio with audible artifacts in the wet path and
//! blends it under the dry signal.
//!
//! # Why peaks are preserved
//!
//! At a loud peak the wet path is squashed flat (its level barely
//! rises above the threshold-plus-make-up plateau), so adding it back
//! contributes only a small fixed amount; the dry path dominates and
//! the peak survives. At a quiet passage the wet path is *un*-reduced
//! and, with make-up gain, sits well *above* the dry path, so the sum
//! is lifted — quiet detail is enhanced. This is exactly the "adds to
//! the combined gain at low levels only" behaviour the reference
//! describes, and is why parallel compression reads as *upward*
//! compression on the combined output even though the wet path is a
//! conventional *downward* compressor.
//!
//! # Signal flow
//!
//! ```text
//!                 ┌─────────── dry · 10^(dry_db/20) ───────────┐
//!   x[n] ──split──┤                                            +──► y[n]
//!                 └─ compressor ─ · 10^(wet_db/20) ────────────┘
//! ```
//!
//! The compressor stage is the same soft-knee, attack/release,
//! peak/RMS detector used by [`crate::Compressor`]; the make-up gain
//! of the inner compressor is folded into the wet-path trim so the two
//! trims (`dry_db`, `wet_db`) are the only mix controls exposed here.
//! The detector is peak-linked across channels so the stereo image of
//! both paths is preserved.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams, EnvelopeMode};
use oxideav_core::{AudioFrame, Result};

/// Parallel ("New York") compressor: blends a compressed wet copy of
/// the input under the untouched dry signal.
#[derive(Debug, Clone)]
pub struct ParallelCompressor {
    threshold_db: f32,
    /// Wet-path compression ratio (≥ 1). `f32::INFINITY` is a limiter.
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    /// Soft-knee width in dB. `0.0` → hard knee.
    knee_db: f32,
    /// Dry-path output trim in dB (default `0` → unity).
    dry_db: f32,
    /// Wet-path output trim in dB (this folds in any compressor
    /// make-up gain; default `0` → unity).
    wet_db: f32,
    /// Detector sensing mode — peak (default) or RMS (power-averaged).
    detector: EnvelopeMode,
    state: Option<State>,
}

#[derive(Debug, Clone)]
struct State {
    sample_rate: u32,
    alpha_atk: f32,
    alpha_rel: f32,
    /// Linear envelope follower across channels (wet detector).
    env: f32,
}

impl ParallelCompressor {
    /// Build a parallel compressor with unity dry/wet trims.
    ///
    /// * `threshold_db` — wet-path drive in dBFS where compression
    ///   begins.
    /// * `ratio` — wet-path input/output dB slope above the knee
    ///   (≥ 1; `f32::INFINITY` → limiter).
    /// * `attack_ms`, `release_ms` — wet-path one-pole follower time
    ///   constants.
    /// * `knee_db` — wet-path soft-knee width in dB (`0` → hard knee).
    ///
    /// Both the dry and wet path output at unity gain. Use
    /// [`ParallelCompressor::with_mix`] to trim the blend.
    pub fn new(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
    ) -> Self {
        Self {
            threshold_db: crate::clamp_param(threshold_db, 0.0, -144.0, 24.0),
            ratio: crate::clamp_param(ratio, 1.0, 1.0, 1_000.0),
            attack_ms: attack_ms.max(0.0),
            release_ms: release_ms.max(0.0),
            knee_db: crate::clamp_param(knee_db, 0.0, 0.0, 96.0),
            dry_db: 0.0,
            wet_db: 0.0,
            detector: EnvelopeMode::Peak,
            state: None,
        }
    }

    /// Build a parallel compressor with explicit dry/wet trims and
    /// detector mode.
    ///
    /// `dry_db` / `wet_db` are linear-in-dB trims applied to the two
    /// paths before the sum. The classic "New York" recipe is a high
    /// ratio (e.g. `10:1`+), a low threshold, fast-ish attack, and
    /// `wet_db` pulled down so the compressed copy sits *under* the
    /// dry signal — adding body to quiet passages without touching the
    /// peaks. `EnvelopeMode::Rms` swaps the detector for the
    /// power-averaged perceptually-relaxed sensing.
    #[allow(clippy::too_many_arguments)]
    pub fn with_mix(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
        dry_db: f32,
        wet_db: f32,
        detector: EnvelopeMode,
    ) -> Self {
        Self {
            dry_db,
            wet_db,
            detector,
            ..Self::new(threshold_db, ratio, attack_ms, release_ms, knee_db)
        }
    }

    /// Current dry-path trim in dB.
    pub fn dry_db(&self) -> f32 {
        self.dry_db
    }

    /// Current wet-path trim in dB.
    pub fn wet_db(&self) -> f32 {
        self.wet_db
    }

    /// Current detector sensing mode (peak vs RMS).
    pub fn detector(&self) -> EnvelopeMode {
        self.detector
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if needs_rebuild {
            self.state = Some(State {
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
        let dry_lin = 10.0f32.powf(self.dry_db / 20.0);
        let wet_lin = 10.0f32.powf(self.wet_db / 20.0);
        let threshold_db = self.threshold_db;
        let knee = self.knee_db;
        let inv_ratio = if self.ratio.is_infinite() {
            0.0
        } else {
            1.0 / self.ratio
        };
        let rms = self.detector == EnvelopeMode::Rms;
        let state = self.state.as_mut().expect("ensure_state ran");

        for s in 0..n_samples {
            // Peak-linked sidechain drive across channels (drives the
            // wet path's compressor; identical to `Compressor`).
            let mut drive = 0.0f32;
            for ch in channels.iter().take(n_chan) {
                let v = if rms { ch[s] * ch[s] } else { ch[s].abs() };
                if v > drive {
                    drive = v;
                }
            }

            if drive > state.env {
                state.env = state.alpha_atk * state.env + (1.0 - state.alpha_atk) * drive;
            } else {
                state.env = state.alpha_rel * state.env + (1.0 - state.alpha_rel) * drive;
            }

            let level = if rms {
                state.env.max(0.0).sqrt()
            } else {
                state.env
            };

            let env_db = 20.0 * level.max(1.0e-10).log10();
            let gr_db = static_gain_reduction_db(env_db, threshold_db, knee, inv_ratio);
            // Wet-path per-sample gain = compression gain · wet trim.
            let wet_gain = 10.0f32.powf(gr_db / 20.0) * wet_lin;

            for ch in channels.iter_mut().take(n_chan) {
                let dry = ch[s];
                // y = dry·dry_trim + (dry·comp_gain)·wet_trim
                ch[s] = dry * dry_lin + dry * wet_gain;
            }
        }
    }
}

/// Static compression curve: dB gain reduction (≤ 0) for a given drive
/// in dB. `inv_ratio = 1/ratio` (use `0.0` for ratio = ∞). `knee` is
/// the soft-knee width in dB; `knee == 0` is a hard knee. Identical to
/// the curve used by [`crate::Compressor`].
fn static_gain_reduction_db(drive_db: f32, threshold_db: f32, knee: f32, inv_ratio: f32) -> f32 {
    let over = drive_db - threshold_db;
    let slope = inv_ratio - 1.0; // ≤ 0
    if knee > 0.0 && over > -knee * 0.5 && over < knee * 0.5 {
        let x = over + knee * 0.5;
        slope * x * x / (2.0 * knee)
    } else if over >= knee * 0.5 {
        slope * over
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

impl AudioFilter for ParallelCompressor {
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

    /// With the wet path muted (`wet_db = -∞`) and the dry path at
    /// unity, the output is bit-identical to the input — the dry split
    /// is a pure passthrough.
    #[test]
    fn dry_only_is_passthrough() {
        let fs = 48_000u32;
        let mut pc = ParallelCompressor::with_mix(
            -30.0,
            8.0,
            5.0,
            50.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            EnvelopeMode::Peak,
        );
        let x = sine(0.5, 1_000.0, fs, 4_096);
        let mut ch = vec![x.clone()];
        pc.ensure_state(fs);
        pc.process_block(&mut ch);
        for (a, b) in ch[0].iter().zip(x.iter()) {
            assert!(
                (a - b).abs() < 1.0e-7,
                "dry-only must passthrough: {a} vs {b}"
            );
        }
    }

    /// Below the threshold the wet path is *not* reduced, so the
    /// combined output is dry + wet ≈ 2× the input (with both trims at
    /// unity) → ≈ +6 dB. This is the "adds to the combined gain at low
    /// levels" behaviour from the reference.
    #[test]
    fn quiet_signal_is_lifted_by_unreduced_wet_path() {
        let fs = 48_000u32;
        let mut pc = ParallelCompressor::new(-10.0, 8.0, 5.0, 50.0, 0.0);
        // -40 dBFS sine, well below the -10 dBFS threshold → wet path
        // passes essentially unchanged.
        let amp = 10.0f32.powf(-40.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 16_384);
        let mut ch = vec![x.clone()];
        pc.ensure_state(fs);
        pc.process_block(&mut ch);
        let tail_out = rms(&ch[0][8_192..]);
        let tail_in = rms(&x[8_192..]);
        let gain_db = db(tail_out) - db(tail_in);
        // dry(unity) + wet(unity, uncompressed) = 2× → +6.02 dB.
        assert!(
            (gain_db - 6.02).abs() < 0.3,
            "quiet-passage combined gain = {gain_db} dB (expected ≈ +6)"
        );
    }

    /// At a loud peak the wet path is heavily reduced, so its
    /// contribution shrinks and the combined gain approaches the dry
    /// path alone (≈ 0 dB for unity dry). The combined gain at a loud
    /// level must be strictly *less* than the quiet-passage gain —
    /// peaks are preserved, quiet detail is lifted.
    #[test]
    fn loud_signal_gains_less_than_quiet_signal() {
        let fs = 48_000u32;
        let thresh = -20.0f32;
        let ratio = 20.0f32;

        let measure = |peak_dbfs: f32| -> f32 {
            let mut pc = ParallelCompressor::new(thresh, ratio, 5.0, 50.0, 0.0);
            let amp = 10.0f32.powf(peak_dbfs / 20.0);
            let x = sine(amp, 1_000.0, fs, 16_384);
            let mut ch = vec![x.clone()];
            pc.ensure_state(fs);
            pc.process_block(&mut ch);
            db(rms(&ch[0][8_192..])) - db(rms(&x[8_192..]))
        };

        let quiet_gain = measure(-50.0); // far below threshold
        let loud_gain = measure(0.0); // far above threshold

        // Quiet passage gets the full +6 dB lift; the loud peak gets
        // markedly less because the wet copy is squashed.
        assert!(
            quiet_gain > loud_gain + 3.0,
            "quiet gain {quiet_gain} dB should exceed loud gain {loud_gain} dB by >3 dB"
        );
        // And the loud combined gain stays modest (dry dominates).
        assert!(
            loud_gain < 3.0,
            "loud combined gain = {loud_gain} dB (peaks should be ≈ preserved)"
        );
    }

    /// Pulling the wet trim down lowers the combined low-level lift:
    /// `wet_db = -6` halves the wet contribution, so a quiet passage
    /// reads dry(1) + wet(0.5) = 1.5× ≈ +3.52 dB instead of +6.
    #[test]
    fn wet_trim_scales_the_low_level_lift() {
        let fs = 48_000u32;
        let mut pc = ParallelCompressor::with_mix(
            -10.0,
            8.0,
            5.0,
            50.0,
            0.0,
            0.0,
            -6.0206, // 0.5 linear
            EnvelopeMode::Peak,
        );
        let amp = 10.0f32.powf(-40.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 16_384);
        let mut ch = vec![x.clone()];
        pc.ensure_state(fs);
        pc.process_block(&mut ch);
        let gain_db = db(rms(&ch[0][8_192..])) - db(rms(&x[8_192..]));
        // 1 + 0.5 = 1.5× → 20·log10(1.5) = 3.522 dB.
        assert!(
            (gain_db - 3.522).abs() < 0.3,
            "wet-trimmed lift = {gain_db} dB (expected ≈ +3.52)"
        );
    }

    /// The detector is peak-linked across channels: a loud transient
    /// on one channel reduces the wet path on *both*, so the stereo
    /// image of the wet copy is preserved. Here a silent right channel
    /// must still receive a (zero) wet contribution and the loud left
    /// channel drives the shared envelope.
    #[test]
    fn detector_is_peak_linked_across_channels() {
        let fs = 48_000u32;
        let mut pc = ParallelCompressor::new(-20.0, 10.0, 1.0, 50.0, 0.0);
        let amp = 10.0f32.powf(-3.0 / 20.0); // loud left
        let left = sine(amp, 1_000.0, fs, 8_192);
        let right = vec![0.0f32; 8_192];
        let mut ch = vec![left.clone(), right.clone()];
        pc.ensure_state(fs);
        pc.process_block(&mut ch);
        // Right stays silent (dry 0 + wet 0).
        for &v in ch[1][4_096..].iter() {
            assert!(v.abs() < 1.0e-7, "silent channel must stay silent: {v}");
        }
        // Left is still audible (dry path always survives).
        assert!(rms(&ch[0][4_096..]) > 0.0);
    }

    /// `dry_db` is a pure trim on the untouched copy: muting the wet
    /// path and setting `dry_db = +6` scales the input by ≈ 2×.
    #[test]
    fn dry_trim_scales_dry_path() {
        let fs = 48_000u32;
        let mut pc = ParallelCompressor::with_mix(
            -30.0,
            8.0,
            5.0,
            50.0,
            0.0,
            6.0206, // +6 dB ≈ 2× linear
            f32::NEG_INFINITY,
            EnvelopeMode::Peak,
        );
        let x = sine(0.25, 1_000.0, fs, 4_096);
        let mut ch = vec![x.clone()];
        pc.ensure_state(fs);
        pc.process_block(&mut ch);
        for (a, b) in ch[0].iter().zip(x.iter()) {
            assert!(
                (a - 2.0 * b).abs() < 1.0e-3,
                "dry +6 dB must double: {a} vs {}",
                2.0 * b
            );
        }
    }

    #[test]
    fn detector_defaults_to_peak_and_unity_trims() {
        let pc = ParallelCompressor::new(-18.0, 4.0, 10.0, 100.0, 6.0);
        assert_eq!(pc.detector(), EnvelopeMode::Peak);
        assert_eq!(pc.dry_db(), 0.0);
        assert_eq!(pc.wet_db(), 0.0);
    }

    /// Empty / channel-less input is a no-op.
    #[test]
    fn empty_block_is_noop() {
        let mut pc = ParallelCompressor::new(-18.0, 4.0, 10.0, 100.0, 0.0);
        pc.ensure_state(48_000);
        let mut ch: Vec<Vec<f32>> = vec![];
        pc.process_block(&mut ch);
        assert!(ch.is_empty());
    }

    /// Changing the sample rate rebuilds the follower coefficients.
    #[test]
    fn sample_rate_change_rebuilds_state() {
        let mut pc = ParallelCompressor::new(-18.0, 4.0, 10.0, 100.0, 0.0);
        pc.ensure_state(44_100);
        let a0 = pc.state.as_ref().unwrap().alpha_atk;
        pc.ensure_state(96_000);
        let a1 = pc.state.as_ref().unwrap().alpha_atk;
        assert!(a0 != a1, "attack alpha must re-derive on rate change");
    }
}
