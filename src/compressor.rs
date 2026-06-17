//! Compressor with soft-knee, attack/release follower, make-up gain, and
//! a selectable peak / RMS detector.
//!
//! # Detector
//!
//! The detector tracks the level across all channels with a one-pole
//! envelope follower. Two sensing modes are offered (see the "Peak vs
//! RMS sensing" section of `docs/audio/filter/`'s dynamic-range
//! compression reference):
//!
//! * **Peak** (default) — the follower smooths the rectified amplitude
//!   `max(|x_0|, |x_1|, …)`. Provides tight peak-level control but does
//!   not necessarily track perceived loudness.
//! * **RMS** — the follower runs on the squared drive `max(x_0², x_1², …)`
//!   and the detector reports `√env`, a power (root-mean-square)
//!   measurement that the spec notes "more closely relates to human
//!   perception of loudness", giving a more relaxed compression.
//!
//! ```text
//! drive       = max(|x_0|, |x_1|, …)            (Peak)
//!             = max(x_0², x_1², …)              (RMS, on power)
//! if drive > env: env ← α_atk · env + (1 - α_atk) · drive
//! else:           env ← α_rel · env + (1 - α_rel) · drive
//! level       = env            (Peak)
//!             = √env           (RMS — convert power back to amplitude)
//! ```
//!
//! with coefficients derived from the classical one-pole IIR time
//! constant relation
//!
//! ```text
//! α = exp(-1 / (τ · f_s))
//! ```
//!
//! where `τ` is the attack or release time in seconds.
//!
//! # Static curve
//!
//! Over-threshold drive in dB is `over = env_db - threshold_db`. With a
//! soft-knee of width `W` centred on the threshold, the compression
//! gain reduction is
//!
//! ```text
//! gr = 0                                              if over ≤ -W/2
//!    = (1/R - 1) · (over + W/2)² / (2W)               if -W/2 < over < W/2
//!    = (1/R - 1) · (over)                             if over ≥ W/2
//! ```
//!
//! Linear output sample = `x · 10^((gr + makeup_gain_db) / 20)`. Limiter
//! mode (`ratio = ∞`) collapses `1/R → 0`, giving `gr = -over` above
//! the knee.
//!
//! # Multi-channel
//!
//! The detector is shared across channels (peak-link) so the stereo
//! image is preserved. Channels are otherwise scaled by the same
//! per-sample gain.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams, EnvelopeMode};
use oxideav_core::{AudioFrame, Result};

/// Peak / RMS compressor.
#[derive(Debug, Clone)]
pub struct Compressor {
    threshold_db: f32,
    /// Compression ratio (≥ 1). `f32::INFINITY` is a brickwall limiter.
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    /// Soft-knee width in dB. `0.0` → hard knee.
    knee_db: f32,
    makeup_gain_db: f32,
    /// Sidechain sensing mode — peak (default) or RMS (power-averaged).
    detector: EnvelopeMode,
    state: Option<CompressorState>,
}

#[derive(Debug, Clone)]
struct CompressorState {
    sample_rate: u32,
    /// `exp(-1 / (τ · fs))` for attack.
    alpha_atk: f32,
    /// `exp(-1 / (τ · fs))` for release.
    alpha_rel: f32,
    /// Linear envelope follower across channels.
    env: f32,
}

impl Compressor {
    /// Build a new compressor.
    ///
    /// * `threshold_db` — drive in dBFS where compression begins.
    /// * `ratio` — input/output dB slope above the knee. Must be ≥ 1.
    ///   Use `f32::INFINITY` for a limiter.
    /// * `attack_ms`, `release_ms` — one-pole follower time constants.
    /// * `knee_db` — soft-knee width in dB (`0` → hard knee).
    /// * `makeup_gain_db` — post-compression linear gain in dB.
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
            detector: EnvelopeMode::Peak,
            state: None,
        }
    }

    /// Build a compressor with an explicit detector sensing mode.
    ///
    /// Identical to [`Compressor::new`] but lets the caller select
    /// [`EnvelopeMode::Rms`] (power-averaged, perceptually-relaxed) in
    /// place of the default [`EnvelopeMode::Peak`]. See the module-level
    /// "Detector" section.
    #[allow(clippy::too_many_arguments)]
    pub fn with_detector(
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        knee_db: f32,
        makeup_gain_db: f32,
        detector: EnvelopeMode,
    ) -> Self {
        Self {
            detector,
            ..Self::new(
                threshold_db,
                ratio,
                attack_ms,
                release_ms,
                knee_db,
                makeup_gain_db,
            )
        }
    }

    /// Current detector sensing mode (peak vs RMS).
    pub fn detector(&self) -> EnvelopeMode {
        self.detector
    }

    /// Brickwall limiter constructor — ratio = ∞.
    pub fn limiter(threshold_db: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self::new(threshold_db, f32::INFINITY, attack_ms, release_ms, 0.0, 0.0)
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if needs_rebuild {
            self.state = Some(CompressorState {
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
        // Snapshot scalar config so the mutable borrow of `state`
        // doesn't conflict with reading `self.*`.
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
            // Sidechain drive across channels. Peak mode smooths the
            // rectified amplitude `|x|`; RMS mode smooths the power `x²`
            // and converts back to amplitude after the follower.
            let mut drive = 0.0f32;
            for ch in channels.iter().take(n_chan) {
                let v = if rms { ch[s] * ch[s] } else { ch[s].abs() };
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

            // Convert the smoothed estimate to an amplitude level. For
            // RMS the follower holds power, so √env recovers amplitude.
            let level = if rms {
                state.env.max(0.0).sqrt()
            } else {
                state.env
            };

            // dB drive of level (floor ≈ -200 dBFS).
            let env_db = 20.0 * level.max(1.0e-10).log10();
            let gr_db = static_gain_reduction_db(env_db, threshold_db, knee, inv_ratio);
            let gain = 10.0f32.powf(gr_db / 20.0) * makeup_lin;

            for ch in channels.iter_mut().take(n_chan) {
                ch[s] *= gain;
            }
        }
    }
}

/// Static compression curve: dB gain reduction (≤ 0) for a given drive
/// in dB. `inv_ratio = 1/ratio` (use `0.0` for ratio = ∞). `knee` is
/// the soft-knee width in dB; `knee == 0` is a hard knee.
fn static_gain_reduction_db(drive_db: f32, threshold_db: f32, knee: f32, inv_ratio: f32) -> f32 {
    let over = drive_db - threshold_db;
    let slope = inv_ratio - 1.0; // ≤ 0
    if knee > 0.0 && over > -knee * 0.5 && over < knee * 0.5 {
        // Smooth-knee quadratic interpolation between (knee-low, 0)
        // and (knee-high, slope * over).
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

impl AudioFilter for Compressor {
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
    fn below_threshold_passes_through_with_only_makeup_gain() {
        let fs = 48_000u32;
        let mut comp = Compressor::new(-10.0, 4.0, 5.0, 50.0, 0.0, 0.0);
        // -20 dBFS sine, threshold = -10 dBFS → no compression.
        let amp = 10.0f32.powf(-20.0 / 20.0);
        let x = sine(amp, 1_000.0, fs, 8_192);
        let mut channels = vec![x.clone()];
        comp.ensure_state(fs);
        comp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][4_096..]);
        let in_rms = rms(&x[4_096..]);
        let gain_db = db(tail_rms) - db(in_rms);
        assert!(
            gain_db.abs() < 0.6,
            "below-threshold gain = {} dB (expected 0)",
            gain_db
        );
    }

    #[test]
    fn four_to_one_at_12db_over_yields_9db_reduction() {
        let fs = 48_000u32;
        // Threshold chosen so a -10 dBFS sine sits 12 dB above it.
        let comp_t_db = -22.0f32;
        let mut comp = Compressor::new(comp_t_db, 4.0, 5.0, 50.0, 0.0, 0.0);
        let amp = 10.0f32.powf(-10.0 / 20.0); // sine at -10 dBFS
        let x = sine(amp, 1_000.0, fs, 32_768);
        let mut channels = vec![x.clone()];
        comp.ensure_state(fs);
        comp.process_block(&mut channels);
        let tail_rms = rms(&channels[0][24_576..]);
        let in_rms = rms(&x[24_576..]);
        let gain_db = db(tail_rms) - db(in_rms);
        // Expected: over = 12 dB, gr = over*(1/R - 1) = 12 * (-0.75) = -9 dB.
        // Steady-state RMS reduction tracks envelope reduction within
        // a fraction of a dB (peak detector ≠ RMS detector, allow ~1 dB).
        assert!(
            (gain_db + 9.0).abs() < 1.5,
            "steady-state gain reduction = {} dB (expected ≈ -9)",
            gain_db
        );
    }

    #[test]
    fn attack_window_reaches_steady_state() {
        let fs = 48_000u32;
        let attack_ms = 20.0f32;
        let release_ms = 200.0f32;
        let mut comp = Compressor::new(-20.0, 4.0, attack_ms, release_ms, 0.0, 0.0);
        comp.ensure_state(fs);
        // Step to constant +12 dB-over input.
        let amp = 10.0f32.powf(-8.0 / 20.0); // -8 dBFS = 12 dB over -20
        let x = vec![amp; (fs as usize) / 2];
        let mut channels = vec![x.clone()];
        comp.process_block(&mut channels);

        // Find the gain at sample 0 vs at the very end (steady).
        let g0 = channels[0][0] / x[0];
        let g_end = channels[0][channels[0].len() - 1] / x[channels[0].len() - 1];
        let g0_db = db(g0);
        let g_end_db = db(g_end);

        // At t=0 the envelope is 0; first sample's gain should be ~1
        // (no reduction). After many attack windows the gain converges
        // to ≈ -9 dB (12 over, 4:1).
        assert!(
            g0_db > -1.0,
            "first-sample gain should be near unity, got {} dB",
            g0_db
        );
        assert!(
            (g_end_db + 9.0).abs() < 2.0,
            "steady-state gain = {} dB",
            g_end_db
        );

        // And: the attack rate is bounded by the attack time constant,
        // so at a sample count *much* less than attack_ms the gain
        // must still be closer to unity than to steady-state.
        let n_quarter_attack = (attack_ms / 4.0 / 1000.0 * fs as f32) as usize;
        let g_quarter = channels[0][n_quarter_attack] / x[n_quarter_attack];
        let g_q_db = db(g_quarter);
        assert!(
            g_q_db > g_end_db + 1.0,
            "gain at t/4 = {} dB should be less reductive than steady {} dB",
            g_q_db,
            g_end_db
        );
    }

    #[test]
    fn detector_defaults_to_peak() {
        let comp = Compressor::new(-10.0, 4.0, 5.0, 50.0, 0.0, 0.0);
        assert_eq!(comp.detector(), EnvelopeMode::Peak);
        let lim = Compressor::limiter(-6.0, 0.5, 30.0);
        assert_eq!(lim.detector(), EnvelopeMode::Peak);
    }

    #[test]
    fn rms_detector_settles_on_true_rms_of_sine() {
        // A sine of amplitude A has RMS = A/√2 (≈ −3.01 dB below the
        // peak). With time constants long relative to the sine period the
        // power follower averages x² and the detector level converges to
        // the true RMS. The over-threshold drive is then computed from
        // the RMS level, so the steady gain reduction is set by how far
        // the *RMS* sits above the threshold — independent of the peak.
        //
        // Threshold chosen so the sine's RMS sits exactly 12 dB over:
        // RMS_dB = -10 (peak) - 3.01 = -13.01 dBFS, threshold = -25.01.
        let fs = 48_000u32;
        let peak_amp = 10.0f32.powf(-10.0 / 20.0); // sine peak at -10 dBFS
                                                   // Sine RMS is 20·log10(1/√2) = -3.0103 dB below the peak.
        let rms_db = -10.0 + 20.0 * (1.0f32 / 2.0f32.sqrt()).log10(); // ≈ -13.01 dBFS
        let thresh_db = rms_db - 12.0; // RMS is 12 dB over
        let ratio = 4.0f32;
        // 50 ms time constants ≈ 50 periods of the 1 kHz tone → the x²
        // ripple is well averaged.
        let x = sine(peak_amp, 1_000.0, fs, 96_000);
        let mut comp =
            Compressor::with_detector(thresh_db, ratio, 50.0, 50.0, 0.0, 0.0, EnvelopeMode::Rms);
        comp.ensure_state(fs);
        let mut ch = vec![x.clone()];
        comp.process_block(&mut ch);

        // The applied gain is nearly constant once settled (ripple is
        // small), so measure it directly at the tail amplitude peaks.
        let tail = 48_000usize;
        let mut acc = 0.0f64;
        let mut cnt = 0usize;
        for i in tail..x.len() {
            if x[i].abs() > peak_amp * 0.9 {
                acc += db(ch[0][i] / x[i]) as f64;
                cnt += 1;
            }
        }
        let gr = (acc / cnt as f64) as f32;
        // RMS 12 dB over, 4:1 → gr = 12 · (1/4 − 1) = −9 dB.
        assert!(
            (gr + 9.0).abs() < 1.0,
            "RMS-detector steady reduction = {gr} dB (expected ≈ -9 from RMS-over-threshold)"
        );
    }

    #[test]
    fn rms_detector_steady_state_matches_dc_amplitude() {
        // For a constant (DC) drive there is no peak-vs-RMS gap, so an
        // RMS detector must converge to the same gain reduction as a
        // peak detector. 12 dB over, 4:1 → ≈ -9 dB either way.
        let fs = 48_000u32;
        let amp = 10.0f32.powf(-8.0 / 20.0); // -8 dBFS = 12 dB over -20
        let x = vec![amp; fs as usize / 2];
        let mut comp =
            Compressor::with_detector(-20.0, 4.0, 5.0, 50.0, 0.0, 0.0, EnvelopeMode::Rms);
        comp.ensure_state(fs);
        let mut ch = vec![x.clone()];
        comp.process_block(&mut ch);
        let g_end = db(ch[0][ch[0].len() - 1] / x[x.len() - 1]);
        assert!(
            (g_end + 9.0).abs() < 1.0,
            "RMS steady-state on DC = {g_end} dB (expected ≈ -9)"
        );
    }

    #[test]
    fn limiter_caps_peak_near_threshold() {
        let fs = 48_000u32;
        // Brickwall at -6 dBFS with a fast attack.
        let mut comp = Compressor::limiter(-6.0, 0.5, 30.0);
        comp.ensure_state(fs);
        // 0 dBFS impulse train spaced widely so the envelope has time
        // to ramp; check the *settled* portion.
        let amp = 1.0f32;
        let x = sine(amp, 1_000.0, fs, 16_384);
        let mut channels = vec![x.clone()];
        comp.process_block(&mut channels);

        // Threshold linear:
        let thresh_lin = 10.0f32.powf(-6.0 / 20.0); // ≈ 0.501
                                                    // Look at the *tail* (skip the attack overshoot window).
        let tail = &channels[0][8_000..];
        let peak = tail.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // Allow a modest overshoot due to the smoothed envelope (the
        // detector is not zero-attack here).
        assert!(
            peak < thresh_lin * 1.10,
            "limiter peak {} exceeds ceiling {} + 10% margin",
            peak,
            thresh_lin
        );
    }
}
