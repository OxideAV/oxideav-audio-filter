//! Amplitude-envelope follower (side-chain detector).
//!
//! The filter tracks the rectified-and-smoothed amplitude of the input
//! signal so downstream stages (or external consumers) can sidechain on
//! it. The audio passes through unchanged — listeners read the current
//! envelope via [`EnvelopeFollower::current`] / [`current_db`] after
//! each `process()` call.
//!
//! # Detector
//!
//! The detector is a classical one-pole peak follower applied to the
//! per-channel absolute value, with separate attack/release coefficients:
//!
//! ```text
//! drive = max(|x_0[n]|, |x_1[n]|, …)          (peak link across channels)
//! coeff = α_atk    if drive > env
//!         α_rel    otherwise
//! env  ← (1 - coeff) · env + coeff · drive
//! ```
//!
//! The coefficients follow the textbook exp-mapping
//!
//! ```text
//! α = 1 - exp(-1 / (τ · f_s))
//! ```
//!
//! where `τ` is `attack_ms` / `release_ms` (converted to seconds). The
//! result is `1 - 1/e` (≈ 63 %) of the target value reached after one
//! time constant, matching most analogue compressor/follower spec
//! sheets.
//!
//! # Modes
//!
//! [`EnvelopeMode::Peak`] (default) tracks `max |x|` per sample.
//! [`EnvelopeMode::Rms`] runs the same one-pole filter on `x²` and
//! reports `sqrt(env_sq)` so the user-visible envelope reads in
//! amplitude (RMS) rather than power (RMS²).
//!
//! # Parameters
//!
//! * `attack_ms` — rise time, default 5 ms. Clamped ≥ 0.01 ms.
//! * `release_ms` — fall time, default 50 ms. Clamped ≥ 0.01 ms.
//! * `mode` — `Peak` (default) or `Rms`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Detector mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeMode {
    /// Peak detector — `env ← LP(|x|)`.
    Peak,
    /// RMS detector — `env ← sqrt(LP(x²))`.
    Rms,
}

/// Streaming amplitude-envelope follower.
#[derive(Debug, Clone)]
pub struct EnvelopeFollower {
    attack_ms: f32,
    release_ms: f32,
    mode: EnvelopeMode,
    state: Option<DetectorState>,
}

#[derive(Debug, Clone, Copy)]
struct DetectorState {
    sample_rate: u32,
    /// Internal smoothed estimate (`|x|` for Peak, `x²` for Rms).
    env: f32,
    /// Last externally-visible value after `process()`.
    last: f32,
}

impl EnvelopeFollower {
    /// New peak follower with default 5 ms / 50 ms attack/release.
    pub fn new(attack_ms: f32, release_ms: f32) -> Self {
        Self {
            attack_ms: attack_ms.max(0.01),
            release_ms: release_ms.max(0.01),
            mode: EnvelopeMode::Peak,
            state: None,
        }
    }

    /// New follower with explicit mode (`Peak` vs `Rms`).
    pub fn with_mode(attack_ms: f32, release_ms: f32, mode: EnvelopeMode) -> Self {
        Self {
            attack_ms: attack_ms.max(0.01),
            release_ms: release_ms.max(0.01),
            mode,
            state: None,
        }
    }

    /// Current detector mode.
    pub fn mode(&self) -> EnvelopeMode {
        self.mode
    }

    /// Latest amplitude estimate. `0.0` before any sample has been
    /// observed. For `Peak` mode this is `LP(|x|)`; for `Rms` mode
    /// this is `sqrt(LP(x²))`.
    pub fn current(&self) -> f32 {
        self.state.as_ref().map(|s| s.last).unwrap_or(0.0)
    }

    /// Latest envelope in dBFS (`20·log10(current)`, with a floor of
    /// −120 dB to keep the value finite when the signal is silent).
    pub fn current_db(&self) -> f32 {
        20.0 * self.current().max(1e-6).log10()
    }

    /// Reset internal state. After this `current()` returns 0 until
    /// the next sample arrives.
    pub fn reset(&mut self) {
        self.state = None;
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if rebuild {
            self.state = Some(DetectorState {
                sample_rate,
                env: 0.0,
                last: 0.0,
            });
        }
    }
}

impl AudioFilter for EnvelopeFollower {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate);
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);

        let state = self.state.as_mut().expect("state ensured above");
        let fs = state.sample_rate as f32;
        let alpha_atk = 1.0 - (-1.0 / (self.attack_ms * 1e-3 * fs)).exp();
        let alpha_rel = 1.0 - (-1.0 / (self.release_ms * 1e-3 * fs)).exp();

        for i in 0..n {
            // Peak-link across channels.
            let mut drive: f32 = 0.0;
            for ch in &channels {
                let v = ch[i];
                let m = match self.mode {
                    EnvelopeMode::Peak => v.abs(),
                    EnvelopeMode::Rms => v * v,
                };
                if m > drive {
                    drive = m;
                }
            }
            let coeff = if drive > state.env {
                alpha_atk
            } else {
                alpha_rel
            };
            state.env = (1.0 - coeff) * state.env + coeff * drive;
        }
        state.last = match self.mode {
            EnvelopeMode::Peak => state.env,
            EnvelopeMode::Rms => state.env.sqrt(),
        };

        // Pass-through: re-encode the decoded channels unchanged.
        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::SampleFormat;

    fn f32_mono(rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 1,
            sample_rate: rate,
        }
    }

    fn make_f32_mono(samples: &[f32]) -> AudioFrame {
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        AudioFrame {
            samples: samples.len() as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    #[test]
    fn peak_envelope_approaches_dc_level() {
        // Constant 0.5 input — peak detector should converge to 0.5
        // after a handful of attack-time constants. With 5 ms / 48 kHz,
        // 100 ms is 20+ τ.
        let fs = 48_000u32;
        let frame = make_f32_mono(&vec![0.5f32; 4800]);
        let mut det = EnvelopeFollower::new(5.0, 50.0);
        det.process(&frame, f32_mono(fs)).unwrap();
        let env = det.current();
        assert!(
            (env - 0.5).abs() < 0.01,
            "peak envelope of DC=0.5 expected ≈ 0.5, got {env}"
        );
    }

    #[test]
    fn release_drops_envelope_to_zero() {
        let fs = 48_000u32;
        let mut det = EnvelopeFollower::new(5.0, 20.0);
        // Charge to 0.5.
        det.process(&make_f32_mono(&vec![0.5f32; 4800]), f32_mono(fs))
            .unwrap();
        assert!((det.current() - 0.5).abs() < 0.01);
        // Then 200 ms of zeros (release τ = 20 ms → 10 τ).
        det.process(&make_f32_mono(&vec![0.0f32; 9600]), f32_mono(fs))
            .unwrap();
        let env = det.current();
        assert!(env < 0.01, "envelope did not decay; got {env}");
    }

    #[test]
    fn rms_mode_reports_rms_of_sine() {
        // A 0.5-amplitude sine has RMS = 0.5/√2 ≈ 0.3536.
        let fs = 48_000u32;
        let n = 4800usize; // 100 ms
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let mut det = EnvelopeFollower::with_mode(20.0, 20.0, EnvelopeMode::Rms);
        // Feed several frames so the slow attack on x² has settled.
        for _ in 0..4 {
            det.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        }
        let env = det.current();
        assert!(
            (env - 0.3536).abs() < 0.05,
            "RMS of half-amp sine ≈ 0.3536, got {env}"
        );
    }

    #[test]
    fn current_db_floor_finite() {
        let det = EnvelopeFollower::new(5.0, 50.0);
        let db = det.current_db();
        assert!(db.is_finite(), "current_db must be finite; got {db}");
        assert!(db <= -100.0, "fresh detector should read very quiet");
    }
}
