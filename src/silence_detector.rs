//! Silence detector — observes signal RMS and reports `silent / loud`
//! transitions with hysteresis.
//!
//! The filter is **observation-only**: it passes the input through unchanged
//! while updating an internal one-pole RMS estimator. Callers can query
//! [`SilenceDetector::is_silent`] / [`SilenceDetector::current_db`] after each
//! `process()` to learn the current state.
//!
//! # Recurrence
//!
//! Per channel, the squared-sample mean is tracked with a leaky integrator:
//!
//! ```text
//! attack  = 1 - exp(-2.2 / (attack_ms · 1e-3 · fs))
//! release = 1 - exp(-2.2 / (release_ms · 1e-3 · fs))
//! sq      = x · x
//! coeff   = attack  if sq > env else release
//! env     = (1 - coeff) · env + coeff · sq
//! ```
//!
//! After mixing all channels' env-sums:
//!
//! ```text
//! rms_dbfs = 10 · log10(sum_env / n_channels)
//! ```
//!
//! State transition (with hysteresis):
//!
//! ```text
//! if rms_dbfs < threshold_dbfs:
//!     elapsed_below_ms += frame_duration
//!     if elapsed_below_ms >= hold_ms:
//!         is_silent = true
//! else:
//!     elapsed_below_ms = 0
//!     is_silent = false
//! ```
//!
//! # Parameters
//!
//! * `threshold_dbfs` — typical −40 to −60 dBFS. Default −60.
//! * `hold_ms` — how long RMS must stay below threshold before flipping
//!   `is_silent = true`. Default 100 ms.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming silence detector.
#[derive(Debug, Clone)]
pub struct SilenceDetector {
    threshold_dbfs: f32,
    hold_ms: f32,
    attack_ms: f32,
    release_ms: f32,
    state: Option<DetectorState>,
}

#[derive(Debug, Clone)]
struct DetectorState {
    sample_rate: u32,
    /// One squared-sample envelope per channel.
    env: Vec<f32>,
    /// Coalesced "below threshold for this long" counter, milliseconds.
    elapsed_below_ms: f32,
    /// Latest evaluated state.
    is_silent: bool,
    /// Latest summed-channel RMS in dBFS (after process()).
    last_db: f32,
}

impl SilenceDetector {
    /// New detector with default 100 ms hold + 5 ms / 50 ms env times.
    pub fn new(threshold_dbfs: f32, hold_ms: f32) -> Self {
        Self {
            threshold_dbfs,
            hold_ms: hold_ms.max(0.0),
            attack_ms: 5.0,
            release_ms: 50.0,
            state: None,
        }
    }

    /// New detector with explicit envelope times.
    pub fn with_env(threshold_dbfs: f32, hold_ms: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            threshold_dbfs,
            hold_ms: hold_ms.max(0.0),
            attack_ms: attack_ms.max(0.01),
            release_ms: release_ms.max(0.01),
            state: None,
        }
    }

    /// `true` if the most recent process() left the detector in the silent
    /// state. Returns `false` before any sample has been observed.
    pub fn is_silent(&self) -> bool {
        self.state.as_ref().map(|s| s.is_silent).unwrap_or(false)
    }

    /// Most recent estimated RMS in dBFS, summed across channels.
    pub fn current_db(&self) -> f32 {
        self.state.as_ref().map(|s| s.last_db).unwrap_or(-120.0)
    }

    /// Reset internal state. After this the detector reports
    /// `is_silent = false` until enough below-threshold material has
    /// passed to re-arm the silence flag.
    pub fn reset(&mut self) {
        self.state = None;
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.env.len() != channels,
            None => true,
        };
        if needs_rebuild {
            self.state = Some(DetectorState {
                sample_rate,
                env: vec![0.0; channels],
                elapsed_below_ms: 0.0,
                is_silent: false,
                last_db: -120.0,
            });
        }
    }
}

impl AudioFilter for SilenceDetector {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let n_chan = params.channels as usize;
        self.ensure_state(params.sample_rate, n_chan);
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n_samples = channels.first().map(|c| c.len()).unwrap_or(0);

        let state = self.state.as_mut().expect("state ensured above");
        let fs = state.sample_rate as f32;
        // 2.2 ≈ ln(10^0.95) → reach 95 % of target in attack/release_ms.
        let attack_coeff = 1.0 - (-2.2 / (self.attack_ms.max(0.01) * 1e-3 * fs)).exp();
        let release_coeff = 1.0 - (-2.2 / (self.release_ms.max(0.01) * 1e-3 * fs)).exp();

        for (ch_idx, buf) in channels.iter().enumerate().take(n_chan) {
            let env = &mut state.env[ch_idx];
            for &x in buf.iter().take(n_samples) {
                let sq = x * x;
                let coeff = if sq > *env {
                    attack_coeff
                } else {
                    release_coeff
                };
                *env = (1.0 - coeff) * *env + coeff * sq;
            }
        }
        // Average envelopes across channels; convert to dBFS.
        let sum_env: f32 = state.env.iter().sum();
        let mean_env = (sum_env / n_chan as f32).max(1e-12);
        let db = 10.0 * mean_env.log10();
        state.last_db = db;

        let frame_ms = (n_samples as f32 / fs) * 1000.0;
        if db < self.threshold_dbfs {
            state.elapsed_below_ms += frame_ms;
            if state.elapsed_below_ms >= self.hold_ms {
                state.is_silent = true;
            }
        } else {
            state.elapsed_below_ms = 0.0;
            state.is_silent = false;
        }

        // Pass-through: re-encode input unchanged.
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
    fn pure_silence_reads_silent_after_hold() {
        let fs = 48_000u32;
        let mut det = SilenceDetector::new(-60.0, 50.0);
        // 100 ms of silence (4800 samples).
        let frame = make_f32_mono(&vec![0.0f32; 4800]);
        // First half — should fully accumulate the hold time.
        det.process(&frame, f32_mono(fs)).unwrap();
        assert!(
            det.is_silent(),
            "silence not detected after 100 ms of zeros"
        );
    }

    #[test]
    fn loud_signal_reads_not_silent() {
        let fs = 48_000u32;
        let mut det = SilenceDetector::new(-40.0, 50.0);
        let in_samples: Vec<f32> = (0..4800)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let frame = make_f32_mono(&in_samples);
        det.process(&frame, f32_mono(fs)).unwrap();
        assert!(!det.is_silent(), "loud sine misclassified as silent");
        // Current RMS for a 0.5-amplitude sine = 0.5/√2 ≈ 0.354 → ~ −9 dBFS.
        let db = det.current_db();
        assert!(
            (-12.0..-6.0).contains(&db),
            "expected ~ −9 dB for half-amp sine; got {db}"
        );
    }

    #[test]
    fn pass_through_is_identity() {
        let fs = 48_000u32;
        let mut det = SilenceDetector::new(-60.0, 50.0);
        let in_samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin() * 0.4).collect();
        let frame = make_f32_mono(&in_samples);
        let out = det.process(&frame, f32_mono(fs)).unwrap();
        let got: Vec<f32> = out[0].data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for i in 0..in_samples.len() {
            assert!(
                (got[i] - in_samples[i]).abs() < 1.0e-6,
                "pass-through differs at {i}: got={} want={}",
                got[i],
                in_samples[i]
            );
        }
    }

    #[test]
    fn loud_then_silent_transitions_after_hold() {
        let fs = 48_000u32;
        let mut det = SilenceDetector::new(-40.0, 30.0);
        // Stage 1: loud sine — should leave is_silent=false.
        let loud: Vec<f32> = (0..4800)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let loud_frame = make_f32_mono(&loud);
        det.process(&loud_frame, f32_mono(fs)).unwrap();
        assert!(!det.is_silent());
        // Stage 2: zeros — after ≥ hold_ms (30 ms = 1440 samples) the
        // detector should switch to silent. Feed 50 ms of zeros to be
        // safely past the envelope release.
        let mut zeros = vec![0.0f32; 4800]; // 100 ms
                                            // Decay envelope. Feed enough zero-frames for the envelope
                                            // (release_ms = 50 ms) plus hold to settle.
        for _ in 0..5 {
            let frame = make_f32_mono(&zeros);
            det.process(&frame, f32_mono(fs)).unwrap();
            zeros.fill(0.0);
        }
        assert!(
            det.is_silent(),
            "did not switch to silent after sustained zeros; db={}",
            det.current_db()
        );
    }
}
