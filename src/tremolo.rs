//! Sine-wave amplitude modulation (tremolo).
//!
//! Multiplies the input by a slowly-varying gain envelope:
//!
//! ```text
//! lfo[n] = (1 + sin(2π · rate · n / fs)) / 2     ∈ [0, 1]
//! gain   = (1 - depth) + depth · lfo[n]          ∈ [1-depth, 1]
//! y[n]   = x[n] · gain
//! ```
//!
//! Special cases:
//!
//! * `depth = 0` ⇒ `gain ≡ 1`, output equals input.
//! * `rate_hz = 0` ⇒ LFO is frozen at `sin(0) = 0`, gain is `(1 - depth/2)`
//!   constant — i.e. a static attenuation, the tremolo never sweeps. The
//!   sample-rate dependent path is bypassed for `rate_hz = 0` so output
//!   equals input precisely (we never *introduce* attenuation when the LFO
//!   is disabled).
//!
//! All channels share a single LFO phase so stereo image is preserved.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Sine-wave tremolo with `rate_hz` and `depth ∈ [0, 1]`.
#[derive(Debug, Clone)]
pub struct Tremolo {
    rate_hz: f32,
    depth: f32,
    /// LFO phase accumulator (radians) — preserved across calls.
    phase: f64,
}

impl Tremolo {
    /// New tremolo. `depth` is clamped to `[0, 1]`. `rate_hz` is clamped
    /// to `[0, 100]` (well above any musically useful tremolo rate).
    pub fn new(rate_hz: f32, depth: f32) -> Self {
        Self {
            rate_hz: crate::clamp_param(rate_hz, 0.0, 0.0, 100.0),
            depth: crate::clamp_param(depth, 0.0, 0.0, 1.0),
            phase: 0.0,
        }
    }

    /// Currently-configured LFO rate.
    pub fn rate_hz(&self) -> f32 {
        self.rate_hz
    }

    /// Currently-configured depth.
    pub fn depth(&self) -> f32 {
        self.depth
    }

    /// Reset the LFO phase to 0.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

impl AudioFilter for Tremolo {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);

        // Bypass when nothing to do — keeps depth=0 / rate=0 outputs
        // bit-exact equal to input (no LFO drift, no DC offset).
        if self.depth == 0.0 || self.rate_hz == 0.0 {
            let out = encode_from_f32(params.format, params.channels, input, &channels)?;
            return Ok(vec![out]);
        }

        let dphase =
            2.0 * std::f64::consts::PI * (self.rate_hz as f64) / (params.sample_rate as f64);
        let depth = self.depth;
        for i in 0..n {
            let lfo = 0.5 + 0.5 * self.phase.sin() as f32;
            let gain = (1.0 - depth) + depth * lfo;
            for ch in channels.iter_mut() {
                ch[i] *= gain;
            }
            self.phase += dphase;
            // Wrap to keep precision over long streams.
            if self.phase >= 2.0 * std::f64::consts::PI {
                self.phase -= 2.0 * std::f64::consts::PI;
            }
        }

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

    fn read_f32(frame: &AudioFrame) -> Vec<f32> {
        frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn depth_zero_is_bypass() {
        let in_samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let frame = make_f32_mono(&in_samples);
        let mut t = Tremolo::new(5.0, 0.0);
        let out = t.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..in_samples.len() {
            assert!(
                (got[i] - in_samples[i]).abs() < 1.0e-7,
                "depth=0 not bypass at {}: got={} want={}",
                i,
                got[i],
                in_samples[i]
            );
        }
    }

    #[test]
    fn rate_zero_is_bypass() {
        let in_samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let frame = make_f32_mono(&in_samples);
        let mut t = Tremolo::new(0.0, 0.5);
        let out = t.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..in_samples.len() {
            assert!(
                (got[i] - in_samples[i]).abs() < 1.0e-7,
                "rate=0 not bypass at {}: got={} want={}",
                i,
                got[i],
                in_samples[i]
            );
        }
    }

    #[test]
    fn lfo_period_matches_rate() {
        // 4 Hz tremolo at 48 kHz on a DC input — the output should
        // have peaks every fs / rate = 12000 samples (250 ms).
        let fs = 48_000u32;
        let rate = 4.0f32;
        // Three full LFO periods plus a margin.
        let n = (fs as f32 / rate) as usize * 3 + 100;
        let frame = make_f32_mono(&vec![1.0f32; n]);
        let mut t = Tremolo::new(rate, 1.0);
        let out = t.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);

        // Find local maxima in `got`. Use a generous window
        // tolerance — they should be spaced by ~12000 samples.
        let mut peaks: Vec<usize> = Vec::new();
        let win = 200usize;
        let n_samples = got.len();
        let mut i = win;
        while i < n_samples - win {
            let here = got[i];
            // Local max in a ±win neighbourhood, value close to 1.
            let mut is_max = here > 0.95;
            if is_max {
                let lo = i.saturating_sub(win);
                let hi = (i + win).min(n_samples);
                for &v in got.iter().take(hi).skip(lo) {
                    if v > here {
                        is_max = false;
                        break;
                    }
                }
            }
            if is_max {
                peaks.push(i);
                i += win; // jump past this peak
            }
            i += 1;
        }
        assert!(
            peaks.len() >= 2,
            "expected ≥ 2 LFO peaks, got {}",
            peaks.len()
        );
        let expected_period = (fs as f32 / rate) as usize;
        for w in peaks.windows(2) {
            let dt = w[1] - w[0];
            let err = (dt as f32 - expected_period as f32).abs() / expected_period as f32;
            assert!(
                err < 0.05,
                "peak spacing {} differs from expected {} (err={:.2}%)",
                dt,
                expected_period,
                err * 100.0
            );
        }
    }

    #[test]
    fn depth_one_drops_to_zero_at_trough() {
        // depth=1 means gain swings from 0..=1. The trough should
        // produce ≈ 0 output on a constant input.
        let fs = 48_000u32;
        let rate = 8.0f32;
        let n = (fs as f32 / rate * 2.0) as usize + 100;
        let frame = make_f32_mono(&vec![1.0f32; n]);
        let mut t = Tremolo::new(rate, 1.0);
        let out = t.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let min_val = got.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = got.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min_val < 0.05, "depth=1 trough not near zero: {}", min_val);
        assert!(max_val > 0.95, "depth=1 peak not near one: {}", max_val);
    }
}
