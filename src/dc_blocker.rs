//! DC blocker — first-order IIR high-pass at sub-audible cutoff.
//!
//! Removes any constant offset and very-low-frequency drift from an audio
//! stream while leaving the audio band untouched.
//!
//! # Recurrence
//!
//! Per channel:
//!
//! ```text
//! y[n] = x[n] - x[n-1] + R · y[n-1]
//! ```
//!
//! where `R ∈ [0, 1)` is the pole radius. With `R` close to (but slightly
//! less than) `1.0` the filter has a single zero at DC and a pole just
//! inside the unit circle on the positive real axis, giving an extremely
//! narrow notch at 0 Hz with essentially flat response everywhere else.
//!
//! # Cutoff selection
//!
//! For a given pole radius `R`, the −3 dB cutoff (in radians per sample)
//! satisfies `R = 1 - π · f_c / f_s` to first order. Concretely the
//! default `R = 0.995` at `f_s = 48 kHz` puts the cutoff at roughly
//! `(1 - 0.995) · f_s / π ≈ 76 Hz`, but the response above ~5 Hz is
//! essentially flat (≤ 0.1 dB attenuation) so for *DC blocking* (as
//! opposed to a true HPF) the audible band is left alone. A user-supplied
//! `pole` knob lets callers trade audio-band ripple for DC-rejection
//! settling time.
//!
//! # Per-channel state
//!
//! Each channel keeps its own `(x[n-1], y[n-1])` history pair so stereo
//! input does not cross-talk through the filter.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Per-channel single-pole DC-blocker state.
#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    x_prev: f32,
    y_prev: f32,
}

/// Streaming DC blocker.
#[derive(Debug, Clone)]
pub struct DcBlocker {
    pole: f32,
    state: Vec<ChState>,
}

impl DcBlocker {
    /// Default pole `R = 0.995`. At 48 kHz this gives a ~5–10 Hz effective
    /// settling cutoff with negligible audio-band attenuation.
    pub fn new() -> Self {
        Self {
            pole: 0.995,
            state: Vec::new(),
        }
    }

    /// Custom pole radius `R` (clamped to `[0, 0.99999]`). Higher `R`
    /// → narrower notch, longer DC-decay time. `R = 0.95` gives faster
    /// decay (a few ms) but a wider HPF skirt; `R = 0.999` gives a
    /// near-pure DC blocker with seconds of decay.
    pub fn with_pole(pole: f32) -> Self {
        Self {
            pole: pole.clamp(0.0, 0.99999),
            state: Vec::new(),
        }
    }

    /// Currently-configured pole radius.
    pub fn pole(&self) -> f32 {
        self.pole
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.state.len() != channels {
            self.state = vec![ChState::default(); channels];
        }
    }

    /// Reset per-channel histories to zero.
    pub fn reset(&mut self) {
        for st in self.state.iter_mut() {
            *st = ChState::default();
        }
    }
}

impl Default for DcBlocker {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for DcBlocker {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(channels.len());
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.state[ch_idx];
            for s in buf.iter_mut() {
                let x = *s;
                let y = x - st.x_prev + self.pole * st.y_prev;
                st.x_prev = x;
                st.y_prev = y;
                *s = y;
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
    fn dc_input_decays_to_zero() {
        // Constant 0.5 input — the DC blocker must drive output toward 0.
        // 200 ms at 48 kHz = 9600 samples is well past the settling time
        // for R = 0.995 (~30 ms time constant).
        let frame = make_f32_mono(&vec![0.5f32; 9_600]);
        let mut bl = DcBlocker::new();
        let out = bl.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[got.len() - 100..];
        let peak = tail.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(peak < 0.01, "DC residual peak after 200 ms = {}", peak);
    }

    #[test]
    fn audio_band_sine_passes_through() {
        // 1 kHz sine should pass with ≤ 0.1 dB loss.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let n = 16_384;
        let w = 2.0 * std::f32::consts::PI * f / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut bl = DcBlocker::new();
        let out = bl.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip the first ~2 ms transient.
        let warm = (fs as f32 * 0.005) as usize;
        let in_rms = {
            let s: f64 = samples[warm..]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum();
            (s / (samples.len() - warm) as f64).sqrt()
        };
        let out_rms = {
            let s: f64 = got[warm..].iter().map(|&v| (v as f64) * (v as f64)).sum();
            (s / (got.len() - warm) as f64).sqrt()
        };
        let ratio = out_rms / in_rms;
        let g_db = 20.0 * ratio.log10();
        assert!(
            g_db.abs() < 0.1,
            "1 kHz pass-through gain = {} dB (expected ≈ 0)",
            g_db
        );
    }

    #[test]
    fn channels_do_not_cross_talk() {
        // Stereo: L = constant DC, R = silence. After settling R must
        // remain silence (left-channel DC must not leak through state).
        let fs = 48_000u32;
        let n = 9_600usize;
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&0.5f32.to_le_bytes()); // L
            bytes.extend_from_slice(&0.0f32.to_le_bytes()); // R
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut bl = DcBlocker::new();
        let out = bl
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 2,
                    sample_rate: fs,
                },
            )
            .unwrap();
        // De-interleave R channel.
        let bytes = &out[0].data[0];
        let mut r_peak = 0.0f32;
        for s in 0..n {
            let off = (s * 2 + 1) * 4;
            let v =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            r_peak = r_peak.max(v.abs());
        }
        assert!(r_peak < 1.0e-6, "R-channel polluted, peak = {}", r_peak);
    }

    #[test]
    fn custom_pole_clamped() {
        let bl = DcBlocker::with_pole(2.0);
        assert!(bl.pole() < 1.0);
        let bl = DcBlocker::with_pole(-1.0);
        assert_eq!(bl.pole(), 0.0);
    }
}
