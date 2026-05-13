//! Hum filter — narrow notches at mains-frequency fundamentals and
//! harmonics (50 Hz or 60 Hz).
//!
//! Cascaded series of [`Biquad`](crate::Biquad) notch sections placed at
//! `fundamental_hz · k` for `k = 1, 2, …, n_harmonics`. Each notch is
//! narrow (high `Q`) so audio just outside the line-hum bands passes
//! through with minimal coloration.
//!
//! # Why series?
//!
//! Hum is normally fundamental + odd-harmonic-dominated (transformer
//! saturation, ground loops, half-wave-rectified line noise). A
//! single notch on the fundamental leaves the harmonics audible; a
//! cascade pulls them all down by the same factor.
//!
//! Each section has its own per-channel state, so stereo input does
//! not cross-talk through the filter.
//!
//! # Parameters
//!
//! * `fundamental_hz` — line frequency, typically 50 (EU/Asia/AU) or
//!   60 (US/JP-east) Hz. Clamped to `[10, 200]`.
//! * `q` — notch sharpness for each section. Higher `Q` → narrower
//!   notch → less colouration of nearby audio, but slower transient
//!   response. Clamped to `[1, 100]`. Default ≈ 30 gives a ~2 Hz
//!   notch bandwidth at 60 Hz.
//! * `n_harmonics` — number of multiples to suppress (`1 = fundamental
//!   only`, `5` covers 60/120/180/240/300 Hz). Clamped to `[1, 16]`.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming line-hum suppression filter.
#[derive(Debug, Clone)]
pub struct HumFilter {
    fundamental_hz: f32,
    q: f32,
    n_harmonics: u8,
    /// One Biquad notch per harmonic.
    sections: Vec<Biquad>,
}

impl HumFilter {
    /// New hum filter. `fundamental_hz` clamped to `[10, 200]`, `q` to
    /// `[1, 100]`, `n_harmonics` to `[1, 16]`.
    pub fn new(fundamental_hz: f32, q: f32, n_harmonics: u8) -> Self {
        let fundamental_hz = fundamental_hz.clamp(10.0, 200.0);
        let q = q.clamp(1.0, 100.0);
        let n_harmonics = n_harmonics.clamp(1, 16);
        let sections = (1..=n_harmonics)
            .map(|k| {
                Biquad::new(BiquadKind::Notch {
                    center_hz: fundamental_hz * k as f32,
                    q,
                })
            })
            .collect();
        Self {
            fundamental_hz,
            q,
            n_harmonics,
            sections,
        }
    }

    /// Convenience: 50 Hz mains (EU/Asia/AU). `q = 60`, 5 harmonics.
    pub fn eu_50() -> Self {
        Self::new(50.0, 60.0, 5)
    }

    /// Convenience: 60 Hz mains (US/JP-east). `q = 60`, 5 harmonics.
    pub fn us_60() -> Self {
        Self::new(60.0, 60.0, 5)
    }

    /// Currently-configured fundamental frequency.
    pub fn fundamental_hz(&self) -> f32 {
        self.fundamental_hz
    }

    /// Currently-configured notch Q (shared across sections).
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Currently-configured harmonic count.
    pub fn n_harmonics(&self) -> u8 {
        self.n_harmonics
    }

    /// Reset all section states to zero.
    pub fn reset(&mut self) {
        for s in self.sections.iter_mut() {
            s.reset();
        }
    }
}

impl AudioFilter for HumFilter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = params.channels;
        let fs = params.sample_rate;
        // Skip harmonics that lie above Nyquist — they're unreachable as
        // analog frequencies anyway and would cause coefficient blow-up.
        let nyq = fs as f32 / 2.0;
        // Interleave channels for the section's `process_in_place` helper.
        let n_samples = channels.first().map(|c| c.len()).unwrap_or(0);
        let mut inter = vec![0.0f32; n_samples * n_chan as usize];
        for (ch_idx, buf) in channels.iter().enumerate() {
            for (i, &v) in buf.iter().enumerate() {
                inter[i * n_chan as usize + ch_idx] = v;
            }
        }
        for (k, sec) in self.sections.iter_mut().enumerate() {
            let kk = k + 1;
            let centre = self.fundamental_hz * kk as f32;
            if centre >= nyq {
                continue;
            }
            sec.process_in_place(&mut inter, n_chan, fs);
        }
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            for (i, v) in buf.iter_mut().enumerate() {
                *v = inter[i * n_chan as usize + ch_idx];
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

    fn rms(samples: &[f32]) -> f32 {
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    fn sine_at(fs: u32, freq: f32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| (i as f32 * w).sin()).collect()
    }

    #[test]
    fn fundamental_60_hz_is_attenuated() {
        // 60 Hz tone at 48 kHz fed into a US-60 filter should be cut by
        // at least 15 dB after the IIR settles. (Discrete IIR notch
        // attenuation at fundamental is limited by Q, sample rate, and
        // settling time of the recurrence.)
        let fs = 48_000u32;
        let n = (fs as usize) * 2; // 2 seconds
        let samples = sine_at(fs, 60.0, n);
        let frame = make_f32_mono(&samples);
        let mut hf = HumFilter::us_60();
        let out = hf.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip 1 s warm-up.
        let warm = fs as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let g_db = 20.0 * (out_r / in_r).log10();
        assert!(
            g_db < -15.0,
            "60 Hz tone not attenuated (gain = {} dB)",
            g_db
        );
    }

    #[test]
    fn harmonic_180_hz_is_attenuated() {
        // 3rd harmonic of 60 Hz mains; default n_harmonics=5 covers it.
        let fs = 48_000u32;
        let n = (fs as usize) * 2;
        let samples = sine_at(fs, 180.0, n);
        let frame = make_f32_mono(&samples);
        let mut hf = HumFilter::us_60();
        let out = hf.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let warm = fs as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let g_db = 20.0 * (out_r / in_r).log10();
        assert!(g_db < -15.0, "180 Hz harmonic not attenuated: {}", g_db);
    }

    #[test]
    fn audio_band_1khz_passes_through() {
        // 1 kHz is far from any notch — should pass with < 1 dB loss.
        let fs = 48_000u32;
        let n = 48_000usize;
        let samples = sine_at(fs, 1_000.0, n);
        let frame = make_f32_mono(&samples);
        let mut hf = HumFilter::us_60();
        let out = hf.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let g_db = 20.0 * (out_r / in_r).log10();
        assert!(
            g_db.abs() < 1.0,
            "1 kHz pass-through gain too far from 0 dB: {} dB",
            g_db
        );
    }

    #[test]
    fn eu_50_attenuates_50_hz() {
        // Use a 2-second signal so the IIR notch fully settles at the
        // low fundamental (50 Hz at fs=48k → ω ≈ 6.5e-3, settling is
        // slow). Default Q=60 / 5 harmonics.
        let fs = 48_000u32;
        let n = (fs as usize) * 2;
        let samples = sine_at(fs, 50.0, n);
        let frame = make_f32_mono(&samples);
        let mut hf = HumFilter::eu_50();
        let out = hf.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip 1 s of warmup.
        let warm = fs as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let g_db = 20.0 * (out_r / in_r).log10();
        assert!(g_db < -15.0, "50 Hz mains not attenuated: {} dB", g_db);
    }
}
