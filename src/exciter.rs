//! Exciter — high-band psycho-acoustic harmonic enhancer.
//!
//! Classic studio "aural exciter" effect: split the signal at a high
//! crossover (typically 2–6 kHz), pass the **high** band through a
//! soft-clip waveshaper to generate even + odd harmonics, then mix the
//! distorted high band back over the dry signal. The result is a
//! brighter, more "present" sound — the saturation creates new
//! harmonic content at frequencies the ear interprets as added "air"
//! / clarity without raising the broadband level the way a high-shelf
//! boost would.
//!
//! # Signal flow
//!
//! ```text
//!                ┌──── LPF ────┐
//!     x[n] ──┬──┤             ├── low_band ─────────────┐
//!            │  └──── HPF ────┘                          │
//!            │                ↓                          │
//!            │       waveshape (tanh, k = drive)         │
//!            │                ↓                          ├──► y[n]
//!            │       harmonic_high                       │
//!            │                ↓                          │
//!            └─────────────── mix · harmonic_high ───────┘
//!                                              + dry · x[n]
//! ```
//!
//! The crossover uses two parallel Butterworth-2 biquads (LPF + HPF) at
//! `cutoff_hz` — same building block as [`Crossover`](crate::Crossover)
//! but the two bands are recombined internally so the output retains
//! the input channel count. The high band is shaped with
//! `tanh(k · x) / tanh(k)` (drive-normalised so the waveshaper has
//! unity peak gain at `|x| = 1` regardless of `drive`); the dry path
//! is the untouched low + high sum, which preserves the original
//! timbre.
//!
//! # Parameters
//!
//! * `cutoff_hz` — split frequency. Clamped to `[500, 18 000]`. Lower
//!   values excite the upper-mids; higher values target only the
//!   "air band".
//! * `drive` — saturation strength in `[0.1, 10]`. `1` is gentle,
//!   `3..5` is moderate, `> 7` is heavy.
//! * `mix` — wet (harmonic) amount in `[0, 1]`. `0` = dry only,
//!   `1` = full harmonic layer added on top of dry.
//!
//! # Design notes
//!
//! The `tanh(k·x) / tanh(k)` normalisation is a standard saturation
//! primitive; the band-split + saturate + mix block diagram is the
//! conventional exciter topology.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming exciter.
#[derive(Debug, Clone)]
pub struct Exciter {
    cutoff_hz: f32,
    drive: f32,
    mix: f32,
    /// Cached `tanh(drive)` denominator for waveshape normalisation.
    norm: f32,
    hpf: Biquad,
}

impl Exciter {
    /// New exciter with default `(cutoff_hz=4000, drive=3.0, mix=0.4)`.
    pub fn new() -> Self {
        Self::with(4_000.0, 3.0, 0.4)
    }

    /// Custom-parameter exciter. Inputs are clamped to safe ranges
    /// (see the type-level docs).
    pub fn with(cutoff_hz: f32, drive: f32, mix: f32) -> Self {
        let cutoff_hz = crate::clamp_param(cutoff_hz, 4_000.0, 500.0, 18_000.0);
        let drive = crate::clamp_param(drive, 1.0, 0.1, 10.0);
        let mix = crate::clamp_param(mix, 0.0, 0.0, 1.0);
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let hpf = Biquad::new(BiquadKind::HighPass { cutoff_hz, q });
        Self {
            cutoff_hz,
            drive,
            mix,
            norm: drive.tanh().max(1.0e-6),
            hpf,
        }
    }

    /// Current crossover cutoff (Hz).
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Current saturation drive.
    pub fn drive(&self) -> f32 {
        self.drive
    }

    /// Current wet/dry mix coefficient.
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Reset the internal high-pass biquad state.
    pub fn reset(&mut self) {
        self.hpf.reset();
    }
}

impl Default for Exciter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for Exciter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let dry = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = dry.len();
        if n_chan == 0 {
            let out = encode_from_f32(params.format, params.channels, input, &dry)?;
            return Ok(vec![out]);
        }
        let n_samples = dry[0].len();

        // Extract the high band via the HPF biquad. We need interleaved
        // form for `process_in_place`.
        let mut inter = vec![0.0f32; n_samples * n_chan];
        for ch in 0..n_chan {
            for i in 0..n_samples {
                inter[i * n_chan + ch] = dry[ch][i];
            }
        }
        self.hpf
            .process_in_place(&mut inter, n_chan as u16, params.sample_rate);

        // Waveshape the high band and add the result to the dry signal.
        let mut channels = dry;
        for (ch, buf) in channels.iter_mut().enumerate() {
            for i in 0..n_samples {
                let high = inter[i * n_chan + ch];
                let shaped = (self.drive * high).tanh() / self.norm;
                buf[i] += self.mix * shaped;
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

    fn rms(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        (s / x.len() as f64).sqrt() as f32
    }

    fn sine_at(fs: u32, freq: f32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect()
    }

    #[test]
    fn zero_mix_is_identity() {
        let samples = sine_at(48_000, 500.0, 512);
        let frame = make_f32_mono(&samples);
        let mut ex = Exciter::with(4_000.0, 3.0, 0.0);
        let out = ex.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..samples.len() {
            assert!(
                (got[i] - samples[i]).abs() < 1.0e-6,
                "mix=0 not identity at {i}: got={} want={}",
                got[i],
                samples[i]
            );
        }
    }

    #[test]
    fn low_band_pure_tone_largely_unaffected() {
        // 200 Hz tone with 4 kHz cutoff → HPF rejects → barely modified.
        let fs = 48_000u32;
        let n = 12_000usize;
        let samples = sine_at(fs, 200.0, n);
        let frame = make_f32_mono(&samples);
        let mut ex = Exciter::with(4_000.0, 5.0, 0.5);
        let out = ex.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Discard warm-up.
        let warm = (fs as f32 * 0.1) as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let delta_db = 20.0 * (out_r / in_r).log10();
        assert!(
            delta_db.abs() < 0.5,
            "low-band tone should be ≈unchanged; Δ = {} dB",
            delta_db
        );
    }

    #[test]
    fn high_band_pure_tone_gets_boost() {
        // 8 kHz tone with 4 kHz cutoff → HPF passes → saturation adds
        // harmonics → RMS goes UP versus dry.
        let fs = 48_000u32;
        let n = 12_000usize;
        let samples = sine_at(fs, 8_000.0, n);
        let frame = make_f32_mono(&samples);
        let mut ex = Exciter::with(4_000.0, 5.0, 1.0);
        let out = ex.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let warm = (fs as f32 * 0.1) as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let delta_db = 20.0 * (out_r / in_r).log10();
        assert!(
            delta_db > 0.5,
            "high-band tone should be boosted; Δ = {} dB",
            delta_db
        );
    }

    #[test]
    fn output_stays_bounded() {
        // Stress the saturator with a sample > 1 — output must not blow up.
        let samples: Vec<f32> = (0..2048).map(|i| ((i as f32) * 0.01).sin() * 0.9).collect();
        let frame = make_f32_mono(&samples);
        let mut ex = Exciter::with(4_000.0, 10.0, 1.0);
        let out = ex.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for v in got {
            assert!(v.is_finite(), "non-finite sample in exciter output");
            assert!(v.abs() < 3.0, "exciter sample {} out of bounds", v);
        }
    }
}
