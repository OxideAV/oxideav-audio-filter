//! Brown / red noise generator (1/f² spectrum).
//!
//! Produces a noise stream with a power spectral density that falls at
//! approximately −6 dB / octave — the result of integrating a white-noise
//! source through a leaky one-pole low-pass IIR.
//!
//! # Recurrence
//!
//! Given a fresh white-noise sample `w` per step:
//!
//! ```text
//! y[n] = α · y[n-1] + (1 - α) · w[n]
//! ```
//!
//! with `α ≈ 0.99` for a strongly red-biased spectrum. The output of the
//! integrator is then rescaled to land back in `[-amplitude, +amplitude]` —
//! a fixed `output_gain` of 3.5 (empirical) normalises the RMS of the
//! integrator chain. A hard clamp keeps occasional outliers within the
//! amplitude bound.
//!
//! # Comparison to pink and white noise
//!
//! | Color | PSD slope | Bias    |
//! |-------|-----------|---------|
//! | white | 0 dB/oct  | flat    |
//! | pink  | −3 dB/oct | mid-low |
//! | brown | −6 dB/oct | bass    |
//!
//! Brown noise sounds rumbly / sub-bass-heavy compared to pink.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming brown-noise (1/f²) generator.
#[derive(Debug, Clone)]
pub struct BrownNoise {
    amplitude: f32,
    alpha: f32,
    output_gain: f32,
    state: u64,
    integrator: f32,
}

impl BrownNoise {
    /// New generator with default seed `0x12345678` and `α = 0.99`.
    /// `amplitude` is clamped to `[0, 1]`.
    pub fn new(amplitude: f32) -> Self {
        Self::with_seed(amplitude, 0x1234_5678)
    }

    /// New generator with an explicit 64-bit seed.
    pub fn with_seed(amplitude: f32, seed: u64) -> Self {
        Self {
            amplitude: amplitude.clamp(0.0, 1.0),
            alpha: 0.99,
            output_gain: 3.5,
            state: seed.max(1),
            integrator: 0.0,
        }
    }

    /// Currently-configured amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Currently-configured integrator pole.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Reset filter state and re-seed the PRNG.
    pub fn reseed(&mut self, seed: u64) {
        self.state = seed.max(1);
        self.integrator = 0.0;
    }

    #[inline]
    fn next_white(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let u = (z >> 11) as f64 / (1u64 << 53) as f64;
        (2.0 * u - 1.0) as f32
    }

    #[inline]
    fn next_sample(&mut self) -> f32 {
        let w = self.next_white();
        self.integrator = self.alpha * self.integrator + (1.0 - self.alpha) * w;
        (self.integrator * self.output_gain * self.amplitude).clamp(-self.amplitude, self.amplitude)
    }
}

impl AudioFilter for BrownNoise {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);
        let n_chan = params.channels as usize;
        let mut out_channels: Vec<Vec<f32>> = (0..n_chan).map(|_| Vec::with_capacity(n)).collect();
        for _ in 0..n {
            let s = self.next_sample();
            for ch in out_channels.iter_mut() {
                ch.push(s);
            }
        }
        let out = encode_from_f32(params.format, params.channels, input, &out_channels)?;
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

    fn silence_frame(n: usize) -> AudioFrame {
        AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![vec![0u8; n * 4]],
        }
    }

    fn read_f32(frame: &AudioFrame) -> Vec<f32> {
        frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn output_bounded_by_amplitude() {
        let mut g = BrownNoise::new(0.5);
        let frame = silence_frame(4096);
        let out = g.process(&frame, f32_mono(48_000)).unwrap();
        let s = read_f32(&out[0]);
        for v in &s {
            assert!(v.abs() <= 0.5001, "sample out of range: {v}");
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut a = BrownNoise::with_seed(0.5, 77);
        let mut b = BrownNoise::with_seed(0.5, 77);
        let frame = silence_frame(1024);
        let oa = a.process(&frame, f32_mono(48_000)).unwrap();
        let ob = b.process(&frame, f32_mono(48_000)).unwrap();
        assert_eq!(read_f32(&oa[0]), read_f32(&ob[0]));
    }

    #[test]
    fn spectral_slope_steeper_than_pink() {
        // Brown noise PSD falls ≈ 6 dB/oct → 4× steeper than pink.
        // Power at 200..400 Hz vs 1600..3200 Hz: ratio of per-bin power
        // should be ≈ 2^(2·3) = 64 (three octaves apart, 6 dB/oct).
        use crate::fft::real_fft;
        let mut g = BrownNoise::with_seed(1.0, 33);
        let frame = silence_frame(8 * 4096);
        let out = g.process(&frame, f32_mono(48_000)).unwrap();
        let s = read_f32(&out[0]);
        let n = 4096usize;
        let mut power = vec![0.0f64; n / 2 + 1];
        let mut chunks = 0;
        for chunk in s.chunks_exact(n) {
            let bins = real_fft(chunk);
            for (i, b) in bins.iter().enumerate() {
                power[i] += (b.magnitude() as f64).powi(2);
            }
            chunks += 1;
        }
        assert!(chunks >= 4);
        let bin_hz = 48_000.0f64 / n as f64;
        let lo = (200.0 / bin_hz) as usize;
        let hi_lo = (400.0 / bin_hz) as usize;
        let lo2 = (1600.0 / bin_hz) as usize;
        let hi2 = (3200.0 / bin_hz) as usize;
        let p_lo_per_bin: f64 = power[lo..hi_lo].iter().sum::<f64>() / (hi_lo - lo) as f64;
        let p_hi_per_bin: f64 = power[lo2..hi2].iter().sum::<f64>() / (hi2 - lo2) as f64;
        let ratio = p_lo_per_bin / p_hi_per_bin;
        // Generous tolerance: leaky-integrator + finite α + FFT variance.
        // Pink gives ~8; brown must clearly exceed it.
        assert!(
            ratio > 16.0,
            "brown PSD ratio {ratio} not steeper than pink (must be > 16)"
        );
    }

    #[test]
    fn brown_low_band_dominates() {
        use crate::fft::real_fft;
        let mut g = BrownNoise::with_seed(1.0, 51);
        let frame = silence_frame(4 * 4096);
        let out = g.process(&frame, f32_mono(48_000)).unwrap();
        let s = read_f32(&out[0]);
        let n = 4096usize;
        let mut power = vec![0.0f64; n / 2 + 1];
        for chunk in s.chunks_exact(n) {
            let bins = real_fft(chunk);
            for (i, b) in bins.iter().enumerate() {
                power[i] += (b.magnitude() as f64).powi(2);
            }
        }
        let half = power.len() / 4;
        let low_total: f64 = power[1..half].iter().sum();
        let high_total: f64 = power[half..].iter().sum();
        assert!(
            low_total > 10.0 * high_total,
            "brown low/high ratio not dominant: low={low_total} high={high_total}"
        );
    }
}
