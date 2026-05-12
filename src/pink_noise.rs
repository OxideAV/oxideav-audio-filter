//! Pink-noise generator (1/f spectrum).
//!
//! Implements **Paul Kellet's economy filter approximation** of pink noise:
//! seven cascaded first-order one-pole low-pass sections summed with carefully
//! chosen coefficients to produce a noise floor whose power spectral density
//! falls at approximately −3 dB / octave from ~10 Hz to ~20 kHz. The
//! coefficients below come from the standard published filter recipe — used
//! as math, not as code transcription.
//!
//! # Recurrence
//!
//! Given a white-noise sample `w` per step:
//!
//! ```text
//! b0 = 0.99886 · b0 + w · 0.0555179
//! b1 = 0.99332 · b1 + w · 0.0750759
//! b2 = 0.96900 · b2 + w · 0.1538520
//! b3 = 0.86650 · b3 + w · 0.3104856
//! b4 = 0.55000 · b4 + w · 0.5329522
//! b5 = -0.7616 · b5 - w · 0.0168980
//! pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w · 0.5362
//! b6 = w · 0.115926
//! ```
//!
//! The output is normalised so a unit-amplitude input white-noise stream
//! produces approximately ±1.0 pink output. Final-stage gain `0.11` is folded
//! into the per-channel `amplitude` knob.
//!
//! # Source
//!
//! The white-noise driver is the same splitmix64 PRNG as [`crate::WhiteNoise`].

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Kellet filter state — six recursive one-pole sections plus the
/// terminal feed-through `b6`.
#[derive(Debug, Clone, Copy, Default)]
struct Kellet {
    b0: f32,
    b1: f32,
    b2: f32,
    b3: f32,
    b4: f32,
    b5: f32,
    b6: f32,
}

/// Streaming pink-noise generator.
#[derive(Debug, Clone)]
pub struct PinkNoise {
    amplitude: f32,
    state: u64,
    kellet: Kellet,
    /// Final gain to bring summed Kellet output back to roughly ±1 with
    /// unit-amplitude white input. Empirically ≈ 0.11; see module docs.
    final_gain: f32,
}

impl PinkNoise {
    /// New generator with default seed `0x12345678`.
    /// `amplitude` is clamped to `[0, 1]`.
    pub fn new(amplitude: f32) -> Self {
        Self::with_seed(amplitude, 0x1234_5678)
    }

    /// New generator with an explicit 64-bit seed.
    pub fn with_seed(amplitude: f32, seed: u64) -> Self {
        Self {
            amplitude: amplitude.clamp(0.0, 1.0),
            state: seed.max(1),
            kellet: Kellet::default(),
            final_gain: 0.11,
        }
    }

    /// Currently-configured amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Reset filter state and re-seed the PRNG.
    pub fn reseed(&mut self, seed: u64) {
        self.state = seed.max(1);
        self.kellet = Kellet::default();
    }

    /// Next white sample in `[-1, 1)`.
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

    /// Next pink sample after running the Kellet recurrence on a fresh
    /// white sample. Output is `final_gain · sum_of_sections · amplitude`.
    #[inline]
    fn next_sample(&mut self) -> f32 {
        let w = self.next_white();
        let k = &mut self.kellet;
        k.b0 = 0.99886 * k.b0 + w * 0.055_517_9;
        k.b1 = 0.99332 * k.b1 + w * 0.075_075_9;
        k.b2 = 0.96900 * k.b2 + w * 0.153_852;
        k.b3 = 0.86650 * k.b3 + w * 0.310_485_6;
        k.b4 = 0.55000 * k.b4 + w * 0.532_952_2;
        k.b5 = -0.7616 * k.b5 - w * 0.016_898;
        let pink = k.b0 + k.b1 + k.b2 + k.b3 + k.b4 + k.b5 + k.b6 + w * 0.5362;
        k.b6 = w * 0.115_926;
        pink * self.final_gain * self.amplitude
    }
}

impl AudioFilter for PinkNoise {
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
                // Clamp to keep within stated amplitude bound even on
                // rare outlier samples; Kellet's filter can occasionally
                // exceed 1.0 by a few percent with the 0.11 gain.
                ch.push(s.clamp(-self.amplitude, self.amplitude));
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
        let mut g = PinkNoise::new(0.4);
        let frame = silence_frame(4096);
        let out = g.process(&frame, f32_mono(48_000)).unwrap();
        let s = read_f32(&out[0]);
        for v in &s {
            assert!(v.abs() <= 0.4001, "sample out of range: {v}");
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut a = PinkNoise::with_seed(0.5, 99);
        let mut b = PinkNoise::with_seed(0.5, 99);
        let frame = silence_frame(1024);
        let oa = a.process(&frame, f32_mono(48_000)).unwrap();
        let ob = b.process(&frame, f32_mono(48_000)).unwrap();
        assert_eq!(read_f32(&oa[0]), read_f32(&ob[0]));
    }

    #[test]
    fn spectral_slope_is_minus_three_db_per_octave() {
        // For pink noise the power spectral density falls 3 dB / octave —
        // i.e. power at 2·f is half the power at f. Run a long generator
        // sample, FFT it, average power in two bands an octave apart
        // (200..400 Hz and 1600..3200 Hz at fs=48 kHz with n=4096 →
        // bin width = 11.72 Hz), and verify the slope.
        use crate::fft::real_fft;
        let mut g = PinkNoise::with_seed(1.0, 7);
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
        let p_lo: f64 = power[lo..hi_lo].iter().sum();
        let p_hi: f64 = power[lo2..hi2].iter().sum();
        // Both windows are 200..400 vs 1600..3200 — same bin width
        // ratio (factor of 2 = one octave wider for the upper band)
        // so we normalise by bin count first.
        let p_lo_per_bin = p_lo / (hi_lo - lo) as f64;
        let p_hi_per_bin = p_hi / (hi2 - lo2) as f64;
        // Lower band is 3 octaves below the upper band (200→1600 = 3 oct).
        // Expected power ratio = 2^3 = 8 (i.e. 9 dB).
        let ratio = p_lo_per_bin / p_hi_per_bin;
        // Allow generous tolerance: Kellet is an approximation, the FFT
        // averaging variance is high with finite data.
        assert!(
            (4.0..16.0).contains(&ratio),
            "pink-noise PSD ratio {ratio} not in 4..16 (expected ≈ 8 for −3 dB/oct over 3 oct)"
        );
    }

    #[test]
    fn pink_has_more_low_energy_than_high() {
        // Sanity: pink noise should weight low frequencies more.
        use crate::fft::real_fft;
        let mut g = PinkNoise::with_seed(1.0, 17);
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
        let half = power.len() / 2;
        let low_total: f64 = power[1..half].iter().sum();
        let high_total: f64 = power[half..].iter().sum();
        assert!(
            low_total > high_total,
            "low total {low_total} not > high total {high_total}"
        );
    }
}
