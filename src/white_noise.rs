//! White-noise generator (flat-spectrum random source).
//!
//! Replaces the input frame's samples with uniformly-distributed pseudo-random
//! values in `[-amplitude, +amplitude]`. The input frame's PTS and sample count
//! are preserved so [`WhiteNoise`] can act as a drop-in tone replacement inside
//! an existing filter graph (e.g. for silence-padding, test signal injection,
//! or A/B perceptual comparison).
//!
//! # Algorithm
//!
//! A 64-bit `splitmix64` PRNG drives the per-sample uniform output:
//!
//! ```text
//! z      = state += 0x9E37_79B9_7F4A_7C15
//! z     ^= z >> 30; z *= 0xBF58_476D_1CE4_E5B9
//! z     ^= z >> 27; z *= 0x94D0_49BB_1331_11EB
//! z     ^= z >> 31
//! u      = (z >> 11) / 2^53                        ∈ [0, 1)
//! sample = (2·u - 1) · amplitude                   ∈ [-A, +A)
//! ```
//!
//! The constants are the published splitmix64 mix constants. State is held
//! across `process` calls so the generator stream is continuous across frame
//! boundaries.
//!
//! # Determinism
//!
//! With a fixed `seed`, the entire emitted sequence is reproducible — useful
//! for unit tests, deterministic mixing, and a/b regression checks. Omitting
//! the seed (or [`WhiteNoise::new`]) uses the default seed `0x12345678`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming white-noise generator.
#[derive(Debug, Clone)]
pub struct WhiteNoise {
    amplitude: f32,
    state: u64,
}

impl WhiteNoise {
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
        }
    }

    /// Currently-configured amplitude.
    pub fn amplitude(&self) -> f32 {
        self.amplitude
    }

    /// Re-seed the generator. Useful for restarting a deterministic stream.
    pub fn reseed(&mut self, seed: u64) {
        self.state = seed.max(1);
    }

    /// Next uniform sample in `[-amplitude, +amplitude)`.
    #[inline]
    fn next_sample(&mut self) -> f32 {
        // splitmix64 — single-call PRNG step (published mix constants).
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map top 53 bits → [0, 1)
        let u = (z >> 11) as f64 / (1u64 << 53) as f64;
        ((2.0 * u - 1.0) as f32) * self.amplitude
    }
}

impl AudioFilter for WhiteNoise {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        // Decode just to learn the per-channel sample count; the input
        // values are discarded — this is a *generator*.
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);
        let n_chan = params.channels as usize;
        let mut out_channels: Vec<Vec<f32>> = (0..n_chan).map(|_| Vec::with_capacity(n)).collect();
        for _ in 0..n {
            // All channels share the same per-step sample → decorrelated channels
            // would need separate PRNG instances; mono-correlated is the
            // documented default. Callers wanting independent L/R noise can
            // instantiate two `WhiteNoise` with different seeds and feed mono.
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
        let mut g = WhiteNoise::new(0.3);
        let frame = silence_frame(4096);
        let out = g.process(&frame, f32_mono(48_000)).unwrap();
        let s = read_f32(&out[0]);
        for v in &s {
            assert!(
                v.abs() <= 0.3001,
                "sample out of range: {v} > amplitude=0.3"
            );
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut a = WhiteNoise::with_seed(0.5, 42);
        let mut b = WhiteNoise::with_seed(0.5, 42);
        let frame = silence_frame(1024);
        let oa = a.process(&frame, f32_mono(48_000)).unwrap();
        let ob = b.process(&frame, f32_mono(48_000)).unwrap();
        let sa = read_f32(&oa[0]);
        let sb = read_f32(&ob[0]);
        assert_eq!(sa, sb, "same seed must produce identical streams");
    }

    #[test]
    fn flat_ish_spectrum() {
        // White noise should have roughly equal energy in any
        // sufficiently-wide frequency band. Run a 4096-point FFT of
        // a long white-noise stream and check that the energy ratio
        // between two non-overlapping equal-width bands is ≈ 1.
        use crate::fft::real_fft;
        let mut g = WhiteNoise::with_seed(1.0, 12345);
        let frame = silence_frame(8192);
        let out = g.process(&frame, f32_mono(48_000)).unwrap();
        let s = read_f32(&out[0]);
        // Average windowed FFTs over 2 chunks of 4096 to reduce variance.
        let n = 4096usize;
        let mut energy = vec![0.0f64; n / 2 + 1];
        let mut chunks = 0;
        for chunk in s.chunks_exact(n) {
            let bins = real_fft(chunk);
            for (i, b) in bins.iter().enumerate() {
                energy[i] += (b.magnitude() as f64).powi(2);
            }
            chunks += 1;
        }
        assert!(chunks >= 2);
        // Compare band [100..500) vs band [1500..1900). Skip DC.
        let lo_band: f64 = energy[100..500].iter().sum();
        let hi_band: f64 = energy[1500..1900].iter().sum();
        let ratio = lo_band / hi_band;
        assert!(
            (0.5..2.0).contains(&ratio),
            "white-noise band ratio {ratio} not ≈ 1"
        );
    }

    #[test]
    fn different_seeds_decorrelate() {
        let mut a = WhiteNoise::with_seed(1.0, 1);
        let mut b = WhiteNoise::with_seed(1.0, 2);
        let frame = silence_frame(2048);
        let oa = a.process(&frame, f32_mono(48_000)).unwrap();
        let ob = b.process(&frame, f32_mono(48_000)).unwrap();
        let sa = read_f32(&oa[0]);
        let sb = read_f32(&ob[0]);
        let mut diff = 0usize;
        for i in 0..sa.len() {
            if (sa[i] - sb[i]).abs() > 1.0e-6 {
                diff += 1;
            }
        }
        // Different seeds should produce different samples almost everywhere.
        assert!(diff > sa.len() * 95 / 100, "seeds 1 vs 2 too similar");
    }
}
