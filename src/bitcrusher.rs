//! Bitcrusher — bit-depth and sample-rate reduction.
//!
//! Two independent degradation stages applied per sample, per channel:
//!
//! 1. **Bit-depth quantisation** — round each sample to a grid of
//!    `2^bits` steps in `[-1, +1]`. The quantisation step is
//!    `Δ = 2 / 2^bits`. `bits = 16` is essentially transparent for
//!    audio sourced from CD; `bits ∈ {1..8}` give characteristic
//!    "lo-fi" / "8-bit console" artefacts.
//!
//! 2. **Sample-rate reduction (sample-and-hold)** — only update the
//!    held output value every `decimation` input samples; otherwise
//!    replay the last held sample. `decimation = 1` is bypass.
//!    Equivalent to running through a downsampler of factor
//!    `decimation` followed by zero-order-hold upsampling back to
//!    `fs`, *without* any anti-aliasing — the aliased spectrum is
//!    the entire point of the effect.
//!
//! # Recurrence (per channel)
//!
//! ```text
//! counter ← counter + 1
//! if counter % decimation == 0:
//!     hold = quantise(x[n], bits)
//! y[n]  = hold
//! ```
//!
//! Quantisation uses round-to-nearest with `clamp(-1, +1)`. Channels
//! share `bits` and `decimation` parameters but each maintains its own
//! `counter` and `hold` so they don't crosstalk through state.
//!
//! # Parameters
//!
//! * `bits` — target word-length, clamped to `[1, 24]`. (≥ 24 is
//!   visually-lossless at single precision, so 24 caps the bias.)
//! * `decimation` — sample-and-hold factor, clamped to `[1, 4096]`.
//!   `decimation = 1` is bypass for the SRR stage.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Per-channel bitcrusher state.
#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    counter: u32,
    hold: f32,
}

/// Streaming bitcrusher.
#[derive(Debug, Clone)]
pub struct Bitcrusher {
    bits: u8,
    decimation: u32,
    state: Vec<ChState>,
}

impl Bitcrusher {
    /// New bitcrusher. `bits` clamped to `[1, 24]`, `decimation` clamped
    /// to `[1, 4096]`.
    pub fn new(bits: u8, decimation: u32) -> Self {
        Self {
            bits: bits.clamp(1, 24),
            decimation: decimation.clamp(1, 4_096),
            state: Vec::new(),
        }
    }

    /// Currently-configured target bit depth.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Currently-configured sample-and-hold decimation factor.
    pub fn decimation(&self) -> u32 {
        self.decimation
    }

    /// Reset internal sample-and-hold state for all channels.
    pub fn reset(&mut self) {
        for st in self.state.iter_mut() {
            *st = ChState::default();
        }
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.state.len() != channels {
            self.state = vec![ChState::default(); channels];
        }
    }

    /// Quantise one sample to `bits` resolution in `[-1, +1]`.
    pub fn quantise(&self, x: f32) -> f32 {
        Self::quantise_with(self.bits, x)
    }

    /// Static helper so the per-sample inner loop can call without
    /// re-borrowing `&self` (the loop already holds a mutable borrow on
    /// `self.state`).
    fn quantise_with(bits: u8, x: f32) -> f32 {
        let xc = x.clamp(-1.0, 1.0);
        // 2^bits levels across [-1, +1] → step = 2 / 2^bits = 2^(1-bits).
        let levels = (1u32 << (bits as u32)) as f32;
        let step = 2.0 / levels;
        // Round to nearest grid value, then re-clamp.
        let q = (xc / step).round() * step;
        q.clamp(-1.0, 1.0)
    }
}

impl AudioFilter for Bitcrusher {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(channels.len());
        let dec = self.decimation;
        let bits = self.bits;
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.state[ch_idx];
            for s in buf.iter_mut() {
                // Update held value at boundaries.
                if st.counter % dec == 0 {
                    st.hold = Self::quantise_with(bits, *s);
                }
                st.counter = st.counter.wrapping_add(1);
                *s = st.hold;
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
    fn one_bit_clips_to_extremes() {
        // bits=1 → grid step = 2/2 = 1.0; round-to-nearest snaps to
        // {-1, 0, +1}. Drive the input firmly past 0.5 so the output
        // saturates to ±1 only.
        let samples: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { -0.9f32 } else { 0.9f32 })
            .collect();
        let frame = make_f32_mono(&samples);
        let mut bc = Bitcrusher::new(1, 1);
        let out = bc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, &v) in got.iter().enumerate() {
            // For inputs ≥ |0.5| the rounding snaps strictly to ±1.
            assert!(
                (v - 1.0).abs() < 1e-5 || (v + 1.0).abs() < 1e-5,
                "1-bit not at ±1 at {}: got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn decimation_holds_sample() {
        // decimation = 4 → output is sample-and-hold of every 4th input.
        let samples: Vec<f32> = (0..16).map(|i| i as f32 * 0.01).collect();
        let frame = make_f32_mono(&samples);
        let mut bc = Bitcrusher::new(24, 4);
        let out = bc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // Indices 0..4 should hold quantise(samples[0]) = 0.
        // Indices 4..8 should hold quantise(samples[4]) = 0.04, etc.
        let centres = [0, 4, 8, 12];
        for &c in &centres {
            let expected = bc.quantise(samples[c]);
            for k in 0..4 {
                let i = c + k;
                if i >= got.len() {
                    break;
                }
                assert!(
                    (got[i] - expected).abs() < 1e-5,
                    "decimation hold off at {}: got={}, want={}",
                    i,
                    got[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn bits_24_decimation_1_near_bypass() {
        // bits=24 + decimation=1 should be near-identity for normal-range
        // audio. Quantisation error ≤ Δ/2 = 2^(-24) ≈ 6e-8.
        let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let frame = make_f32_mono(&samples);
        let mut bc = Bitcrusher::new(24, 1);
        let out = bc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..samples.len() {
            let err = (got[i] - samples[i]).abs();
            assert!(err < 1e-6, "bits=24 too lossy at {}: err={}", i, err);
        }
    }

    #[test]
    fn quantisation_grid_8_bit() {
        // 8 bits → 256 levels across [-1, +1] → step = 2/256 = 1/128.
        // Every output sample must be a multiple of 1/128.
        let samples: Vec<f32> = (0..256).map(|i| (i as f32 / 256.0) - 0.5).collect();
        let frame = make_f32_mono(&samples);
        let mut bc = Bitcrusher::new(8, 1);
        let out = bc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for &v in got.iter() {
            let units = v * 128.0;
            assert!(
                (units - units.round()).abs() < 1e-4,
                "8-bit output not on 1/128 grid: {} (units={})",
                v,
                units
            );
        }
    }
}
