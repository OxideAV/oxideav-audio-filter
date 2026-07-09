//! Mid/Side stereo widener.
//!
//! Decomposes the input into mid (`M`) and side (`S`) signals,
//! scales the side component by a `width` factor, and recomposes
//! the L/R signal:
//!
//! ```text
//! M  = (L + R) / 2
//! S  = (L - R) / 2
//! L' = M + width · S
//! R' = M - width · S
//! ```
//!
//! Special cases:
//!
//! * `width = 0` collapses to mono (`L = R = M`).
//! * `width = 1` is bit-exact bypass (`L' = L`, `R' = R`).
//! * `width = 2` doubles the side energy, exaggerating stereo width.
//!
//! Inputs with fewer than 2 channels pass through unchanged.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Mid/Side stereo widener with `width ∈ [0, 2]`.
#[derive(Debug, Clone)]
pub struct StereoWidener {
    width: f32,
}

impl StereoWidener {
    /// New widener. `width` is clamped to `[0, 2]`.
    pub fn new(width: f32) -> Self {
        Self {
            width: crate::clamp_param(width, 1.0, 0.0, 2.0),
        }
    }

    /// Currently-configured width.
    pub fn width(&self) -> f32 {
        self.width
    }

    /// Update the width factor.
    pub fn set_width(&mut self, width: f32) {
        self.width = width.clamp(0.0, 2.0);
    }
}

impl AudioFilter for StereoWidener {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;

        // Only operate on stereo (or higher) inputs. Mono passes
        // through unchanged.
        if channels.len() >= 2 {
            let n = channels[0].len();
            let w = self.width;
            // Borrow split via split_at_mut to avoid double-mut.
            let (l_slice, rest) = channels.split_at_mut(1);
            let l_buf = &mut l_slice[0];
            let r_buf = &mut rest[0];
            for i in 0..n {
                let l = l_buf[i];
                let r = r_buf[i];
                let mid = 0.5 * (l + r);
                let side = 0.5 * (l - r);
                l_buf[i] = mid + w * side;
                r_buf[i] = mid - w * side;
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

    fn stereo_params(rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 2,
            sample_rate: rate,
        }
    }

    fn make_f32_stereo(l: &[f32], r: &[f32]) -> AudioFrame {
        assert_eq!(l.len(), r.len());
        let mut bytes = Vec::with_capacity(l.len() * 2 * 4);
        for i in 0..l.len() {
            bytes.extend_from_slice(&l[i].to_le_bytes());
            bytes.extend_from_slice(&r[i].to_le_bytes());
        }
        AudioFrame {
            samples: l.len() as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    fn read_stereo(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let mut l = Vec::new();
        let mut r = Vec::new();
        for chunk in frame.data[0].chunks_exact(8) {
            l.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            r.push(f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]));
        }
        (l, r)
    }

    #[test]
    fn width_zero_monaurises() {
        // L = +0.5, R = -0.5 → mid = 0; with width=0, L' = R' = 0.
        let frame = make_f32_stereo(&[0.5, 0.3, -0.2], &[-0.5, 0.1, 0.4]);
        let mut w = StereoWidener::new(0.0);
        let out = w.process(&frame, stereo_params(48_000)).unwrap();
        let (l, r) = read_stereo(&out[0]);
        for i in 0..l.len() {
            assert!(
                (l[i] - r[i]).abs() < 1.0e-6,
                "width=0 channels diverge at {}: L={} R={}",
                i,
                l[i],
                r[i]
            );
            // L = R = mid = (Lin + Rin) / 2
            let want = 0.5
                * (frame.data[0][i * 8..i * 8 + 8]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .sum::<f32>());
            assert!(
                (l[i] - want).abs() < 1.0e-6,
                "mid mismatch at {}: got={} want={}",
                i,
                l[i],
                want
            );
        }
    }

    #[test]
    fn width_one_is_bit_exact_bypass() {
        let l_in: Vec<f32> = (0..256).map(|i| (i as f32 * 0.07).sin() * 0.7).collect();
        let r_in: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).cos() * 0.6).collect();
        let frame = make_f32_stereo(&l_in, &r_in);
        let mut w = StereoWidener::new(1.0);
        let out = w.process(&frame, stereo_params(48_000)).unwrap();
        let (l, r) = read_stereo(&out[0]);
        for i in 0..l_in.len() {
            assert!(
                (l[i] - l_in[i]).abs() < 1.0e-5,
                "L not bypassed at {}: got={} in={}",
                i,
                l[i],
                l_in[i]
            );
            assert!(
                (r[i] - r_in[i]).abs() < 1.0e-5,
                "R not bypassed at {}: got={} in={}",
                i,
                r[i],
                r_in[i]
            );
        }
    }

    #[test]
    fn width_two_widens() {
        // L = +0.4, R = -0.4 → mid = 0, side = 0.4.
        // width = 2 → L' = 0 + 2·0.4 = 0.8, R' = 0 - 2·0.4 = -0.8.
        let frame = make_f32_stereo(&[0.4f32; 32], &[-0.4f32; 32]);
        let mut w = StereoWidener::new(2.0);
        let out = w.process(&frame, stereo_params(48_000)).unwrap();
        let (l, r) = read_stereo(&out[0]);
        for i in 0..l.len() {
            assert!(
                (l[i] - 0.8).abs() < 1.0e-5,
                "L widened wrong at {}: got={}",
                i,
                l[i]
            );
            assert!(
                (r[i] + 0.8).abs() < 1.0e-5,
                "R widened wrong at {}: got={}",
                i,
                r[i]
            );
        }
    }

    #[test]
    fn mono_input_unchanged() {
        // Even with width=2, a mono input must pass through unchanged.
        let samples = vec![0.1f32, 0.5, -0.3, 0.7];
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in &samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: samples.len() as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut w = StereoWidener::new(2.0);
        let out = w
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 1,
                    sample_rate: 48_000,
                },
            )
            .unwrap();
        let bytes = &out[0].data[0];
        for (i, want) in samples.iter().enumerate() {
            let off = i * 4;
            let v =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            assert!(
                (v - *want).abs() < 1.0e-6,
                "mono altered at {}: got={} want={}",
                i,
                v,
                want
            );
        }
    }
}
