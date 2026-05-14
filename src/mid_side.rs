//! Explicit Mid/Side ↔ Left/Right transcoder.
//!
//! Stereo processing in many DSP contexts is more natural in the
//! sum/difference domain than in the native L/R domain:
//!
//! ```text
//! M = (L + R) / 2            L = M + S
//! S = (L - R) / 2            R = M - S
//! ```
//!
//! Both directions are unitary up to the `1/2` scaling: forward then
//! inverse is bit-identity (modulo round-off). The factor-of-two
//! convention is the standard "decoder-friendly" form — `M` and `S`
//! both lie in `[-1, +1]` whenever both `L` and `R` do.
//!
//! # Mode
//!
//! [`MidSideMode::Encode`] takes an L/R input frame and emits an M/S
//! output (channel 0 = M, channel 1 = S). [`MidSideMode::Decode`] is
//! the inverse. Channel counts must be exactly two. Mono input is
//! rejected — for mono → stereo upmix use the [`crate::downmix`] family.
//!
//! No internal state is kept between frames; the transform is
//! sample-wise.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Error, Result};

/// Direction of conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidSideMode {
    /// L/R → M/S. Channel 0 holds the mid (sum/2), channel 1 the side
    /// (difference/2).
    Encode,
    /// M/S → L/R. Channel 0 holds the recovered left, channel 1 the
    /// recovered right.
    Decode,
}

/// Stateless M/S ↔ L/R transcoder.
#[derive(Debug, Clone, Copy)]
pub struct MidSide {
    mode: MidSideMode,
}

impl MidSide {
    /// New M/S encoder (L/R → M/S).
    pub fn encoder() -> Self {
        Self {
            mode: MidSideMode::Encode,
        }
    }

    /// New M/S decoder (M/S → L/R).
    pub fn decoder() -> Self {
        Self {
            mode: MidSideMode::Decode,
        }
    }

    /// Currently configured direction.
    pub fn mode(&self) -> MidSideMode {
        self.mode
    }
}

impl AudioFilter for MidSide {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        if params.channels != 2 {
            return Err(Error::invalid(format!(
                "mid_side: requires stereo input (channels=2), got channels={}",
                params.channels
            )));
        }
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        if channels.len() != 2 {
            return Err(Error::invalid("mid_side: decoded frame is not 2-channel"));
        }
        let n = channels[0].len().min(channels[1].len());
        let (left, right_rest) = channels.split_at_mut(1);
        let left = &mut left[0];
        let right = &mut right_rest[0];
        match self.mode {
            MidSideMode::Encode => {
                for (l_ref, r_ref) in left.iter_mut().zip(right.iter_mut()).take(n) {
                    let l = *l_ref;
                    let r = *r_ref;
                    *l_ref = 0.5 * (l + r);
                    *r_ref = 0.5 * (l - r);
                }
            }
            MidSideMode::Decode => {
                for (l_ref, r_ref) in left.iter_mut().zip(right.iter_mut()).take(n) {
                    let m = *l_ref;
                    let s = *r_ref;
                    *l_ref = m + s;
                    *r_ref = m - s;
                }
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

    fn stereo_params() -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 2,
            sample_rate: 48_000,
        }
    }

    /// Build an interleaved-stereo f32 frame from L and R sample slices.
    fn make_stereo(l: &[f32], r: &[f32]) -> AudioFrame {
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

    fn split_stereo(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let n = frame.samples as usize;
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        let bytes = &frame.data[0];
        for i in 0..n {
            let lo = i * 8;
            l.push(f32::from_le_bytes([
                bytes[lo],
                bytes[lo + 1],
                bytes[lo + 2],
                bytes[lo + 3],
            ]));
            r.push(f32::from_le_bytes([
                bytes[lo + 4],
                bytes[lo + 5],
                bytes[lo + 6],
                bytes[lo + 7],
            ]));
        }
        (l, r)
    }

    #[test]
    fn encode_then_decode_roundtrips() {
        let l: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin() * 0.4).collect();
        let r: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).cos() * 0.3).collect();
        let frame = make_stereo(&l, &r);
        let mut enc = MidSide::encoder();
        let ms = enc.process(&frame, stereo_params()).unwrap();
        let mut dec = MidSide::decoder();
        let lr = dec.process(&ms[0], stereo_params()).unwrap();
        let (lo, ro) = split_stereo(&lr[0]);
        for i in 0..l.len() {
            assert!(
                (lo[i] - l[i]).abs() < 1e-6,
                "L mismatch at {i}: got={} want={}",
                lo[i],
                l[i]
            );
            assert!(
                (ro[i] - r[i]).abs() < 1e-6,
                "R mismatch at {i}: got={} want={}",
                ro[i],
                r[i]
            );
        }
    }

    #[test]
    fn mid_channel_is_average() {
        let l = vec![1.0f32, 0.5, -0.25, 0.0];
        let r = vec![1.0f32, 0.5, -0.25, 0.0];
        let frame = make_stereo(&l, &r);
        let mut enc = MidSide::encoder();
        let out = enc.process(&frame, stereo_params()).unwrap();
        let (m, s) = split_stereo(&out[0]);
        for i in 0..l.len() {
            // L == R → M == L, S == 0
            assert!(
                (m[i] - l[i]).abs() < 1e-6,
                "M expected {} got {}",
                l[i],
                m[i]
            );
            assert!(s[i].abs() < 1e-6, "S expected 0 got {}", s[i]);
        }
    }

    #[test]
    fn opposite_channels_concentrate_in_side() {
        // L = +x, R = -x → M = 0, S = x.
        let l = vec![0.5f32, 0.25, -0.5, 0.1];
        let r: Vec<f32> = l.iter().map(|v| -v).collect();
        let frame = make_stereo(&l, &r);
        let mut enc = MidSide::encoder();
        let out = enc.process(&frame, stereo_params()).unwrap();
        let (m, s) = split_stereo(&out[0]);
        for i in 0..l.len() {
            assert!(m[i].abs() < 1e-6, "M expected 0 got {}", m[i]);
            assert!(
                (s[i] - l[i]).abs() < 1e-6,
                "S expected {} got {}",
                l[i],
                s[i]
            );
        }
    }

    #[test]
    fn mono_input_rejected() {
        let mono = make_stereo(&[0.0f32; 4], &[0.0f32; 4]);
        let mut enc = MidSide::encoder();
        let mono_params = AudioStreamParams {
            format: SampleFormat::F32,
            channels: 1,
            sample_rate: 48_000,
        };
        assert!(enc.process(&mono, mono_params).is_err());
    }
}
