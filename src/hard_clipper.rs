//! Hard clipper — memoryless symmetric clipping distortion.
//!
//! Applies a pre-gain (`drive`) and then clamps the result to a fixed
//! `±ceiling`. Unlike [`TapeSaturation`](crate::tape_saturation) (which
//! uses a smooth `tanh` knee) the transfer curve here is piecewise
//! linear with two hard corners, so it generates strong odd harmonics —
//! the classic "fuzz / overdrive" timbre. It is also distinct from
//! [`Volume`](crate::volume) (gain followed by a *fixed* `±1.0` clip):
//! here the clip threshold is a separate, configurable knob and the
//! drive is applied *before* the clamp so the curve flattens
//! independently of where the output ceiling sits.
//!
//! # Transfer function
//!
//! Per sample, per channel (memoryless — no state carried across
//! samples or channels):
//!
//! ```text
//! y[n] = clamp(drive · x[n], -ceiling, +ceiling)
//!      = max(-ceiling, min(ceiling, drive · x[n]))
//! ```
//!
//! For `|drive · x| ≤ ceiling` the curve is exactly linear with slope
//! `drive`; beyond that the output saturates flat at `±ceiling`. With
//! `drive = 1` and `ceiling ≥ 1` the filter is a bit-exact pass-through
//! for in-range samples.
//!
//! # Harmonic content
//!
//! Symmetric clipping of a sine produces a curve that is an *odd*
//! function (`f(-x) = -f(x)`), so its Fourier series contains only odd
//! harmonics (`3f, 5f, 7f, …`) and no DC term — the same property that
//! makes a fully-saturated clipper approximate a square wave (whose
//! spectrum is `Σ (1/k)·sin(k·2πf t)` over odd `k`).
//!
//! # Parameters
//!
//! * `drive` — pre-clip gain. `1.0` = unity; `> 1.0` pushes more of the
//!   waveform into the flat region (more distortion). Clamped to
//!   `[0.0, 64.0]`.
//! * `ceiling` — clip threshold (the flat output level, in the same
//!   `[-1, 1]`-normalised sample domain). Clamped to `[1e-6, 1.0]`;
//!   the lower bound keeps the threshold strictly positive so the curve
//!   never collapses to a constant-zero output.
//!
//! # General DSP literature
//!
//! Hard (memoryless) clipping is the elementary nonlinear distortion
//! primitive, implemented from the first-principles `clamp` definition.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming memoryless hard clipper.
#[derive(Debug, Clone, Copy)]
pub struct HardClipper {
    drive: f32,
    ceiling: f32,
}

impl HardClipper {
    /// New hard clipper. `drive` clamped to `[0.0, 64.0]`; `ceiling`
    /// clamped to `[1e-6, 1.0]` (kept strictly positive so the transfer
    /// curve always has a non-degenerate linear region).
    pub fn new(drive: f32, ceiling: f32) -> Self {
        Self {
            drive: drive.clamp(0.0, 64.0),
            ceiling: ceiling.clamp(1e-6, 1.0),
        }
    }

    /// Currently-configured pre-clip drive.
    pub fn drive(&self) -> f32 {
        self.drive
    }

    /// Currently-configured clip ceiling.
    pub fn ceiling(&self) -> f32 {
        self.ceiling
    }

    /// Apply the transfer function to one sample.
    #[inline]
    fn shape(&self, x: f32) -> f32 {
        (self.drive * x).clamp(-self.ceiling, self.ceiling)
    }
}

impl Default for HardClipper {
    /// Unity drive, full-scale `±1.0` ceiling — a no-op for in-range
    /// audio (only out-of-range excursions get clamped).
    fn default() -> Self {
        Self::new(1.0, 1.0)
    }
}

impl AudioFilter for HardClipper {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        for buf in channels.iter_mut() {
            for s in buf.iter_mut() {
                *s = self.shape(*s);
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
    fn hand_verified_transfer_curve() {
        // drive=2, ceiling=1.0. Expected per the closed form
        // y = clamp(2·x, -1, 1):
        //   x= 0.1 →  0.2   (linear, 2·0.1)
        //   x= 0.4 →  0.8   (linear, 2·0.4)
        //   x= 0.6 →  1.0   (2·0.6=1.2 clamped to ceiling)
        //   x= 1.0 →  1.0   (2·1.0=2.0 clamped)
        //   x=-0.3 → -0.6   (linear, 2·-0.3)
        //   x=-0.9 → -1.0   (2·-0.9=-1.8 clamped to -ceiling)
        let input = [0.1, 0.4, 0.6, 1.0, -0.3, -0.9];
        let expected = [0.2, 0.8, 1.0, 1.0, -0.6, -1.0];
        let frame = make_f32_mono(&input);
        let mut hc = HardClipper::new(2.0, 1.0);
        let out = hc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "sample {i}: got {g}, expected {e} (in={})",
                input[i]
            );
        }
    }

    #[test]
    fn custom_ceiling_clamps_lower() {
        // ceiling=0.5, drive=1: anything ≥0.5 saturates at 0.5.
        let input = [0.2, 0.5, 0.7, -0.49, -0.5, -0.8];
        let expected = [0.2, 0.5, 0.5, -0.49, -0.5, -0.5];
        let frame = make_f32_mono(&input);
        let mut hc = HardClipper::new(1.0, 0.5);
        let out = hc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "sample {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn unity_drive_full_scale_is_passthrough_in_range() {
        // drive=1, ceiling=1: every in-range sample passes through
        // bit-exactly; only |x|>1 gets clamped.
        let input = [0.0, 0.25, -0.5, 0.999, -1.0, 1.0];
        let frame = make_f32_mono(&input);
        let mut hc = HardClipper::default();
        let out = hc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
            assert!((g - x).abs() < 1e-6, "sample {i}: got {g}, expected {x}");
        }
    }

    #[test]
    fn output_bounded_by_ceiling() {
        // Heavy drive on a loud sine — no output sample may exceed the
        // ceiling in magnitude.
        let n = 2048;
        let w = 2.0 * std::f32::consts::PI * 13.0 / n as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.95).collect();
        let frame = make_f32_mono(&samples);
        let ceiling = 0.7;
        let mut hc = HardClipper::new(8.0, ceiling);
        let out = hc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let peak = got.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            peak <= ceiling + 1e-6,
            "output peak {peak} exceeds ceiling {ceiling}"
        );
        // A 0.95-amplitude sine × drive 8 = 7.6 peak >> ceiling, so the
        // clipper MUST actually be hitting the rail (output peak ≈ ceiling).
        assert!(
            peak > ceiling - 1e-3,
            "clipper never reached the rail: peak={peak}"
        );
    }

    #[test]
    fn symmetric_clip_no_dc_offset() {
        // Symmetric clipping is an odd function, so a full-period sine
        // input yields zero-mean output (no even harmonics, no DC).
        let n = 4096;
        let w = 2.0 * std::f32::consts::PI * 16.0 / n as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.9).collect();
        let frame = make_f32_mono(&samples);
        let mut hc = HardClipper::new(4.0, 0.6);
        let out = hc.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let mean: f32 = got.iter().sum::<f32>() / n as f32;
        assert!(
            mean.abs() < 1e-4,
            "symmetric clip introduced DC: mean={mean}"
        );
    }

    #[test]
    fn stereo_channels_independent() {
        // L = loud (must clip), R = quiet (must pass). Each channel
        // shaped independently with no cross-talk (filter is memoryless).
        let n = 256usize;
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&0.9f32.to_le_bytes()); // L
            bytes.extend_from_slice(&0.1f32.to_le_bytes()); // R
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut hc = HardClipper::new(2.0, 1.0);
        let out = hc
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 2,
                    sample_rate: 48_000,
                },
            )
            .unwrap();
        let bytes = &out[0].data[0];
        let rd = |s: usize, c: usize| {
            let off = (s * 2 + c) * 4;
            f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
        };
        for s in 0..n {
            assert!((rd(s, 0) - 1.0).abs() < 1e-6, "L should clip to 1.0"); // 2·0.9=1.8→1.0
            assert!((rd(s, 1) - 0.2).abs() < 1e-6, "R should be 2·0.1=0.2"); // 2·0.1=0.2
        }
    }

    #[test]
    fn parameters_clamped() {
        let hc = HardClipper::new(-5.0, 10.0);
        assert_eq!(hc.drive(), 0.0); // negative drive clamped up to 0
        assert_eq!(hc.ceiling(), 1.0); // ceiling clamped down to 1.0
        let hc = HardClipper::new(1000.0, 0.0);
        assert_eq!(hc.drive(), 64.0); // huge drive clamped to 64
        assert!(hc.ceiling() > 0.0); // ceiling kept strictly positive
    }
}
