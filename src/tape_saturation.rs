//! Tape saturation — soft-clip waveshaper with asymmetric drive.
//!
//! Models magnetic-tape's gentle compression-into-saturation curve with a
//! `tanh`-based soft-clipper plus a configurable positive/negative gain
//! bias to emulate the asymmetric magnetisation curve of real tape
//! (different polarities saturate at slightly different rates).
//!
//! # Waveshaper
//!
//! ```text
//! pre   = drive · x[n]
//! pre_a = if pre ≥ 0 { pre }
//!         else        { pre · (1 + asymmetry) }    (negative limb gain)
//! y_pre = tanh(pre_a)
//! y[n]  = y_pre / drive_norm
//! ```
//!
//! `drive_norm = tanh(drive)` so output peaks stay near `±1` regardless of
//! drive (the curve "fills out" but doesn't overflow). For drive → 0 this
//! reduces to identity (output = input / 1).
//!
//! # Parameters
//!
//! * `drive` — input gain feeding the saturator. `drive = 1` is "kiss"
//!   saturation (low THD); `drive = 4` is heavy soft-clip. Clamped to
//!   `[0.1, 16]`.
//! * `asymmetry` — adds positive bias to negative-going samples. `0`
//!   gives a symmetric `tanh` curve (odd harmonics only); `> 0` adds
//!   even harmonics. Clamped to `[0, 1]`.
//!
//! # No filtering
//!
//! Real tape rolls off treble (head-gap loss) and rolls off bass
//! (head-bump dip). Those colourations belong in a separate `Biquad`
//! shelf stage in front of and behind this filter — this module is the
//! pure non-linear shaper only.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming tape-saturation waveshaper.
#[derive(Debug, Clone)]
pub struct TapeSaturation {
    drive: f32,
    asymmetry: f32,
    /// Pre-computed `tanh(drive)` so we don't recompute per sample.
    drive_norm: f32,
}

impl TapeSaturation {
    /// New tape saturator. `drive` clamped to `[0.1, 16]`, `asymmetry`
    /// clamped to `[0, 1]`.
    pub fn new(drive: f32, asymmetry: f32) -> Self {
        let drive = crate::clamp_param(drive, 1.0, 0.1, 16.0);
        let asymmetry = crate::clamp_param(asymmetry, 0.0, 0.0, 1.0);
        // tanh(drive) → 1 for drive ≥ ~3, so for high drive output peaks
        // approach the literal x ≈ 1 input → 1 output mapping.
        let drive_norm = drive.tanh().max(1e-6);
        Self {
            drive,
            asymmetry,
            drive_norm,
        }
    }

    /// Currently-configured drive.
    pub fn drive(&self) -> f32 {
        self.drive
    }

    /// Currently-configured asymmetry.
    pub fn asymmetry(&self) -> f32 {
        self.asymmetry
    }
}

impl AudioFilter for TapeSaturation {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let drive = self.drive;
        let asym = self.asymmetry;
        let norm = self.drive_norm;
        for buf in channels.iter_mut() {
            for s in buf.iter_mut() {
                let pre = drive * *s;
                let pre_a = if pre >= 0.0 { pre } else { pre * (1.0 + asym) };
                *s = pre_a.tanh() / norm;
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
    fn output_bounded() {
        // For any input, output magnitude ≤ 1 / drive_norm. With
        // drive=1, norm=tanh(1)≈0.7616 → max output ≈ 1.31.
        let n = 1024;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin() * 10.0).collect();
        let frame = make_f32_mono(&samples);
        let mut ts = TapeSaturation::new(1.0, 0.0);
        let out = ts.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let peak = got.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let bound = 1.0 / 1.0f32.tanh();
        assert!(
            peak <= bound + 1e-4,
            "output peak {} exceeds bound {}",
            peak,
            bound
        );
    }

    #[test]
    fn small_signal_near_linear() {
        // For small inputs `tanh(drive·x) ≈ drive·x` so y ≈ x / tanh(drive) · drive ≈ x
        // (because tanh is locally linear with slope 1 around 0).
        let n = 64;
        let samples: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 1e-3).collect();
        let frame = make_f32_mono(&samples);
        let mut ts = TapeSaturation::new(2.0, 0.0);
        let out = ts.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..n {
            // Effective small-signal gain = drive / tanh(drive).
            let g = 2.0 / 2.0f32.tanh();
            let expected = samples[i] * g;
            let err = (got[i] - expected).abs();
            assert!(
                err < 1e-5,
                "small-signal non-linearity at {}: got={}, expect={}",
                i,
                got[i],
                expected
            );
        }
    }

    #[test]
    fn symmetric_drive_no_dc_offset() {
        // For asymmetry=0 + symmetric input (sine spanning an integer
        // number of periods), output mean should be ≈ 0 (no DC).
        // 4096 samples × ω = 16 × 2π → 16 full cycles.
        let n = 4096;
        let w = 2.0 * std::f32::consts::PI * 16.0 / (n as f32);
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.9).collect();
        let frame = make_f32_mono(&samples);
        let mut ts = TapeSaturation::new(3.0, 0.0);
        let out = ts.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let mean: f32 = got.iter().sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-4, "symmetric output DC offset: {}", mean);
    }

    #[test]
    fn asymmetry_introduces_dc() {
        // With asymmetry > 0 the negative limb is amplified before
        // saturation → after symmetric input, output mean becomes < 0
        // (negative side hits the soft-clip harder).
        let n = 4096;
        let w = 2.0 * std::f32::consts::PI * 16.0 / (n as f32);
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.9).collect();
        let frame = make_f32_mono(&samples);
        let mut ts = TapeSaturation::new(3.0, 0.7);
        let out = ts.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let mean: f32 = got.iter().sum::<f32>() / n as f32;
        assert!(
            mean < -0.01,
            "asymmetry didn't introduce expected -DC: mean={}",
            mean
        );
    }
}
