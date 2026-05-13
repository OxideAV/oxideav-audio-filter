//! Auto-pan — LFO-modulated stereo placement.
//!
//! Sweeps a (typically mono or centred) source between the two stereo
//! channels at a slow LFO rate. The pan law is the standard linear
//! constant-power "raw amplitude" law, *not* `cos/sin` constant-power:
//! we want the dry signal to swing between full-L and full-R, with
//! `depth = 0` collapsing both channels to half-amplitude centre.
//!
//! # Recurrence
//!
//! Per output sample (using interleaved left/right):
//!
//! ```text
//! lfo[n]   = sin(2π · rate · n / fs)                   ∈ [-1, +1]
//! pan[n]   = depth · lfo[n]                            ∈ [-depth, +depth]
//! gain_L   = (1 - pan[n]) · 0.5
//! gain_R   = (1 + pan[n]) · 0.5
//! y_L[n]   = (x_L[n] + x_R[n]) · gain_L    (mono-sum × L gain)
//! y_R[n]   = (x_L[n] + x_R[n]) · gain_R    (mono-sum × R gain)
//! ```
//!
//! Mono input is duplicated to both channels first, so `x_L = x_R = x`.
//! Surround-channel layouts (channels > 2) pan their **first two** channels
//! and pass the rest through unchanged.
//!
//! # Parameters
//!
//! * `rate_hz` — LFO frequency. Clamped `[0, 20]`. Typical 0.5..4 Hz.
//! * `depth` — pan-swing amplitude. Clamped `[0, 1]`. `depth = 1` swings
//!   fully L↔R; `depth = 0` parks the LFO at centre (output is the
//!   half-amplitude mono sum).
//! * Phase between L and R LFO is zero (both swing together — *true* pan,
//!   not stereo trémolo).

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming auto-panner.
#[derive(Debug, Clone)]
pub struct AutoPan {
    rate_hz: f32,
    depth: f32,
    phase: f64,
}

impl AutoPan {
    /// New auto-pan. `rate_hz` clamped to `[0, 20]`, `depth` clamped to
    /// `[0, 1]`.
    pub fn new(rate_hz: f32, depth: f32) -> Self {
        Self {
            rate_hz: rate_hz.clamp(0.0, 20.0),
            depth: depth.clamp(0.0, 1.0),
            phase: 0.0,
        }
    }

    /// Currently-configured LFO rate.
    pub fn rate_hz(&self) -> f32 {
        self.rate_hz
    }

    /// Currently-configured pan depth.
    pub fn depth(&self) -> f32 {
        self.depth
    }

    /// Reset LFO phase to zero.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

impl AudioFilter for AutoPan {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n_samples = channels.first().map(|c| c.len()).unwrap_or(0);
        let n_chan = channels.len();

        // No-op for zero-channel pathological input.
        if n_chan == 0 {
            let out = encode_from_f32(params.format, params.channels, input, &channels)?;
            return Ok(vec![out]);
        }

        let fs = params.sample_rate as f64;
        let dphase = 2.0 * std::f64::consts::PI * (self.rate_hz as f64) / fs;
        let depth = self.depth;

        // Mono input: duplicate channel 0 to a virtual right path so the
        // pan can swing it; output stays mono if input is mono.
        if n_chan == 1 {
            // For mono: just amplitude-modulate with the L-gain half of
            // the LFO (mono-pan is degenerate — collapse to tremolo at
            // half-depth, but offset so depth=0 is bypass).
            for s in channels[0].iter_mut().take(n_samples) {
                let lfo = self.phase.sin() as f32;
                // Half-amplitude swing keeps mono output bounded ≤ peak in.
                let gain = 1.0 - depth * 0.5 * (1.0 - lfo);
                *s *= gain;
                self.phase += dphase;
                if self.phase >= 2.0 * std::f64::consts::PI {
                    self.phase -= 2.0 * std::f64::consts::PI;
                }
            }
            let out = encode_from_f32(params.format, params.channels, input, &channels)?;
            return Ok(vec![out]);
        }

        // Stereo (or surround). Operate on channels [0]=L and [1]=R only;
        // leave channels ≥ 2 untouched.
        let (left, rest) = channels.split_first_mut().expect("≥ 2 channels here");
        let right = &mut rest[0];
        for i in 0..n_samples {
            let lfo = self.phase.sin() as f32;
            let pan = depth * lfo;
            let gl = (1.0 - pan) * 0.5;
            let gr = (1.0 + pan) * 0.5;
            let xl = left[i];
            let xr = right[i];
            let m = xl + xr;
            left[i] = m * gl;
            right[i] = m * gr;
            self.phase += dphase;
            if self.phase >= 2.0 * std::f64::consts::PI {
                self.phase -= 2.0 * std::f64::consts::PI;
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

    fn f32_stereo(rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 2,
            sample_rate: rate,
        }
    }

    fn make_stereo(left: &[f32], right: &[f32]) -> AudioFrame {
        assert_eq!(left.len(), right.len());
        let mut bytes = Vec::with_capacity(left.len() * 2 * 4);
        for i in 0..left.len() {
            bytes.extend_from_slice(&left[i].to_le_bytes());
            bytes.extend_from_slice(&right[i].to_le_bytes());
        }
        AudioFrame {
            samples: left.len() as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    fn read_stereo(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let bytes = &frame.data[0];
        let n = frame.samples as usize;
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 2 * 4;
            l.push(f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
            r.push(f32::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]));
        }
        (l, r)
    }

    #[test]
    fn depth_zero_is_centre() {
        // Mono-mixed input on both L/R: output should be 0.5·(L+R) on
        // each channel (depth=0 ⇒ pan=0 ⇒ gain=0.5 both sides).
        let n = 512;
        let lin = vec![0.6f32; n];
        let rin = vec![0.2f32; n];
        let frame = make_stereo(&lin, &rin);
        let mut ap = AutoPan::new(2.0, 0.0);
        let out = ap.process(&frame, f32_stereo(48_000)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        let m = (0.6 + 0.2) * 0.5;
        for i in 0..n {
            assert!(
                (lo[i] - m).abs() < 1e-5,
                "depth=0 L not centre at {}: got={}, want={}",
                i,
                lo[i],
                m
            );
            assert!(
                (ro[i] - m).abs() < 1e-5,
                "depth=0 R not centre at {}: got={}, want={}",
                i,
                ro[i],
                m
            );
        }
    }

    #[test]
    fn full_depth_swings_l_to_r() {
        // Constant DC input, depth=1 — output L and R must both swing
        // through 0 once per LFO period.
        let fs = 48_000u32;
        let rate = 2.0f32;
        let n = (fs as f32 / rate) as usize + 100;
        let frame = make_stereo(&vec![1.0f32; n], &vec![1.0f32; n]);
        let mut ap = AutoPan::new(rate, 1.0);
        let out = ap.process(&frame, f32_stereo(fs)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        // L gain = (1 - sin(φ)) · 0.5 ∈ [0, 1] for sin ∈ [-1, +1],
        // mono sum = 2 → L_out ∈ [0, 2].
        let l_min = lo.iter().cloned().fold(f32::INFINITY, f32::min);
        let l_max = lo.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let r_min = ro.iter().cloned().fold(f32::INFINITY, f32::min);
        let r_max = ro.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(l_min < 0.05, "L did not swing to zero: min={}", l_min);
        assert!(l_max > 1.95, "L did not swing to full: max={}", l_max);
        assert!(r_min < 0.05, "R did not swing to zero: min={}", r_min);
        assert!(r_max > 1.95, "R did not swing to full: max={}", r_max);
    }

    #[test]
    fn l_and_r_are_anti_correlated() {
        // With a constant DC mono-summed input, L+R must stay constant at
        // (xl+xr) — pan is conservative across the pair.
        let fs = 48_000u32;
        let n = 2048;
        let frame = make_stereo(&vec![0.5f32; n], &vec![0.3f32; n]);
        let mut ap = AutoPan::new(3.0, 1.0);
        let out = ap.process(&frame, f32_stereo(fs)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        let total_in = 0.5 + 0.3;
        for i in 0..n {
            let s = lo[i] + ro[i];
            assert!(
                (s - total_in).abs() < 1e-5,
                "L+R = {} not {} at {}",
                s,
                total_in,
                i
            );
        }
    }

    #[test]
    fn rate_zero_static() {
        // rate=0 → LFO frozen at sin(0)=0 → centre pan independent of depth.
        let n = 256;
        let frame = make_stereo(&vec![0.7f32; n], &vec![0.3f32; n]);
        let mut ap = AutoPan::new(0.0, 1.0);
        let out = ap.process(&frame, f32_stereo(48_000)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        let m = (0.7 + 0.3) * 0.5;
        for i in 0..n {
            assert!((lo[i] - m).abs() < 1e-5);
            assert!((ro[i] - m).abs() < 1e-5);
        }
    }
}
