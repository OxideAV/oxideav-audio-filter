//! Stereo Imager — frequency-dependent width control.
//!
//! Unlike the broadband [`StereoWidener`](crate::StereoWidener), which
//! scales the M/S side signal by a single coefficient, the **imager**
//! splits the side channel into a low and a high band and applies a
//! different width factor to each. The classic mastering use is:
//!
//! * `low_width ≈ 0` — collapse the bass to mono (sub frequencies
//!   below ~150 Hz benefit from being phase-coherent for translation
//!   to small speakers + vinyl cutting).
//! * `high_width ≈ 1..2` — leave or expand the top end, where the
//!   ear's HRTF localisation is strongest.
//!
//! # Signal flow
//!
//! ```text
//!     L,R  ──►  M = (L+R)/2,   S = (L-R)/2          (encode)
//!
//!     S         ─► LPF_xover ─► S_low  · low_width   ┐
//!                                                    ├── S' = S_low' + S_high'
//!                ─► HPF_xover ─► S_high · high_width ┘
//!
//!     M (untouched, mono content is never widened)
//!
//!     L' = M + S'    R' = M - S'                     (decode)
//! ```
//!
//! Crossover uses two parallel Butterworth-2 biquads at `cutoff_hz`.
//! Like the simpler [`Crossover`](crate::Crossover) the sum is not
//! magnitude-flat near the corner (~+3 dB lump) but the artefact is
//! confined to the side channel and largely inaudible because the
//! mid channel carries the same energy in phase.
//!
//! # Parameters
//!
//! * `cutoff_hz` — width-split frequency. Clamped to `[20, 20_000]`.
//!   Typical 100–300 Hz for mastering, 1000 Hz for "broadcast wide".
//! * `low_width` / `high_width` — multipliers in `[0, 2]` applied to
//!   the side signal below / above the cutoff. `0` = mono in that
//!   band, `1` = bypass, `2` = doubly-wide.
//!
//! Mono input is detected (channel count = 1) and the filter
//! degrades to a pass-through.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming stereo imager.
#[derive(Debug, Clone)]
pub struct StereoImager {
    cutoff_hz: f32,
    low_width: f32,
    high_width: f32,
    lpf: Biquad,
    hpf: Biquad,
}

impl StereoImager {
    /// Default: `cutoff = 250 Hz`, `low = 0` (mono bass),
    /// `high = 1.5` (slight widening above the corner).
    pub fn new() -> Self {
        Self::with(250.0, 0.0, 1.5)
    }

    /// Custom-parameter constructor.
    pub fn with(cutoff_hz: f32, low_width: f32, high_width: f32) -> Self {
        let cutoff_hz = crate::clamp_param(cutoff_hz, 250.0, 20.0, 20_000.0);
        let low_width = crate::clamp_param(low_width, 1.0, 0.0, 2.0);
        let high_width = crate::clamp_param(high_width, 1.0, 0.0, 2.0);
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let lpf = Biquad::new(BiquadKind::LowPass { cutoff_hz, q });
        let hpf = Biquad::new(BiquadKind::HighPass { cutoff_hz, q });
        Self {
            cutoff_hz,
            low_width,
            high_width,
            lpf,
            hpf,
        }
    }

    /// Current crossover frequency.
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }
    /// Current low-band side-channel multiplier.
    pub fn low_width(&self) -> f32 {
        self.low_width
    }
    /// Current high-band side-channel multiplier.
    pub fn high_width(&self) -> f32 {
        self.high_width
    }

    /// Reset both crossover filter states.
    pub fn reset(&mut self) {
        self.lpf.reset();
        self.hpf.reset();
    }
}

impl Default for StereoImager {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for StereoImager {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let dry = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = dry.len();
        // Anything other than stereo: pass-through. Mono has no side
        // signal; surround layouts have non-trivial M/S definitions
        // we don't want to silently get wrong.
        if n_chan != 2 {
            let out = encode_from_f32(params.format, params.channels, input, &dry)?;
            return Ok(vec![out]);
        }
        let n_samples = dry[0].len();
        let l = &dry[0];
        let r = &dry[1];

        // Encode to M/S.
        let mut mid = vec![0.0f32; n_samples];
        let mut side = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            mid[i] = 0.5 * (l[i] + r[i]);
            side[i] = 0.5 * (l[i] - r[i]);
        }

        // Split side into low + high bands.
        let mut side_low = side.clone();
        let mut side_high = side.clone();
        self.lpf
            .process_in_place(&mut side_low, 1, params.sample_rate);
        self.hpf
            .process_in_place(&mut side_high, 1, params.sample_rate);

        // Recombine with per-band width.
        let mut new_side = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            new_side[i] = self.low_width * side_low[i] + self.high_width * side_high[i];
        }

        // Decode back to L/R.
        let mut new_l = vec![0.0f32; n_samples];
        let mut new_r = vec![0.0f32; n_samples];
        for i in 0..n_samples {
            new_l[i] = mid[i] + new_side[i];
            new_r[i] = mid[i] - new_side[i];
        }

        let channels = vec![new_l, new_r];
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

    fn make_f32_stereo(l: &[f32], r: &[f32]) -> AudioFrame {
        assert_eq!(l.len(), r.len());
        let n = l.len();
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for i in 0..n {
            bytes.extend_from_slice(&l[i].to_le_bytes());
            bytes.extend_from_slice(&r[i].to_le_bytes());
        }
        AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    fn read_stereo(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let n = frame.samples as usize;
        let mut l = Vec::with_capacity(n);
        let mut r = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 2 * 4;
            l.push(f32::from_le_bytes([
                frame.data[0][off],
                frame.data[0][off + 1],
                frame.data[0][off + 2],
                frame.data[0][off + 3],
            ]));
            r.push(f32::from_le_bytes([
                frame.data[0][off + 4],
                frame.data[0][off + 5],
                frame.data[0][off + 6],
                frame.data[0][off + 7],
            ]));
        }
        (l, r)
    }

    fn rms(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        (s / x.len() as f64).sqrt() as f32
    }

    fn sine(amp: f32, freq: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| amp * (i as f32 * w).sin()).collect()
    }

    #[test]
    fn unity_widths_preserves_total_rms_far_from_crossover() {
        // Both widths = 1 → side LPF + side HPF are summed back into the
        // side channel. The Butterworth-2 LPF/HPF *magnitude* sum has the
        // usual +3 dB ripple at the crossover and a phase shift that
        // prevents sample-by-sample identity — but FAR from the corner the
        // sum collapses to the input (LPF ≈ 1 below, HPF ≈ 1 above).
        // Verify that for tones placed well below and well above the
        // crossover, output RMS matches input RMS to within 0.5 dB.
        let fs = 48_000u32;
        let n = 24_000usize;
        let cutoff = 500.0f32;
        // Tones two octaves away from the crossover on each side.
        let l = sine(0.4, cutoff / 4.0, fs, n); // 125 Hz
        let r = sine(0.4, cutoff * 4.0, fs, n); // 2 kHz
        let frame = make_f32_stereo(&l, &r);
        let mut im = StereoImager::with(cutoff, 1.0, 1.0);
        let out = im.process(&frame, f32_stereo(fs)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let l_db = 20.0 * (rms(&lo[warm..]) / rms(&l[warm..])).log10();
        let r_db = 20.0 * (rms(&ro[warm..]) / rms(&r[warm..])).log10();
        assert!(
            l_db.abs() < 0.5,
            "off-corner L RMS drift = {} dB (unity widths)",
            l_db
        );
        assert!(
            r_db.abs() < 0.5,
            "off-corner R RMS drift = {} dB (unity widths)",
            r_db
        );
    }

    #[test]
    fn low_width_zero_collapses_bass_to_mono() {
        // Pure 60 Hz with opposite polarity in L and R → mid = 0, side
        // is the full signal at 60 Hz. With low_width = 0 and a 500 Hz
        // cutoff, the LPF passes the entire 60 Hz side into the
        // attenuated bucket, so |L - R| should drop to ~0.
        let fs = 48_000u32;
        let n = 24_000usize;
        let l = sine(0.5, 60.0, fs, n);
        let r: Vec<f32> = l.iter().map(|v| -*v).collect();
        let frame = make_f32_stereo(&l, &r);
        let mut im = StereoImager::with(500.0, 0.0, 1.0);
        let out = im.process(&frame, f32_stereo(fs)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        let diff: Vec<f32> = lo.iter().zip(ro.iter()).map(|(a, b)| a - b).collect();
        let warm = (fs as f32 * 0.2) as usize;
        let in_diff: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| a - b).collect();
        let in_r = rms(&in_diff[warm..]);
        let out_r = rms(&diff[warm..]);
        let delta_db = 20.0 * (out_r / in_r).max(1.0e-12).log10();
        assert!(
            delta_db < -20.0,
            "low-band side not collapsed: {} dB",
            delta_db
        );
    }

    #[test]
    fn high_width_two_widens_top() {
        // 8 kHz tone, anti-phase in L/R. high_width = 2.0 should double
        // the side amplitude in the top band.
        let fs = 48_000u32;
        let n = 24_000usize;
        let l = sine(0.5, 8_000.0, fs, n);
        let r: Vec<f32> = l.iter().map(|v| -*v).collect();
        let frame = make_f32_stereo(&l, &r);
        let mut im = StereoImager::with(500.0, 1.0, 2.0);
        let out = im.process(&frame, f32_stereo(fs)).unwrap();
        let (lo, ro) = read_stereo(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let in_diff: Vec<f32> = l.iter().zip(r.iter()).map(|(a, b)| a - b).collect();
        let out_diff: Vec<f32> = lo.iter().zip(ro.iter()).map(|(a, b)| a - b).collect();
        let in_r = rms(&in_diff[warm..]);
        let out_r = rms(&out_diff[warm..]);
        let delta_db = 20.0 * (out_r / in_r).log10();
        // Doubled side ⇒ +6 dB on L-R. Filter sum ripple ⇒ within ±2 dB.
        assert!(
            (4.0..=8.0).contains(&delta_db),
            "high-band widening not ≈ +6 dB: {} dB",
            delta_db
        );
    }

    #[test]
    fn mono_input_passes_through() {
        let fs = 48_000u32;
        let samples = sine(0.4, 440.0, fs, 512);
        let mut bytes = Vec::with_capacity(samples.len() * 4);
        for s in &samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: samples.len() as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut im = StereoImager::new();
        let mono_params = AudioStreamParams {
            format: SampleFormat::F32,
            channels: 1,
            sample_rate: fs,
        };
        let out = im.process(&frame, mono_params).unwrap();
        let got: Vec<f32> = out[0].data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for i in 0..samples.len() {
            assert!((got[i] - samples[i]).abs() < 1.0e-6);
        }
    }
}
