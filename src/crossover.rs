//! Two-way crossover — splits one audio input into separate low-pass
//! and high-pass bands at a chosen cutoff.
//!
//! Useful as the front end of a multi-band processor: feed each band
//! into a separate downstream filter (compressor / EQ / delay) and
//! sum them again at the back.
//!
//! # Output layout
//!
//! The output frame has `2 · channels` interleaved channels:
//!
//! ```text
//! [ low_L  low_R  ...  high_L  high_R  ... ]
//! ```
//!
//! That is, the first `channels` channels carry the low band and the
//! next `channels` channels carry the high band. Stereo input → 4-channel
//! output; mono input → 2-channel output `[low, high]`.
//!
//! # Recurrence
//!
//! Two independent [`Biquad`](crate::Biquad) chains run in parallel per
//! channel — one low-pass, one high-pass, both at `cutoff_hz`:
//!
//! ```text
//! low_band[n]  = lpf(x[n])
//! high_band[n] = hpf(x[n])
//! ```
//!
//! With `q = 1/√2 ≈ 0.7071` (Butterworth) the magnitude responses cross
//! at −3 dB so `low(f) + high(f) ≈ x(f)` near the crossover. The sum is
//! **not** perfectly flat (LR2 / LR4 require all-pass-equivalent
//! topologies); a magnitude-summation Butterworth-2 crossover has a
//! ~+3 dB lump at the crossover frequency. For perfect reconstruction
//! sum the two bands phase-aware: a fourth-order Linkwitz-Riley
//! crossover (two cascaded Butterworth-2 of opposite polarity) is the
//! standard fix; this implementation gives the simpler two-pole pair.
//!
//! # Parameters
//!
//! * `cutoff_hz` — crossover frequency. Clamped to `[1, 24 000]`.
//! * `q` — shared Q for both filters. Default `1/√2` (Butterworth-2).
//!   Clamped to `[0.1, 10]`.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming two-way crossover.
#[derive(Debug, Clone)]
pub struct Crossover {
    cutoff_hz: f32,
    q: f32,
    lpf: Biquad,
    hpf: Biquad,
}

impl Crossover {
    /// New crossover at `cutoff_hz`. `cutoff_hz` clamped to `[1, 24 000]`,
    /// `q` clamped to `[0.1, 10]`.
    pub fn new(cutoff_hz: f32, q: f32) -> Self {
        let cutoff_hz = cutoff_hz.clamp(1.0, 24_000.0);
        let q = q.clamp(0.1, 10.0);
        Self {
            cutoff_hz,
            q,
            lpf: Biquad::new(BiquadKind::LowPass { cutoff_hz, q }),
            hpf: Biquad::new(BiquadKind::HighPass { cutoff_hz, q }),
        }
    }

    /// Butterworth-Q (`1/√2`) two-way crossover.
    pub fn butterworth(cutoff_hz: f32) -> Self {
        Self::new(cutoff_hz, std::f32::consts::FRAC_1_SQRT_2)
    }

    /// Currently-configured crossover frequency.
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Currently-configured Q.
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Reset both filter states.
    pub fn reset(&mut self) {
        self.lpf.reset();
        self.hpf.reset();
    }
}

impl AudioFilter for Crossover {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let channels_f32 = decode_to_f32(input, params.format, params.channels)?;
        let n_chan_in = channels_f32.len();
        if n_chan_in == 0 {
            // Pathological — pass through.
            let out = encode_from_f32(params.format, params.channels, input, &channels_f32)?;
            return Ok(vec![out]);
        }

        // Run LPF and HPF over copies of the input.
        let mut low: Vec<Vec<f32>> = channels_f32.to_vec();
        let mut high: Vec<Vec<f32>> = channels_f32.to_vec();

        // Interleave each side, run, de-interleave.
        let n_samples = low[0].len();
        let n_chan = n_chan_in as u16;
        let mut inter_low = vec![0.0f32; n_samples * n_chan_in];
        let mut inter_high = vec![0.0f32; n_samples * n_chan_in];
        for ch in 0..n_chan_in {
            for i in 0..n_samples {
                inter_low[i * n_chan_in + ch] = low[ch][i];
                inter_high[i * n_chan_in + ch] = high[ch][i];
            }
        }
        self.lpf
            .process_in_place(&mut inter_low, n_chan, params.sample_rate);
        self.hpf
            .process_in_place(&mut inter_high, n_chan, params.sample_rate);
        for ch in 0..n_chan_in {
            for i in 0..n_samples {
                low[ch][i] = inter_low[i * n_chan_in + ch];
                high[ch][i] = inter_high[i * n_chan_in + ch];
            }
        }

        // Pack as `[low_chans..., high_chans...]` and synthesise a
        // frame with 2× channels.
        let mut combined: Vec<Vec<f32>> = Vec::with_capacity(n_chan_in * 2);
        combined.extend(low);
        combined.extend(high);
        let new_channels = (n_chan_in as u16) * 2;
        let out = encode_from_f32(params.format, new_channels, input, &combined)?;
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

    /// De-interleave a `2`-channel `f32` frame into `(low, high)` mono
    /// buffers. Mono input → 2-channel output (low, high).
    fn read_split(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let bytes = &frame.data[0];
        let n = frame.samples as usize;
        let mut lo = Vec::with_capacity(n);
        let mut hi = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 2 * 4;
            lo.push(f32::from_le_bytes([
                bytes[off],
                bytes[off + 1],
                bytes[off + 2],
                bytes[off + 3],
            ]));
            hi.push(f32::from_le_bytes([
                bytes[off + 4],
                bytes[off + 5],
                bytes[off + 6],
                bytes[off + 7],
            ]));
        }
        (lo, hi)
    }

    fn rms(samples: &[f32]) -> f32 {
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    fn sine_at(fs: u32, freq: f32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| (i as f32 * w).sin()).collect()
    }

    #[test]
    fn output_has_double_channel_count() {
        let frame = make_f32_mono(&vec![0.5f32; 256]);
        let mut xo = Crossover::butterworth(1_000.0);
        let out = xo.process(&frame, f32_mono(48_000)).unwrap();
        let bytes = &out[0].data[0];
        // 256 samples × 2 channels × 4 B = 2048 B.
        assert_eq!(bytes.len(), 256 * 2 * 4);
    }

    #[test]
    fn low_tone_appears_in_low_band() {
        // 100 Hz tone with 1 kHz crossover → present in low, absent in high.
        let fs = 48_000u32;
        let n = 48_000usize;
        let samples = sine_at(fs, 100.0, n);
        let frame = make_f32_mono(&samples);
        let mut xo = Crossover::butterworth(1_000.0);
        let out = xo.process(&frame, f32_mono(fs)).unwrap();
        let (lo, hi) = read_split(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let lo_r = rms(&lo[warm..]);
        let hi_r = rms(&hi[warm..]);
        let in_r = rms(&samples[warm..]);
        let lo_db = 20.0 * (lo_r / in_r).log10();
        let hi_db = 20.0 * (hi_r / in_r).log10();
        assert!(lo_db > -1.5, "low band lost 100 Hz tone: {} dB", lo_db);
        assert!(hi_db < -20.0, "high band leaked 100 Hz tone: {} dB", hi_db);
    }

    #[test]
    fn high_tone_appears_in_high_band() {
        // 10 kHz tone with 1 kHz crossover → present in high, absent in low.
        let fs = 48_000u32;
        let n = 48_000usize;
        let samples = sine_at(fs, 10_000.0, n);
        let frame = make_f32_mono(&samples);
        let mut xo = Crossover::butterworth(1_000.0);
        let out = xo.process(&frame, f32_mono(fs)).unwrap();
        let (lo, hi) = read_split(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let lo_r = rms(&lo[warm..]);
        let hi_r = rms(&hi[warm..]);
        let in_r = rms(&samples[warm..]);
        let lo_db = 20.0 * (lo_r / in_r).log10();
        let hi_db = 20.0 * (hi_r / in_r).log10();
        assert!(hi_db > -1.5, "high band lost 10 kHz tone: {} dB", hi_db);
        assert!(lo_db < -20.0, "low band leaked 10 kHz tone: {} dB", lo_db);
    }

    #[test]
    fn cutoff_tone_is_minus_3db_each_band() {
        // At the crossover frequency, each Butterworth-2 band should
        // pass ~ −3 dB (1/√2 amplitude).
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let n = 48_000usize;
        let samples = sine_at(fs, fc, n);
        let frame = make_f32_mono(&samples);
        let mut xo = Crossover::butterworth(fc);
        let out = xo.process(&frame, f32_mono(fs)).unwrap();
        let (lo, hi) = read_split(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let lo_db = 20.0 * (rms(&lo[warm..]) / rms(&samples[warm..])).log10();
        let hi_db = 20.0 * (rms(&hi[warm..]) / rms(&samples[warm..])).log10();
        // Allow a 2 dB window around the −3 dB nominal point.
        assert!(
            (-5.0..=-1.0).contains(&lo_db),
            "low-band gain at fc not near −3 dB: {} dB",
            lo_db
        );
        assert!(
            (-5.0..=-1.0).contains(&hi_db),
            "high-band gain at fc not near −3 dB: {} dB",
            hi_db
        );
    }
}
