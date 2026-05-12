//! Multi-band parametric equalizer — N [`Biquad`] sections in series.
//!
//! Builder-style construction:
//!
//! ```
//! use oxideav_audio_filter::Equalizer;
//! let _eq = Equalizer::new(48_000)
//!     .with_low_shelf(100.0, 0.707, 3.0)
//!     .with_peaking(1_000.0, 1.5, -2.0)
//!     .with_high_shelf(10_000.0, 0.707, 4.0);
//! ```
//!
//! Each `with_*` call appends a [`Biquad`] section configured for the given
//! kind / cutoff / Q / gain. Sections are evaluated in chain order (output of
//! section `i` feeds input of section `i+1`).
//!
//! State (per-channel `s1, s2`) is preserved across `process` calls so the
//! filter can be used in streaming mode.
//!
//! # Why not just use multiple `Biquad`s directly?
//!
//! You can — but a dedicated `Equalizer` saves the "decode → process → encode"
//! sample-conversion round-trip between sections. The chain runs entirely in
//! `f32` between sections, encoding only once at the output.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Multi-band parametric EQ.
#[derive(Debug, Clone)]
pub struct Equalizer {
    sample_rate: u32,
    sections: Vec<Biquad>,
}

impl Equalizer {
    /// New empty EQ. Use [`Equalizer::add_band`] (or the typed `with_*`
    /// helpers) to populate sections.
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            sample_rate: sample_rate_hz.max(1),
            sections: Vec::new(),
        }
    }

    /// Number of bands.
    pub fn band_count(&self) -> usize {
        self.sections.len()
    }

    /// Append a band by [`BiquadKind`]. Consumes and returns `self` for
    /// builder-style chaining.
    pub fn add_band(mut self, kind: BiquadKind) -> Self {
        self.sections.push(Biquad::new(kind));
        self
    }

    /// Builder: 2-pole low-pass.
    pub fn with_low_pass(self, cutoff_hz: f32, q: f32) -> Self {
        self.add_band(BiquadKind::LowPass { cutoff_hz, q })
    }

    /// Builder: 2-pole high-pass.
    pub fn with_high_pass(self, cutoff_hz: f32, q: f32) -> Self {
        self.add_band(BiquadKind::HighPass { cutoff_hz, q })
    }

    /// Builder: constant-skirt band-pass.
    pub fn with_band_pass(self, center_hz: f32, q: f32) -> Self {
        self.add_band(BiquadKind::BandPass { center_hz, q })
    }

    /// Builder: notch (band-stop).
    pub fn with_notch(self, center_hz: f32, q: f32) -> Self {
        self.add_band(BiquadKind::Notch { center_hz, q })
    }

    /// Builder: parametric peaking bell.
    pub fn with_peaking(self, center_hz: f32, q: f32, gain_db: f32) -> Self {
        self.add_band(BiquadKind::Peaking {
            center_hz,
            q,
            gain_db,
        })
    }

    /// Builder: low shelf.
    pub fn with_low_shelf(self, cutoff_hz: f32, q: f32, gain_db: f32) -> Self {
        self.add_band(BiquadKind::LowShelf {
            cutoff_hz,
            q,
            gain_db,
        })
    }

    /// Builder: high shelf.
    pub fn with_high_shelf(self, cutoff_hz: f32, q: f32, gain_db: f32) -> Self {
        self.add_band(BiquadKind::HighShelf {
            cutoff_hz,
            q,
            gain_db,
        })
    }

    /// Reset every band's internal state (delay-line memory) to zero.
    pub fn reset(&mut self) {
        for s in self.sections.iter_mut() {
            s.reset();
        }
    }

    /// Apply the chain to an interleaved sample buffer in place.
    pub fn process_in_place(&mut self, samples: &mut [f32], channels: u16, sample_rate_hz: u32) {
        for sec in self.sections.iter_mut() {
            sec.process_in_place(samples, channels, sample_rate_hz);
        }
    }
}

impl AudioFilter for Equalizer {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        // Build an interleaved view for each section call. Sections share
        // the per-band per-channel state internally; we iterate over them
        // sequentially.
        let n_chan = params.channels as usize;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);
        let mut interleaved = vec![0.0f32; n * n_chan];
        for s in 0..n {
            for (ch_idx, buf) in channels.iter().enumerate().take(n_chan) {
                interleaved[s * n_chan + ch_idx] = buf[s];
            }
        }
        for sec in self.sections.iter_mut() {
            sec.process_in_place(&mut interleaved, params.channels, self.sample_rate);
        }
        // Scatter back.
        for s in 0..n {
            for (ch_idx, buf) in channels.iter_mut().enumerate().take(n_chan) {
                buf[s] = interleaved[s * n_chan + ch_idx];
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

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    fn sine(freq_hz: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq_hz / fs as f32;
        (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect()
    }

    #[test]
    fn zero_bands_is_bypass() {
        let in_samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 0.3).collect();
        let frame = make_f32_mono(&in_samples);
        let mut eq = Equalizer::new(48_000);
        let out = eq.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..in_samples.len() {
            assert!(
                (got[i] - in_samples[i]).abs() < 1.0e-6,
                "empty EQ not bypass at {i}: got={} want={}",
                got[i],
                in_samples[i]
            );
        }
    }

    #[test]
    fn peaking_boost_amplifies_band() {
        // +12 dB at 1 kHz, Q=2 → strongly amplifies 1 kHz tone.
        let fs = 48_000u32;
        let n = 4096usize;
        let dry = sine(1_000.0, fs, n);
        let frame = make_f32_mono(&dry);
        let mut eq = Equalizer::new(fs).with_peaking(1_000.0, 2.0, 12.0);
        let out = eq.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip startup transient.
        let dry_rms = rms(&dry[1024..]);
        let wet_rms = rms(&got[1024..]);
        let gain_db = 20.0 * (wet_rms / dry_rms).log10();
        // Expect ~+12 dB; allow generous tolerance.
        assert!(
            (8.0..15.0).contains(&gain_db),
            "peaking boost = {gain_db} dB, expected ~12"
        );
    }

    #[test]
    fn three_band_eq_at_flat_settings_preserves_wideband() {
        // Three bands set to 0 dB gain each should leave a wideband signal
        // approximately unchanged.
        let fs = 48_000u32;
        let n = 4096usize;
        // Multi-tone test signal.
        let dry: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.1 * (2.0 * std::f32::consts::PI * 200.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 5_000.0 * t).sin()
            })
            .collect();
        let frame = make_f32_mono(&dry);
        let mut eq = Equalizer::new(fs)
            .with_peaking(100.0, 0.707, 0.0)
            .with_peaking(1_000.0, 0.707, 0.0)
            .with_peaking(10_000.0, 0.707, 0.0);
        let out = eq.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip transient.
        let dry_rms = rms(&dry[1024..]);
        let wet_rms = rms(&got[1024..]);
        let diff_db = 20.0 * (wet_rms / dry_rms).log10();
        assert!(
            diff_db.abs() < 0.1,
            "flat 3-band EQ altered wideband signal: diff_db = {diff_db}"
        );
    }

    #[test]
    fn band_count_tracks_adds() {
        let eq = Equalizer::new(48_000);
        assert_eq!(eq.band_count(), 0);
        let eq = eq
            .with_low_shelf(100.0, 0.707, 3.0)
            .with_peaking(1_000.0, 1.0, 0.0)
            .with_high_shelf(10_000.0, 0.707, 6.0);
        assert_eq!(eq.band_count(), 3);
    }

    #[test]
    fn low_pass_attenuates_high_frequency() {
        let fs = 48_000u32;
        let n = 4096usize;
        // 8 kHz tone, LPF at 1 kHz → strong attenuation.
        let dry = sine(8_000.0, fs, n);
        let frame = make_f32_mono(&dry);
        let mut eq = Equalizer::new(fs).with_low_pass(1_000.0, 0.707);
        let out = eq.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let in_rms = rms(&dry[1024..]);
        let out_rms = rms(&got[1024..]);
        let atten_db = 20.0 * (out_rms / in_rms).log10();
        // 3 octaves above cutoff → expect at least −20 dB attenuation.
        assert!(
            atten_db < -20.0,
            "LPF did not attenuate 8 kHz: atten_db = {atten_db}"
        );
    }
}
