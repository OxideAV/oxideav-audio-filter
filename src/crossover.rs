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
//! # Slopes
//!
//! Two topologies are offered via [`CrossoverSlope`]:
//!
//! * [`Butterworth2`](CrossoverSlope::Butterworth2) — a single
//!   Butterworth-2 (`q = 1/√2 ≈ 0.7071`) per band, 12 dB/oct. The
//!   magnitude responses cross at −3 dB, but the sum is **not** flat:
//!   a parallel Butterworth-2 LPF and HPF at the same `cutoff_hz` are
//!   180° apart there, so `low(f) + high(f)` *cancels* at the crossover
//!   — a deep null rather than unity. (Recombine the two bands through
//!   downstream processing, not by direct addition, if you use this
//!   slope.) This is the original behaviour and stays byte-for-byte
//!   identical via [`Crossover::new`] / [`Crossover::butterworth`].
//! * [`LinkwitzRiley4`](CrossoverSlope::LinkwitzRiley4) — a 4th-order
//!   Linkwitz-Riley split (24 dB/oct), built as **two cascaded**
//!   Butterworth-2 sections per band at the same `q = 1/√2`. Each band
//!   is then −6 dB at `cutoff_hz` and *in phase* with the other, so the
//!   summed output is a 2nd-order all-pass: magnitude-flat
//!   reconstruction (`|low(f) + high(f)| = 1` for all `f`), with only a
//!   frequency-dependent phase shift. This is the standard
//!   perfect-(magnitude)-reconstruction crossover used in multi-band
//!   processors and loudspeaker management. LR4 keeps both bands at the
//!   same polarity (the even cascade order cancels the per-section
//!   inversion an LR2 would need).
//!
//! # Parameters
//!
//! * `cutoff_hz` — crossover frequency. Clamped to `[1, 24 000]`.
//! * `q` — shared Q for both filters. Default `1/√2` (Butterworth-2).
//!   Clamped to `[0.1, 10]`. For [`LinkwitzRiley4`](CrossoverSlope::LinkwitzRiley4)
//!   the per-section `q` is forced to `1/√2` regardless (two cascaded
//!   Butterworth-2 is what *defines* LR4), so the `q` field only varies
//!   the [`Butterworth2`](CrossoverSlope::Butterworth2) slope.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Crossover filter slope / topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverSlope {
    /// Single Butterworth-2 per band, 12 dB/oct, −3 dB at `cutoff_hz`.
    /// Direct summation nulls at the crossover (LPF/HPF are 180° apart);
    /// recombine through downstream processing, not by addition.
    Butterworth2,
    /// 4th-order Linkwitz-Riley, 24 dB/oct: two cascaded Butterworth-2
    /// sections per band, −6 dB at `cutoff_hz`, in-phase summation →
    /// magnitude-flat all-pass reconstruction.
    LinkwitzRiley4,
}

/// Streaming two-way crossover.
#[derive(Debug, Clone)]
pub struct Crossover {
    cutoff_hz: f32,
    q: f32,
    slope: CrossoverSlope,
    /// Low-pass cascade: 1 section for Butterworth-2, 2 for LR4.
    lpf: Vec<Biquad>,
    /// High-pass cascade: 1 section for Butterworth-2, 2 for LR4.
    hpf: Vec<Biquad>,
}

impl Crossover {
    /// New Butterworth-2 crossover at `cutoff_hz`. `cutoff_hz` clamped to
    /// `[1, 24 000]`, `q` clamped to `[0.1, 10]`. Behaviour identical to
    /// all prior releases.
    pub fn new(cutoff_hz: f32, q: f32) -> Self {
        Self::with_slope(cutoff_hz, q, CrossoverSlope::Butterworth2)
    }

    /// New crossover at `cutoff_hz` with an explicit [`CrossoverSlope`].
    ///
    /// For [`CrossoverSlope::LinkwitzRiley4`] the per-section `q` is
    /// forced to `1/√2` (two cascaded Butterworth-2 *is* LR4); the `q`
    /// argument is retained for [`Crossover::q`] reporting and only
    /// affects the [`CrossoverSlope::Butterworth2`] slope.
    pub fn with_slope(cutoff_hz: f32, q: f32, slope: CrossoverSlope) -> Self {
        let cutoff_hz = cutoff_hz.clamp(1.0, 24_000.0);
        let q = q.clamp(0.1, 10.0);
        let (lpf, hpf) = Self::build_chains(cutoff_hz, q, slope);
        Self {
            cutoff_hz,
            q,
            slope,
            lpf,
            hpf,
        }
    }

    fn build_chains(cutoff_hz: f32, q: f32, slope: CrossoverSlope) -> (Vec<Biquad>, Vec<Biquad>) {
        match slope {
            CrossoverSlope::Butterworth2 => (
                vec![Biquad::new(BiquadKind::LowPass { cutoff_hz, q })],
                vec![Biquad::new(BiquadKind::HighPass { cutoff_hz, q })],
            ),
            CrossoverSlope::LinkwitzRiley4 => {
                // LR4 = two cascaded Butterworth-2 (q = 1/√2) per band.
                let bq = std::f32::consts::FRAC_1_SQRT_2;
                (
                    vec![
                        Biquad::new(BiquadKind::LowPass { cutoff_hz, q: bq }),
                        Biquad::new(BiquadKind::LowPass { cutoff_hz, q: bq }),
                    ],
                    vec![
                        Biquad::new(BiquadKind::HighPass { cutoff_hz, q: bq }),
                        Biquad::new(BiquadKind::HighPass { cutoff_hz, q: bq }),
                    ],
                )
            }
        }
    }

    /// Butterworth-Q (`1/√2`) two-way crossover (12 dB/oct).
    pub fn butterworth(cutoff_hz: f32) -> Self {
        Self::new(cutoff_hz, std::f32::consts::FRAC_1_SQRT_2)
    }

    /// 4th-order Linkwitz-Riley two-way crossover (24 dB/oct,
    /// magnitude-flat summation).
    pub fn linkwitz_riley(cutoff_hz: f32) -> Self {
        Self::with_slope(
            cutoff_hz,
            std::f32::consts::FRAC_1_SQRT_2,
            CrossoverSlope::LinkwitzRiley4,
        )
    }

    /// Currently-configured crossover frequency.
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Currently-configured Q.
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Currently-configured slope / topology.
    pub fn slope(&self) -> CrossoverSlope {
        self.slope
    }

    /// Reset all filter states (every cascade section).
    pub fn reset(&mut self) {
        for s in self.lpf.iter_mut() {
            s.reset();
        }
        for s in self.hpf.iter_mut() {
            s.reset();
        }
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
        for section in self.lpf.iter_mut() {
            section.process_in_place(&mut inter_low, n_chan, params.sample_rate);
        }
        for section in self.hpf.iter_mut() {
            section.process_in_place(&mut inter_high, n_chan, params.sample_rate);
        }
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

    // ---- Linkwitz-Riley 4th-order (LR4) ----

    #[test]
    fn lr4_reports_slope_and_q() {
        let xo = Crossover::linkwitz_riley(1_000.0);
        assert_eq!(xo.slope(), CrossoverSlope::LinkwitzRiley4);
        // Reported q is the Butterworth section q.
        assert!((xo.q() - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        // `new` / `butterworth` stay Butterworth-2.
        assert_eq!(
            Crossover::butterworth(1_000.0).slope(),
            CrossoverSlope::Butterworth2
        );
    }

    #[test]
    fn lr4_is_minus_6db_each_band_at_cutoff() {
        // Two cascaded −3 dB Butterworth-2 sections → −6 dB per band at fc.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let n = 48_000usize;
        let samples = sine_at(fs, fc, n);
        let frame = make_f32_mono(&samples);
        let mut xo = Crossover::linkwitz_riley(fc);
        let out = xo.process(&frame, f32_mono(fs)).unwrap();
        let (lo, hi) = read_split(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let lo_db = 20.0 * (rms(&lo[warm..]) / rms(&samples[warm..])).log10();
        let hi_db = 20.0 * (rms(&hi[warm..]) / rms(&samples[warm..])).log10();
        // −6 dB nominal; allow a 2 dB window.
        assert!(
            (-8.0..=-4.0).contains(&lo_db),
            "LR4 low-band gain at fc not near −6 dB: {} dB",
            lo_db
        );
        assert!(
            (-8.0..=-4.0).contains(&hi_db),
            "LR4 high-band gain at fc not near −6 dB: {} dB",
            hi_db
        );
    }

    #[test]
    fn lr4_rejects_steeper_than_butterworth2() {
        // One octave below fc, the high band of LR4 (24 dB/oct) should
        // reject markedly more than the Butterworth-2 high band (12 dB/oct).
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let probe = 250.0f32; // two octaves below fc → big LR4 advantage
        let n = 48_000usize;
        let samples = sine_at(fs, probe, n);
        let frame = make_f32_mono(&samples);
        let warm = (fs as f32 * 0.2) as usize;

        let mut bw = Crossover::butterworth(fc);
        let out_bw = bw.process(&frame, f32_mono(fs)).unwrap();
        let (_, hi_bw) = read_split(&out_bw[0]);
        let bw_db = 20.0 * (rms(&hi_bw[warm..]) / rms(&samples[warm..])).log10();

        let mut lr = Crossover::linkwitz_riley(fc);
        let out_lr = lr.process(&frame, f32_mono(fs)).unwrap();
        let (_, hi_lr) = read_split(&out_lr[0]);
        let lr_db = 20.0 * (rms(&hi_lr[warm..]) / rms(&samples[warm..])).log10();

        // LR4 rejection should beat Butterworth-2 by well over 10 dB.
        assert!(
            lr_db < bw_db - 10.0,
            "LR4 high-band ({} dB) not >10 dB below Butterworth-2 ({} dB) at {} Hz",
            lr_db,
            bw_db,
            probe
        );
    }

    #[test]
    fn lr4_reconstruction_is_magnitude_flat() {
        // The defining LR4 property: low(f) + high(f) is a pure all-pass,
        // so the summed magnitude is ~0 dB at EVERY frequency, including
        // the crossover — unlike the Butterworth-2 ~+3 dB lump there.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let n = 48_000usize;
        let warm = (fs as f32 * 0.2) as usize;

        for &probe in &[120.0f32, 500.0, 1_000.0, 2_000.0, 8_000.0] {
            let samples = sine_at(fs, probe, n);
            let frame = make_f32_mono(&samples);
            let mut xo = Crossover::linkwitz_riley(fc);
            let out = xo.process(&frame, f32_mono(fs)).unwrap();
            let (lo, hi) = read_split(&out[0]);
            let sum: Vec<f32> = lo.iter().zip(hi.iter()).map(|(&l, &h)| l + h).collect();
            let sum_db = 20.0 * (rms(&sum[warm..]) / rms(&samples[warm..])).log10();
            // All-pass reconstruction: |L+H| = |in| to within ~1 dB.
            assert!(
                sum_db.abs() < 1.0,
                "LR4 L+H not magnitude-flat at {} Hz: {} dB",
                probe,
                sum_db
            );
        }
    }

    #[test]
    fn butterworth2_summation_has_crossover_null() {
        // Contrast / sanity: a parallel Butterworth-2 LPF + HPF at the
        // same fc are 180° apart there, so their *sum* cancels — a deep
        // null at the crossover, NOT a flat response. This is the
        // reconstruction defect LR4 removes; confirms the two topologies
        // genuinely differ.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let n = 48_000usize;
        let warm = (fs as f32 * 0.2) as usize;
        let samples = sine_at(fs, fc, n);
        let frame = make_f32_mono(&samples);
        let mut xo = Crossover::butterworth(fc);
        let out = xo.process(&frame, f32_mono(fs)).unwrap();
        let (lo, hi) = read_split(&out[0]);
        let sum: Vec<f32> = lo.iter().zip(hi.iter()).map(|(&l, &h)| l + h).collect();
        let sum_db = 20.0 * (rms(&sum[warm..]) / rms(&samples[warm..])).log10();
        // Expect a deep cancellation null (far below unity).
        assert!(
            sum_db < -20.0,
            "Butterworth-2 summation should null at fc, got {} dB",
            sum_db
        );
    }
}
