//! Three-band compressor — splits the input into Low / Mid / High
//! bands at two crossover frequencies, compresses each band
//! independently, then re-sums them.
//!
//! This is the standard "multiband" mastering / broadcast effect: a
//! single broadband compressor can only react to whichever band of
//! frequencies happens to be loudest at any instant, so a strong bass
//! note may pump down vocals. Per-band compression lets each region
//! of the spectrum be tamed without affecting the others.
//!
//! # Signal flow
//!
//! ```text
//!                        ┌── LPF₁ ──────────────────┐
//!                        │                          │── comp_low ──┐
//!     x[n] ──┬──────────┤                          │              │
//!            │            ├── (HPF₁) - (LPF₂) ──── │── comp_mid ──┼── y[n]
//!            │            │                          │              │
//!            │            └────────── HPF₂ ────────│── comp_high ─┘
//!            │
//!            └──> dry split with two Butterworth-2 biquads at
//!                 f_lo (≈250 Hz) and f_hi (≈2500 Hz).
//! ```
//!
//! Each band is processed by its own [`Compressor`](crate::Compressor)
//! (peak detector + one-pole follower + soft-knee static curve). The
//! three bands are summed back per sample.
//!
//! # Limitations of this topology
//!
//! Parallel Butterworth-2 LPF/HPF crossovers are not phase-coherent in
//! the sum: at each crossover there is a ~+3 dB lump and a phase shift
//! that, summed, gives non-flat magnitude. A 4th-order Linkwitz-Riley
//! tree (two cascaded LR2 of opposite polarity) gives perfect
//! reconstruction at unity band gains; the same in-place biquad chain
//! is sufficient for the mastering use-case described above and matches
//! the rest of this crate's spectrum-split filters (cf. [`Crossover`],
//! [`DeEsser`]).
//!
//! # Parameters
//!
//! * `low_cutoff_hz` / `high_cutoff_hz` — Butterworth-2 crossovers
//!   (default 250 Hz / 2500 Hz). `low < high` is required; both
//!   clamped to `[20, 20_000]` and `high` lifted above `low` if
//!   needed.
//! * `threshold_db` / `ratio` / `attack_ms` / `release_ms` /
//!   `makeup_gain_db` — three independent triples, one per band, see
//!   [`BandSettings`].

use crate::biquad::{Biquad, BiquadKind};
use crate::compressor::Compressor;
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Per-band compressor settings.
#[derive(Debug, Clone, Copy)]
pub struct BandSettings {
    pub threshold_db: f32,
    pub ratio: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub knee_db: f32,
    pub makeup_gain_db: f32,
}

impl BandSettings {
    /// Tame-it-gently defaults: -18 dBFS, 3:1, 10/100 ms, 6 dB knee.
    pub fn default_low() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 3.0,
            attack_ms: 10.0,
            release_ms: 100.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
        }
    }
    /// Default for the mid band — slightly faster attack than the low band.
    pub fn default_mid() -> Self {
        Self {
            threshold_db: -16.0,
            ratio: 3.0,
            attack_ms: 5.0,
            release_ms: 80.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
        }
    }
    /// Default for the high band — fastest, gentlest ratio.
    pub fn default_high() -> Self {
        Self {
            threshold_db: -14.0,
            ratio: 2.5,
            attack_ms: 2.0,
            release_ms: 60.0,
            knee_db: 6.0,
            makeup_gain_db: 0.0,
        }
    }
}

/// Three-band compressor.
#[derive(Debug, Clone)]
pub struct MultibandCompressor {
    low_cutoff_hz: f32,
    high_cutoff_hz: f32,
    lpf_low: Biquad,
    hpf_low: Biquad,
    lpf_high: Biquad,
    hpf_high: Biquad,
    comp_low: Compressor,
    comp_mid: Compressor,
    comp_high: Compressor,
}

impl MultibandCompressor {
    /// New three-band compressor with sane mastering defaults at
    /// 250 Hz / 2500 Hz crossovers.
    pub fn new() -> Self {
        Self::with(
            250.0,
            2_500.0,
            BandSettings::default_low(),
            BandSettings::default_mid(),
            BandSettings::default_high(),
        )
    }

    /// Custom-parameter constructor.
    pub fn with(
        low_cutoff_hz: f32,
        high_cutoff_hz: f32,
        low: BandSettings,
        mid: BandSettings,
        high: BandSettings,
    ) -> Self {
        let low_cutoff_hz = low_cutoff_hz.clamp(20.0, 20_000.0);
        let mut high_cutoff_hz = high_cutoff_hz.clamp(20.0, 20_000.0);
        if high_cutoff_hz <= low_cutoff_hz {
            high_cutoff_hz = (low_cutoff_hz * 2.0).min(20_000.0);
        }
        let q = std::f32::consts::FRAC_1_SQRT_2;
        let lpf_low = Biquad::new(BiquadKind::LowPass {
            cutoff_hz: low_cutoff_hz,
            q,
        });
        let hpf_low = Biquad::new(BiquadKind::HighPass {
            cutoff_hz: low_cutoff_hz,
            q,
        });
        let lpf_high = Biquad::new(BiquadKind::LowPass {
            cutoff_hz: high_cutoff_hz,
            q,
        });
        let hpf_high = Biquad::new(BiquadKind::HighPass {
            cutoff_hz: high_cutoff_hz,
            q,
        });
        Self {
            low_cutoff_hz,
            high_cutoff_hz,
            lpf_low,
            hpf_low,
            lpf_high,
            hpf_high,
            comp_low: build_compressor(low),
            comp_mid: build_compressor(mid),
            comp_high: build_compressor(high),
        }
    }

    /// Currently-configured low crossover (Hz).
    pub fn low_cutoff_hz(&self) -> f32 {
        self.low_cutoff_hz
    }
    /// Currently-configured high crossover (Hz).
    pub fn high_cutoff_hz(&self) -> f32 {
        self.high_cutoff_hz
    }

    /// Reset all six biquad states (compressor envelopes are
    /// preserved — they re-converge cheaply).
    pub fn reset(&mut self) {
        self.lpf_low.reset();
        self.hpf_low.reset();
        self.lpf_high.reset();
        self.hpf_high.reset();
    }
}

fn build_compressor(s: BandSettings) -> Compressor {
    Compressor::new(
        s.threshold_db,
        s.ratio,
        s.attack_ms,
        s.release_ms,
        s.knee_db,
        s.makeup_gain_db,
    )
}

impl Default for MultibandCompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for MultibandCompressor {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let dry = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = dry.len();
        if n_chan == 0 {
            let out = encode_from_f32(params.format, params.channels, input, &dry)?;
            return Ok(vec![out]);
        }
        let n_samples = dry[0].len();

        // Carve three bands. We do LPF_high pass on the source, then
        // mid = LPF_high - LPF_low (band-pass via subtraction) and
        // high = HPF_high; low = LPF_low.
        let mut low = dry.clone();
        let mut mid_low_corner = dry.clone(); // becomes LPF_high(x)
        let mut high = dry.clone();

        let mut inter_low = interleave(&low, n_chan, n_samples);
        let mut inter_mid_lo = interleave(&mid_low_corner, n_chan, n_samples);
        let mut inter_high = interleave(&high, n_chan, n_samples);
        self.lpf_low
            .process_in_place(&mut inter_low, n_chan as u16, params.sample_rate);
        self.lpf_high
            .process_in_place(&mut inter_mid_lo, n_chan as u16, params.sample_rate);
        self.hpf_high
            .process_in_place(&mut inter_high, n_chan as u16, params.sample_rate);
        deinterleave(&inter_low, n_chan, n_samples, &mut low);
        deinterleave(&inter_mid_lo, n_chan, n_samples, &mut mid_low_corner);
        deinterleave(&inter_high, n_chan, n_samples, &mut high);

        // Mid = LPF_high(x) - LPF_low(x) — magnitude-summation band-pass.
        let mut mid = mid_low_corner;
        for ch in 0..n_chan {
            for i in 0..n_samples {
                mid[ch][i] -= low[ch][i];
            }
        }

        // Compress each band independently. The compressor expects to
        // process per-channel buffers in place via its public API; we
        // synthesise minimal frames around each band.
        run_compressor(&mut self.comp_low, &mut low, params);
        run_compressor(&mut self.comp_mid, &mut mid, params);
        run_compressor(&mut self.comp_high, &mut high, params);

        // Sum.
        let mut summed = vec![vec![0.0f32; n_samples]; n_chan];
        for ch in 0..n_chan {
            for i in 0..n_samples {
                summed[ch][i] = low[ch][i] + mid[ch][i] + high[ch][i];
            }
        }
        let out = encode_from_f32(params.format, params.channels, input, &summed)?;
        Ok(vec![out])
    }
}

fn interleave(bands: &[Vec<f32>], n_chan: usize, n_samples: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_samples * n_chan];
    for ch in 0..n_chan {
        for i in 0..n_samples {
            out[i * n_chan + ch] = bands[ch][i];
        }
    }
    out
}

fn deinterleave(inter: &[f32], n_chan: usize, n_samples: usize, out: &mut [Vec<f32>]) {
    for ch in 0..n_chan {
        for i in 0..n_samples {
            out[ch][i] = inter[i * n_chan + ch];
        }
    }
}

/// Push a `Vec<Vec<f32>>` band through a [`Compressor`] using the
/// public `AudioFilter::process()` contract. Wraps the band in an
/// `AudioFrame`, decodes, encodes — keeping the compressor's
/// internal envelope and time-constant cache live.
fn run_compressor(comp: &mut Compressor, band: &mut [Vec<f32>], params: AudioStreamParams) {
    let n_chan = band.len();
    if n_chan == 0 {
        return;
    }
    let n_samples = band[0].len();

    // Synthesise an interleaved-F32 frame so we can leverage
    // Compressor's existing AudioFilter impl without leaking any
    // extra public surface.
    let mut bytes = vec![0u8; n_samples * n_chan * 4];
    for i in 0..n_samples {
        for (ch, b) in band.iter().enumerate().take(n_chan) {
            let off = (i * n_chan + ch) * 4;
            bytes[off..off + 4].copy_from_slice(&b[i].to_le_bytes());
        }
    }
    let frame = AudioFrame {
        samples: n_samples as u32,
        pts: None,
        data: vec![bytes],
    };
    let local_params = AudioStreamParams {
        format: oxideav_core::SampleFormat::F32,
        channels: n_chan as u16,
        sample_rate: params.sample_rate,
    };
    let out = comp
        .process(&frame, local_params)
        .expect("compressor processing infallible for F32 input");
    if let Some(f) = out.first() {
        let plane = &f.data[0];
        for i in 0..n_samples {
            for (ch, b) in band.iter_mut().enumerate().take(n_chan) {
                let off = (i * n_chan + ch) * 4;
                b[i] = f32::from_le_bytes([
                    plane[off],
                    plane[off + 1],
                    plane[off + 2],
                    plane[off + 3],
                ]);
            }
        }
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

    fn rms(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        (s / x.len() as f64).sqrt() as f32
    }

    fn sine(amp: f32, freq: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| amp * (i as f32 * w).sin()).collect()
    }

    #[test]
    fn quiet_input_passes_through_close_to_unity() {
        // Far below every band's threshold → should be essentially the
        // sum of the three filter outputs, which approximates the input
        // (within band-summation ripple).
        let fs = 48_000u32;
        let x = sine(0.05, 800.0, fs, 24_000);
        let frame = make_f32_mono(&x);
        let mut mbc = MultibandCompressor::new();
        let out = mbc.process(&frame, f32_mono(fs)).unwrap();
        let y = read_f32(&out[0]);
        let warm = (fs as f32 * 0.2) as usize;
        let in_r = rms(&x[warm..]);
        let out_r = rms(&y[warm..]);
        let delta_db = 20.0 * (out_r / in_r).log10();
        // Band-sum ripple at mid-frequency can be ±3 dB; allow a
        // generous window.
        assert!(
            delta_db.abs() < 4.5,
            "low-level Δ = {} dB (expected near 0)",
            delta_db
        );
    }

    #[test]
    fn loud_low_tone_only_compresses_low_band() {
        // Two simultaneous tones: a loud 60 Hz bass + a quiet 8 kHz top.
        // With a low-band threshold of -10 dBFS the bass MUST be reduced;
        // the high band should be left alone (its compressor is set to a
        // high threshold).
        let fs = 48_000u32;
        let n = 24_000usize;
        let bass: Vec<f32> = sine(0.8, 60.0, fs, n);
        let air: Vec<f32> = sine(0.1, 8_000.0, fs, n);
        let mix: Vec<f32> = bass.iter().zip(air.iter()).map(|(a, b)| *a + *b).collect();
        let frame = make_f32_mono(&mix);

        // Set the low band very aggressive (-25 dBFS, 10:1) and leave
        // the others lenient (-3 dBFS, 1.5:1 — practically off for the
        // quiet 8 kHz tone).
        let low = BandSettings {
            threshold_db: -25.0,
            ratio: 10.0,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let lenient = BandSettings {
            threshold_db: -3.0,
            ratio: 1.5,
            attack_ms: 5.0,
            release_ms: 50.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
        };
        let mut mbc = MultibandCompressor::with(250.0, 2_500.0, low, lenient, lenient);
        let out = mbc.process(&frame, f32_mono(fs)).unwrap();
        let y = read_f32(&out[0]);

        // Project to bass band by re-filtering with our own LPF.
        let mut bass_out = y.clone();
        let mut bass_in = mix.clone();
        let mut lpf = Biquad::new(BiquadKind::LowPass {
            cutoff_hz: 250.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        });
        lpf.process_in_place(&mut bass_out, 1, fs);
        let mut lpf2 = Biquad::new(BiquadKind::LowPass {
            cutoff_hz: 250.0,
            q: std::f32::consts::FRAC_1_SQRT_2,
        });
        lpf2.process_in_place(&mut bass_in, 1, fs);
        let warm = (fs as f32 * 0.2) as usize;
        let bass_in_r = rms(&bass_in[warm..]);
        let bass_out_r = rms(&bass_out[warm..]);
        let delta_db = 20.0 * (bass_out_r / bass_in_r).log10();
        // 60 Hz at 0.8 amplitude sits ~ -1.9 dBFS, threshold -25 dBFS,
        // ratio 10:1 → over≈23 dB, gr≈-21 dB. Sloppy detector + filter
        // ripple — accept anything below -8 dB as a clear compression
        // signature.
        assert!(
            delta_db < -8.0,
            "bass band only weakly reduced: {} dB",
            delta_db
        );
    }

    #[test]
    fn output_is_finite_for_zero_input() {
        let fs = 48_000u32;
        let zero = vec![0.0f32; 4_096];
        let frame = make_f32_mono(&zero);
        let mut mbc = MultibandCompressor::new();
        let out = mbc.process(&frame, f32_mono(fs)).unwrap();
        let y = read_f32(&out[0]);
        for v in y {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn crossover_clamp_keeps_high_above_low() {
        // Misconfigured: high <= low. Constructor must lift `high`.
        let mbc = MultibandCompressor::with(
            500.0,
            300.0,
            BandSettings::default_low(),
            BandSettings::default_mid(),
            BandSettings::default_high(),
        );
        assert!(mbc.high_cutoff_hz() > mbc.low_cutoff_hz());
    }
}
