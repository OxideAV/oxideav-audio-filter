//! Biquadratic IIR EQ filter family.
//!
//! Implements eight second-order IIR configurations sharing a single
//! direct-form-II-transposed core. State is held in `f64` to keep the
//! recurrence numerically stable for low cutoff / high-Q settings; the
//! per-sample input and output remain `f32`.
//!
//! # Recurrence (DF-II-T)
//!
//! After normalising by `a0` the difference equation is
//!
//! ```text
//! y[n]  = b0*x[n] + s1[n-1]
//! s1[n] = b1*x[n] - a1*y[n] + s2[n-1]
//! s2[n] = b2*x[n] - a2*y[n]
//! ```
//!
//! Each channel keeps its own `(s1, s2)` pair so stereo (or higher
//! channel-count) inputs do not cross-talk.
//!
//! # Coefficient derivation
//!
//! All seven configurations are derived from the **bilinear transform**
//! of an analog prototype: starting from `H(s)` for the analog filter,
//! pre-warp the cutoff with `ω = 2π·f_c / f_s`, substitute
//! `s ← (1 - z⁻¹)/(1 + z⁻¹)` (pre-warped), and gather the
//! `(b0, b1, b2, a0, a1, a2)` polynomial coefficients in `z⁻¹`. The
//! resulting algebra is the well-known "RBJ Audio EQ Cookbook" form;
//! we use the equations as math (in our own variable names) — no
//! reference C source is consulted.
//!
//! Working variables shared by every kind:
//!
//! ```text
//! ω      = 2π · f_c / f_s
//! cosω   = cos(ω)
//! sinω   = sin(ω)
//! α      = sinω / (2Q)
//! ```
//!
//! For shelving / peaking configurations we also use
//! `A = 10^(gain_db / 40)` (square root of linear gain — shelving
//! filters operate on amplitude, the `/40` instead of `/20` accounts
//! for the half-gain at the corner).
//!
//! # Public API
//!
//! ```
//! use oxideav_audio_filter::biquad::{Biquad, BiquadKind};
//! let mut bq = Biquad::new(BiquadKind::LowPass {
//!     cutoff_hz: 1_000.0,
//!     q: 0.707,
//! });
//! let mut buf = [0.0f32; 1024];
//! bq.process_in_place(&mut buf, 1, 48_000);
//! ```

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// One of the eight supported biquad configurations.
///
/// All variants are derived from the bilinear transform of their analog
/// prototypes; see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BiquadKind {
    /// 2-pole low-pass (Butterworth-style when `Q = 1/√2 ≈ 0.707`).
    LowPass { cutoff_hz: f32, q: f32 },
    /// 2-pole high-pass.
    HighPass { cutoff_hz: f32, q: f32 },
    /// Constant-skirt-gain band-pass; peak gain ≈ Q at the centre.
    BandPass { center_hz: f32, q: f32 },
    /// Notch (band-stop).
    Notch { center_hz: f32, q: f32 },
    /// Parametric mid-EQ bell. Positive `gain_db` boosts, negative cuts.
    Peaking {
        center_hz: f32,
        q: f32,
        gain_db: f32,
    },
    /// Low shelf — gain applied below the cutoff, unity above.
    LowShelf {
        cutoff_hz: f32,
        q: f32,
        gain_db: f32,
    },
    /// High shelf — gain applied above the cutoff, unity below.
    HighShelf {
        cutoff_hz: f32,
        q: f32,
        gain_db: f32,
    },
    /// Second-order all-pass — `|H(e^{jω})| ≡ 1` for every `ω` (flat
    /// magnitude response), but the phase rotates through `−2π` as
    /// frequency sweeps from DC to Nyquist, crossing `−π` at the
    /// centre frequency. Width of the phase-rotation transition is
    /// set by `Q`: higher `Q` → sharper sweep. Used as a phase-
    /// alignment / decorrelation primitive in reverb tanks, phaser
    /// stages, and crossover phase-correction networks.
    AllPass { center_hz: f32, q: f32 },
}

/// Normalised second-order section coefficients (`b0, b1, b2, a1, a2`)
/// after dividing through by `a0`.
#[derive(Debug, Clone, Copy)]
struct Coeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Coeffs {
    fn from_kind(kind: BiquadKind, sample_rate_hz: u32) -> Self {
        let fs = sample_rate_hz.max(1) as f64;
        match kind {
            BiquadKind::LowPass { cutoff_hz, q } => low_pass(fs, cutoff_hz as f64, q as f64),
            BiquadKind::HighPass { cutoff_hz, q } => high_pass(fs, cutoff_hz as f64, q as f64),
            BiquadKind::BandPass { center_hz, q } => band_pass(fs, center_hz as f64, q as f64),
            BiquadKind::Notch { center_hz, q } => notch(fs, center_hz as f64, q as f64),
            BiquadKind::Peaking {
                center_hz,
                q,
                gain_db,
            } => peaking(fs, center_hz as f64, q as f64, gain_db as f64),
            BiquadKind::LowShelf {
                cutoff_hz,
                q,
                gain_db,
            } => low_shelf(fs, cutoff_hz as f64, q as f64, gain_db as f64),
            BiquadKind::HighShelf {
                cutoff_hz,
                q,
                gain_db,
            } => high_shelf(fs, cutoff_hz as f64, q as f64, gain_db as f64),
            BiquadKind::AllPass { center_hz, q } => all_pass(fs, center_hz as f64, q as f64),
        }
    }
}

/// Pre-warped angular frequency and skirt parameter.
struct WarpVars {
    cosw: f64,
    sinw: f64,
    alpha: f64,
}

fn warp(fs: f64, fc: f64, q: f64) -> WarpVars {
    let w = 2.0 * std::f64::consts::PI * (fc.max(1.0e-6) / fs);
    let cosw = w.cos();
    let sinw = w.sin();
    let q = q.max(1.0e-6);
    let alpha = sinw / (2.0 * q);
    WarpVars { cosw, sinw, alpha }
}

fn normalise(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Coeffs {
    Coeffs {
        b0: b0 / a0,
        b1: b1 / a0,
        b2: b2 / a0,
        a1: a1 / a0,
        a2: a2 / a0,
    }
}

/// 2-pole low-pass. Analog prototype `H(s) = 1 / (s² + s/Q + 1)`,
/// bilinear → `b = ((1-cosω)/2, 1-cosω, (1-cosω)/2)`,
/// `a = (1+α, -2cosω, 1-α)`.
fn low_pass(fs: f64, fc: f64, q: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let one_minus_cos = 1.0 - v.cosw;
    let b0 = one_minus_cos * 0.5;
    let b1 = one_minus_cos;
    let b2 = one_minus_cos * 0.5;
    let a0 = 1.0 + v.alpha;
    let a1 = -2.0 * v.cosw;
    let a2 = 1.0 - v.alpha;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// 2-pole high-pass. Analog `H(s) = s² / (s² + s/Q + 1)`,
/// bilinear → `b = ((1+cosω)/2, -(1+cosω), (1+cosω)/2)`.
fn high_pass(fs: f64, fc: f64, q: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let one_plus_cos = 1.0 + v.cosw;
    let b0 = one_plus_cos * 0.5;
    let b1 = -one_plus_cos;
    let b2 = one_plus_cos * 0.5;
    let a0 = 1.0 + v.alpha;
    let a1 = -2.0 * v.cosw;
    let a2 = 1.0 - v.alpha;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// Constant-skirt-gain band-pass. Peak gain ≈ Q at centre.
/// Analog `H(s) = (s/Q) / (s² + s/Q + 1)`, bilinear →
/// `b = (sinω/2, 0, -sinω/2)`.
fn band_pass(fs: f64, fc: f64, q: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let b0 = v.sinw * 0.5;
    let b1 = 0.0;
    let b2 = -v.sinw * 0.5;
    let a0 = 1.0 + v.alpha;
    let a1 = -2.0 * v.cosw;
    let a2 = 1.0 - v.alpha;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// Notch. Analog `H(s) = (s² + 1) / (s² + s/Q + 1)`, bilinear →
/// `b = (1, -2cosω, 1)`.
fn notch(fs: f64, fc: f64, q: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let b0 = 1.0;
    let b1 = -2.0 * v.cosw;
    let b2 = 1.0;
    let a0 = 1.0 + v.alpha;
    let a1 = -2.0 * v.cosw;
    let a2 = 1.0 - v.alpha;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// Parametric peaking. `A = 10^(gain_db/40)`. Analog
/// `H(s) = (s² + A·s/Q + 1) / (s² + s/(A·Q) + 1)`, bilinear →
/// `b = (1 + α·A, -2cosω, 1 - α·A)`,
/// `a = (1 + α/A, -2cosω, 1 - α/A)`.
fn peaking(fs: f64, fc: f64, q: f64, gain_db: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    let b0 = 1.0 + v.alpha * a_gain;
    let b1 = -2.0 * v.cosw;
    let b2 = 1.0 - v.alpha * a_gain;
    let a0 = 1.0 + v.alpha / a_gain;
    let a1 = -2.0 * v.cosw;
    let a2 = 1.0 - v.alpha / a_gain;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// Low shelf. Uses `A = 10^(gain_db/40)` and a `β = 2√A·α` skirt term;
/// derivation: bilinear of `H(s) = A · (s² + (√A/Q)·s + A) /
/// (A·s² + (√A/Q)·s + 1)`.
fn low_shelf(fs: f64, fc: f64, q: f64, gain_db: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    let sqrt_a = a_gain.sqrt();
    let beta = 2.0 * sqrt_a * v.alpha;
    let b0 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * v.cosw + beta);
    let b1 = 2.0 * a_gain * ((a_gain - 1.0) - (a_gain + 1.0) * v.cosw);
    let b2 = a_gain * ((a_gain + 1.0) - (a_gain - 1.0) * v.cosw - beta);
    let a0 = (a_gain + 1.0) + (a_gain - 1.0) * v.cosw + beta;
    let a1 = -2.0 * ((a_gain - 1.0) + (a_gain + 1.0) * v.cosw);
    let a2 = (a_gain + 1.0) + (a_gain - 1.0) * v.cosw - beta;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// High shelf — symmetric to [`low_shelf`] with the cosω signs flipped.
fn high_shelf(fs: f64, fc: f64, q: f64, gain_db: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    let sqrt_a = a_gain.sqrt();
    let beta = 2.0 * sqrt_a * v.alpha;
    let b0 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * v.cosw + beta);
    let b1 = -2.0 * a_gain * ((a_gain - 1.0) + (a_gain + 1.0) * v.cosw);
    let b2 = a_gain * ((a_gain + 1.0) + (a_gain - 1.0) * v.cosw - beta);
    let a0 = (a_gain + 1.0) - (a_gain - 1.0) * v.cosw + beta;
    let a1 = 2.0 * ((a_gain - 1.0) - (a_gain + 1.0) * v.cosw);
    let a2 = (a_gain + 1.0) - (a_gain - 1.0) * v.cosw - beta;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// Second-order all-pass. Analog prototype `H(s) = (s² − s/Q + 1) /
/// (s² + s/Q + 1)` — numerator and denominator are mirror images so
/// `|H(jω)| ≡ 1` for every analog frequency. Bilinear transform gives
/// `b = (1 − α, −2cosω, 1 + α)`, `a = (1 + α, −2cosω, 1 − α)`; the
/// numerator is the bit-reversal of the denominator, which preserves
/// the flat-magnitude property in the digital domain. Phase rotates
/// from `0` at DC down through `−π` at `ω = ω_c` to `−2π` at Nyquist.
///
/// Formula derived from the standard Audio EQ Cookbook expressions
/// (Robert Bristow-Johnson, public-domain canonical reference for
/// bilinear-transformed cookbook biquads); written here in our own
/// variable names from the documented analog `H(s)` above, no
/// reference C source consulted.
fn all_pass(fs: f64, fc: f64, q: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let b0 = 1.0 - v.alpha;
    let b1 = -2.0 * v.cosw;
    let b2 = 1.0 + v.alpha;
    let a0 = 1.0 + v.alpha;
    let a1 = -2.0 * v.cosw;
    let a2 = 1.0 - v.alpha;
    normalise(b0, b1, b2, a0, a1, a2)
}

/// Per-channel DF-II-transposed state.
#[derive(Debug, Clone, Copy, Default)]
struct State {
    s1: f64,
    s2: f64,
}

/// Streaming biquad. Holds the configuration, the most-recent compiled
/// coefficients, and one `(s1, s2)` state pair per channel.
#[derive(Debug, Clone)]
pub struct Biquad {
    kind: BiquadKind,
    /// Coefficients cached from the last `(kind, sample_rate)` combo.
    coeffs: Option<Coeffs>,
    /// Sample rate the cached coefficients were built for.
    cached_rate: u32,
    /// Kind the cached coefficients were built for.
    cached_kind: Option<BiquadKind>,
    /// One state per channel; resized lazily.
    states: Vec<State>,
}

impl Biquad {
    /// New biquad in the given configuration.
    pub fn new(kind: BiquadKind) -> Self {
        Self {
            kind,
            coeffs: None,
            cached_rate: 0,
            cached_kind: None,
            states: Vec::new(),
        }
    }

    /// Convenience: 2-pole low-pass.
    pub fn low_pass(sample_rate_hz: u32, cutoff_hz: f32, q: f32) -> Self {
        let mut bq = Self::new(BiquadKind::LowPass { cutoff_hz, q });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: 2-pole high-pass.
    pub fn high_pass(sample_rate_hz: u32, cutoff_hz: f32, q: f32) -> Self {
        let mut bq = Self::new(BiquadKind::HighPass { cutoff_hz, q });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: constant-skirt band-pass.
    pub fn band_pass(sample_rate_hz: u32, center_hz: f32, q: f32) -> Self {
        let mut bq = Self::new(BiquadKind::BandPass { center_hz, q });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: notch.
    pub fn notch(sample_rate_hz: u32, center_hz: f32, q: f32) -> Self {
        let mut bq = Self::new(BiquadKind::Notch { center_hz, q });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: parametric peaking EQ.
    pub fn peaking(sample_rate_hz: u32, center_hz: f32, q: f32, gain_db: f32) -> Self {
        let mut bq = Self::new(BiquadKind::Peaking {
            center_hz,
            q,
            gain_db,
        });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: low shelf.
    pub fn low_shelf(sample_rate_hz: u32, cutoff_hz: f32, q: f32, gain_db: f32) -> Self {
        let mut bq = Self::new(BiquadKind::LowShelf {
            cutoff_hz,
            q,
            gain_db,
        });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: high shelf.
    pub fn high_shelf(sample_rate_hz: u32, cutoff_hz: f32, q: f32, gain_db: f32) -> Self {
        let mut bq = Self::new(BiquadKind::HighShelf {
            cutoff_hz,
            q,
            gain_db,
        });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: second-order all-pass (flat magnitude, frequency-
    /// dependent phase rotation centred at `center_hz`; `Q` sets the
    /// width of the phase-rotation transition).
    pub fn all_pass(sample_rate_hz: u32, center_hz: f32, q: f32) -> Self {
        let mut bq = Self::new(BiquadKind::AllPass { center_hz, q });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }

    /// Swap to a new configuration. Existing state is preserved across
    /// the swap so coefficient updates do not pop.
    pub fn set_kind(&mut self, kind: BiquadKind) {
        self.kind = kind;
        // Force recompile on next process()/ensure_coeffs() call.
        self.cached_kind = None;
    }

    /// Currently-active configuration.
    pub fn kind(&self) -> BiquadKind {
        self.kind
    }

    fn ensure_coeffs(&mut self, sample_rate_hz: u32) {
        let needs_recompile = self.cached_kind != Some(self.kind)
            || self.cached_rate != sample_rate_hz
            || self.coeffs.is_none();
        if needs_recompile {
            self.coeffs = Some(Coeffs::from_kind(self.kind, sample_rate_hz));
            self.cached_kind = Some(self.kind);
            self.cached_rate = sample_rate_hz;
        }
    }

    fn ensure_states(&mut self, channels: usize) {
        if self.states.len() != channels {
            self.states = vec![State::default(); channels];
        }
    }

    /// Reset filter state (delay-line memory) to zero. Coefficients are
    /// not touched.
    pub fn reset(&mut self) {
        for st in self.states.iter_mut() {
            *st = State::default();
        }
    }

    /// Apply the recurrence in-place to an interleaved sample buffer.
    /// `channels` must divide `samples.len()` evenly.
    pub fn process_in_place(&mut self, samples: &mut [f32], channels: u16, sample_rate_hz: u32) {
        if channels == 0 || samples.is_empty() {
            return;
        }
        self.ensure_coeffs(sample_rate_hz);
        self.ensure_states(channels as usize);

        let c = self.coeffs.expect("ensure_coeffs ran above");
        let n_chan = channels as usize;
        let n_frames = samples.len() / n_chan;

        for frame in 0..n_frames {
            for ch in 0..n_chan {
                let i = frame * n_chan + ch;
                let st = &mut self.states[ch];
                let x = samples[i] as f64;
                let y = c.b0 * x + st.s1;
                st.s1 = c.b1 * x - c.a1 * y + st.s2;
                st.s2 = c.b2 * x - c.a2 * y;
                samples[i] = y as f32;
            }
        }
    }
}

impl AudioFilter for Biquad {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_coeffs(params.sample_rate);
        self.ensure_states(channels.len());
        let c = self.coeffs.expect("ensure_coeffs ran above");

        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.states[ch_idx];
            for s in buf.iter_mut() {
                let x = *s as f64;
                let y = c.b0 * x + st.s1;
                st.s1 = c.b1 * x - c.a1 * y + st.s2;
                st.s2 = c.b2 * x - c.a2 * y;
                *s = y as f32;
            }
        }

        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    fn db(linear: f32) -> f32 {
        20.0 * linear.max(1.0e-12).log10()
    }

    /// Half-period sine generator at `freq_hz` for `n` samples at `fs`.
    fn sine(freq_hz: f32, fs: u32, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let w = 2.0 * std::f32::consts::PI * freq_hz / fs as f32;
        for i in 0..n {
            out.push((i as f32 * w).sin());
        }
        out
    }

    /// Run a stream through a biquad. `tail_keep` is the number of
    /// tail samples to return (skipping the start-up transient).
    fn run(bq: &mut Biquad, input: &[f32], fs: u32, tail_keep: usize) -> Vec<f32> {
        let mut buf = input.to_vec();
        bq.process_in_place(&mut buf, 1, fs);
        let n = buf.len();
        buf[n.saturating_sub(tail_keep)..].to_vec()
    }

    #[test]
    fn lowpass_impulse_has_finite_energy() {
        // Cutoff at fs/4 = 12 kHz, fs=48 kHz, Butterworth Q.
        let mut bq = Biquad::low_pass(48_000, 12_000.0, std::f32::consts::FRAC_1_SQRT_2);
        let mut x = vec![0.0f32; 4096];
        x[0] = 1.0;
        bq.process_in_place(&mut x, 1, 48_000);
        let l1: f64 = x.iter().map(|v| (*v as f64).abs()).sum();
        // Sum of |h[n]| for a passive LPF is finite; theoretical DC
        // gain is 1 so the *signed* sum should be ≈ 1. The L1 norm is
        // bounded by that of the impulse response of the prototype.
        assert!(l1.is_finite() && l1 > 0.5 && l1 < 5.0, "L1={}", l1);
        let signed: f64 = x.iter().map(|v| *v as f64).sum();
        assert!(
            (signed - 1.0).abs() < 0.05,
            "DC gain via summed impulse = {}",
            signed
        );
    }

    #[test]
    fn lowpass_minus3db_at_cutoff() {
        let fs = 48_000u32;
        let fc = 4_000.0f32;
        let mut bq = Biquad::low_pass(fs, fc, std::f32::consts::FRAC_1_SQRT_2);
        let x = sine(fc, fs, 8192);
        let y = run(&mut bq, &x, fs, 4096);
        let g_db = db(rms(&y)) - db(rms(&sine(fc, fs, 4096)));
        // Butterworth -3 dB at fc, tolerance ±0.5 dB.
        assert!(
            (g_db + 3.0).abs() < 0.6,
            "gain at fc = {} dB (expected ≈ -3)",
            g_db
        );
    }

    #[test]
    fn lowpass_strong_rejection_above_band() {
        let fs = 48_000u32;
        let fc = 2_000.0f32;
        let mut bq = Biquad::low_pass(fs, fc, std::f32::consts::FRAC_1_SQRT_2);
        let probe_freq = 10.0 * fc; // = 20 kHz; just under Nyquist
        let x = sine(probe_freq, fs, 8192);
        let y = run(&mut bq, &x, fs, 4096);
        let g_db = db(rms(&y)) - db(rms(&sine(probe_freq, fs, 4096)));
        assert!(g_db < -30.0, "rejection at 10×fc = {} dB", g_db);
    }

    #[test]
    fn peaking_plus6db_returns_plus6db() {
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let mut bq = Biquad::peaking(fs, fc, 2.0, 6.0);
        let x = sine(fc, fs, 16_384);
        let y = run(&mut bq, &x, fs, 8_192);
        let g_db = db(rms(&y)) - db(rms(&sine(fc, fs, 8_192)));
        assert!(
            (g_db - 6.0).abs() < 0.6,
            "peaking gain = {} dB (expected ≈ 6)",
            g_db
        );
    }

    #[test]
    fn low_shelf_passes_band_correctly() {
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let mut bq_lo = Biquad::low_shelf(fs, fc, std::f32::consts::FRAC_1_SQRT_2, 6.0);
        let mut bq_hi = Biquad::low_shelf(fs, fc, std::f32::consts::FRAC_1_SQRT_2, 6.0);

        // Far below cutoff: shelf is in band, gain ≈ +6 dB.
        let f_lo = 80.0f32;
        let x_lo = sine(f_lo, fs, 65_536);
        let y_lo = run(&mut bq_lo, &x_lo, fs, 32_768);
        let g_lo_db = db(rms(&y_lo)) - db(rms(&sine(f_lo, fs, 32_768)));
        assert!(
            (g_lo_db - 6.0).abs() < 0.6,
            "low_shelf in-band gain = {}",
            g_lo_db
        );

        // Well above cutoff: shelf is out-of-band, gain ≈ 0 dB.
        let f_hi = 16_000.0f32;
        let x_hi = sine(f_hi, fs, 16_384);
        let y_hi = run(&mut bq_hi, &x_hi, fs, 8_192);
        let g_hi_db = db(rms(&y_hi)) - db(rms(&sine(f_hi, fs, 8_192)));
        assert!(
            g_hi_db.abs() < 0.6,
            "low_shelf out-of-band gain = {}",
            g_hi_db
        );
    }

    #[test]
    fn highpass_rejects_dc() {
        let fs = 48_000u32;
        let mut bq = Biquad::high_pass(fs, 200.0, std::f32::consts::FRAC_1_SQRT_2);
        let mut x = vec![1.0f32; 4096];
        bq.process_in_place(&mut x, 1, fs);
        // After settling the output of a HPF applied to DC is ≈ 0.
        let tail = &x[3_500..];
        let peak = tail.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(peak < 0.05, "HPF DC residual peak = {}", peak);
    }

    #[test]
    fn notch_attenuates_centre_frequency() {
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let mut bq = Biquad::notch(fs, fc, 8.0);
        let x = sine(fc, fs, 16_384);
        let y = run(&mut bq, &x, fs, 8_192);
        let g_db = db(rms(&y)) - db(rms(&sine(fc, fs, 8_192)));
        assert!(g_db < -20.0, "notch attenuation at fc = {}", g_db);
    }

    #[test]
    fn all_pass_flat_magnitude_at_three_frequencies() {
        // The defining property of an all-pass is `|H(e^{jω})| ≡ 1` at
        // every frequency, regardless of `Q`. We probe at three points
        // around a `center_hz = 1 kHz` design — well below (passband-
        // style), at the centre (the transition / phase-flip point),
        // and well above (stopband-style) — and assert the steady-state
        // gain is within ±0.1 dB of unity at all three.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let q = 2.0f32;
        let probes = [200.0f32, 1_000.0f32, 8_000.0f32];
        for &probe in &probes {
            let mut bq = Biquad::all_pass(fs, fc, q);
            let x = sine(probe, fs, 16_384);
            // Skip a generous start-up window so the IIR transient has
            // decayed; an APF with Q=2 settles within ~10 ms (≈ 480
            // samples at 48 kHz), so 8 192 samples is comfortable.
            let y = run(&mut bq, &x, fs, 8_192);
            let g_db = db(rms(&y)) - db(rms(&sine(probe, fs, 8_192)));
            assert!(
                g_db.abs() < 0.1,
                "all-pass magnitude at {} Hz = {} dB (expected ≈ 0)",
                probe,
                g_db
            );
        }
    }

    #[test]
    fn all_pass_phase_inverts_at_center_frequency() {
        // Sanity check on the phase response: at `ω = ω_c` the cookbook
        // APF has phase `−π`, i.e. a sign flip. Cross-correlating the
        // input sine with the output should give a strongly negative
        // peak (output is the negated, magnitude-preserved input plus
        // a settled transient). We probe at the centre frequency with
        // a generous tail-skip and expect the inner product to be
        // close to `−E[x²]` (= `−1/2` for a unit-amplitude sine).
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let mut bq = Biquad::all_pass(fs, fc, std::f32::consts::FRAC_1_SQRT_2);
        let n = 16_384usize;
        let x = sine(fc, fs, n);
        let mut buf = x.clone();
        bq.process_in_place(&mut buf, 1, fs);
        // Skip the first half to bypass start-up transient; correlate
        // tail.
        let tail_x = &x[n / 2..];
        let tail_y = &buf[n / 2..];
        let mut dot = 0.0f64;
        let mut nrm = 0.0f64;
        for (xi, yi) in tail_x.iter().zip(tail_y.iter()) {
            dot += (*xi as f64) * (*yi as f64);
            nrm += (*xi as f64) * (*xi as f64);
        }
        // Cosine of the phase shift in steady state.
        // For `−π` phase, expected `r = cos(−π) = −1`. Tolerance ±0.1
        // covers the bilinear pre-warp + finite-window leakage.
        let r = dot / nrm;
        assert!(
            (r + 1.0).abs() < 0.1,
            "all-pass phase at fc gave correlation {} (expected ≈ -1)",
            r
        );
    }

    #[test]
    fn all_pass_high_q_impulse_stable() {
        // With `Q = 50` (very narrow phase-rotation skirt) the poles
        // sit close to the unit circle; the impulse response is a long
        // damped sinusoid. We confirm: (a) every sample is finite (no
        // NaN / Inf from DF-II-T numerics), and (b) the response
        // decays — the L² energy in the last quarter of the response
        // is strictly less than the first quarter. Also check the L1
        // norm is bounded so an external compiler can prove BIBO
        // stability.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let q = 50.0f32;
        let mut bq = Biquad::all_pass(fs, fc, q);
        let mut x = vec![0.0f32; 16_384];
        x[0] = 1.0;
        bq.process_in_place(&mut x, 1, fs);
        // Finite-everywhere check.
        assert!(
            x.iter().all(|v| v.is_finite()),
            "high-Q all-pass impulse response produced non-finite sample"
        );
        // Decay check: first-quarter L² energy > last-quarter L².
        let q_len = x.len() / 4;
        let head: f64 = x[..q_len].iter().map(|v| (*v as f64).powi(2)).sum();
        let tail: f64 = x[3 * q_len..].iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(
            head > tail,
            "energy did not decay: head L²={} tail L²={}",
            head,
            tail
        );
        // L1 bound — a stable all-pass has finite L1 norm of the
        // impulse response (cookbook APF: ≈ 1/(1-r) where r ≈ poles'
        // radius; at Q=50 the bound is generous but still finite).
        let l1: f64 = x.iter().map(|v| (*v as f64).abs()).sum();
        assert!(
            l1.is_finite() && l1 < 1_000.0,
            "high-Q all-pass L1 = {} (expected finite, < 1000)",
            l1
        );
    }

    #[test]
    fn process_in_place_handles_stereo_without_crosstalk() {
        let fs = 48_000u32;
        let mut bq = Biquad::low_pass(fs, 1_000.0, std::f32::consts::FRAC_1_SQRT_2);
        // Two interleaved channels: L has a DC step, R is silent.
        let frames = 256;
        let mut buf = vec![0.0f32; frames * 2];
        for i in 0..frames {
            buf[i * 2] = 1.0; // L
                              // R stays 0
        }
        bq.process_in_place(&mut buf, 2, fs);
        let r_peak = (0..frames)
            .map(|i| buf[i * 2 + 1].abs())
            .fold(0.0f32, f32::max);
        assert!(r_peak < 1.0e-6, "R channel polluted by L: peak={}", r_peak);
        // L should respond.
        let l_peak = (0..frames).map(|i| buf[i * 2].abs()).fold(0.0f32, f32::max);
        assert!(l_peak > 0.5, "L channel under-responding: peak={}", l_peak);
    }
}
