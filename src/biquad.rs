//! Biquadratic IIR EQ filter family.
//!
//! Implements eleven second-order IIR configurations sharing a single
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
//! All configurations are derived from the **bilinear transform**
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
//! The cookbook offers three equivalent parameterisations of the skirt
//! term `α`; this module exposes two of them:
//!
//! ```text
//! α = sinω / (2Q)                                   (case: Q)
//! α = (sinω / 2) · √((A + 1/A)·(1/S − 1) + 2)       (case: S, shelves)
//! ```
//!
//! `S` is the cookbook *shelf slope*: at `S = 1` the shelf is as steep
//! as it can be while the gain remains monotonic in frequency; the
//! dB/octave slope at the midpoint stays proportional to `S` for fixed
//! `f_c / f_s` and `gain_db`. The two cases are related by
//! `1/Q = √((A + 1/A)·(1/S − 1) + 2)`.
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

/// One of the eleven supported biquad configurations.
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
    /// Constant-0-dB-peak-gain band-pass. Same pole pair as
    /// [`BandPass`](BiquadKind::BandPass) but the numerator is scaled
    /// by `1/Q` (analog prototype `H(s) = (s/Q) / (s² + s/Q + 1)`), so
    /// the magnitude at the centre frequency is exactly unity for
    /// every `Q` — `Q` only sets the bandwidth. This is the variant to
    /// reach for when band-passing should not change programme level.
    BandPassConstantPeak { center_hz: f32, q: f32 },
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
    /// Low shelf parameterised by the cookbook *shelf slope* `S`
    /// instead of `Q`. `slope = 1.0` gives the steepest transition
    /// that keeps the gain monotonic in frequency; `slope < 1`
    /// relaxes (widens) the transition; `slope > 1` steepens it
    /// further at the cost of response overshoot around the corner.
    /// `cutoff_hz` is the shelf *midpoint* frequency (gain is exactly
    /// `gain_db / 2` there).
    LowShelfSlope {
        cutoff_hz: f32,
        slope: f32,
        gain_db: f32,
    },
    /// High shelf parameterised by the shelf slope `S` — mirror of
    /// [`LowShelfSlope`](BiquadKind::LowShelfSlope).
    HighShelfSlope {
        cutoff_hz: f32,
        slope: f32,
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
            BiquadKind::BandPassConstantPeak { center_hz, q } => {
                band_pass_constant_peak(fs, center_hz as f64, q as f64)
            }
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
            BiquadKind::LowShelfSlope {
                cutoff_hz,
                slope,
                gain_db,
            } => low_shelf_slope(fs, cutoff_hz as f64, slope as f64, gain_db as f64),
            BiquadKind::HighShelfSlope {
                cutoff_hz,
                slope,
                gain_db,
            } => high_shelf_slope(fs, cutoff_hz as f64, slope as f64, gain_db as f64),
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

/// Shelf-slope (case S) skirt parameter:
/// `α = (sinω/2)·√((A + 1/A)(1/S − 1) + 2)`. The radicand goes
/// negative when `S` is pushed past the largest value the chosen gain
/// supports (`1/S ≥ 1 − 2/(A + 1/A)`); we clamp it at a tiny positive
/// floor so `α` stays real (equivalent to capping `S` at that maximum).
fn warp_slope(fs: f64, fc: f64, slope: f64, a_gain: f64) -> WarpVars {
    let w = 2.0 * std::f64::consts::PI * (fc.max(1.0e-6) / fs);
    let cosw = w.cos();
    let sinw = w.sin();
    let s = slope.max(1.0e-6);
    let radicand = ((a_gain + 1.0 / a_gain) * (1.0 / s - 1.0) + 2.0).max(1.0e-12);
    let alpha = (sinw / 2.0) * radicand.sqrt();
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

/// Constant-0-dB-peak-gain band-pass. Analog
/// `H(s) = (s/Q) / (s² + s/Q + 1)` — the constant-skirt numerator
/// divided by `Q` — bilinear → `b = (α, 0, -α)` with the same
/// denominator, so `|H| = 1` exactly at the centre frequency for any
/// `Q`.
fn band_pass_constant_peak(fs: f64, fc: f64, q: f64) -> Coeffs {
    let v = warp(fs, fc, q);
    let b0 = v.alpha;
    let b1 = 0.0;
    let b2 = -v.alpha;
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

/// Low-shelf coefficient assembly from precomputed warp variables.
/// Uses `A = 10^(gain_db/40)` and a `β = 2√A·α` skirt term;
/// derivation: bilinear of `H(s) = A · (s² + (√A/Q)·s + A) /
/// (A·s² + (√A/Q)·s + 1)`.
fn low_shelf_from(v: WarpVars, a_gain: f64) -> Coeffs {
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

/// High-shelf coefficient assembly — symmetric to [`low_shelf_from`]
/// with the cosω signs flipped.
fn high_shelf_from(v: WarpVars, a_gain: f64) -> Coeffs {
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

/// Low shelf, `Q`-parameterised skirt.
fn low_shelf(fs: f64, fc: f64, q: f64, gain_db: f64) -> Coeffs {
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    low_shelf_from(warp(fs, fc, q), a_gain)
}

/// High shelf, `Q`-parameterised skirt.
fn high_shelf(fs: f64, fc: f64, q: f64, gain_db: f64) -> Coeffs {
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    high_shelf_from(warp(fs, fc, q), a_gain)
}

/// Low shelf, slope-(S)-parameterised skirt (see [`warp_slope`]).
fn low_shelf_slope(fs: f64, fc: f64, slope: f64, gain_db: f64) -> Coeffs {
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    low_shelf_from(warp_slope(fs, fc, slope, a_gain), a_gain)
}

/// High shelf, slope-(S)-parameterised skirt (see [`warp_slope`]).
fn high_shelf_slope(fs: f64, fc: f64, slope: f64, gain_db: f64) -> Coeffs {
    let a_gain = 10.0_f64.powf(gain_db / 40.0);
    high_shelf_from(warp_slope(fs, fc, slope, a_gain), a_gain)
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
    /// Convenience: constant-0-dB-peak band-pass (`Q` sets bandwidth
    /// only; centre-frequency gain is exactly unity).
    pub fn band_pass_constant_peak(sample_rate_hz: u32, center_hz: f32, q: f32) -> Self {
        let mut bq = Self::new(BiquadKind::BandPassConstantPeak { center_hz, q });
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
    /// Convenience: low shelf parameterised by shelf slope `S`
    /// (`slope = 1.0` → steepest monotonic transition).
    pub fn low_shelf_slope(sample_rate_hz: u32, cutoff_hz: f32, slope: f32, gain_db: f32) -> Self {
        let mut bq = Self::new(BiquadKind::LowShelfSlope {
            cutoff_hz,
            slope,
            gain_db,
        });
        bq.ensure_coeffs(sample_rate_hz);
        bq
    }
    /// Convenience: high shelf parameterised by shelf slope `S`.
    pub fn high_shelf_slope(sample_rate_hz: u32, cutoff_hz: f32, slope: f32, gain_db: f32) -> Self {
        let mut bq = Self::new(BiquadKind::HighShelfSlope {
            cutoff_hz,
            slope,
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

    /// Closed-form magnitude response in dB at `freq_hz` for a stream
    /// at `sample_rate_hz`, evaluated directly from the compiled
    /// coefficients:
    ///
    /// ```text
    /// |H(e^{jω})| = |b0 + b1·e^{-jω} + b2·e^{-j2ω}|
    ///             / |1  + a1·e^{-jω} + a2·e^{-j2ω}|,   ω = 2π·f/fs
    /// ```
    ///
    /// Pure function of the configuration — does not touch filter
    /// state. Useful for response plotting and for asserting the
    /// design formulas without running samples through the recurrence.
    pub fn magnitude_response_db(&self, freq_hz: f32, sample_rate_hz: u32) -> f64 {
        let c = Coeffs::from_kind(self.kind, sample_rate_hz);
        let w = 2.0 * std::f64::consts::PI * freq_hz as f64 / sample_rate_hz.max(1) as f64;
        let (cos1, sin1) = (w.cos(), w.sin());
        let (cos2, sin2) = ((2.0 * w).cos(), (2.0 * w).sin());
        // Numerator / denominator evaluated at z = e^{jω} (so z^{-1} =
        // e^{-jω} contributes -sin to the imaginary part).
        let num_re = c.b0 + c.b1 * cos1 + c.b2 * cos2;
        let num_im = -(c.b1 * sin1 + c.b2 * sin2);
        let den_re = 1.0 + c.a1 * cos1 + c.a2 * cos2;
        let den_im = -(c.a1 * sin1 + c.a2 * sin2);
        let mag2 = (num_re * num_re + num_im * num_im) / (den_re * den_re + den_im * den_im);
        10.0 * mag2.max(1.0e-30).log10()
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

    /// Apply the recurrence in-place to ONE channel's contiguous
    /// (planar) buffer, using the per-channel state slot `channel` out
    /// of `total_channels` slots.
    ///
    /// Callers that decode to planar per-channel buffers and drive the
    /// same `Biquad` over each channel in turn (swept-filter effects
    /// that share one coefficient set across channels) must use this
    /// instead of repeated `process_in_place(.., 1, ..)` calls — the
    /// latter reuses state slot 0 for every channel, so channel k's
    /// delay-line history leaks into channel k+1.
    pub fn process_channel_in_place(
        &mut self,
        samples: &mut [f32],
        channel: usize,
        total_channels: usize,
        sample_rate_hz: u32,
    ) {
        if total_channels == 0 || channel >= total_channels || samples.is_empty() {
            return;
        }
        self.ensure_coeffs(sample_rate_hz);
        self.ensure_states(total_channels);

        let c = self.coeffs.expect("ensure_coeffs ran above");
        let st = &mut self.states[channel];
        for s in samples.iter_mut() {
            let x = *s as f64;
            let y = c.b0 * x + st.s1;
            st.s1 = c.b1 * x - c.a1 * y + st.s2;
            st.s2 = c.b2 * x - c.a2 * y;
            *s = y as f32;
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

    // ---- Cookbook completion (round 284): constant-peak BPF + slope
    // shelves, verified with closed-form frequency-response assertions
    // via `magnitude_response_db` (no sample-domain estimation error).

    #[test]
    fn constant_peak_bpf_unity_at_center_for_every_q() {
        // Per the staged cookbook BPF (constant 0 dB peak gain):
        // b = (α, 0, −α) over the standard denominator makes
        // |H(e^{jω0})| = 1 *exactly*, independent of Q. Tolerance is
        // 1e-9 dB — this is an algebraic identity, not an estimate.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        for &q in &[0.3f32, std::f32::consts::FRAC_1_SQRT_2, 2.0, 8.0, 32.0] {
            let bq = Biquad::band_pass_constant_peak(fs, fc, q);
            let g_db = bq.magnitude_response_db(fc, fs);
            assert!(
                g_db.abs() < 1.0e-9,
                "constant-peak BPF gain at fc (Q={}) = {} dB (expected 0)",
                q,
                g_db
            );
        }
    }

    #[test]
    fn constant_skirt_vs_constant_peak_differ_by_exactly_20log10_q() {
        // The two cookbook BPF variants share the same denominator and
        // their numerators differ by the exact factor Q
        // (skirt: b0 = sinω0/2 = Q·α; peak: b0 = α), so at *every*
        // frequency the responses differ by 20·log10(Q) dB.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let q = 4.0f32;
        let skirt = Biquad::band_pass(fs, fc, q);
        let peak = Biquad::band_pass_constant_peak(fs, fc, q);
        let expected = 20.0 * (q as f64).log10(); // ≈ 12.04 dB
        for &f in &[fc / 4.0, fc / 2.0, fc, fc * 2.0, fc * 4.0] {
            let d = skirt.magnitude_response_db(f, fs) - peak.magnitude_response_db(f, fs);
            assert!(
                (d - expected).abs() < 1.0e-9,
                "skirt − peak at {} Hz = {} dB (expected {})",
                f,
                d,
                expected
            );
        }
        // And the constant-skirt variant's centre gain is therefore
        // exactly 20·log10(Q) dB above unity.
        let g_skirt = skirt.magnitude_response_db(fc, fs);
        assert!(
            (g_skirt - expected).abs() < 1.0e-9,
            "constant-skirt BPF centre gain = {} dB (expected {})",
            g_skirt,
            expected
        );
    }

    #[test]
    fn constant_peak_bpf_unity_at_center_through_samples() {
        // End-to-end check through the recurrence: a high-Q (Q = 8)
        // constant-peak BPF passes a centre-frequency sine at 0 dB.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let mut bq = Biquad::band_pass_constant_peak(fs, fc, 8.0);
        let x = sine(fc, fs, 32_768);
        let y = run(&mut bq, &x, fs, 16_384);
        let g_db = db(rms(&y)) - db(rms(&sine(fc, fs, 16_384)));
        assert!(
            g_db.abs() < 0.1,
            "processed constant-peak BPF gain at fc = {} dB (expected ≈ 0)",
            g_db
        );
    }

    #[test]
    fn slope_one_shelf_equals_q_inv_sqrt2_shelf() {
        // Cookbook S↔Q mapping: 1/Q² = (A + 1/A)(1/S − 1) + 2, so at
        // S = 1 the radicand is exactly 2 → Q = 1/√2 for *any* gain.
        // The slope-parameterised shelf must therefore reproduce the
        // Q-parameterised one bit-for-bit (modulo f32→f64 of the Q
        // constant; assert ≤ 1e-6 dB at probe points).
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        for &gain_db in &[-15.0f32, -6.0, 6.0, 15.0] {
            let s_lo = Biquad::low_shelf_slope(fs, fc, 1.0, gain_db);
            let q_lo = Biquad::low_shelf(fs, fc, std::f32::consts::FRAC_1_SQRT_2, gain_db);
            let s_hi = Biquad::high_shelf_slope(fs, fc, 1.0, gain_db);
            let q_hi = Biquad::high_shelf(fs, fc, std::f32::consts::FRAC_1_SQRT_2, gain_db);
            for &f in &[50.0f32, 250.0, 1_000.0, 4_000.0, 16_000.0] {
                let d_lo = s_lo.magnitude_response_db(f, fs) - q_lo.magnitude_response_db(f, fs);
                let d_hi = s_hi.magnitude_response_db(f, fs) - q_hi.magnitude_response_db(f, fs);
                assert!(
                    d_lo.abs() < 1.0e-6 && d_hi.abs() < 1.0e-6,
                    "S=1 vs Q=1/√2 mismatch at {} Hz (gain {} dB): lo {} dB, hi {} dB",
                    f,
                    gain_db,
                    d_lo,
                    d_hi
                );
            }
        }
    }

    #[test]
    fn shelf_midpoint_gain_is_exactly_half_dbgain() {
        // f0 is the cookbook shelf *midpoint*: |H| at f0 is exactly A,
        // i.e. gain_db / 2 in dB (the analog prototype's numerator and
        // denominator magnitudes coincide at ω = 1 apart from the
        // leading A, and the BLT pre-warp preserves the f0 mapping).
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        for &gain_db in &[-12.0f32, -3.0, 3.0, 12.0] {
            for &slope in &[0.5f32, 1.0] {
                let lo = Biquad::low_shelf_slope(fs, fc, slope, gain_db);
                let hi = Biquad::high_shelf_slope(fs, fc, slope, gain_db);
                let expected = gain_db as f64 / 2.0;
                let g_lo = lo.magnitude_response_db(fc, fs);
                let g_hi = hi.magnitude_response_db(fc, fs);
                assert!(
                    (g_lo - expected).abs() < 1.0e-6,
                    "low-shelf midpoint gain (S={}, {} dB) = {} (expected {})",
                    slope,
                    gain_db,
                    g_lo,
                    expected
                );
                assert!(
                    (g_hi - expected).abs() < 1.0e-6,
                    "high-shelf midpoint gain (S={}, {} dB) = {} (expected {})",
                    slope,
                    gain_db,
                    g_hi,
                    expected
                );
            }
        }
    }

    #[test]
    fn slope_shelf_dc_and_nyquist_plateaus() {
        // Low shelf: DC gain is exactly gain_db, Nyquist gain exactly
        // 0 dB. High shelf mirrors. Probe just inside the edges (1 Hz
        // and fs/2 − 1 Hz) with a 0.01 dB tolerance.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let gain_db = 12.0f32;
        let lo = Biquad::low_shelf_slope(fs, fc, 1.0, gain_db);
        let hi = Biquad::high_shelf_slope(fs, fc, 1.0, gain_db);
        let near_dc = 1.0f32;
        let near_nyq = fs as f32 / 2.0 - 1.0;
        let lo_dc = lo.magnitude_response_db(near_dc, fs);
        let lo_ny = lo.magnitude_response_db(near_nyq, fs);
        let hi_dc = hi.magnitude_response_db(near_dc, fs);
        let hi_ny = hi.magnitude_response_db(near_nyq, fs);
        assert!(
            (lo_dc - gain_db as f64).abs() < 0.01,
            "low-shelf DC = {} dB",
            lo_dc
        );
        assert!(lo_ny.abs() < 0.01, "low-shelf Nyquist = {} dB", lo_ny);
        assert!(hi_dc.abs() < 0.01, "high-shelf DC = {} dB", hi_dc);
        assert!(
            (hi_ny - gain_db as f64).abs() < 0.01,
            "high-shelf Nyquist = {} dB",
            hi_ny
        );
    }

    #[test]
    fn slope_one_is_monotonic_slope_two_overshoots() {
        // Per the staged cookbook, S = 1 is the steepest shelf that
        // remains monotonic. Sweep 121 log-spaced points (20 Hz →
        // 20 kHz, 1/12-octave): the +12 dB low shelf at S = 1 must be
        // non-increasing throughout, while S = 2 must overshoot — its
        // maximum exceeds the shelf gain and its minimum dips below
        // unity (measured ≈ +1.38 / −1.38 dB for this design; assert
        // a conservative ±1 dB excursion).
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let gain_db = 12.0f32;
        let freqs: Vec<f32> = (0..121)
            .map(|i| 20.0f32 * 2.0f32.powf(i as f32 / 12.0))
            .filter(|f| *f < fs as f32 / 2.0)
            .collect();

        let s1 = Biquad::low_shelf_slope(fs, fc, 1.0, gain_db);
        let mags1: Vec<f64> = freqs
            .iter()
            .map(|&f| s1.magnitude_response_db(f, fs))
            .collect();
        for w in mags1.windows(2) {
            assert!(
                w[1] <= w[0] + 1.0e-6,
                "S=1 low shelf not monotonic: {} dB → {} dB",
                w[0],
                w[1]
            );
        }

        let s2 = Biquad::low_shelf_slope(fs, fc, 2.0, gain_db);
        let mags2: Vec<f64> = freqs
            .iter()
            .map(|&f| s2.magnitude_response_db(f, fs))
            .collect();
        let max2 = mags2.iter().cloned().fold(f64::MIN, f64::max);
        let min2 = mags2.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            max2 > gain_db as f64 + 1.0,
            "S=2 low shelf max = {} dB (expected overshoot > {})",
            max2,
            gain_db as f64 + 1.0
        );
        assert!(
            min2 < -1.0,
            "S=2 low shelf min = {} dB (expected undershoot < -1)",
            min2
        );
    }

    #[test]
    fn slope_shelf_processes_samples_at_expected_gain() {
        // End-to-end: +6 dB high shelf (S = 1) at 1 kHz boosts an
        // 8 kHz sine by ≈ 6 dB and leaves an 80 Hz sine at ≈ 0 dB.
        let fs = 48_000u32;
        let mut hi = Biquad::high_shelf_slope(fs, 1_000.0, 1.0, 6.0);
        let x_hi = sine(8_000.0, fs, 16_384);
        let y_hi = run(&mut hi, &x_hi, fs, 8_192);
        let g_hi = db(rms(&y_hi)) - db(rms(&sine(8_000.0, fs, 8_192)));
        assert!(
            (g_hi - 6.0).abs() < 0.6,
            "high-shelf-slope in-band gain = {} dB (expected ≈ 6)",
            g_hi
        );

        let mut hi2 = Biquad::high_shelf_slope(fs, 1_000.0, 1.0, 6.0);
        let x_lo = sine(80.0, fs, 65_536);
        let y_lo = run(&mut hi2, &x_lo, fs, 32_768);
        let g_lo = db(rms(&y_lo)) - db(rms(&sine(80.0, fs, 32_768)));
        assert!(
            g_lo.abs() < 0.6,
            "high-shelf-slope out-of-band gain = {} dB (expected ≈ 0)",
            g_lo
        );
    }

    #[test]
    fn magnitude_response_matches_measured_gain_for_legacy_kinds() {
        // Cross-check the analytic evaluator against the recurrence on
        // an existing configuration: peaking +6 dB at 1 kHz, probed at
        // the centre. Analytic must be exactly 6 dB — at ω0 the
        // peaking numerator reduces to 2jαA·sinω0 and the denominator
        // to 2j(α/A)·sinω0, so |H(e^{jω0})| = A² = 10^(gain_db/20)
        // exactly. The measured sine gain agrees within the
        // estimation tolerance.
        let fs = 48_000u32;
        let fc = 1_000.0f32;
        let bq = Biquad::peaking(fs, fc, 2.0, 6.0);
        let analytic = bq.magnitude_response_db(fc, fs);
        assert!(
            (analytic - 6.0).abs() < 1.0e-9,
            "analytic peaking gain at fc = {} dB",
            analytic
        );
        let mut bq2 = Biquad::peaking(fs, fc, 2.0, 6.0);
        let x = sine(fc, fs, 16_384);
        let y = run(&mut bq2, &x, fs, 8_192);
        let measured = db(rms(&y)) as f64 - db(rms(&sine(fc, fs, 8_192))) as f64;
        assert!(
            (measured - analytic).abs() < 0.6,
            "analytic {} dB vs measured {} dB",
            analytic,
            measured
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
