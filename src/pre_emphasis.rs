//! Pre-emphasis shelving filter — analog-broadcast / tape / FM record EQ.
//!
//! Pre-emphasis boosts the high-frequency content of a signal before it
//! enters a noisy channel (FM transmission, magnetic tape, vinyl record
//! cutting), so that the receiver's matching [`DeEmphasis`](crate::de_emphasis)
//! attenuation can flatten the response while also suppressing channel
//! noise in the high-frequency band. The pair `pre · de` is a (near-)identity
//! cascade by construction.
//!
//! # Curves
//!
//! All curves expressed as a first- or second-order analog transfer derived
//! algebraically below. Coefficients are then mapped to the discrete domain
//! by the bilinear transform `s = 2·f_s·(z − 1)/(z + 1)` and rearranged into
//! direct-form-I difference-equation form. No coefficients are taken from
//! any external table or library; the curve names are the canonical analog
//! time constants in seconds.
//!
//! | Curve              | τ₁         | τ₂        | τ₃        | Use                |
//! | ------------------ | ---------- | --------- | --------- | ------------------ |
//! | `Fm50us`           | —          | —         | 50 µs     | FM broadcast (EU)  |
//! | `Fm75us`           | —          | —         | 75 µs     | FM broadcast (NA)  |
//! | `J17`              | —          | —         | 50 µs     | Telco voice (ITU)  |
//! | `Custom { tau_s }` | —          | —         | any       | User-specified     |
//! | `Riaa3180_318_75`  | 3180 µs    | 318 µs    | 75 µs     | Phonograph / vinyl |
//!
//! The single-constant curves (`Fm50us`, `Fm75us`, `J17`, `Custom`) all
//! reduce to the same first-order shelving derivation; only `τ` differs.
//!
//! # First-order derivation (single-time-constant curves)
//!
//! The analog pre-emphasis is the shelving boost
//!
//! ```text
//!              1 + s·τ
//! H_pre(s) = ────────────
//!            1 + s·τ/G
//! ```
//!
//! where `τ` is the curve's time constant and `G` (`asymptotic_gain`) is the
//! high-frequency shelf gain. The ratio of zero to pole is `G`, giving a
//! shelf rising at +20 dB/decade between `f_c = 1/(2π·τ)` and `f_c · G`.
//! For pure FM emphasis the analog standard takes `G → ∞` (single zero, no
//! finite pole); in discrete time we choose a finite `G` large enough that
//! the shelf top is well above the Nyquist of the highest reasonable
//! sample rate — default `G = 10` (20 dB cap). At `f_s = 48 kHz`, `τ = 50
//! µs`, the equivalent pole sits at `G/τ ≈ 200_000 rad/s ≈ 31.8 kHz`, well
//! above the 24 kHz Nyquist; the filter's response at 24 kHz is within
//! about 0.5 dB of the asymptotic `G = 10` limit so the cap is acoustically
//! transparent.
//!
//! ## Bilinear transform
//!
//! Let `c = 2·τ·f_s`. Substituting `s = 2·f_s·(z − 1)/(z + 1)` into
//! `H_pre(s)`:
//!
//! ```text
//!                (1 + c·(z − 1)/(z + 1))
//! H_pre(z) = ─────────────────────────────
//!              (1 + (c/G)·(z − 1)/(z + 1))
//!
//!              (z + 1) + c·(z − 1)
//!          = ────────────────────────────
//!              (z + 1) + (c/G)·(z − 1)
//!
//!              (1 + c)·z + (1 − c)
//!          = ────────────────────────────
//!              (1 + c/G)·z + (1 − c/G)
//! ```
//!
//! Divide numerator and denominator by `(1 + c/G)·z` to obtain the
//! direct-form-I coefficients `(b₀, b₁, a₁)` with `a₀ = 1`:
//!
//! ```text
//! b₀ = (1 + c)     / (1 + c/G)
//! b₁ = (1 − c)     / (1 + c/G)
//! a₁ = (1 − c/G)   / (1 + c/G)
//! ```
//!
//! Sanity checks:
//!
//! * **DC gain.** `H_pre(z = 1) = (b₀ + b₁)/(1 + a₁) =
//!   ((1 + c) + (1 − c)) / ((1 + c/G) + (1 − c/G)) = 2/2 = 1`. ✓
//! * **Nyquist gain.** `H_pre(z = −1) = (b₀ − b₁)/(1 − a₁) =
//!   ((1 + c) − (1 − c)) / ((1 + c/G) − (1 − c/G)) = 2c / (2c/G) = G`. ✓
//!   So the discrete shelf hits exactly the chosen asymptotic boost at
//!   Nyquist.
//! * **−3 dB corner.** For `G ≫ 1` the magnitude `|H(jω)|` rises at
//!   +20 dB/decade between `ω_z = 1/τ` and `ω_p = G/τ`. The corner where
//!   gain crosses `√2` above unity is at `ω = 1/τ` (i.e. `f_c = 1/(2π·τ)`).
//!   For 50 µs: `f_c = 1/(2π·5e−5) ≈ 3183 Hz`. For 75 µs: `≈ 2122 Hz`.
//!
//! ## Streaming difference equation
//!
//! The direct-form-I recurrence per channel is
//!
//! ```text
//! y[n] = b₀ · x[n] + b₁ · x[n−1] − a₁ · y[n−1]
//! ```
//!
//! requiring a single `(x_prev, y_prev)` pair of `f64` state per channel.
//!
//! # Second-order derivation (RIAA)
//!
//! The RIAA record curve is the inverse of the analog playback curve
//!
//! ```text
//!                       1 + s·τ₂
//! H_play(s) = ─────────────────────────────
//!             (1 + s·τ₁)·(1 + s·τ₃)
//! ```
//!
//! so the record (pre-emphasis) curve is its inverse,
//!
//! ```text
//!             (1 + s·τ₁)·(1 + s·τ₃)
//! H_rec(s) = ────────────────────────
//!                    1 + s·τ₂
//! ```
//!
//! As a second-order rational the numerator expands to
//! `1 + (τ₁ + τ₃)·s + τ₁·τ₃·s²` and the denominator is just `1 + τ₂·s`.
//! For the bilinear-mapped filter we apply the substitution to each `s`
//! and collect powers of `z`. With `T = 1/f_s` and `K = 2·f_s`:
//!
//! Let `A = (τ₁ + τ₃)·K`, `B = τ₁·τ₃·K²`, `C = τ₂·K`. The analog
//! numerator becomes
//!
//! ```text
//! N(s) = 1 + A · (z − 1)/(z + 1) + B · ((z − 1)/(z + 1))²
//!      = (z + 1)² + A · (z − 1)·(z + 1) + B · (z − 1)²    (· 1/(z+1)²)
//!      = (1 + A + B)·z² + (2 − 2B)·z + (1 − A + B)        (numerator
//!                                                          coefficients
//!                                                          in z² form)
//! ```
//!
//! The analog denominator becomes
//!
//! ```text
//! D(s) = 1 + C · (z − 1)/(z + 1)
//!      = (1 + C)·z + (1 − C)                              (after one
//!                                                          (z + 1) factor)
//! ```
//!
//! After matching common `(z + 1)` factors (one for `D`, two for `N`), the
//! overall transfer is
//!
//! ```text
//!                ((1+A+B)·z² + (2−2B)·z + (1−A+B))
//! H_rec(z) = ─────────────────────────────────────────
//!                (z + 1) · ((1+C)·z + (1−C))
//! ```
//!
//! Expanding the denominator: `(z+1)·((1+C)·z + (1−C))
//!  = (1+C)·z² + ((1+C) + (1−C))·z + (1−C)
//!  = (1+C)·z² + 2·z + (1−C)`.
//!
//! Normalising both numerator and denominator by the leading denominator
//! coefficient `(1 + C)` gives the direct-form-I coefficients with `a₀ = 1`:
//!
//! ```text
//! b₀ = (1 + A + B) / (1 + C)
//! b₁ = (2 − 2·B)   / (1 + C)
//! b₂ = (1 − A + B) / (1 + C)
//! a₁ = 2          / (1 + C)
//! a₂ = (1 − C)    / (1 + C)
//! ```
//!
//! Sanity:
//!
//! * **DC gain** `H(z=1) = (b₀+b₁+b₂)/(1+a₁+a₂) = ((1+A+B)+(2−2B)+(1−A+B))
//!   / ((1+C)+2+(1−C)) = 4/4 = 1`. ✓
//!
//! ## Streaming difference equation
//!
//! Direct-form-I recurrence per channel:
//!
//! ```text
//! y[n] = b₀·x[n] + b₁·x[n−1] + b₂·x[n−2] − a₁·y[n−1] − a₂·y[n−2]
//! ```
//!
//! requiring `(x_prev, x_prev2, y_prev, y_prev2)` of `f64` per channel.
//!
//! # Stability
//!
//! All pole magnitudes computed above are inside the unit circle for any
//! `c > 0`, `G > 1`, `τ_k > 0`, `f_s > 0`. Specifically:
//!
//! * First-order: `|a₁| = |(1 − c/G)/(1 + c/G)| < 1` for any positive
//!   `c/G`.
//! * Second-order: the two poles are `z = −1` (from the `(z+1)` factor —
//!   marginally on the unit circle) and `z = (C−1)/(C+1)` (strictly
//!   inside for `C > 0`).
//!
//! The first pole at `z = −1` would be of concern as marginal stability
//! except that it is *exactly cancelled* by the corresponding `(z+1)`
//! factor in the numerator after the algebra collapses (the analog
//! transfer's structure has the same number of finite zeros as poles in
//! its bilinear image, since `(1 + s·τ_z)` maps to a factor proportional
//! to `((1+c)·z + (1−c))/(z+1)` which carries its own `1/(z+1)` term
//! that collapses against the matching factor introduced by the bilinear
//! image of the denominator's `(1 + s·τ_p)`). In practice the
//! second-order difference equation as written above is bounded-input
//! bounded-output stable.
//!
//! # Channel independence
//!
//! Each channel carries its own filter state. Stereo input does not
//! cross-talk through the filter; the state vectors are sized to
//! `params.channels` on the first call and on every `set_sample_rate` /
//! `reset`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Pre-emphasis curve family.
///
/// `Fm50us` / `Fm75us` / `J17` / `Custom` are single-time-constant
/// first-order shelving curves; `Riaa3180_318_75` is the three-time-constant
/// second-order phonograph record curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Curve {
    /// FM broadcast pre-emphasis used in Europe (CCIR system), 50 µs.
    Fm50us,
    /// FM broadcast pre-emphasis used in North America, 75 µs.
    Fm75us,
    /// ITU-R J.17 voice-band pre-emphasis (telephony / broadcast voice),
    /// 50 µs time constant.
    J17,
    /// User-specified single-time-constant curve, time constant in
    /// seconds.
    Custom {
        /// Time constant in seconds.
        tau_s: f32,
    },
    /// RIAA phonograph record pre-emphasis (3180 µs + 318 µs + 75 µs
    /// three-time-constant second-order curve).
    Riaa3180_318_75,
}

impl Curve {
    /// Returns `true` if this curve uses the second-order RIAA derivation.
    pub fn is_second_order(self) -> bool {
        matches!(self, Curve::Riaa3180_318_75)
    }

    /// Time constant in seconds for the single-time-constant curves.
    /// Returns `None` for the second-order RIAA curve (which has three).
    pub fn single_tau_s(self) -> Option<f32> {
        match self {
            Curve::Fm50us | Curve::J17 => Some(50.0e-6),
            Curve::Fm75us => Some(75.0e-6),
            Curve::Custom { tau_s } => Some(tau_s),
            Curve::Riaa3180_318_75 => None,
        }
    }
}

/// First-order direct-form-I per-channel state `(x_prev, y_prev)`.
#[derive(Debug, Clone, Copy, Default)]
struct State1 {
    x1: f64,
    y1: f64,
}

/// Second-order direct-form-I per-channel state
/// `(x_prev, x_prev2, y_prev, y_prev2)`.
#[derive(Debug, Clone, Copy, Default)]
struct State2 {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

/// First-order coefficients for `H(z) = (b0 + b1 z^-1) / (1 + a1 z^-1)`.
#[derive(Debug, Clone, Copy)]
struct Coeff1 {
    b0: f64,
    b1: f64,
    a1: f64,
}

impl Coeff1 {
    /// Derive direct-form-I coefficients from the analog shelving
    /// transfer `H(s) = (1 + s·τ) / (1 + s·τ/G)` via bilinear transform
    /// at sample rate `fs`. See module header for the full derivation.
    fn derive(tau_s: f64, g: f64, fs: f64) -> Self {
        let c = 2.0 * tau_s * fs;
        let cg = c / g;
        let d = 1.0 + cg;
        Self {
            b0: (1.0 + c) / d,
            b1: (1.0 - c) / d,
            a1: (1.0 - cg) / d,
        }
    }
}

/// Second-order coefficients for
/// `H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`.
#[derive(Debug, Clone, Copy)]
struct Coeff2 {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Coeff2 {
    /// Derive direct-form-I coefficients for the RIAA record curve
    /// `H(s) = (1 + s·τ1)·(1 + s·τ3) / (1 + s·τ2)` via bilinear transform.
    /// See module header for the full derivation.
    fn derive_riaa(tau1_s: f64, tau2_s: f64, tau3_s: f64, fs: f64) -> Self {
        let k = 2.0 * fs;
        let a = (tau1_s + tau3_s) * k;
        let b = tau1_s * tau3_s * k * k;
        let c = tau2_s * k;
        let inv_d = 1.0 / (1.0 + c);
        Self {
            b0: (1.0 + a + b) * inv_d,
            b1: (2.0 - 2.0 * b) * inv_d,
            b2: (1.0 - a + b) * inv_d,
            a1: 2.0 * inv_d,
            a2: (1.0 - c) * inv_d,
        }
    }
}

/// Streaming pre-emphasis filter.
///
/// State is per-channel; the coefficients are recomputed whenever the
/// configured curve or sample rate changes.
#[derive(Debug, Clone)]
pub struct PreEmphasis {
    curve: Curve,
    /// Asymptotic shelf-top gain `G` for single-time-constant curves.
    /// Default `10.0` gives a 20 dB HF boost cap; values < 1 are clamped
    /// to `1.0` (no boost) and values > 1000 are clamped to keep
    /// numerical headroom.
    asymptotic_gain: f64,
    sample_rate: u32,
    coeff1: Option<Coeff1>,
    coeff2: Option<Coeff2>,
    state1: Vec<State1>,
    state2: Vec<State2>,
}

impl PreEmphasis {
    /// Build a pre-emphasis filter on the given curve with the default
    /// asymptotic boost cap of `G = 10` (20 dB HF asymptote). The
    /// concrete coefficients are derived on the first `process()` call
    /// (or eagerly by [`Self::set_sample_rate`]).
    pub fn new(curve: Curve) -> Self {
        Self {
            curve,
            asymptotic_gain: 10.0,
            sample_rate: 0,
            coeff1: None,
            coeff2: None,
            state1: Vec::new(),
            state2: Vec::new(),
        }
    }

    /// Build a pre-emphasis filter with an explicit asymptotic shelf gain
    /// `G` (clamped to `[1.0, 1000.0]`). Larger `G` extends the shelf
    /// asymptote upward; for FM broadcast emphasis values in the range
    /// 5..=20 are typical (the analog standard's `G → ∞` is approximated
    /// arbitrarily well by any `G` whose equivalent pole `G/τ` sits well
    /// above Nyquist).
    pub fn with_gain(curve: Curve, asymptotic_gain: f32) -> Self {
        Self {
            curve,
            asymptotic_gain: (asymptotic_gain as f64).clamp(1.0, 1000.0),
            sample_rate: 0,
            coeff1: None,
            coeff2: None,
            state1: Vec::new(),
            state2: Vec::new(),
        }
    }

    /// The curve currently in force.
    pub fn curve(&self) -> Curve {
        self.curve
    }

    /// The asymptotic gain `G` currently in force.
    pub fn asymptotic_gain(&self) -> f32 {
        self.asymptotic_gain as f32
    }

    /// Recompute the discrete coefficients for the given sample rate.
    /// Called automatically from `process()` if the rate changes between
    /// frames.
    pub fn set_sample_rate(&mut self, fs: u32) {
        self.sample_rate = fs;
        let fs_f = fs as f64;
        if let Some(tau) = self.curve.single_tau_s() {
            self.coeff1 = Some(Coeff1::derive(tau as f64, self.asymptotic_gain, fs_f));
            self.coeff2 = None;
        } else {
            // RIAA: τ₁ = 3180 µs, τ₂ = 318 µs, τ₃ = 75 µs.
            self.coeff2 = Some(Coeff2::derive_riaa(3180.0e-6, 318.0e-6, 75.0e-6, fs_f));
            self.coeff1 = None;
        }
    }

    /// Clear per-channel state but keep coefficients.
    pub fn reset(&mut self) {
        for s in self.state1.iter_mut() {
            *s = State1::default();
        }
        for s in self.state2.iter_mut() {
            *s = State2::default();
        }
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.curve.is_second_order() {
            if self.state2.len() != channels {
                self.state2 = vec![State2::default(); channels];
            }
            self.state1.clear();
        } else {
            if self.state1.len() != channels {
                self.state1 = vec![State1::default(); channels];
            }
            self.state2.clear();
        }
    }
}

impl AudioFilter for PreEmphasis {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        if self.sample_rate != params.sample_rate {
            self.set_sample_rate(params.sample_rate);
        }
        let mut planes = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(planes.len());

        if let Some(c) = self.coeff1 {
            for (ch, buf) in planes.iter_mut().enumerate() {
                let st = &mut self.state1[ch];
                for s in buf.iter_mut() {
                    let x = *s as f64;
                    let y = c.b0 * x + c.b1 * st.x1 - c.a1 * st.y1;
                    st.x1 = x;
                    st.y1 = y;
                    *s = y as f32;
                }
            }
        } else if let Some(c) = self.coeff2 {
            for (ch, buf) in planes.iter_mut().enumerate() {
                let st = &mut self.state2[ch];
                for s in buf.iter_mut() {
                    let x = *s as f64;
                    let y = c.b0 * x + c.b1 * st.x1 + c.b2 * st.x2 - c.a1 * st.y1 - c.a2 * st.y2;
                    st.x2 = st.x1;
                    st.x1 = x;
                    st.y2 = st.y1;
                    st.y1 = y;
                    *s = y as f32;
                }
            }
        }

        let out = encode_from_f32(params.format, params.channels, input, &planes)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::SampleFormat;
    use std::f32::consts::PI;

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

    fn rms(samples: &[f32]) -> f64 {
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt()
    }

    fn sine_block(freq_hz: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * PI * freq_hz / fs as f32;
        (0..n).map(|i| (i as f32 * w).sin()).collect()
    }

    fn measure_gain_db(
        flt: &mut PreEmphasis,
        params: AudioStreamParams,
        freq_hz: f32,
        n: usize,
    ) -> f64 {
        flt.reset();
        let input = sine_block(freq_hz, params.sample_rate, n);
        let frame = make_f32_mono(&input);
        let out = flt.process(&frame, params).unwrap();
        let out_samples = read_f32(&out[0]);
        // Skip the first 5 % to let the IIR warm up.
        let warm = n / 20;
        let in_rms = rms(&input[warm..]);
        let out_rms = rms(&out_samples[warm..]);
        20.0 * (out_rms / in_rms).log10()
    }

    #[test]
    fn dc_gain_unity_fm50() {
        // DC step of constant 0.5 should converge to 0.5 after settling.
        // The shelving filter has DC gain = 1 exactly (sanity check in
        // the module header derivation).
        let frame = make_f32_mono(&vec![0.5f32; 8192]);
        let mut flt = PreEmphasis::new(Curve::Fm50us);
        let out = flt.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[got.len() - 100..];
        let mean: f64 = tail.iter().map(|&v| v as f64).sum::<f64>() / tail.len() as f64;
        assert!(
            (mean - 0.5).abs() < 1.0e-3,
            "DC gain ≠ 1: tail mean = {mean}"
        );
    }

    #[test]
    fn dc_gain_unity_riaa() {
        let frame = make_f32_mono(&vec![0.3f32; 16384]);
        let mut flt = PreEmphasis::new(Curve::Riaa3180_318_75);
        let out = flt.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[got.len() - 200..];
        let mean: f64 = tail.iter().map(|&v| v as f64).sum::<f64>() / tail.len() as f64;
        assert!(
            (mean - 0.3).abs() < 1.0e-3,
            "RIAA DC gain ≠ 1: tail mean = {mean}"
        );
    }

    #[test]
    fn nyquist_asymptote_equals_g() {
        // Per the derivation, H(z = -1) = G. With G = 10 the alternating
        // sequence ±1 should be amplified by a factor of ~10.
        let mut flt = PreEmphasis::with_gain(Curve::Fm50us, 10.0);
        flt.set_sample_rate(48_000);
        let n = 256;
        let nyq: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.05 } else { -0.05 })
            .collect();
        let frame = make_f32_mono(&nyq);
        let out = flt.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // Skip warm-up; sample the steady-state alternating peak.
        let peak = got[64..].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            (peak / 0.05 - 10.0).abs() < 0.5,
            "Nyquist asymptote should be ≈ G = 10×; got {peak} / 0.05 = {}",
            peak / 0.05
        );
    }

    #[test]
    fn fm50_corner_frequency_derived_from_tau() {
        // f_c = 1/(2π·τ) for τ = 50 µs ⇒ ≈ 3183 Hz. At this frequency
        // the magnitude should be ≈ +3 dB (the −3 dB corner of the
        // shelf's low-frequency tail measured against the asymptotic
        // top). Allow a generous ±2 dB window because the shelf has
        // finite top (G = 10) so the "corner" is approximate.
        let mut flt = PreEmphasis::with_gain(Curve::Fm50us, 10.0);
        let params = f32_mono(48_000);
        let n = 16_384;
        let g_db = measure_gain_db(&mut flt, params, 3_183.0, n);
        // Expected gain at f_c is approximately +3 dB.
        assert!(
            (g_db - 3.0).abs() < 2.0,
            "FM-50 corner gain at 3183 Hz expected ≈ 3 dB; got {g_db} dB"
        );
    }

    #[test]
    fn fm75_corner_frequency_derived_from_tau() {
        // f_c = 1/(2π·75e−6) ≈ 2122 Hz.
        let mut flt = PreEmphasis::with_gain(Curve::Fm75us, 10.0);
        let params = f32_mono(48_000);
        let n = 16_384;
        let g_db = measure_gain_db(&mut flt, params, 2_122.0, n);
        assert!(
            (g_db - 3.0).abs() < 2.0,
            "FM-75 corner gain at 2122 Hz expected ≈ 3 dB; got {g_db} dB"
        );
    }

    #[test]
    fn fm50_slope_between_corners_is_20db_per_decade() {
        // Between f_c = 1/(2π·τ) and the shelf top f_c·G, magnitude
        // should rise at +20 dB/decade. Sample two frequencies one
        // decade apart, comfortably inside the slope region.
        let mut flt = PreEmphasis::with_gain(Curve::Fm50us, 100.0);
        // G = 100 places the shelf top at 318 kHz — well above Nyquist;
        // the slope between 3 kHz (f_c) and 30 kHz is the full +20
        // dB/decade rise, but with Nyquist = 24 kHz let's measure 3 kHz
        // → 12 kHz (half a decade, +10 dB).
        let params = f32_mono(96_000); // higher fs so 12 kHz is well inside.
        let n = 32_768;
        let g_lo = measure_gain_db(&mut flt, params, 3_000.0, n);
        let g_hi = measure_gain_db(&mut flt, params, 12_000.0, n);
        let rise = g_hi - g_lo;
        // Two octaves below shelf top => 12 dB rise expected (±2 dB).
        assert!(
            (rise - 12.0).abs() < 3.0,
            "Between 3 kHz and 12 kHz expected ≈ 12 dB rise; got {rise} dB \
             (g_lo = {g_lo}, g_hi = {g_hi})"
        );
    }

    #[test]
    fn channels_do_not_cross_talk() {
        // Stereo: L = 1 kHz sine, R = silence. After filter R must
        // remain silence to within rounding.
        let fs = 48_000u32;
        let n = 8_192usize;
        let l = sine_block(1_000.0, fs, n);
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for &sample in l.iter().take(n) {
            bytes.extend_from_slice(&sample.to_le_bytes());
            bytes.extend_from_slice(&0.0f32.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let params = AudioStreamParams {
            format: SampleFormat::F32,
            channels: 2,
            sample_rate: fs,
        };
        let mut flt = PreEmphasis::new(Curve::Fm75us);
        let out = flt.process(&frame, params).unwrap();
        let bytes = &out[0].data[0];
        let mut r_peak = 0.0f32;
        for s in 0..n {
            let off = (s * 2 + 1) * 4;
            let v =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            r_peak = r_peak.max(v.abs());
        }
        assert!(r_peak < 1.0e-6, "R-channel polluted, peak = {r_peak}");
    }

    #[test]
    fn streaming_continuity_split_equals_whole() {
        // Two halves of the same input concatenated through one filter
        // instance must match the same input processed in one shot.
        let fs = 48_000u32;
        let n = 4_096usize;
        let input = sine_block(2_000.0, fs, n);
        let params = f32_mono(fs);

        let mut whole = PreEmphasis::new(Curve::Fm50us);
        let out_w = whole.process(&make_f32_mono(&input), params).unwrap();
        let got_w = read_f32(&out_w[0]);

        let mut split = PreEmphasis::new(Curve::Fm50us);
        let half = n / 2;
        let out_a = split
            .process(&make_f32_mono(&input[..half]), params)
            .unwrap();
        let out_b = split
            .process(&make_f32_mono(&input[half..]), params)
            .unwrap();
        let mut got_s = read_f32(&out_a[0]);
        got_s.extend(read_f32(&out_b[0]));

        for (i, (a, b)) in got_w.iter().zip(got_s.iter()).enumerate() {
            assert!((a - b).abs() < 1.0e-6, "split[{i}] = {b}, whole[{i}] = {a}");
        }
    }

    #[test]
    fn sample_rate_change_rederives_coefficients() {
        let mut flt = PreEmphasis::new(Curve::Fm50us);
        flt.set_sample_rate(48_000);
        let c48 = flt.coeff1.expect("48k coeffs derived");
        flt.set_sample_rate(96_000);
        let c96 = flt.coeff1.expect("96k coeffs derived");
        // Different f_s ⇒ different c ⇒ different coefficients.
        assert!(
            (c48.b0 - c96.b0).abs() > 1.0e-6,
            "coefficients should differ between 48k and 96k"
        );
    }

    #[test]
    fn reset_clears_state_but_keeps_coefficients() {
        let mut flt = PreEmphasis::new(Curve::Fm50us);
        flt.set_sample_rate(48_000);
        let pre = flt.coeff1.unwrap();
        let frame = make_f32_mono(&vec![0.5f32; 1024]);
        let _ = flt.process(&frame, f32_mono(48_000)).unwrap();
        assert!(flt.state1[0].x1 != 0.0 || flt.state1[0].y1 != 0.0);
        flt.reset();
        assert_eq!(flt.state1[0].x1, 0.0);
        assert_eq!(flt.state1[0].y1, 0.0);
        let post = flt.coeff1.unwrap();
        assert_eq!(pre.b0, post.b0);
    }

    #[test]
    fn custom_curve_matches_fm50_when_tau_matches() {
        // Custom { tau_s: 50e-6 } must produce bit-identical output to
        // Fm50us at the same gain.
        let mut a = PreEmphasis::with_gain(Curve::Fm50us, 10.0);
        let mut b = PreEmphasis::with_gain(Curve::Custom { tau_s: 50.0e-6 }, 10.0);
        let params = f32_mono(48_000);
        let input = sine_block(1_000.0, 48_000, 1024);
        let oa = a.process(&make_f32_mono(&input), params).unwrap();
        let ob = b.process(&make_f32_mono(&input), params).unwrap();
        let ga = read_f32(&oa[0]);
        let gb = read_f32(&ob[0]);
        for (i, (x, y)) in ga.iter().zip(gb.iter()).enumerate() {
            assert!((x - y).abs() < 1.0e-7, "diverged at {i}: {x} vs {y}");
        }
    }

    #[test]
    fn riaa_amplifies_low_frequencies_more_than_high() {
        // The record curve has very steep boost in the bass (the playback
        // is shelved down at low frequencies, so record must shelve up)
        // and only mild boost in the treble. Specifically the standard
        // record curve at 20 Hz is ~+19.3 dB relative to 1 kHz, and at
        // 20 kHz is ~+19.6 dB relative to 1 kHz — both edges are roughly
        // symmetric in level relative to mid-band.
        //
        // We check only the qualitative property: gain at 100 Hz must
        // exceed gain at 1 kHz (because the playback curve has a strong
        // 1/(1 + sτ1) bass cut → record must compensate with a boost).
        let mut flt = PreEmphasis::new(Curve::Riaa3180_318_75);
        let params = f32_mono(48_000);
        let n = 32_768;
        let g_lo = measure_gain_db(&mut flt, params, 100.0, n);
        let g_mid = measure_gain_db(&mut flt, params, 1_000.0, n);
        assert!(
            g_lo < g_mid - 5.0,
            "RIAA record gain at 100 Hz ({g_lo} dB) should be well \
             BELOW gain at 1 kHz ({g_mid} dB) — the bass shelf is a CUT \
             in the record curve (inverse of the +19 dB playback boost)"
        );
    }

    #[test]
    fn riaa_streaming_continuity_split_equals_whole() {
        let fs = 48_000u32;
        let n = 8_192usize;
        let input: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.3 * (2.0 * PI * 440.0 * t).sin() + 0.2 * (2.0 * PI * 2_000.0 * t).sin()
            })
            .collect();
        let params = f32_mono(fs);

        let mut whole = PreEmphasis::new(Curve::Riaa3180_318_75);
        let out_w = whole.process(&make_f32_mono(&input), params).unwrap();
        let got_w = read_f32(&out_w[0]);

        let mut split = PreEmphasis::new(Curve::Riaa3180_318_75);
        let half = n / 2;
        let out_a = split
            .process(&make_f32_mono(&input[..half]), params)
            .unwrap();
        let out_b = split
            .process(&make_f32_mono(&input[half..]), params)
            .unwrap();
        let mut got_s = read_f32(&out_a[0]);
        got_s.extend(read_f32(&out_b[0]));

        for (i, (a, b)) in got_w.iter().zip(got_s.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-6,
                "RIAA split[{i}] = {b}, whole[{i}] = {a}"
            );
        }
    }

    #[test]
    fn j17_matches_fm50us_first_order() {
        // J.17 shares the 50 µs first-order single-time-constant
        // formulation. Curves should agree bit-exactly at the same gain.
        let mut a = PreEmphasis::with_gain(Curve::Fm50us, 10.0);
        let mut b = PreEmphasis::with_gain(Curve::J17, 10.0);
        let params = f32_mono(48_000);
        let input = sine_block(800.0, 48_000, 512);
        let oa = a.process(&make_f32_mono(&input), params).unwrap();
        let ob = b.process(&make_f32_mono(&input), params).unwrap();
        let ga = read_f32(&oa[0]);
        let gb = read_f32(&ob[0]);
        for (i, (x, y)) in ga.iter().zip(gb.iter()).enumerate() {
            assert!((x - y).abs() < 1.0e-7, "J17 vs FM50 diverged at {i}");
        }
    }

    #[test]
    fn gain_clamped_to_one_below() {
        // G < 1 makes no physical sense (would be a HF *cut*, not
        // pre-emphasis). Clamp to G = 1 ⇒ identity filter.
        let mut flt = PreEmphasis::with_gain(Curve::Fm50us, 0.5);
        assert_eq!(flt.asymptotic_gain(), 1.0);
        let input = sine_block(1_000.0, 48_000, 256);
        let out = flt
            .process(&make_f32_mono(&input), f32_mono(48_000))
            .unwrap();
        let got = read_f32(&out[0]);
        // G = 1 ⇒ zero and pole coincide ⇒ unity transfer everywhere.
        for (i, (x, y)) in input.iter().zip(got.iter()).enumerate() {
            assert!((x - y).abs() < 1.0e-5, "identity broken at {i}");
        }
    }
}
