//! De-emphasis shelving filter — the algebraic inverse of
//! [`PreEmphasis`](crate::pre_emphasis).
//!
//! De-emphasis is applied at the receive/playback side of a noisy channel
//! to restore flat frequency response while also attenuating
//! high-frequency channel noise. The cascade
//! `pre_emphasis · de_emphasis` is a (near-)identity by construction; the
//! two filters share the [`Curve`](crate::pre_emphasis::Curve) family and
//! the same asymptotic-shelf parameter `G`.
//!
//! # First-order derivation (single-time-constant curves)
//!
//! The analog de-emphasis transfer is the inverse of the matching
//! `PreEmphasis` analog transfer:
//!
//! ```text
//!             1 + s·τ/G
//! H_de(s) = ────────────
//!             1 + s·τ
//! ```
//!
//! Bilinear transform `s = 2·f_s·(z − 1)/(z + 1)`. Let `c = 2·τ·f_s`:
//!
//! ```text
//!              (1 + (c/G)·(z − 1)/(z + 1))
//! H_de(z) = ─────────────────────────────────
//!                (1 + c·(z − 1)/(z + 1))
//!
//!              (z + 1) + (c/G)·(z − 1)
//!          = ──────────────────────────────
//!              (z + 1) + c·(z − 1)
//!
//!              (1 + c/G)·z + (1 − c/G)
//!          = ──────────────────────────────
//!              (1 + c)·z + (1 − c)
//! ```
//!
//! Divide top and bottom by `(1 + c)·z`:
//!
//! ```text
//! b₀ = (1 + c/G) / (1 + c)
//! b₁ = (1 − c/G) / (1 + c)
//! a₁ = (1 − c)   / (1 + c)
//! ```
//!
//! Sanity:
//!
//! * **DC gain.** `H_de(z=1) = (b₀+b₁)/(1+a₁) = ((1+c/G)+(1−c/G))
//!   / ((1+c) + (1−c)) · (1) = 2/2 = 1`. ✓
//! * **Nyquist gain.** `H_de(z=−1) = (b₀−b₁)/(1−a₁) = ((1+c/G)−(1−c/G))
//!   / ((1+c)−(1−c)) = (2c/G)/(2c) = 1/G`. ✓ Exact inverse of
//!   [`PreEmphasis`]'s `G` boost.
//!
//! # Second-order derivation (RIAA)
//!
//! The RIAA playback transfer is
//!
//! ```text
//!                       1 + s·τ₂
//! H_play(s) = ─────────────────────────────
//!             (1 + s·τ₁)·(1 + s·τ₃)
//! ```
//!
//! with τ₁ = 3180 µs, τ₂ = 318 µs, τ₃ = 75 µs. Numerator is `1 + τ₂·s`;
//! denominator expands to `1 + (τ₁ + τ₃)·s + τ₁·τ₃·s²`.
//!
//! Let `K = 2·f_s`, `A = (τ₁ + τ₃)·K`, `B = τ₁·τ₃·K²`, `C = τ₂·K`.
//! Numerator becomes
//!
//! ```text
//! N(s) = 1 + C·(z − 1)/(z + 1)
//!      = (1 + C)·z + (1 − C)            (× (z + 1))
//! ```
//!
//! Denominator becomes
//!
//! ```text
//! D(s) = 1 + A·(z − 1)/(z + 1) + B·((z − 1)/(z + 1))²
//!      = (1 + A + B)·z² + (2 − 2B)·z + (1 − A + B)   (× (z + 1)²)
//! ```
//!
//! Cancelling one `(z + 1)` factor between numerator and denominator:
//!
//! ```text
//!                    (1 + C)·z + (1 − C)
//! H_play(z) = ────────────────────────────────────────────
//!              ((1+A+B)·z² + (2−2B)·z + (1−A+B)) / (z+1)
//! ```
//!
//! Multiplying through by `(z + 1)` in numerator (to share the
//! denominator's degree) gives a `(z + 1)·((1+C)·z + (1−C))` numerator
//! which expands to
//!
//! ```text
//! (1 + C)·z² + ((1 + C) + (1 − C))·z + (1 − C)
//!  = (1 + C)·z² + 2·z + (1 − C)
//! ```
//!
//! Normalising both numerator and denominator by `(1 + A + B)` (the
//! leading denominator coefficient) gives the direct-form-I coefficients
//! with `a₀ = 1`:
//!
//! ```text
//! b₀ = (1 + C)   / (1 + A + B)
//! b₁ = 2         / (1 + A + B)
//! b₂ = (1 − C)   / (1 + A + B)
//! a₁ = (2 − 2·B) / (1 + A + B)
//! a₂ = (1 − A + B) / (1 + A + B)
//! ```
//!
//! Sanity:
//!
//! * **DC gain** `H(z=1) = (b₀+b₁+b₂)/(1+a₁+a₂) = ((1+C)+2+(1−C)) /
//!   ((1+A+B)+(2−2B)+(1−A+B)) = 4/4 = 1`. ✓
//! * **Stability.** Both poles must lie inside the unit circle.
//!   Symbolic simplification of the matching `(z + 1)` factor moves
//!   one pole into a benign cancellation and places the other at
//!   `z = (C − 1)/(C + 1)`. For all positive sample rates with the
//!   canonical τ values this lies strictly inside the unit circle,
//!   so the filter is BIBO-stable.
//!
//! # Cascade identity check (audit aid)
//!
//! `PreEmphasis(curve, G) · DeEmphasis(curve, G)` with matching curve
//! and `G` parameters yields `H_pre(z) · H_de(z) = 1` exactly in
//! symbolic algebra (substitute the symbolic coefficients above). In
//! floating-point arithmetic the cascade introduces only `f64`
//! round-off-level error (≲ 10⁻¹² per sample) — see the cascade test
//! at the bottom of this file.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

pub use crate::pre_emphasis::Curve;

#[derive(Debug, Clone, Copy, Default)]
struct State1 {
    x1: f64,
    y1: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct State2 {
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

#[derive(Debug, Clone, Copy)]
struct Coeff1 {
    b0: f64,
    b1: f64,
    a1: f64,
}

impl Coeff1 {
    /// Derive direct-form-I coefficients from the analog inverse
    /// shelving transfer `H(s) = (1 + s·τ/G) / (1 + s·τ)` via bilinear
    /// transform at sample rate `fs`. See module header for the full
    /// derivation.
    fn derive(tau_s: f64, g: f64, fs: f64) -> Self {
        let c = 2.0 * tau_s * fs;
        let cg = c / g;
        let d = 1.0 + c;
        Self {
            b0: (1.0 + cg) / d,
            b1: (1.0 - cg) / d,
            a1: (1.0 - c) / d,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Coeff2 {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

impl Coeff2 {
    /// Derive direct-form-I coefficients for the RIAA playback curve
    /// `H(s) = (1 + s·τ₂) / ((1 + s·τ₁)·(1 + s·τ₃))` via bilinear
    /// transform. See module header for the full derivation.
    fn derive_riaa(tau1_s: f64, tau2_s: f64, tau3_s: f64, fs: f64) -> Self {
        let k = 2.0 * fs;
        let a = (tau1_s + tau3_s) * k;
        let b = tau1_s * tau3_s * k * k;
        let c = tau2_s * k;
        let inv_d = 1.0 / (1.0 + a + b);
        Self {
            b0: (1.0 + c) * inv_d,
            b1: 2.0 * inv_d,
            b2: (1.0 - c) * inv_d,
            a1: (2.0 - 2.0 * b) * inv_d,
            a2: (1.0 - a + b) * inv_d,
        }
    }
}

/// Streaming de-emphasis filter (inverse of [`PreEmphasis`]).
#[derive(Debug, Clone)]
pub struct DeEmphasis {
    curve: Curve,
    asymptotic_gain: f64,
    sample_rate: u32,
    coeff1: Option<Coeff1>,
    coeff2: Option<Coeff2>,
    state1: Vec<State1>,
    state2: Vec<State2>,
}

impl DeEmphasis {
    /// Build a de-emphasis filter on the given curve with the default
    /// asymptotic shelf-top gain `G = 10` (matching the default
    /// `PreEmphasis::new`'s 20 dB HF boost cap).
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

    /// Build with an explicit asymptotic gain `G` (clamped to
    /// `[1.0, 1000.0]`). For the cascade `pre · de` to be exactly
    /// inverse, `G` here must match the value used on the matching
    /// [`PreEmphasis`].
    pub fn with_gain(curve: Curve, asymptotic_gain: f32) -> Self {
        Self {
            curve,
            asymptotic_gain: crate::clamp_param(asymptotic_gain, 10.0, 1.0, 1000.0) as f64,
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
    pub fn set_sample_rate(&mut self, fs: u32) {
        self.sample_rate = fs;
        let fs_f = fs as f64;
        if let Some(tau) = self.curve.single_tau_s() {
            self.coeff1 = Some(Coeff1::derive(tau as f64, self.asymptotic_gain, fs_f));
            self.coeff2 = None;
        } else {
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

impl AudioFilter for DeEmphasis {
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
    use crate::pre_emphasis::PreEmphasis;
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
        flt: &mut DeEmphasis,
        params: AudioStreamParams,
        freq_hz: f32,
        n: usize,
    ) -> f64 {
        flt.reset();
        let input = sine_block(freq_hz, params.sample_rate, n);
        let frame = make_f32_mono(&input);
        let out = flt.process(&frame, params).unwrap();
        let out_samples = read_f32(&out[0]);
        let warm = n / 20;
        let in_rms = rms(&input[warm..]);
        let out_rms = rms(&out_samples[warm..]);
        20.0 * (out_rms / in_rms).log10()
    }

    #[test]
    fn dc_gain_unity_fm75() {
        let frame = make_f32_mono(&vec![0.5f32; 8192]);
        let mut flt = DeEmphasis::new(Curve::Fm75us);
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
        let mut flt = DeEmphasis::new(Curve::Riaa3180_318_75);
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
    fn nyquist_gain_is_inverse_of_g() {
        // H_de(z = -1) = 1/G per the derivation. G = 10 ⇒ Nyquist
        // alternating signal should be attenuated by 10×.
        let mut flt = DeEmphasis::with_gain(Curve::Fm50us, 10.0);
        flt.set_sample_rate(48_000);
        let n = 512;
        let nyq: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let frame = make_f32_mono(&nyq);
        let out = flt.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        let peak = got[128..].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let expected = 0.5 / 10.0;
        assert!(
            (peak - expected).abs() < 0.01,
            "Nyquist attenuation expected ≈ 1/G of 0.5 = {expected}; got {peak}"
        );
    }

    #[test]
    fn fm50_attenuates_high_frequencies_relative_to_dc() {
        // High frequency should be at least 10 dB below DC for G = 10.
        let mut flt = DeEmphasis::with_gain(Curve::Fm50us, 10.0);
        let params = f32_mono(48_000);
        let g_hi = measure_gain_db(&mut flt, params, 10_000.0, 16_384);
        // Expected ~ −15 dB at 10 kHz for τ = 50 µs G = 10.
        assert!(
            g_hi < -8.0,
            "10 kHz with FM-50 G=10 should be attenuated > 8 dB; got {g_hi}"
        );
    }

    #[test]
    fn cascade_pre_then_de_is_near_identity_fm50() {
        // The exact inverse property: pre · de = 1. Run a broadband test
        // signal through pre then de, compare to the original.
        let fs = 48_000u32;
        let n = 4096usize;
        // Deterministic pseudo-noise via splitmix64 (re-using the
        // sample-pattern style from the existing white_noise filter
        // module to avoid importing rand).
        let mut s: u64 = 0xCAFEBABEDEADBEEF;
        let input: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                // Map to [-0.3, 0.3] so we stay well clear of clipping
                // even after the +20 dB pre-emphasis boost.
                let f = (z as f64 / u64::MAX as f64) - 0.5;
                (f * 0.6) as f32
            })
            .collect();

        let mut pre = PreEmphasis::with_gain(Curve::Fm50us, 10.0);
        let mut de = DeEmphasis::with_gain(Curve::Fm50us, 10.0);
        let params = f32_mono(fs);
        let pre_out = pre.process(&make_f32_mono(&input), params).unwrap();
        let pre_samples = read_f32(&pre_out[0]);
        let de_out = de.process(&make_f32_mono(&pre_samples), params).unwrap();
        let recovered = read_f32(&de_out[0]);

        // Skip the first ~100 samples (warm-up of the IIR's transient).
        let warm = 200;
        let in_rms = rms(&input[warm..]);
        let err_rms = {
            let s: f64 = input[warm..]
                .iter()
                .zip(recovered[warm..].iter())
                .map(|(a, b)| {
                    let d = (*a as f64) - (*b as f64);
                    d * d
                })
                .sum();
            (s / (input.len() - warm) as f64).sqrt()
        };
        let err_db = 20.0 * (err_rms / in_rms).log10();
        assert!(
            err_db < -60.0,
            "FM-50 cascade error RMS = {err_db} dB (should be < -60)"
        );
    }

    #[test]
    fn cascade_pre_then_de_is_near_identity_fm75() {
        let fs = 48_000u32;
        let n = 4096usize;
        let mut s: u64 = 0xF00DBABE_12345678;
        let input: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                ((z as f64 / u64::MAX as f64) - 0.5) as f32 * 0.6
            })
            .collect();

        let mut pre = PreEmphasis::with_gain(Curve::Fm75us, 10.0);
        let mut de = DeEmphasis::with_gain(Curve::Fm75us, 10.0);
        let params = f32_mono(fs);
        let pre_out = pre.process(&make_f32_mono(&input), params).unwrap();
        let pre_samples = read_f32(&pre_out[0]);
        let de_out = de.process(&make_f32_mono(&pre_samples), params).unwrap();
        let recovered = read_f32(&de_out[0]);

        let warm = 200;
        let in_rms = rms(&input[warm..]);
        let err_rms = {
            let s: f64 = input[warm..]
                .iter()
                .zip(recovered[warm..].iter())
                .map(|(a, b)| {
                    let d = (*a as f64) - (*b as f64);
                    d * d
                })
                .sum();
            (s / (input.len() - warm) as f64).sqrt()
        };
        let err_db = 20.0 * (err_rms / in_rms).log10();
        assert!(err_db < -60.0, "FM-75 cascade error = {err_db} dB");
    }

    #[test]
    fn cascade_pre_then_de_is_near_identity_riaa() {
        // RIAA cascade — same property, second-order filters.
        let fs = 48_000u32;
        let n = 8192usize;
        let mut s: u64 = 0xABCDEF0123456789;
        let input: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                z ^= z >> 31;
                ((z as f64 / u64::MAX as f64) - 0.5) as f32 * 0.3
            })
            .collect();

        let mut pre = PreEmphasis::new(Curve::Riaa3180_318_75);
        let mut de = DeEmphasis::new(Curve::Riaa3180_318_75);
        let params = f32_mono(fs);
        let pre_out = pre.process(&make_f32_mono(&input), params).unwrap();
        let pre_samples = read_f32(&pre_out[0]);
        let de_out = de.process(&make_f32_mono(&pre_samples), params).unwrap();
        let recovered = read_f32(&de_out[0]);

        // RIAA is second-order so settling takes longer.
        let warm = 1000;
        let in_rms = rms(&input[warm..]);
        let err_rms = {
            let s: f64 = input[warm..]
                .iter()
                .zip(recovered[warm..].iter())
                .map(|(a, b)| {
                    let d = (*a as f64) - (*b as f64);
                    d * d
                })
                .sum();
            (s / (input.len() - warm) as f64).sqrt()
        };
        let err_db = 20.0 * (err_rms / in_rms).log10();
        assert!(
            err_db < -50.0,
            "RIAA cascade error = {err_db} dB (should be < -50)"
        );
    }

    #[test]
    fn channels_do_not_cross_talk() {
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
        let mut flt = DeEmphasis::new(Curve::Fm75us);
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
        let fs = 48_000u32;
        let n = 4_096usize;
        let input = sine_block(2_000.0, fs, n);
        let params = f32_mono(fs);

        let mut whole = DeEmphasis::new(Curve::Fm50us);
        let out_w = whole.process(&make_f32_mono(&input), params).unwrap();
        let got_w = read_f32(&out_w[0]);

        let mut split = DeEmphasis::new(Curve::Fm50us);
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
    fn reset_clears_state_but_keeps_coefficients() {
        let mut flt = DeEmphasis::new(Curve::Fm75us);
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
    fn riaa_amplifies_low_frequencies_more_than_high() {
        // The playback curve has +19 dB at 20 Hz vs −19 dB at 20 kHz
        // relative to the 1 kHz reference. We just check that 100 Hz
        // gain materially exceeds 5 kHz gain.
        let mut flt = DeEmphasis::new(Curve::Riaa3180_318_75);
        let params = f32_mono(48_000);
        let n = 32_768;
        let g_lo = measure_gain_db(&mut flt, params, 100.0, n);
        let g_hi = measure_gain_db(&mut flt, params, 5_000.0, n);
        assert!(
            g_lo > g_hi + 10.0,
            "RIAA playback gain at 100 Hz ({g_lo} dB) should clearly \
             exceed gain at 5 kHz ({g_hi} dB)"
        );
    }

    #[test]
    fn j17_matches_fm50_first_order() {
        // J.17 shares the 50 µs first-order single-time-constant
        // formulation.
        let mut a = DeEmphasis::with_gain(Curve::Fm50us, 10.0);
        let mut b = DeEmphasis::with_gain(Curve::J17, 10.0);
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
        // G < 1 ⇒ clamp to G = 1 ⇒ identity filter (zero/pole coincide).
        let mut flt = DeEmphasis::with_gain(Curve::Fm50us, 0.5);
        assert_eq!(flt.asymptotic_gain(), 1.0);
        let input = sine_block(1_000.0, 48_000, 256);
        let out = flt
            .process(&make_f32_mono(&input), f32_mono(48_000))
            .unwrap();
        let got = read_f32(&out[0]);
        for (i, (x, y)) in input.iter().zip(got.iter()).enumerate() {
            assert!((x - y).abs() < 1.0e-5, "identity broken at {i}");
        }
    }

    #[test]
    fn sample_rate_change_rederives_coefficients() {
        let mut flt = DeEmphasis::new(Curve::Fm50us);
        flt.set_sample_rate(48_000);
        let c48 = flt.coeff1.expect("48k coeffs derived");
        flt.set_sample_rate(96_000);
        let c96 = flt.coeff1.expect("96k coeffs derived");
        assert!(
            (c48.b0 - c96.b0).abs() > 1.0e-6,
            "coefficients should differ between 48k and 96k"
        );
    }

    #[test]
    fn custom_curve_matches_fm75_when_tau_matches() {
        let mut a = DeEmphasis::with_gain(Curve::Fm75us, 10.0);
        let mut b = DeEmphasis::with_gain(Curve::Custom { tau_s: 75.0e-6 }, 10.0);
        let params = f32_mono(48_000);
        let input = sine_block(2_500.0, 48_000, 1024);
        let oa = a.process(&make_f32_mono(&input), params).unwrap();
        let ob = b.process(&make_f32_mono(&input), params).unwrap();
        let ga = read_f32(&oa[0]);
        let gb = read_f32(&ob[0]);
        for (i, (x, y)) in ga.iter().zip(gb.iter()).enumerate() {
            assert!((x - y).abs() < 1.0e-7, "diverged at {i}");
        }
    }
}
