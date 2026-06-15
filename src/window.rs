//! FIR analysis-window catalogue.
//!
//! A reusable, allocation-light library of the common finite-length
//! analysis windows used for spectral analysis (STFT / FFT) and for
//! windowed-sinc FIR design. Where the [`spectrogram`](crate::spectrogram)
//! renderer historically hard-coded a three-entry Hann / Hamming /
//! Blackman switch and the resampler / true-peak detector each grew
//! their own one-off Kaiser taper, this module gathers the full
//! closed-form catalogue in one place so any consumer can request a
//! window by name and get a numerically-checked taper back.
//!
//! # Convention
//!
//! All windows here use the **symmetric** definition with denominator
//! `L - 1` for a length-`L` window: sample `n` (for `0 ≤ n ≤ L-1`)
//! evaluates the staged closed form with `N = L - 1`, so the first and
//! last samples sit at the window's two endpoints. This matches the
//! `0 ≤ n ≤ N` convention used throughout the staged reference and the
//! `denom = n - 1` convention the spectrogram already used.
//!
//! For the cosine-sum family the general form is
//!
//! ```text
//! w[n] = a0 − a1·cos(2πn/N) + a2·cos(4πn/N) − a3·cos(6πn/N) + a4·cos(8πn/N)
//! ```
//!
//! with the per-window coefficient vectors transcribed from the staged
//! reference. Single-cosine windows (Hann, Hamming) are the `L = 2`
//! special case.
//!
//! # Provenance
//!
//! Every coefficient set and closed form is transcribed from
//! `docs/audio/filter/wikipedia-window-function.html` (the
//! cosine-sum / Tukey / Gaussian / Kaiser / sine / Lanczos
//! definitions). No external library source was consulted.

use std::f64::consts::PI;

/// A finite-length analysis window.
///
/// Variants that carry no parameter use the staged closed form
/// directly; the parameterised variants (`Gaussian`, `Tukey`,
/// `Kaiser`) carry the single shape parameter the staged reference
/// names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Window {
    /// All-ones window. `w[n] = 1`.
    Rectangular,
    /// Triangular / Bartlett window (zero-valued endpoints): the
    /// `L`-point triangle peaking at the centre. `w[n] = 1 − |(n − N/2)/(N/2)|`.
    /// (the 1st-order B-spline window.)
    Triangular,
    /// Welch (parabolic) window — a single parabolic section:
    /// `w[n] = 1 − ((n − N/2)/(N/2))²` for `0 ≤ n ≤ N`. The defining
    /// quadratic reaches zero just outside the window span, so the
    /// endpoints are nulled. Close to (and slightly wider main-lobe than)
    /// the `Sine` window; the canonical window of Welch's periodogram-
    /// averaging power-spectral-density estimate.
    Welch,
    /// Parzen window — the 4th-order B-spline (de la Vallée Poussin)
    /// window. A piecewise-cubic, twice-continuously-differentiable
    /// taper with strictly non-negative spectrum; the smoothest of the
    /// polynomial B-spline family (Triangular = 1st order, Welch ≈ 2nd
    /// order, Parzen = 4th order). Defined zero-phase on `|m| ≤ L/2`
    /// (with `m = n − N/2`) by the two-segment cubic in the staged
    /// reference and nulled at the endpoints.
    Parzen,
    /// Hann (raised-cosine). `w[n] = 0.5·(1 − cos(2πn/N))`.
    Hann,
    /// Hamming, optimal coefficients `a0 = 0.53836`, `a1 = 0.46164`.
    Hamming,
    /// Classic ("α = 0.16") Blackman: `a = [0.42, 0.5, 0.08]`.
    Blackman,
    /// "Exact" Blackman: `a = [7938/18608, 9240/18608, 1430/18608]`.
    BlackmanExact,
    /// Nuttall window with continuous first derivative.
    Nuttall,
    /// Blackman–Nuttall window.
    BlackmanNuttall,
    /// Blackman–Harris window (three-term form).
    BlackmanHarris,
    /// Flat-top window (five-term cosine sum, minimal scalloping loss).
    FlatTop,
    /// Sine (a.k.a. cosine / half-sine) window. `w[n] = sin(πn/N)`.
    Sine,
    /// Lanczos window. `w[n] = sinc(2n/N − 1)`.
    Lanczos,
    /// Gaussian window with standard-deviation factor `sigma`
    /// (`σ ≤ 0.5`): `w[n] = exp(−½·((n − N/2)/(σ·N/2))²)`.
    Gaussian(f64),
    /// Tukey (cosine-tapered) window with taper fraction `alpha`
    /// (`α ∈ [0, 1]`); `α = 0` → rectangular, `α = 1` → Hann.
    Tukey(f64),
    /// Kaiser window with shape `beta`:
    /// `w[n] = I₀(β·√(1 − (2n/N − 1)²)) / I₀(β)`.
    Kaiser(f64),
}

impl Window {
    /// Evaluate the window at integer index `n` of a length-`len`
    /// window. Returns `1.0` for `len <= 1` (degenerate one- or
    /// zero-point window). Out-of-range `n` is clamped into `[0, len-1]`.
    pub fn value(self, n: usize, len: usize) -> f64 {
        if len <= 1 {
            return 1.0;
        }
        let nn = n.min(len - 1) as f64;
        let big_n = (len - 1) as f64; // N in the staged 0..=N convention
        match self {
            Window::Rectangular => 1.0,
            Window::Triangular => 1.0 - ((nn - big_n / 2.0) / (big_n / 2.0)).abs(),
            Window::Welch => {
                // Single parabolic section on the 0..=N convention:
                // w[n] = 1 − ((n − N/2)/(N/2))².
                let r = (nn - big_n / 2.0) / (big_n / 2.0);
                1.0 - r * r
            }
            Window::Parzen => parzen(nn, len),
            Window::Hann => cosine_sum(&[0.5, 0.5], nn, big_n),
            Window::Hamming => cosine_sum(&[0.53836, 0.46164], nn, big_n),
            Window::Blackman => cosine_sum(&[0.42, 0.5, 0.08], nn, big_n),
            Window::BlackmanExact => cosine_sum(
                &[7938.0 / 18608.0, 9240.0 / 18608.0, 1430.0 / 18608.0],
                nn,
                big_n,
            ),
            Window::Nuttall => cosine_sum(&[0.355768, 0.487396, 0.144232, 0.012604], nn, big_n),
            Window::BlackmanNuttall => {
                cosine_sum(&[0.3635819, 0.4891775, 0.1365995, 0.0106411], nn, big_n)
            }
            Window::BlackmanHarris => cosine_sum(&[0.35875, 0.48829, 0.14128, 0.01168], nn, big_n),
            Window::FlatTop => cosine_sum(
                &[
                    0.21557895,
                    0.41663158,
                    0.277263158,
                    0.083578947,
                    0.006947368,
                ],
                nn,
                big_n,
            ),
            Window::Sine => (PI * nn / big_n).sin(),
            Window::Lanczos => sinc(2.0 * nn / big_n - 1.0),
            Window::Gaussian(sigma) => {
                let s = sigma.clamp(1.0e-6, 0.5);
                let num = nn - big_n / 2.0;
                let den = s * big_n / 2.0;
                (-0.5 * (num / den) * (num / den)).exp()
            }
            Window::Tukey(alpha) => {
                let a = alpha.clamp(0.0, 1.0);
                if a == 0.0 {
                    return 1.0;
                }
                // Symmetric Tukey on 0..=N.
                let half = big_n / 2.0;
                let x = if nn <= half { nn } else { big_n - nn };
                let edge = a * big_n / 2.0;
                if x < edge {
                    0.5 * (1.0 - (2.0 * PI * x / (a * big_n)).cos())
                } else {
                    1.0
                }
            }
            Window::Kaiser(beta) => {
                let r = 2.0 * nn / big_n - 1.0; // -1..=+1 across the window
                let arg = 1.0 - r * r;
                let num = if arg <= 0.0 {
                    1.0
                } else {
                    bessel_i0(beta * arg.sqrt())
                };
                num / bessel_i0(beta)
            }
        }
    }

    /// Generate the full length-`len` window as a freshly allocated
    /// `Vec<f64>`. `len == 0` yields an empty vector; `len == 1` yields
    /// `[1.0]`.
    pub fn generate(self, len: usize) -> Vec<f64> {
        (0..len).map(|n| self.value(n, len)).collect()
    }

    /// Generate the window as `f32`, convenient for the `f32` PCM hot
    /// paths in this crate (spectrogram, resampler).
    pub fn generate_f32(self, len: usize) -> Vec<f32> {
        (0..len).map(|n| self.value(n, len) as f32).collect()
    }

    /// Coherent (DC) gain of the window: the mean of its samples.
    /// Multiplying a windowed FFT magnitude by `1 / coherent_gain`
    /// restores the amplitude of a tone that fills the window.
    pub fn coherent_gain(self, len: usize) -> f64 {
        if len == 0 {
            return 0.0;
        }
        self.generate(len).iter().sum::<f64>() / len as f64
    }

    /// Equivalent-noise-bandwidth (ENBW) of the window, in DFT bins:
    /// `N·Σw² / (Σw)²`. A rectangular window is `1.0`; tapered windows
    /// are larger (they trade frequency resolution for lower side-lobes).
    pub fn equivalent_noise_bandwidth(self, len: usize) -> f64 {
        if len == 0 {
            return 0.0;
        }
        let w = self.generate(len);
        let sum: f64 = w.iter().sum();
        let sum_sq: f64 = w.iter().map(|v| v * v).sum();
        if sum == 0.0 {
            return 0.0;
        }
        len as f64 * sum_sq / (sum * sum)
    }
}

/// General cosine-sum evaluation:
/// `a0 − a1·cos(2πn/N) + a2·cos(4πn/N) − a3·cos(6πn/N) + …`,
/// with the sign of term `l` being `(-1)^l`.
fn cosine_sum(a: &[f64], n: f64, big_n: f64) -> f64 {
    let base = 2.0 * PI * n / big_n;
    let mut acc = 0.0;
    for (l, &al) in a.iter().enumerate() {
        let sign = if l % 2 == 0 { 1.0 } else { -1.0 };
        acc += sign * al * (l as f64 * base).cos();
    }
    acc
}

/// Parzen (4th-order B-spline) window, length `len`, index `n`.
///
/// Staged zero-phase form on the centred index `m = n − N/2`
/// (`N = L − 1`), with the half-width `h = L/2`:
///
/// ```text
/// w0(m) = 1 − 6·(m/h)²·(1 − |m|/h)        for 0 ≤ |m| ≤ L/4
/// w0(m) = 2·(1 − |m|/h)³                  for L/4 < |m| ≤ L/2
/// w[n]  = w0(n − N/2),  0 ≤ n ≤ N
/// ```
fn parzen(n: f64, len: usize) -> f64 {
    let big_n = (len - 1) as f64;
    let half = len as f64 / 2.0; // h = L/2
    let m = n - big_n / 2.0; // centred index
    let am = m.abs();
    let r = am / half; // |m| / (L/2) ∈ [0, ~1]
    if am <= len as f64 / 4.0 {
        1.0 - 6.0 * r * r * (1.0 - r)
    } else {
        let s = 1.0 - r;
        2.0 * s * s * s
    }
}

/// Normalised sinc: `sinc(x) = sin(πx)/(πx)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Modified Bessel function of the first kind, order 0, evaluated in
/// `f64`. Series expansion; converges quickly for the moderate `β`
/// arguments used by Kaiser windows.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x_sq = (x * x) / 4.0;
    for k in 1..64 {
        term *= half_x_sq / (k as f64 * k as f64);
        sum += term;
        if term < 1.0e-15 * sum {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-9;

    fn assert_close(a: f64, b: f64, eps: f64, msg: &str) {
        assert!(
            (a - b).abs() <= eps,
            "{msg}: {a} vs {b} (Δ={})",
            (a - b).abs()
        );
    }

    #[test]
    fn rectangular_is_all_ones() {
        let w = Window::Rectangular.generate(8);
        assert_eq!(w.len(), 8);
        for v in w {
            assert_close(v, 1.0, EPS, "rect");
        }
    }

    #[test]
    fn degenerate_lengths() {
        assert!(Window::Hann.generate(0).is_empty());
        assert_eq!(Window::Hann.generate(1), vec![1.0]);
        // Any window at len==1 is the single endpoint value 1.0.
        assert_close(Window::Blackman.value(0, 1), 1.0, EPS, "len1");
    }

    #[test]
    fn hann_endpoints_and_centre() {
        // Hann: zero at both endpoints, unity at the exact centre of an
        // odd-length window.
        let w = Window::Hann.generate(9);
        assert_close(w[0], 0.0, EPS, "hann start");
        assert_close(w[8], 0.0, EPS, "hann end");
        assert_close(w[4], 1.0, EPS, "hann centre");
        // Worked midpoint value: n=2, N=8 -> 0.5(1 - cos(π/2)) = 0.5.
        assert_close(w[2], 0.5, EPS, "hann quarter");
    }

    #[test]
    fn hamming_optimal_coefficients() {
        // Endpoints: a0 - a1 = 0.53836 - 0.46164 = 0.07672.
        let w = Window::Hamming.generate(9);
        assert_close(w[0], 0.07672, EPS, "hamming start");
        assert_close(w[8], 0.07672, EPS, "hamming end");
        // Centre: a0 + a1 = 1.0 (cos(2π·4/8)=cos(π)=-1).
        assert_close(w[4], 1.0, EPS, "hamming centre");
    }

    #[test]
    fn blackman_classic_coefficients() {
        // n=0: a0 - a1 + a2 = 0.42 - 0.5 + 0.08 = 0.0.
        let w = Window::Blackman.generate(9);
        assert_close(w[0], 0.0, EPS, "blackman start");
        assert_close(w[8], 0.0, EPS, "blackman end");
        // Centre (n=4,N=8): cos(2π/8·4)=cos(π)=-1, cos(4π/8·4)=cos(2π)=1
        // -> 0.42 + 0.5 + 0.08 = 1.0.
        assert_close(w[4], 1.0, EPS, "blackman centre");
    }

    #[test]
    fn exact_blackman_centre_is_unity() {
        let w = Window::BlackmanExact.generate(9);
        // a0 + a1 + a2 = 1 exactly (sum of the rational coefficients).
        assert_close(w[4], 1.0, 1.0e-12, "exact blackman centre");
        // exact-Blackman does NOT null its endpoints (discontinuity):
        // a0 - a1 + a2 = (7938 - 9240 + 1430)/18608 = 128/18608.
        assert_close(w[0], 128.0 / 18608.0, 1.0e-12, "exact blackman start");
    }

    #[test]
    fn cosine_sum_families_sum_to_unity_at_centre() {
        // Every alternating cosine-sum window peaks at the centre with
        // value Σ a_l (all cosines hit ±1 with the matching sign).
        for (w, coeffs) in [
            (
                Window::Nuttall,
                vec![0.355768, 0.487396, 0.144232, 0.012604],
            ),
            (
                Window::BlackmanNuttall,
                vec![0.3635819, 0.4891775, 0.1365995, 0.0106411],
            ),
            (
                Window::BlackmanHarris,
                vec![0.35875, 0.48829, 0.14128, 0.01168],
            ),
            (
                Window::FlatTop,
                vec![
                    0.21557895,
                    0.41663158,
                    0.277263158,
                    0.083578947,
                    0.006947368,
                ],
            ),
        ] {
            let expect: f64 = coeffs.iter().sum();
            let s = w.generate(9);
            assert_close(s[4], expect, 1.0e-9, "centre = Σa");
        }
    }

    #[test]
    fn flat_top_goes_negative() {
        // The flat-top window is the only catalogue entry that takes
        // negative values near the edges.
        let w = Window::FlatTop.generate(33);
        assert!(
            w.iter().any(|&v| v < -1.0e-3),
            "flat-top should dip negative"
        );
    }

    #[test]
    fn sine_window_values() {
        // Sine: sin(πn/N). n=0 -> 0, centre (n=N/2) -> sin(π/2) = 1.
        let w = Window::Sine.generate(9);
        assert_close(w[0], 0.0, EPS, "sine start");
        assert_close(w[4], 1.0, EPS, "sine centre");
        // n=2,N=8 -> sin(π/4) = √2/2.
        assert_close(w[2], (2.0_f64).sqrt() / 2.0, EPS, "sine quarter");
    }

    #[test]
    fn lanczos_endpoints_and_centre() {
        // Lanczos: sinc(2n/N - 1). Endpoints sinc(±1)=0, centre sinc(0)=1.
        let w = Window::Lanczos.generate(9);
        assert_close(w[0], 0.0, EPS, "lanczos start");
        assert_close(w[8], 0.0, EPS, "lanczos end");
        assert_close(w[4], 1.0, EPS, "lanczos centre");
    }

    #[test]
    fn gaussian_centre_unity_and_symmetry() {
        let g = Window::Gaussian(0.4);
        let w = g.generate(17);
        assert_close(w[8], 1.0, EPS, "gaussian centre");
        for i in 0..8 {
            assert_close(w[i], w[16 - i], EPS, "gaussian symmetry");
        }
        // Endpoint value: exp(-0.5·(1/σ)²) with σ=0.4 -> exp(-0.5/0.16).
        let expect = (-0.5_f64 / (0.4 * 0.4)).exp();
        assert_close(w[0], expect, EPS, "gaussian endpoint");
    }

    #[test]
    fn tukey_limits_match_rectangular_and_hann() {
        let len = 21;
        let rect = Window::Rectangular.generate(len);
        let hann = Window::Hann.generate(len);
        let t0 = Window::Tukey(0.0).generate(len);
        let t1 = Window::Tukey(1.0).generate(len);
        for i in 0..len {
            assert_close(t0[i], rect[i], EPS, "tukey α=0 == rect");
            assert_close(t1[i], hann[i], EPS, "tukey α=1 == hann");
        }
        // Intermediate α: flat unity plateau in the middle, tapered edges.
        let t = Window::Tukey(0.5).generate(len);
        assert_close(t[10], 1.0, EPS, "tukey centre unity");
        assert_close(t[0], 0.0, EPS, "tukey edge zero");
    }

    #[test]
    fn kaiser_beta_zero_is_rectangular() {
        // I0(0)=1, so β=0 -> every tap is 1.
        let w = Window::Kaiser(0.0).generate(16);
        for v in w {
            assert_close(v, 1.0, EPS, "kaiser β=0 == rect");
        }
    }

    #[test]
    fn kaiser_centre_unity_endpoints_attenuated() {
        let w = Window::Kaiser(8.0).generate(17);
        // Centre r=0 -> I0(β)/I0(β) = 1.
        assert_close(w[8], 1.0, EPS, "kaiser centre");
        // Endpoints r=±1 -> I0(0)/I0(β) = 1/I0(8) (small but positive).
        let expect = 1.0 / bessel_i0(8.0);
        assert_close(w[0], expect, EPS, "kaiser endpoint");
        assert_close(w[16], expect, EPS, "kaiser endpoint sym");
        assert!(w[0] > 0.0 && w[0] < 0.05, "kaiser edge small positive");
    }

    #[test]
    fn bessel_i0_reference_values() {
        // I0(0) = 1 exactly.
        assert_close(bessel_i0(0.0), 1.0, 1.0e-12, "I0(0)");
        // I0(1) ≈ 1.2660658777520084.
        assert_close(bessel_i0(1.0), 1.2660658777520084, 1.0e-10, "I0(1)");
        // I0(2) ≈ 2.2795853023360673.
        assert_close(bessel_i0(2.0), 2.2795853023360673, 1.0e-9, "I0(2)");
    }

    #[test]
    fn triangular_endpoints_and_centre() {
        // Bartlett: zero (or near) at endpoints, unity at the centre.
        let w = Window::Triangular.generate(9);
        assert_close(w[0], 0.0, EPS, "tri start");
        assert_close(w[8], 0.0, EPS, "tri end");
        assert_close(w[4], 1.0, EPS, "tri centre");
        // n=2,N=8 -> 1 - |(2-4)/4| = 0.5.
        assert_close(w[2], 0.5, EPS, "tri quarter");
    }

    #[test]
    fn welch_parabola_values() {
        // Welch: 1 − ((n − N/2)/(N/2))². Nulls both endpoints exactly,
        // unity at the centre of an odd-length window.
        let w = Window::Welch.generate(9);
        assert_close(w[0], 0.0, EPS, "welch start");
        assert_close(w[8], 0.0, EPS, "welch end");
        assert_close(w[4], 1.0, EPS, "welch centre");
        // n=2,N=8 -> r=(2-4)/4=-0.5 -> 1 - 0.25 = 0.75.
        assert_close(w[2], 0.75, EPS, "welch quarter");
        // n=3,N=8 -> r=-0.25 -> 1 - 0.0625 = 0.9375.
        assert_close(w[3], 0.9375, EPS, "welch three-quarter");
    }

    #[test]
    fn welch_is_strictly_concave_and_nonnegative() {
        // A parabola opening downward: monotone increasing to the centre,
        // strictly positive on the interior, never negative anywhere.
        let len = 33;
        let w = Window::Welch.generate(len);
        let mid = len / 2;
        for i in 1..=mid {
            assert!(w[i] >= w[i - 1], "welch rising to centre at {i}");
        }
        for &v in &w[1..len - 1] {
            assert!(v > 0.0, "welch interior strictly positive: {v}");
        }
        assert!(w.iter().all(|&v| v >= 0.0), "welch never negative");
    }

    #[test]
    fn parzen_centre_unity_and_endpoint_formula() {
        // Parzen peaks at unity at the centre.
        let w = Window::Parzen.generate(9);
        assert_close(w[4], 1.0, EPS, "parzen centre");
        // Endpoint: at n=0, |m|=N/2, r=(L-1)/L, (1-r)=1/L, so the cubic
        // segment gives 2·(1/L)³. For L=9 that is 2/729.
        assert_close(w[0], 2.0 / 729.0, EPS, "parzen endpoint");
        assert_close(w[8], 2.0 / 729.0, EPS, "parzen endpoint sym");
    }

    #[test]
    fn parzen_is_smooth_nonnegative_bell() {
        // Parzen is a non-negative, monotone-to-centre bell (smoothest
        // of the polynomial B-spline family).
        let len = 65;
        let w = Window::Parzen.generate(len);
        let mid = len / 2;
        for i in 1..=mid {
            assert!(w[i] >= w[i - 1] - 1.0e-12, "parzen rising to centre at {i}");
        }
        assert!(w.iter().all(|&v| v >= 0.0), "parzen never negative");
        // Endpoint shrinks as the window lengthens: 2·(1/L)³.
        assert_close(
            w[0],
            2.0 / (len as f64).powi(3),
            1.0e-12,
            "parzen endpoint L=65",
        );
    }

    #[test]
    fn polynomial_bspline_enbw_ordering() {
        // Equivalent-noise-bandwidth widens monotonically as the
        // polynomial taper grows smoother:
        // rectangular (1.0) < Welch (≈1.20) < Triangular/Bartlett
        // (≈1.34) < Parzen (≈1.92).
        let len = 256;
        let rect = Window::Rectangular.equivalent_noise_bandwidth(len);
        let welch = Window::Welch.equivalent_noise_bandwidth(len);
        let tri = Window::Triangular.equivalent_noise_bandwidth(len);
        let parzen = Window::Parzen.equivalent_noise_bandwidth(len);
        assert_close(rect, 1.0, 1.0e-9, "rect ENBW == 1");
        assert!(welch > rect, "welch widens vs rect ({welch} > {rect})");
        assert!(tri > welch, "triangular wider than welch ({tri} > {welch})");
        assert!(parzen > tri, "parzen widest ({parzen} > {tri})");
    }

    #[test]
    fn enbw_ordering() {
        // Equivalent-noise-bandwidth: rectangular == 1, and tapered
        // windows widen monotonically Hann < Blackman < flat-top.
        let len = 64;
        let rect = Window::Rectangular.equivalent_noise_bandwidth(len);
        let hann = Window::Hann.equivalent_noise_bandwidth(len);
        let black = Window::Blackman.equivalent_noise_bandwidth(len);
        let flat = Window::FlatTop.equivalent_noise_bandwidth(len);
        assert_close(rect, 1.0, 1.0e-9, "rect ENBW == 1");
        assert!(hann > rect, "hann widens vs rect ({hann} > {rect})");
        assert!(black > hann, "blackman wider than hann ({black} > {hann})");
        assert!(flat > black, "flat-top widest ({flat} > {black})");
        // Hann ENBW approaches the textbook asymptotic 1.5 bins as the
        // window lengthens (symmetric L-1 convention converges from
        // above: ≈ 1.524 at L=64, ≈ 1.5015 at L=1024).
        let hann_long = Window::Hann.equivalent_noise_bandwidth(1024);
        assert!(
            (hann_long - 1.5).abs() < 0.01,
            "hann ENBW → 1.5: {hann_long}"
        );
        assert!(hann > 1.5 && hann < 1.55, "hann ENBW near 1.5: {hann}");
    }

    #[test]
    fn coherent_gain_rectangular_is_one() {
        assert_close(Window::Rectangular.coherent_gain(32), 1.0, EPS, "rect gain");
        // Hann coherent gain -> 0.5 for a long window.
        assert!(
            (Window::Hann.coherent_gain(1024) - 0.5).abs() < 1.0e-3,
            "hann gain ≈ 0.5"
        );
    }

    #[test]
    fn generate_f32_matches_f64() {
        let a = Window::Blackman.generate(31);
        let b = Window::Blackman.generate_f32(31);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((*x as f32 - *y).abs() < 1.0e-6, "f32 == f64 cast");
        }
    }

    #[test]
    fn all_windows_are_symmetric() {
        let len = 23;
        for w in [
            Window::Rectangular,
            Window::Triangular,
            Window::Welch,
            Window::Parzen,
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::BlackmanExact,
            Window::Nuttall,
            Window::BlackmanNuttall,
            Window::BlackmanHarris,
            Window::FlatTop,
            Window::Sine,
            Window::Lanczos,
            Window::Gaussian(0.3),
            Window::Tukey(0.4),
            Window::Kaiser(6.0),
        ] {
            let s = w.generate(len);
            for i in 0..len {
                assert_close(s[i], s[len - 1 - i], 1.0e-9, "symmetry");
            }
        }
    }
}
