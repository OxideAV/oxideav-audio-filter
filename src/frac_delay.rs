//! Fractional-delay interpolated sample reads.
//!
//! Many time-varying effects (chorus, flanger, vibrato, pitch-shift,
//! Doppler, comb resonators) need the value of a discrete signal at a
//! *continuous* position `p` that falls **between** two stored samples.
//! Reconstructing `x(p)` for non-integer `p` is exactly the
//! bandlimited-interpolation problem: a signal sampled at rate `Fs` and
//! bandlimited to `Fs/2` is, by Shannon's sampling theorem, uniquely
//! recoverable at any continuous time from its samples. The reference
//! material describes several ways to approximate that reconstruction
//! and ranks them by audio suitability.
//!
//! This module collects those reconstruction kernels in one allocation-
//! light primitive so the effect modules stop each hand-rolling their
//! own two-tap linear read. A [`FracDelayLine`] owns a per-channel ring
//! buffer; callers `push` one input sample per channel per step and
//! `read` at any fractional delay `d ≥ 0` samples into the past, with
//! the interpolation [`Interp`] kernel chosen at construction.
//!
//! # Reconstruction kernels
//!
//! Let `p` be the continuous read position and write `p = i + f` with
//! integer part `i = floor(p)` and fractional part `f ∈ [0, 1)`.
//!
//! * [`Interp::Linear`] — two-tap straight-line blend
//!   `y = (1 − f)·x[i] + f·x[i+1]`. Cheapest; its error is the textbook
//!   bound that grows with frequency, audible as a gentle treble roll-off
//!   that deepens as `f → 0.5`.
//! * [`Interp::Hermite`] — four-tap cubic through `x[i]` and `x[i+1]`
//!   with the Catmull–Rom tangent estimate (centred finite differences
//!   of the bracketing samples). A C¹ cubic polynomial — markedly flatter
//!   passband than linear for one extra tap pair.
//! * [`Interp::Lagrange(n)`] — the order-`n` polynomial that passes
//!   exactly through the `n + 1` nearest samples, evaluated with the
//!   barycentric / product Lagrange basis
//!   `Lⱼ(x) = Πₘ≠ⱼ (x − xₘ)/(xⱼ − xₘ)`. Order 1 reduces to `Linear`;
//!   order 3 is the four-tap cubic Lagrange the reference relates to a
//!   truncated sinc. The taps are centred on the read position so the
//!   approximation is symmetric about `f = 0.5`.
//! * [`Interp::Sinc { half_taps, beta }`] — a windowed-sinc kernel that
//!   approximates the ideal reconstruction sum
//!   `x(p) = Σₖ x[k]·sinc(p − k)` (Shannon's formula) truncated to
//!   `2·half_taps` neighbours and tapered by a Kaiser window of shape
//!   `beta` to suppress the truncation ripple. The most faithful kernel
//!   here — its flat region extends nearly to Nyquist as `half_taps`
//!   grows — at the cost of `2·half_taps` multiply-adds per channel per
//!   read.
//!
//! # Provenance
//!
//! The ideal reconstruction sum (`x(t) = Σ x(nTs)·hs(t − nTs)`, with
//! `hs` the normalised sinc), the linear-interpolation error framing,
//! and the sinc↔Lagrange relationship are taken from the staged
//! resampling references under `docs/audio/filter/`
//! (`jos-bandlimited-interpolation.html`,
//! `jos-theory-of-sample-rate-conversion.html`, `jos-resample.html`).
//! The Kaiser window shape reuses [`crate::window::Window::Kaiser`].
//! No external library source was consulted.

use crate::window::Window;
use std::f64::consts::PI;

/// Interpolation kernel used to reconstruct a sample at a fractional
/// (between-sample) read position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Interp {
    /// Two-tap linear blend. Cheapest, with a frequency-dependent
    /// treble roll-off deepest at `f = 0.5`.
    Linear,
    /// Four-tap C¹ cubic with Catmull–Rom tangents (centred differences).
    Hermite,
    /// Order-`n` Lagrange polynomial through the `n + 1` nearest taps.
    /// `n` is clamped to `[1, MAX_LAGRANGE_ORDER]`; `n = 1` ≡ `Linear`.
    Lagrange(usize),
    /// Windowed-sinc reconstruction truncated to `2·half_taps`
    /// neighbours, tapered by a Kaiser window of shape `beta`.
    /// `half_taps` is clamped to `[1, MAX_SINC_HALF_TAPS]`.
    Sinc {
        /// Half the number of taps either side of the read position.
        half_taps: usize,
        /// Kaiser shape parameter (larger ⇒ deeper stop-band, wider main lobe).
        beta: f64,
    },
}

/// Largest supported Lagrange order (kept small: high-order polynomial
/// interpolation through equally-spaced points is ill-conditioned).
pub const MAX_LAGRANGE_ORDER: usize = 8;

/// Largest supported sinc half-width (`2·N` taps total).
pub const MAX_SINC_HALF_TAPS: usize = 32;

/// Per-channel fractional-delay ring buffer with a selectable
/// reconstruction kernel.
///
/// `push` advances the line by one input sample per channel; `read`
/// evaluates the signal at a fractional delay into the past. The ring
/// length bounds the maximum delay (`capacity − 1 − kernel_reach`
/// samples; see [`max_delay`](FracDelayLine::max_delay)).
pub struct FracDelayLine {
    channels: usize,
    /// Per-channel ring buffers, each `capacity` long.
    rings: Vec<Vec<f32>>,
    capacity: usize,
    /// Index of the most-recently-written sample in each ring.
    head: usize,
    /// Number of samples pushed so far (saturates at `capacity`).
    filled: usize,
    interp: Interp,
}

impl FracDelayLine {
    /// Build a line for `channels` channels with ring length `capacity`
    /// samples and interpolation kernel `interp`.
    ///
    /// `channels` and `capacity` are forced to at least 1. Kernel
    /// parameters are clamped to their supported ranges.
    pub fn new(channels: usize, capacity: usize, interp: Interp) -> Self {
        let channels = channels.max(1);
        let capacity = capacity.max(1);
        let interp = Self::sanitize(interp);
        Self {
            channels,
            rings: vec![vec![0.0; capacity]; channels],
            capacity,
            head: 0,
            filled: 0,
            interp,
        }
    }

    fn sanitize(interp: Interp) -> Interp {
        match interp {
            Interp::Lagrange(n) => Interp::Lagrange(n.clamp(1, MAX_LAGRANGE_ORDER)),
            Interp::Sinc { half_taps, beta } => Interp::Sinc {
                half_taps: half_taps.clamp(1, MAX_SINC_HALF_TAPS),
                beta: if beta.is_finite() && beta >= 0.0 {
                    beta
                } else {
                    0.0
                },
            },
            other => other,
        }
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Ring capacity in samples.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The active interpolation kernel.
    pub fn interp(&self) -> Interp {
        self.interp
    }

    /// How many integer samples ahead/behind the read position the
    /// kernel reaches. The read position can use up to `reach` future
    /// taps (toward `head`) and `reach` past taps, so the largest
    /// safely-readable delay is `capacity − 1 − reach`.
    fn reach(&self) -> usize {
        match self.interp {
            Interp::Linear => 1,
            Interp::Hermite => 2,
            // Order n needs ceil((n+1)/2) taps on the far side.
            Interp::Lagrange(n) => n.div_ceil(2) + 1,
            Interp::Sinc { half_taps, .. } => half_taps,
        }
    }

    /// Largest fractional delay (in samples) that can be read without
    /// the kernel running off the end of the buffered history.
    pub fn max_delay(&self) -> f32 {
        let reach = self.reach();
        (self.capacity.saturating_sub(1 + reach)) as f32
    }

    /// Reset all state (zero the rings, empty the fill count).
    pub fn reset(&mut self) {
        for r in &mut self.rings {
            r.iter_mut().for_each(|s| *s = 0.0);
        }
        self.head = 0;
        self.filled = 0;
    }

    /// Push one sample per channel, advancing the line by one step.
    ///
    /// `samples.len()` must equal [`channels`](Self::channels); extra
    /// entries are ignored and missing channels are treated as `0.0`.
    pub fn push(&mut self, samples: &[f32]) {
        self.head = (self.head + 1) % self.capacity;
        for (ch, ring) in self.rings.iter_mut().enumerate() {
            ring[self.head] = samples.get(ch).copied().unwrap_or(0.0);
        }
        if self.filled < self.capacity {
            self.filled += 1;
        }
    }

    /// Fetch the stored sample for `channel` at integer delay `d` into
    /// the past (`d = 0` is the most recent `push`). Out-of-range
    /// channel or under-filled history reads return `0.0`.
    fn tap(&self, channel: usize, d: isize) -> f32 {
        if channel >= self.channels {
            return 0.0;
        }
        // d < 0 would read samples newer than the most recent push, which
        // don't exist. A kernel centred near the newest sample reaches a
        // tap or two into that region; clamp to the newest sample (edge
        // extension) so the boundary read degrades gracefully instead of
        // injecting a spurious zero.
        let d = if d < 0 { 0usize } else { d as usize };
        if d >= self.capacity {
            return 0.0;
        }
        // Treat positions beyond the filled history as silence.
        if d >= self.filled {
            return 0.0;
        }
        let idx = (self.head + self.capacity - d) % self.capacity;
        self.rings[channel][idx]
    }

    /// Read `channel` at a fractional delay of `delay` samples into the
    /// past (`delay = 0.0` returns the most recent sample). Negative
    /// delays clamp to `0.0`. The interpolation kernel selected at
    /// construction reconstructs the between-sample value.
    pub fn read(&self, channel: usize, delay: f32) -> f32 {
        let delay = if delay.is_finite() {
            delay.max(0.0)
        } else {
            0.0
        };
        // Split into integer delay `i` and fraction `f` measured *toward
        // the past*: position = i + f samples old, with f in [0, 1).
        let i = delay.floor() as isize;
        let f = (delay - i as f32) as f64;
        match self.interp {
            Interp::Linear => self.read_linear(channel, i, f),
            Interp::Hermite => self.read_hermite(channel, i, f),
            Interp::Lagrange(n) => self.read_lagrange(channel, i, f, n),
            Interp::Sinc { half_taps, beta } => self.read_sinc(channel, i, f, half_taps, beta),
        }
    }

    /// `y = (1 − f)·x[i] + f·x[i+1]`, where larger delay index ⇒ older.
    fn read_linear(&self, channel: usize, i: isize, f: f64) -> f32 {
        let a = self.tap(channel, i) as f64; // newer side (smaller delay)
        let b = self.tap(channel, i + 1) as f64; // older side
        ((1.0 - f) * a + f * b) as f32
    }

    /// Catmull–Rom four-tap cubic. With `s = f` the interval runs from
    /// the newer bracket sample `x1` (delay `i`) to the older `x2`
    /// (delay `i+1`), tangents from the outer samples `x0`/`x3`.
    fn read_hermite(&self, channel: usize, i: isize, f: f64) -> f32 {
        // Order by increasing delay (x0 newest .. x3 oldest). We
        // interpolate between x1 and x2 as the delay grows by `f`.
        let x0 = self.tap(channel, i - 1) as f64;
        let x1 = self.tap(channel, i) as f64;
        let x2 = self.tap(channel, i + 1) as f64;
        let x3 = self.tap(channel, i + 2) as f64;
        // Catmull–Rom basis for the segment x1->x2 at parameter f.
        let c0 = x1;
        let c1 = 0.5 * (x2 - x0);
        let c2 = x0 - 2.5 * x1 + 2.0 * x2 - 0.5 * x3;
        let c3 = 0.5 * (x3 - x0) + 1.5 * (x1 - x2);
        (((c3 * f + c2) * f + c1) * f + c0) as f32
    }

    /// Order-`n` Lagrange polynomial through the `n + 1` taps centred on
    /// the read position. Node `k` sits at delay `base + k` (integer);
    /// the continuous position relative to the node block is
    /// `xq = (i − base) + f`.
    fn read_lagrange(&self, channel: usize, i: isize, f: f64, n: usize) -> f32 {
        let order = n; // sanitized to [1, MAX_LAGRANGE_ORDER]
                       // Centre the (order+1) nodes so the block straddles [i, i+1].
                       // For odd order this is symmetric; for even order it leans one
                       // sample toward the newer side.
        let left = (order / 2) as isize;
        let base = i - left;
        let xq = (i - base) as f64 + f; // query coordinate in node space
        let mut acc = 0.0_f64;
        for j in 0..=order {
            let mut basis = 1.0_f64;
            let xj = j as f64;
            for m in 0..=order {
                if m == j {
                    continue;
                }
                let xm = m as f64;
                basis *= (xq - xm) / (xj - xm);
            }
            acc += basis * self.tap(channel, base + j as isize) as f64;
        }
        acc as f32
    }

    /// Windowed-sinc reconstruction `Σ x[k]·sinc(pos − k)·w[k]`,
    /// truncated to `2·half_taps` taps and tapered by a Kaiser window.
    fn read_sinc(&self, channel: usize, i: isize, f: f64, half_taps: usize, beta: f64) -> f32 {
        let h = half_taps as isize;
        let kaiser = Window::Kaiser(beta);
        // Window length spanning the 2*half_taps taps (symmetric).
        let win_len = 2 * half_taps;
        let mut acc = 0.0_f64;
        let mut wsum = 0.0_f64;
        // Taps run from delay (i - h + 1) .. (i + h), i.e. h newer taps
        // and h older taps bracketing the fractional position.
        for (widx, k) in ((-h + 1)..=h).enumerate() {
            let delay = i + k;
            // Distance from the continuous read position to this tap, in
            // samples: read position is `i + f` (old), tap is at `delay`.
            let dist = (i as f64 + f) - delay as f64;
            let s = sinc(dist);
            let w = kaiser.value(widx, win_len);
            acc += self.tap(channel, delay) as f64 * s * w;
            wsum += s * w;
        }
        // Normalise so a DC input reconstructs to its own value.
        if wsum.abs() > 1e-12 {
            (acc / wsum) as f32
        } else {
            acc as f32
        }
    }
}

/// Normalised sinc `sin(πx)/(πx)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push a slice of mono samples in chronological order.
    fn push_mono(line: &mut FracDelayLine, xs: &[f32]) {
        for &x in xs {
            line.push(&[x]);
        }
    }

    #[test]
    fn integer_delay_returns_exact_sample_all_kernels() {
        for interp in [
            Interp::Linear,
            Interp::Hermite,
            Interp::Lagrange(3),
            Interp::Lagrange(5),
            Interp::Sinc {
                half_taps: 8,
                beta: 8.0,
            },
        ] {
            let mut l = FracDelayLine::new(1, 128, interp);
            let xs: Vec<f32> = (0..40).map(|n| (n as f32 * 0.1).sin()).collect();
            push_mono(&mut l, &xs);
            // delay 0 = most recent, delay d = xs[len-1-d]
            for d in 0..6 {
                let got = l.read(0, d as f32);
                let want = xs[xs.len() - 1 - d];
                assert!(
                    (got - want).abs() < 1e-5,
                    "{interp:?} delay {d}: got {got} want {want}"
                );
            }
        }
    }

    #[test]
    fn linear_midpoint_is_average() {
        let mut l = FracDelayLine::new(1, 16, Interp::Linear);
        push_mono(&mut l, &[1.0, 3.0]); // newest = 3.0 (delay 0), 1.0 (delay 1)
                                        // delay 0.5 sits halfway between the two -> mean.
        assert!((l.read(0, 0.5) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn all_kernels_reproduce_constant_dc() {
        // A constant signal must be reconstructed exactly (partition of
        // unity / DC-gain-one property) at any fractional position.
        for interp in [
            Interp::Linear,
            Interp::Hermite,
            Interp::Lagrange(2),
            Interp::Lagrange(4),
            Interp::Lagrange(7),
            Interp::Sinc {
                half_taps: 12,
                beta: 9.0,
            },
        ] {
            let mut l = FracDelayLine::new(1, 128, interp);
            push_mono(&mut l, &[0.7_f32; 80]);
            for &d in &[0.0, 0.25, 0.5, 0.5, 0.9, 3.3, 7.7] {
                let got = l.read(0, d);
                assert!(
                    (got - 0.7).abs() < 1e-4,
                    "{interp:?} delay {d}: DC reconstructed as {got}"
                );
            }
        }
    }

    #[test]
    fn lagrange_reproduces_low_order_polynomial_exactly() {
        // An order-n Lagrange interpolator is exact on any polynomial of
        // degree <= n. Feed a quadratic and read with order 3.
        let mut l = FracDelayLine::new(1, 64, Interp::Lagrange(3));
        // x[n] = a + b n + c n^2 sampled on the grid.
        let (a, b, c) = (0.5_f64, -0.2_f64, 0.03_f64);
        let n_total = 30usize;
        let xs: Vec<f32> = (0..n_total)
            .map(|n| (a + b * n as f64 + c * (n * n) as f64) as f32)
            .collect();
        push_mono(&mut l, &xs);
        // Read at a few fractional positions and compare to the closed
        // form. delay d -> absolute index (n_total-1) - d. Keep delays a
        // couple of samples off the newest edge so the order-3 block's
        // newer-side tap stays inside real history (no edge clamping).
        for &d in &[2.3_f32, 3.5, 5.7, 8.4] {
            let abs_pos = (n_total - 1) as f64 - d as f64;
            let want = a + b * abs_pos + c * abs_pos * abs_pos;
            let got = l.read(0, d) as f64;
            assert!(
                (got - want).abs() < 1e-3,
                "delay {d}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn lagrange_order_one_matches_linear() {
        let mut a = FracDelayLine::new(1, 32, Interp::Lagrange(1));
        let mut b = FracDelayLine::new(1, 32, Interp::Linear);
        let xs: Vec<f32> = (0..20).map(|n| (0.3 * n as f32).cos()).collect();
        push_mono(&mut a, &xs);
        push_mono(&mut b, &xs);
        for k in 0..20 {
            let d = k as f32 * 0.17;
            assert!((a.read(0, d) - b.read(0, d)).abs() < 1e-6);
        }
    }

    #[test]
    fn hermite_passes_through_endpoints() {
        // At f=0 Hermite must return x1 (delay i) and at the next integer
        // delay return the older sample exactly.
        let mut l = FracDelayLine::new(1, 16, Interp::Hermite);
        push_mono(&mut l, &[1.0, -2.0, 4.0, -1.0, 3.0]);
        // delay 1 -> -1.0, delay 2 -> 4.0
        assert!((l.read(0, 1.0) - (-1.0)).abs() < 1e-6);
        assert!((l.read(0, 2.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn sinc_beats_linear_on_sine_treble() {
        // A windowed-sinc kernel must track a high-frequency sine more
        // faithfully than linear interpolation at the worst-case f=0.5.
        let fs = 48_000.0_f32;
        let freq = 8_000.0_f32; // well below Nyquist but high
        let n_total = 256usize;
        let xs: Vec<f32> = (0..n_total)
            .map(|n| (2.0 * std::f32::consts::PI * freq * n as f32 / fs).sin())
            .collect();

        let mut lin = FracDelayLine::new(1, 512, Interp::Linear);
        let mut snc = FracDelayLine::new(
            1,
            512,
            Interp::Sinc {
                half_taps: 16,
                beta: 9.0,
            },
        );
        push_mono(&mut lin, &xs);
        push_mono(&mut snc, &xs);

        // True value half a sample older than the newest sample.
        let mut lin_err = 0.0_f64;
        let mut snc_err = 0.0_f64;
        let mut count = 0.0_f64;
        for d_int in 1..40 {
            let d = d_int as f32 + 0.5;
            // continuous absolute index
            let abs_pos = (n_total - 1) as f32 - d;
            let truth = (2.0 * std::f32::consts::PI * freq * abs_pos / fs).sin() as f64;
            lin_err += (lin.read(0, d) as f64 - truth).powi(2);
            snc_err += (snc.read(0, d) as f64 - truth).powi(2);
            count += 1.0;
        }
        let lin_rms = (lin_err / count).sqrt();
        let snc_rms = (snc_err / count).sqrt();
        assert!(
            snc_rms < lin_rms * 0.25,
            "sinc rms {snc_rms} not << linear rms {lin_rms}"
        );
    }

    #[test]
    fn channels_are_independent() {
        let mut l = FracDelayLine::new(2, 16, Interp::Linear);
        l.push(&[1.0, 10.0]);
        l.push(&[2.0, 20.0]);
        l.push(&[3.0, 30.0]);
        // delay 0.5 between newest (3,30) and prev (2,20).
        assert!((l.read(0, 0.5) - 2.5).abs() < 1e-6);
        assert!((l.read(1, 0.5) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn underfilled_history_reads_zero() {
        let mut l = FracDelayLine::new(1, 32, Interp::Linear);
        l.push(&[5.0]);
        // Only one sample pushed; delay 0 is 5.0, but delay 3 has no
        // history -> 0.
        assert!((l.read(0, 0.0) - 5.0).abs() < 1e-6);
        assert!(l.read(0, 3.0).abs() < 1e-6);
    }

    #[test]
    fn negative_and_nonfinite_delay_clamp() {
        let mut l = FracDelayLine::new(1, 16, Interp::Linear);
        push_mono(&mut l, &[1.0, 2.0, 3.0]);
        assert!((l.read(0, -5.0) - 3.0).abs() < 1e-6);
        assert!((l.read(0, f32::NAN) - 3.0).abs() < 1e-6);
        assert!((l.read(0, f32::INFINITY) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_state() {
        let mut l = FracDelayLine::new(1, 16, Interp::Linear);
        push_mono(&mut l, &[1.0, 2.0, 3.0]);
        l.reset();
        assert!(l.read(0, 0.0).abs() < 1e-9);
        assert_eq!(l.capacity(), 16);
        assert_eq!(l.channels(), 1);
    }

    #[test]
    fn parameters_are_clamped() {
        let l = FracDelayLine::new(0, 0, Interp::Lagrange(999));
        assert_eq!(l.channels(), 1);
        assert_eq!(l.capacity(), 1);
        assert_eq!(l.interp(), Interp::Lagrange(MAX_LAGRANGE_ORDER));

        let l2 = FracDelayLine::new(
            1,
            16,
            Interp::Sinc {
                half_taps: 9999,
                beta: -1.0,
            },
        );
        assert_eq!(
            l2.interp(),
            Interp::Sinc {
                half_taps: MAX_SINC_HALF_TAPS,
                beta: 0.0
            }
        );
    }

    #[test]
    fn max_delay_accounts_for_kernel_reach() {
        let lin = FracDelayLine::new(1, 100, Interp::Linear);
        let lag = FracDelayLine::new(1, 100, Interp::Lagrange(7));
        let snc = FracDelayLine::new(
            1,
            100,
            Interp::Sinc {
                half_taps: 16,
                beta: 8.0,
            },
        );
        // Larger kernel reach => smaller safe max delay.
        assert!(lin.max_delay() > lag.max_delay());
        assert!(lag.max_delay() > snc.max_delay());
        assert!(snc.max_delay() > 0.0);
    }

    #[test]
    fn sinc_dc_normalisation_holds_for_small_window() {
        let mut l = FracDelayLine::new(
            1,
            64,
            Interp::Sinc {
                half_taps: 2,
                beta: 4.0,
            },
        );
        push_mono(&mut l, &[1.0_f32; 40]);
        assert!((l.read(0, 2.5) - 1.0).abs() < 1e-4);
    }
}
