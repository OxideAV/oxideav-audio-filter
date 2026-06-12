//! Dither — word-length-reduction requantizer with TPDF dither and
//! error-feedback noise shaping.
//!
//! Rounds each sample onto the mid-tread grid of a signed `bits`-wide
//! integer word (`Δ = 2^(1-bits)`, so `bits = 16` lands every output on
//! an exact 16-bit PCM code value), optionally injecting dither noise
//! before the rounder and optionally feeding the rounding error back
//! through a first- or second-order shaping loop. This is the mastering
//! / transcoding primitive that belongs at the very end of a float
//! pipeline right before a fixed-point encode — truncating or plainly
//! rounding a float bus down to 16 bits produces *correlated* error
//! (harmonic distortion, deadband), while a dithered requantisation
//! produces benign uncorrelated noise.
//!
//! Distinct from [`Bitcrusher`](crate::Bitcrusher): the bitcrusher is a
//! *creative* degradation effect (bare quantisation + sample-and-hold
//! rate reduction, aliasing on purpose); `Dither` is a *transparency*
//! tool (statistically decorrelated error, optional psychoacoustic
//! noise tilt, no rate reduction).
//!
//! # Quantiser
//!
//! Mid-tread uniform quantiser over full-scale `±1`:
//!
//! ```text
//! Δ = 2^(1-bits)                      (bits = 16 → Δ = 1/32768)
//! k = round(v / Δ),  k clamped to [-2^(bits-1), 2^(bits-1) - 1]
//! y = k · Δ                           (exactly representable in f32 for bits ≤ 24)
//! ```
//!
//! The bare rounding error `y - v ∈ [-Δ/2, +Δ/2]` is a *deterministic
//! function of the input*: on a low-level periodic signal it repeats
//! with the signal's period and shows up as harmonic distortion, and a
//! sine of amplitude `< Δ/2` rounds to identical zeros — the signal
//! vanishes entirely (the quantiser deadband).
//!
//! # Dither (first-principles derivation)
//!
//! Non-subtractive dither adds an independent random `d` before the
//! rounder: `y = Q(v + d)`. The classical moment analysis (Schuchman
//! condition): the total error's `m`-th moment is independent of the
//! input iff the dither's characteristic function and its first `m`
//! derivatives vanish at every nonzero multiple of `2π/Δ`.
//!
//! * **RPDF** — uniform over `[-Δ/2, +Δ/2)` (characteristic function
//!   `sinc(uΔ/2)`, zeros at the right points but nonzero derivative):
//!   makes the *mean* error zero for every input (kills distortion)
//!   but the error *variance* still tracks the signal — audible "noise
//!   modulation".
//! * **TPDF** — triangular over `[-Δ, +Δ)`, generated as the sum of
//!   two independent RPDF draws (characteristic function `sinc²`, so
//!   both the function and its first derivative vanish): renders mean
//!   *and* variance of the total error signal-independent. Total error
//!   variance is constant at
//!   `Δ²/12 (quantisation) + 2·Δ²/12 (dither) = Δ²/4`.
//!
//! TPDF is therefore the default. The cost is 4.77 dB more noise than
//! bare rounding — flat, constant, and signal-independent.
//!
//! # Noise shaping (error feedback)
//!
//! The shaping loop subtracts filtered past errors before quantising:
//!
//! ```text
//! v[n] = x[n] - c₁·e[n-1] - c₂·e[n-2]
//! y[n] = Q(v[n] + d[n])
//! e[n] = y[n] - v[n]                 (total injected error, incl. dither)
//! ```
//!
//! which gives `y = x + (1 - C(z))·e` — the error reaches the output
//! through the noise transfer function `NTF(z) = 1 - C(z)` while the
//! signal passes untouched.
//!
//! * **First order** (`c = [1]`): `NTF = 1 - z⁻¹`,
//!   `|NTF(e^{jω})|² = 4·sin²(ω/2)` — a zero at DC and a +6 dB/oct
//!   tilt. Total noise power gain over white `e` is
//!   `(1/2π)∫4sin²(ω/2)dω = 2` (+3 dB), all of it pushed above `fs/6`
//!   (where `|NTF| = 1`).
//! * **Second order** (`c = [2, -1]`): `NTF = (1 - z⁻¹)²`,
//!   `|NTF|² = 16·sin⁴(ω/2)` — a double zero at DC. Power gain
//!   `16·⟨sin⁴⟩ = 16·(3/8) = 6` (+7.8 dB), with the crossover
//!   `|NTF| = 1` at `ω = π/3` again: much *less* noise in the low/mid
//!   band where hearing is most sensitive, much more in the top
//!   octave where it is not.
//!
//! Shaping should be combined with TPDF dither: feedback of an
//! undithered error re-correlates it with the signal and the shaped
//! spectrum grows idle tones.
//!
//! # Parameters
//!
//! * `bits` — target word length, clamped to `[2, 24]` (beyond 24 the
//!   grid is finer than an `f32` mantissa and the output could no
//!   longer represent the codes exactly).
//! * `mode` — [`DitherMode::None`] / [`DitherMode::Rpdf`] /
//!   [`DitherMode::Tpdf`] (default `Tpdf`).
//! * `shaping` — [`NoiseShaping::Off`] / [`NoiseShaping::FirstOrder`]
//!   / [`NoiseShaping::SecondOrder`] (default `Off`).
//! * `seed` — splitmix64 PRNG seed (same generator as
//!   [`WhiteNoise`](crate::WhiteNoise)); fixed seed → bit-reproducible
//!   output.
//!
//! Per-channel error-feedback state; the PRNG is shared across
//! channels (each draw advances it), so channels receive mutually
//! independent dither sequences.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Dither probability density applied before the rounder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DitherMode {
    /// Bare rounding — error correlates with the programme
    /// (distortion + deadband). Only for measurement / comparison.
    None,
    /// Rectangular PDF, peak-to-peak `Δ`: zero-mean error for every
    /// input, but signal-dependent error variance (noise modulation).
    Rpdf,
    /// Triangular PDF, peak-to-peak `2Δ` (sum of two RPDF draws):
    /// mean *and* variance of the error are signal-independent.
    /// The canonical mastering default.
    Tpdf,
}

/// Error-feedback noise-shaping order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseShaping {
    /// Flat error spectrum (no feedback).
    Off,
    /// `NTF = 1 - z⁻¹` — +6 dB/oct tilt, total power ×2.
    FirstOrder,
    /// `NTF = (1 - z⁻¹)²` — +12 dB/oct tilt, total power ×6.
    SecondOrder,
}

/// Per-channel error-feedback history.
#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    e1: f64,
    e2: f64,
}

/// Streaming dithered requantizer.
#[derive(Debug, Clone)]
pub struct Dither {
    bits: u8,
    step: f64,
    q_min: i64,
    q_max: i64,
    mode: DitherMode,
    shaping: NoiseShaping,
    rng: u64,
    state: Vec<ChState>,
}

/// Default PRNG seed (same convention as the noise generators).
const DEFAULT_SEED: u64 = 0x1234_5678;

impl Dither {
    /// New requantizer with TPDF dither, no shaping, default seed.
    /// `bits` clamped to `[2, 24]`.
    pub fn new(bits: u8) -> Self {
        Self::with(bits, DitherMode::Tpdf, NoiseShaping::Off)
    }

    /// New requantizer with explicit dither mode + shaping order.
    pub fn with(bits: u8, mode: DitherMode, shaping: NoiseShaping) -> Self {
        Self::with_seed(bits, mode, shaping, DEFAULT_SEED)
    }

    /// New requantizer with an explicit 64-bit PRNG seed.
    pub fn with_seed(bits: u8, mode: DitherMode, shaping: NoiseShaping, seed: u64) -> Self {
        let bits = bits.clamp(2, 24);
        Self {
            bits,
            step: (2.0f64).powi(1 - bits as i32),
            q_min: -(1i64 << (bits - 1)),
            q_max: (1i64 << (bits - 1)) - 1,
            mode,
            shaping,
            rng: seed.max(1),
            state: Vec::new(),
        }
    }

    /// Configured word length in bits.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Configured dither density.
    pub fn mode(&self) -> DitherMode {
        self.mode
    }

    /// Configured shaping order.
    pub fn shaping(&self) -> NoiseShaping {
        self.shaping
    }

    /// Quantiser step `Δ = 2^(1-bits)`.
    pub fn step(&self) -> f64 {
        self.step
    }

    /// Re-seed the PRNG and clear the error-feedback history.
    pub fn reset(&mut self, seed: u64) {
        self.rng = seed.max(1);
        self.state.clear();
    }

    /// splitmix64 step → uniform in `[0, 1)` — published mix constants
    /// (same generator as `WhiteNoise`).
    #[inline]
    fn next_unit(&mut self) -> f64 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    /// One dither draw in units of full scale.
    #[inline]
    fn draw(&mut self) -> f64 {
        match self.mode {
            DitherMode::None => 0.0,
            // Uniform over [-Δ/2, +Δ/2)
            DitherMode::Rpdf => (self.next_unit() - 0.5) * self.step,
            // Triangular over [-Δ, +Δ): sum of two independent RPDFs
            DitherMode::Tpdf => ((self.next_unit() - 0.5) + (self.next_unit() - 0.5)) * self.step,
        }
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.state.len() != channels {
            self.state = vec![ChState::default(); channels];
        }
    }
}

impl AudioFilter for Dither {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.channels as usize);
        let (c1, c2) = match self.shaping {
            NoiseShaping::Off => (0.0, 0.0),
            NoiseShaping::FirstOrder => (1.0, 0.0),
            NoiseShaping::SecondOrder => (2.0, -1.0),
        };
        let step = self.step;
        let mut decoded = decode_to_f32(input, params.format, params.channels)?;
        for (ch, buf) in decoded.iter_mut().enumerate() {
            let ChState { mut e1, mut e2 } = self.state[ch];
            for s in buf.iter_mut() {
                // v = x - C(z)·e  (error feedback; C = 0 when shaping off)
                let v = *s as f64 - c1 * e1 - c2 * e2;
                let d = self.draw();
                // Mid-tread rounder onto the bits-wide grid, clamped to
                // the signed-word code range.
                let k = (((v + d) / step).round() as i64).clamp(self.q_min, self.q_max);
                let y = k as f64 * step;
                // Total injected error (quantisation + dither + clamp)
                // is what the loop shapes.
                e2 = e1;
                e1 = y - v;
                *s = y as f32;
            }
            self.state[ch] = ChState { e1, e2 };
        }
        let out = encode_from_f32(params.format, params.channels, input, &decoded)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::SampleFormat;

    fn f32_params(channels: u16, rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels,
            sample_rate: rate,
        }
    }

    fn make_f32_planar(planes: &[Vec<f32>]) -> AudioFrame {
        let n = planes[0].len();
        let data = planes
            .iter()
            .map(|p| {
                let mut bytes = Vec::with_capacity(p.len() * 4);
                for s in p {
                    bytes.extend_from_slice(&s.to_le_bytes());
                }
                bytes
            })
            .collect();
        AudioFrame {
            samples: n as u32,
            pts: None,
            data,
        }
    }

    fn read_f32(frame: &AudioFrame, plane: usize) -> Vec<f32> {
        frame.data[plane]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Deterministic broadband test signal (xorshift32), |x| ≤ amp.
    fn test_noise(n: usize, amp: f32, mut seed: u32) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            v.push((seed as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp);
        }
        v
    }

    fn sine(n: usize, amp: f64, cycles_per_n: f64) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (amp * (2.0 * std::f64::consts::PI * cycles_per_n * i as f64 / n as f64).sin())
                    as f32
            })
            .collect()
    }

    #[test]
    fn output_lands_exactly_on_code_grid() {
        // Every output sample must be k·Δ for an integer k inside the
        // signed-word code range — even with dither + 2nd-order shaping.
        let bits = 12u8;
        let mut d = Dither::with(bits, DitherMode::Tpdf, NoiseShaping::SecondOrder);
        let input = test_noise(8192, 0.9, 1);
        let frame = make_f32_planar(&[input]);
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let step = d.step();
        let (q_min, q_max) = (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1);
        for (i, y) in read_f32(&out[0], 0).iter().enumerate() {
            let k = (*y as f64 / step).round();
            assert!(
                (*y as f64 - k * step).abs() < 1e-12,
                "sample {i}: {y} is not on the {bits}-bit grid"
            );
            assert!(
                (q_min as f64..=q_max as f64).contains(&k),
                "sample {i}: code {k} outside [{q_min}, {q_max}]"
            );
        }
    }

    #[test]
    fn bare_rounding_error_bounded_by_half_step() {
        // mode None, no shaping: |y - x| ≤ Δ/2 for in-range input.
        let mut d = Dither::with(10, DitherMode::None, NoiseShaping::Off);
        let input = test_noise(4096, 0.95, 2);
        let frame = make_f32_planar(std::slice::from_ref(&input));
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let half = (d.step() / 2.0) as f32;
        for (i, (x, y)) in input.iter().zip(read_f32(&out[0], 0)).enumerate() {
            assert!(
                (y - x).abs() <= half * 1.0001,
                "sample {i}: bare rounding error {} > Δ/2={half}",
                (y - x).abs()
            );
        }
    }

    #[test]
    fn tpdf_error_bounded_by_three_half_steps() {
        // TPDF dither spans ±Δ, rounding adds ±Δ/2 → |y - x| ≤ 3Δ/2.
        let mut d = Dither::with(10, DitherMode::Tpdf, NoiseShaping::Off);
        let input = test_noise(16384, 0.9, 3);
        let frame = make_f32_planar(std::slice::from_ref(&input));
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let bound = (1.5 * d.step()) as f32;
        for (i, (x, y)) in input.iter().zip(read_f32(&out[0], 0)).enumerate() {
            assert!(
                (y - x).abs() <= bound * 1.0001,
                "sample {i}: TPDF error {} > 3Δ/2={bound}",
                (y - x).abs()
            );
        }
    }

    #[test]
    fn tpdf_error_mean_near_zero() {
        // Non-subtractive TPDF renders the first moment of the error
        // signal-independent → the long-run mean error is ≈ 0.
        let mut d = Dither::with(8, DitherMode::Tpdf, NoiseShaping::Off);
        let input = test_noise(65536, 0.7, 4);
        let frame = make_f32_planar(std::slice::from_ref(&input));
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let got = read_f32(&out[0], 0);
        let mean: f64 = input
            .iter()
            .zip(&got)
            .map(|(x, y)| (*y - *x) as f64)
            .sum::<f64>()
            / input.len() as f64;
        // std of the mean = (Δ/2)/√N; Δ/20 is > 6σ of slack.
        assert!(
            mean.abs() < d.step() / 20.0,
            "mean error {mean} not ≈ 0 (Δ = {})",
            d.step()
        );
    }

    #[test]
    fn deadband_swallows_subliminal_tone_without_dither() {
        // A sine of amplitude 0.4Δ < Δ/2 rounds to all-zero codes:
        // the quantiser deadband erases the signal entirely.
        let bits = 12u8;
        let step = (2.0f64).powi(1 - bits as i32);
        let input = sine(4096, 0.4 * step, 64.0);
        let mut d = Dither::with(bits, DitherMode::None, NoiseShaping::Off);
        let frame = make_f32_planar(&[input]);
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        assert!(
            read_f32(&out[0], 0).iter().all(|y| *y == 0.0),
            "undithered sub-Δ/2 sine must round to exact silence"
        );
    }

    #[test]
    fn tpdf_dither_rescues_subliminal_tone() {
        // Same 0.4Δ sine, TPDF dithered: the fundamental survives the
        // requantisation as a clear spectral line above the dither
        // noise floor. Expected bin magnitude A·N/2 = 0.4Δ·2048 ≈ 819Δ;
        // per-bin noise ≈ √(N·Δ²/4) = 32Δ → ~25× headroom.
        use crate::fft::real_fft;
        let bits = 12u8;
        let step = (2.0f64).powi(1 - bits as i32);
        let n = 4096usize;
        let bin = 64usize;
        let input = sine(n, 0.4 * step, bin as f64);
        let mut d = Dither::with(bits, DitherMode::Tpdf, NoiseShaping::Off);
        let frame = make_f32_planar(&[input]);
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let bins = real_fft(&read_f32(&out[0], 0));
        let fundamental = bins[bin].magnitude() as f64;
        assert!(
            fundamental > 400.0 * step,
            "dithered fundamental bin {fundamental} too weak (expected ≈ {})",
            819.0 * step
        );
    }

    #[test]
    fn shaped_noise_power_matches_ntf_closed_form() {
        // Parseval check on the noise transfer function. With white
        // TPDF total error of variance σ² = Δ²/4, the output error
        // power is σ²·(1/2π)∫|NTF|²dω:
        //   Off          → ×1
        //   1 - z⁻¹      → ∫4sin²(ω/2)   → ×2
        //   (1 - z⁻¹)²   → ∫16sin⁴(ω/2)  → ×6
        let n = 65536usize;
        let input = test_noise(n, 0.6, 7);
        let sigma2 = {
            let d = Dither::new(8);
            d.step() * d.step() / 4.0
        };
        let mut measured = Vec::new();
        for shaping in [
            NoiseShaping::Off,
            NoiseShaping::FirstOrder,
            NoiseShaping::SecondOrder,
        ] {
            let mut d = Dither::with_seed(8, DitherMode::Tpdf, shaping, 99);
            let frame = make_f32_planar(std::slice::from_ref(&input));
            let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
            let got = read_f32(&out[0], 0);
            let p: f64 = input
                .iter()
                .zip(&got)
                .map(|(x, y)| ((*y - *x) as f64).powi(2))
                .sum::<f64>()
                / n as f64;
            measured.push(p / sigma2);
        }
        let expected = [1.0, 2.0, 6.0];
        for (i, (m, e)) in measured.iter().zip(expected).enumerate() {
            assert!(
                (0.75..1.35).contains(&(m / e)),
                "shaping #{i}: measured power ratio {m} vs closed-form {e}"
            );
        }
    }

    #[test]
    fn second_order_shaping_tilts_error_spectrum() {
        // |NTF|² = 16sin⁴(ω/2): ≪ 1 below fs/16, = 16 at Nyquist.
        // Compare the error spectrum (averaged 4×4096 FFTs) between
        // shaping Off and SecondOrder: low band must drop hard, top
        // band must rise.
        use crate::fft::real_fft;
        let n = 16384usize;
        let fft_n = 4096usize;
        let input = test_noise(n, 0.6, 11);
        let spectrum = |shaping: NoiseShaping| -> Vec<f64> {
            let mut d = Dither::with_seed(8, DitherMode::Tpdf, shaping, 1234);
            let frame = make_f32_planar(std::slice::from_ref(&input));
            let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
            let got = read_f32(&out[0], 0);
            let err: Vec<f32> = input.iter().zip(&got).map(|(x, y)| *y - *x).collect();
            let mut energy = vec![0.0f64; fft_n / 2 + 1];
            for chunk in err.chunks_exact(fft_n) {
                for (i, b) in real_fft(chunk).iter().enumerate() {
                    energy[i] += (b.magnitude() as f64).powi(2);
                }
            }
            energy
        };
        let flat = spectrum(NoiseShaping::Off);
        let shaped = spectrum(NoiseShaping::SecondOrder);
        // Low band: bins 1..256 (< fs/16, ω < π/8 → |NTF|² ≤ 16sin⁴(π/16) ≈ 0.023)
        let low_flat: f64 = flat[1..256].iter().sum();
        let low_shaped: f64 = shaped[1..256].iter().sum();
        assert!(
            low_shaped < 0.35 * low_flat,
            "low-band error not suppressed: shaped {low_shaped} vs flat {low_flat}"
        );
        // Top band: bins 1792..2048 (> 7fs/16 → |NTF|² ≥ 16sin⁴(7π/16) ≈ 13)
        let hi_flat: f64 = flat[1792..2048].iter().sum();
        let hi_shaped: f64 = shaped[1792..2048].iter().sum();
        assert!(
            hi_shaped > 2.0 * hi_flat,
            "top-band error not lifted: shaped {hi_shaped} vs flat {hi_flat}"
        );
    }

    #[test]
    fn full_scale_codes_clamp_correctly() {
        // x = -1 hits the most-negative code exactly (-2^(b-1)·Δ = -1);
        // x = +1 clamps to the top code (2^(b-1)-1)·Δ = 1 - Δ.
        let mut d = Dither::with(16, DitherMode::None, NoiseShaping::Off);
        let frame = make_f32_planar(&[vec![-1.0, 1.0, 0.0]]);
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let got = read_f32(&out[0], 0);
        assert_eq!(got[0], -1.0);
        assert_eq!(got[1], (1.0 - d.step()) as f32);
        assert_eq!(got[2], 0.0);
    }

    #[test]
    fn bits_clamped_to_supported_range() {
        assert_eq!(Dither::new(0).bits(), 2);
        assert_eq!(Dither::new(99).bits(), 24);
        assert_eq!(Dither::new(16).bits(), 16);
        // Δ for 16 bits is exactly 1/32768.
        assert_eq!(Dither::new(16).step(), 1.0 / 32768.0);
    }

    #[test]
    fn deterministic_with_seed() {
        let input = test_noise(4096, 0.8, 21);
        let run = |seed: u64| {
            let mut d = Dither::with_seed(12, DitherMode::Tpdf, NoiseShaping::SecondOrder, seed);
            let frame = make_f32_planar(std::slice::from_ref(&input));
            read_f32(&d.process(&frame, f32_params(1, 48_000)).unwrap()[0], 0)
        };
        assert_eq!(run(42), run(42), "same seed must reproduce bit-exactly");
        assert_ne!(run(42), run(43), "different seeds must differ");
    }

    #[test]
    fn channels_receive_independent_dither() {
        // Identical content on both channels: the shared-PRNG draws
        // still differ per channel, so the dithered outputs must not
        // be channel-identical — yet both stay on the grid.
        let content = test_noise(4096, 0.5, 31);
        let mut d = Dither::with(8, DitherMode::Tpdf, NoiseShaping::Off);
        let frame = make_f32_planar(&[content.clone(), content]);
        let params = AudioStreamParams {
            format: SampleFormat::F32P,
            channels: 2,
            sample_rate: 48_000,
        };
        let out = d.process(&frame, params).unwrap();
        let l = read_f32(&out[0], 0);
        let r = read_f32(&out[0], 1);
        let differing = l.iter().zip(&r).filter(|(a, b)| a != b).count();
        assert!(
            differing > l.len() / 4,
            "channels share dither: only {differing} samples differ"
        );
    }

    #[test]
    fn streaming_continuity_one_call_equals_two() {
        // Error-feedback state + PRNG stream must carry across frame
        // boundaries: processing 8192 samples in one call must equal
        // 4096 + 4096 in two calls, bit-exactly.
        let input = test_noise(8192, 0.7, 41);
        let p = f32_params(1, 48_000);
        let mut whole = Dither::with_seed(10, DitherMode::Tpdf, NoiseShaping::SecondOrder, 7);
        let one = read_f32(
            &whole
                .process(&make_f32_planar(std::slice::from_ref(&input)), p)
                .unwrap()[0],
            0,
        );
        let mut split = Dither::with_seed(10, DitherMode::Tpdf, NoiseShaping::SecondOrder, 7);
        let mut two = read_f32(
            &split
                .process(&make_f32_planar(&[input[..4096].to_vec()]), p)
                .unwrap()[0],
            0,
        );
        two.extend(read_f32(
            &split
                .process(&make_f32_planar(&[input[4096..].to_vec()]), p)
                .unwrap()[0],
            0,
        ));
        assert_eq!(one, two, "frame split must not change the output");
    }

    #[test]
    fn rpdf_error_bounded_by_step() {
        // RPDF dither spans ±Δ/2, rounding adds ±Δ/2 → |y - x| ≤ Δ.
        let mut d = Dither::with(10, DitherMode::Rpdf, NoiseShaping::Off);
        let input = test_noise(16384, 0.9, 51);
        let frame = make_f32_planar(std::slice::from_ref(&input));
        let out = d.process(&frame, f32_params(1, 48_000)).unwrap();
        let bound = d.step() as f32;
        for (i, (x, y)) in input.iter().zip(read_f32(&out[0], 0)).enumerate() {
            assert!(
                (y - x).abs() <= bound * 1.0001,
                "sample {i}: RPDF error {} > Δ={bound}",
                (y - x).abs()
            );
        }
    }
}
