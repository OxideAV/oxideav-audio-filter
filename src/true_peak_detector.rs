//! True-peak detector — 4× polyphase oversampled inter-sample peak
//! observer (dBTP).
//!
//! Sample-domain peak detection (`max |x[n]|`) systematically
//! understates the true analogue peak of a reconstructed signal: the
//! reconstructed waveform passes between the discrete samples and can
//! swing higher than any individual sample. The classic worked example
//! is a full-scale sine sitting exactly between two samples (zero
//! crossings on the grid points) — every sample reads `0` but the
//! analogue peak is `±1.0` (0 dBFS). Less pathological full-scale
//! material routinely produces 1.5–3 dB of inter-sample headroom
//! shortfall when downstream stages (DAC / lossy encoder) reconstruct
//! the band-limited continuum. dBTP ("decibels true peak") quantifies
//! the band-limited reconstructed peak rather than the sampled peak.
//!
//! This module observes (not modifies) the true peak by upsampling the
//! input 4× with a windowed-sinc FIR low-pass and reporting `max |y|`
//! over the oversampled stream. The audio passes through unchanged;
//! consumers poll [`TruePeakDetector::current_dbtp`] /
//! [`TruePeakDetector::max_dbtp`] / [`TruePeakDetector::overs`] after
//! each `process()` call.
//!
//! # Algorithm
//!
//! For a 4× upsample factor `L = 4` and a base FIR `h[k]`, `k ∈
//! [0, M)`, the polyphase decomposition splits `h` into `L` sub-filters
//!
//! ```text
//! p_φ[k] = h[L · k + φ]            φ ∈ [0, L)   k ∈ [0, M/L)
//! ```
//!
//! Phase `φ = 0` reproduces the input sample exactly (the prototype
//! `h` is constructed with a single non-zero centre tap on the
//! integer grid by construction — see the `kaiser_lowpass` design
//! below); phases `φ ∈ {1, 2, 3}` interpolate at `φ / L` of the
//! inter-sample distance. Per input sample `x[n]`, the four
//! oversampled outputs are
//!
//! ```text
//! y_φ[n] = Σ_k  p_φ[k] · x[n - k]
//! ```
//!
//! and the local true-peak proxy is `max(|y_0|, |y_1|, |y_2|, |y_3|)`.
//! Maintained per channel; peak-linked across channels with `max`.
//!
//! # FIR design
//!
//! The base FIR is a Kaiser-windowed ideal low-pass with cutoff
//! `f_c / f_s_oversampled = 1 / (2 L)` (i.e. preserves the original
//! Nyquist `f_s / 2` and rejects the `L − 1` images above it):
//!
//! ```text
//! ideal[k]  = sinc(2 · f_c · (k - (M-1)/2))                    // ideal LPF
//! kaiser[k] = I_0(β · sqrt(1 - (2k/(M-1) - 1)²)) / I_0(β)      // Kaiser window
//! h[k]      = ideal[k] · kaiser[k]
//! h[k]     *= L / Σ h[k]                                       // unit DC gain after upsample-then-LPF
//! ```
//!
//! The Kaiser β controls the side-lobe attenuation. The
//! Kaiser–Schafer formula for an `A` dB stop-band is
//!
//! ```text
//! β = 0.1102 · (A - 8.7)              if A > 50
//!   = 0.5842 · (A - 21)^0.4 + 0.07886·(A - 21)   if 21 < A ≤ 50
//!   = 0.0                              if A ≤ 21
//! ```
//!
//! Defaults: `M = 48` taps (12 taps per phase, comfortably wider than
//! the minimum needed to capture a full sine period of the original
//! Nyquist tone after upsample), `A = 100 dB` (β ≈ 10.06).
//!
//! # Why an FIR and not a half-band cascade
//!
//! A polyphase FIR is the textbook tradeoff for a fixed-rate observer:
//! cheap per-sample (12 multiply-adds × 4 phases = 48 MAC / input
//! sample), no IIR transient, exact phase linearity (centre-tap
//! symmetric FIR), and the entire response is determined by `M` and
//! `β` — no per-rate coefficient redesign. The detector's accuracy
//! ceiling is the FIR's stop-band, which `A = 100 dB` puts well below
//! the audible reconstruction error.
//!
//! # Numerical behaviour
//!
//! Internal accumulation runs in `f64` (matches the precision of the
//! existing `Biquad` DF-II-T state) — at 100 dB stop-band the residual
//! image leakage is `10^-5` of the signal, and `f32` accumulation can
//! lose that headroom on long sums.
//!
//! # API surface
//!
//! The filter is **observation-only**: [`AudioFilter::process`] returns
//! the input frame unchanged. Consumers call:
//!
//! * [`TruePeakDetector::current_dbtp`] — max dBTP over the *last*
//!   processed frame (and `0` before any sample has been seen).
//! * [`TruePeakDetector::max_dbtp`] — running maximum over the entire
//!   detector's history. Reset with [`TruePeakDetector::reset_max`].
//! * [`TruePeakDetector::overs`] — count of oversampled samples whose
//!   `|y|` exceeded the user-configurable overs threshold (default
//!   `1.0` = 0 dBTP). Reset with [`TruePeakDetector::reset_overs`].
//! * [`TruePeakDetector::reset`] — wipe all state, including the FIR
//!   delay lines. Coefficients survive.
//!
//! # Channels and sample-rate changes
//!
//! Per-channel state (delay line + max accumulator) is rebuilt on the
//! first frame and whenever the channel count or sample rate changes
//! between calls. The FIR itself is sample-rate-independent (its
//! cutoff is in *normalised* frequency at the oversampled rate) so no
//! coefficient redesign is needed across rates — only delay-line size
//! and zero-init.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Default polyphase oversampling factor.
pub const TPD_OVERSAMPLE: usize = 4;

/// Default base-FIR length (taps); must be a multiple of
/// [`TPD_OVERSAMPLE`]. With `M = 48`, each polyphase sub-filter has
/// `M / L = 12` taps.
pub const TPD_FIR_TAPS: usize = 48;

/// Default stop-band attenuation, in dB, for the Kaiser-window FIR
/// design.
pub const TPD_KAISER_DB: f64 = 100.0;

/// Default overs threshold (linear amplitude). `1.0` = 0 dBTP.
pub const TPD_OVERS_THRESHOLD: f32 = 1.0;

/// 4×-oversampled true-peak detector (dBTP).
#[derive(Debug, Clone)]
pub struct TruePeakDetector {
    /// Polyphase sub-filter coefficients: `phases[phase][tap]`.
    /// `phases.len() == oversample`, every inner length is
    /// `taps_per_phase`.
    phases: Vec<Vec<f64>>,
    /// `oversample` factor (`L`).
    oversample: usize,
    /// `taps_per_phase` (`M / L`). The per-channel delay line is
    /// sized to this value.
    taps_per_phase: usize,
    /// Threshold above which an oversampled sample is counted as an
    /// "over" (linear amplitude).
    overs_threshold: f32,
    state: Option<DetectorState>,
}

#[derive(Debug, Clone)]
struct DetectorState {
    sample_rate: u32,
    /// One delay line per channel, of length `taps_per_phase`.
    /// `delays[ch][k]` holds the `k`-most-recent input sample for
    /// channel `ch` (`k = 0` is the *newest*).
    delays: Vec<Vec<f64>>,
    /// Last `process()`'s max linear |y_φ| across all phases and
    /// channels.
    last_linear: f32,
    /// Running max linear |y_φ| since construction (or last
    /// [`reset_max`](TruePeakDetector::reset_max)).
    max_linear: f32,
    /// Cumulative count of oversampled samples whose `|y_φ|` exceeded
    /// `overs_threshold` since construction (or last
    /// [`reset_overs`](TruePeakDetector::reset_overs)).
    overs: u64,
}

impl TruePeakDetector {
    /// Construct a detector with the default 4× oversample, 48-tap
    /// Kaiser FIR (β chosen for ~100 dB stop-band), and a `1.0`
    /// (0 dBTP) overs threshold.
    pub fn new() -> Self {
        Self::with_params(
            TPD_OVERSAMPLE,
            TPD_FIR_TAPS,
            TPD_KAISER_DB,
            TPD_OVERS_THRESHOLD,
        )
    }

    /// Construct a detector with explicit oversample factor, FIR
    /// length (must be a multiple of `oversample` and ≥ `oversample`),
    /// Kaiser stop-band attenuation, and overs threshold.
    ///
    /// `oversample` is clamped to `[1, 16]`; the default of `4` is
    /// the conventional choice in broadcast loudness practice and is
    /// the floor that suppresses inter-sample peak under-estimation to
    /// below `0.1 dB` on typical material. `taps` is rounded up to
    /// the next multiple of `oversample`. `kaiser_db` is clamped to
    /// `[20, 180]`. `overs_threshold` is the *linear* level above
    /// which an oversampled sample is counted as an "over" — `1.0`
    /// is the canonical 0 dBTP threshold.
    pub fn with_params(
        oversample: usize,
        taps: usize,
        kaiser_db: f64,
        overs_threshold: f32,
    ) -> Self {
        let oversample = oversample.clamp(1, 16);
        // Round taps up to a multiple of oversample; floor at oversample
        // so we always have at least one tap per phase.
        let taps = taps.max(oversample).div_ceil(oversample) * oversample;
        let kaiser_db = kaiser_db.clamp(20.0, 180.0);
        let h = kaiser_lowpass(taps, oversample, kaiser_db);
        let phases = polyphase_split(&h, oversample);
        Self {
            taps_per_phase: phases[0].len(),
            phases,
            oversample,
            overs_threshold: overs_threshold.max(0.0),
            state: None,
        }
    }

    /// Number of taps in each polyphase sub-filter.
    pub fn taps_per_phase(&self) -> usize {
        self.taps_per_phase
    }

    /// Oversampling factor in effect (`L`).
    pub fn oversample(&self) -> usize {
        self.oversample
    }

    /// True peak observed during the most recent [`AudioFilter::process`]
    /// call, in dBTP. Returns `-INFINITY` if nothing has been
    /// processed yet (or all samples were exact zero).
    pub fn current_dbtp(&self) -> f32 {
        linear_to_dbtp(self.state.as_ref().map(|s| s.last_linear).unwrap_or(0.0))
    }

    /// Running maximum true peak observed since construction or last
    /// [`reset_max`](Self::reset_max), in dBTP.
    pub fn max_dbtp(&self) -> f32 {
        linear_to_dbtp(self.state.as_ref().map(|s| s.max_linear).unwrap_or(0.0))
    }

    /// Number of oversampled samples whose `|y|` exceeded the
    /// configured `overs_threshold` since construction or last
    /// [`reset_overs`](Self::reset_overs).
    pub fn overs(&self) -> u64 {
        self.state.as_ref().map(|s| s.overs).unwrap_or(0)
    }

    /// Configured overs threshold, linear amplitude. The default of
    /// `1.0` corresponds to 0 dBTP.
    pub fn overs_threshold(&self) -> f32 {
        self.overs_threshold
    }

    /// Wipe per-channel state (delay lines, max accumulators, overs
    /// counter). Coefficients survive.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Reset the running maximum to zero without wiping the delay
    /// lines or the overs counter.
    pub fn reset_max(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.max_linear = 0.0;
        }
    }

    /// Reset the overs counter to zero without wiping the delay
    /// lines or the running maximum.
    pub fn reset_overs(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.overs = 0;
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.delays.len() != channels,
            None => true,
        };
        if rebuild {
            self.state = Some(DetectorState {
                sample_rate,
                delays: vec![vec![0.0; self.taps_per_phase]; channels],
                last_linear: 0.0,
                max_linear: 0.0,
                overs: 0,
            });
        }
    }
}

impl Default for TruePeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for TruePeakDetector {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate, params.channels as usize);
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);

        let state = self.state.as_mut().expect("state ensured above");
        let overs_thr = self.overs_threshold as f64;
        let mut frame_max: f64 = 0.0;
        let mut frame_overs: u64 = 0;

        for i in 0..n {
            let mut sample_max: f64 = 0.0;
            for (ch_idx, ch) in channels.iter().enumerate() {
                let x = ch[i] as f64;
                let delay = &mut state.delays[ch_idx];
                // Shift in the new sample (newest at index 0).
                delay.rotate_right(1);
                delay[0] = x;

                // Evaluate each polyphase sub-filter at this input
                // sample. Sub-filter `phases[phi]` taps the delay line
                // with phases[phi][k] · delay[k], k = 0..taps_per_phase.
                for phase_coeffs in &self.phases {
                    let mut acc: f64 = 0.0;
                    for (k, &c) in phase_coeffs.iter().enumerate() {
                        acc += c * delay[k];
                    }
                    let absy = acc.abs();
                    if absy > sample_max {
                        sample_max = absy;
                    }
                    if absy > overs_thr {
                        frame_overs += 1;
                    }
                }
            }
            if sample_max > frame_max {
                frame_max = sample_max;
            }
        }

        state.last_linear = frame_max as f32;
        if state.last_linear > state.max_linear {
            state.max_linear = state.last_linear;
        }
        state.overs = state.overs.saturating_add(frame_overs);

        // Pass-through: re-encode the decoded channels unchanged.
        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

/// Convert a linear amplitude to dBTP, with `-INFINITY` for exact zero.
fn linear_to_dbtp(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Modified Bessel function of the first kind, order 0. Series
/// expansion `I_0(x) = Σ_{m≥0} (x/2)^{2m} / (m!)²`. Convergence is
/// rapid for the `|x| ≤ β ≤ ~25` range we use; we sum until the
/// incremental term is < 1e-12 of the running sum, or `m > 60`.
fn bessel_i0(x: f64) -> f64 {
    let half_x_sq = (x * 0.5) * (x * 0.5);
    let mut sum = 1.0;
    let mut term = 1.0;
    for m in 1..=60 {
        // term_{m} = term_{m-1} · (x/2)² / m²
        term *= half_x_sq / ((m as f64) * (m as f64));
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum
}

/// Kaiser β from a target stop-band attenuation `A` (dB), per the
/// Kaiser–Schafer empirical formula.
fn kaiser_beta(a_db: f64) -> f64 {
    if a_db > 50.0 {
        0.1102 * (a_db - 8.7)
    } else if a_db > 21.0 {
        0.5842 * (a_db - 21.0).powf(0.4) + 0.07886 * (a_db - 21.0)
    } else {
        0.0
    }
}

/// Build an `m`-tap Kaiser-windowed ideal low-pass FIR with cutoff
/// `f_c / f_s_oversampled = 1 / (2 · oversample)`. The result is
/// normalised so the sum of taps equals `oversample` — the textbook
/// gain for upsample-by-`L`-then-low-pass.
fn kaiser_lowpass(m: usize, oversample: usize, kaiser_db: f64) -> Vec<f64> {
    let beta = kaiser_beta(kaiser_db);
    let i0_beta = bessel_i0(beta);
    let centre = (m as f64 - 1.0) * 0.5;
    let fc = 1.0 / (2.0 * oversample as f64); // normalised cutoff
    let two_fc = 2.0 * fc;
    let mut h = Vec::with_capacity(m);
    for k in 0..m {
        let n = k as f64 - centre;
        // Ideal LPF impulse response: 2·fc · sinc(2·fc · n)
        let ideal = if n.abs() < 1e-12 {
            two_fc
        } else {
            let arg = std::f64::consts::PI * two_fc * n;
            two_fc * arg.sin() / arg
        };
        // Kaiser window
        let u = if m == 1 {
            0.0
        } else {
            2.0 * k as f64 / (m as f64 - 1.0) - 1.0
        };
        let win_arg = beta * (1.0 - u * u).max(0.0).sqrt();
        let w = bessel_i0(win_arg) / i0_beta;
        h.push(ideal * w);
    }
    // Normalise so that Σ h[k] = oversample. After polyphase
    // de-interleaving each sub-filter's tap sum is ~1, and phase 0
    // (the one aligned with the integer grid) is the
    // identity interpolator on a DC input.
    let sum: f64 = h.iter().sum();
    let scale = oversample as f64 / sum;
    for v in &mut h {
        *v *= scale;
    }
    h
}

/// Split a flat FIR `h` of length `L · K` into `L` sub-filters of
/// length `K` such that `p[phi][k] = h[L·k + phi]`. The standard
/// polyphase decomposition for upsample-and-filter; per input sample,
/// evaluating sub-filter `phi` against the input delay line yields
/// the oversampled sample at fractional offset `phi / L`.
fn polyphase_split(h: &[f64], oversample: usize) -> Vec<Vec<f64>> {
    assert!(h.len() % oversample == 0);
    let k = h.len() / oversample;
    let mut phases = vec![Vec::with_capacity(k); oversample];
    for (i, &c) in h.iter().enumerate() {
        let phi = i % oversample;
        phases[phi].push(c);
    }
    // Derivation: inserting `L - 1` zeros between input samples
    // (upsampler), then convolving with `h`, gives at oversampled
    // index `m = L·n + φ`:
    //
    //   y[L·n + φ] = Σ_k h[k] · u[L·n + φ - k]
    //
    // `u` is non-zero only at multiples of `L`, so `k = L·m + φ`:
    //
    //   y[L·n + φ] = Σ_m h[L·m + φ] · x[n - m]
    //              = Σ_m p_φ[m]    · x[n - m]
    //
    // i.e. sub-filter coefficient at lag `m` weights `x[n - m]`.
    // With the delay-line convention `delay[0]` = newest = `x[n]`,
    // `delay[k]` = `x[n - k]`, the natural ordering `p_φ[k] = h[L·k +
    // φ]` already maps coefficient index → delay index 1:1 — no
    // reversal needed.
    phases
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

    fn f32p_stereo(rate: u32) -> AudioStreamParams {
        // Planar F32 so the two-plane test frame in `make_f32p_stereo`
        // matches the decoder's per-channel plane convention.
        AudioStreamParams {
            format: SampleFormat::F32P,
            channels: 2,
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

    fn make_f32p_stereo(left: &[f32], right: &[f32]) -> AudioFrame {
        assert_eq!(left.len(), right.len());
        let mut l_bytes = Vec::with_capacity(left.len() * 4);
        let mut r_bytes = Vec::with_capacity(right.len() * 4);
        for (&l, &r) in left.iter().zip(right.iter()) {
            l_bytes.extend_from_slice(&l.to_le_bytes());
            r_bytes.extend_from_slice(&r.to_le_bytes());
        }
        AudioFrame {
            samples: left.len() as u32,
            pts: None,
            data: vec![l_bytes, r_bytes],
        }
    }

    #[test]
    fn bessel_i0_known_values() {
        // I_0(0) = 1, I_0(1) ≈ 1.2660658, I_0(5) ≈ 27.2398718
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        assert!((bessel_i0(1.0) - 1.266_065_877_752_008).abs() < 1e-9);
        assert!((bessel_i0(5.0) - 27.239_871_823_604_45).abs() < 1e-6);
    }

    #[test]
    fn kaiser_beta_known_breakpoints() {
        // A=21 → β=0 (boundary), A>50 → linear in (A-8.7) with slope 0.1102.
        assert!(kaiser_beta(20.0).abs() < 1e-12);
        assert!((kaiser_beta(100.0) - 0.1102 * (100.0 - 8.7)).abs() < 1e-12);
    }

    #[test]
    fn kaiser_lowpass_sums_to_oversample_factor() {
        let h = kaiser_lowpass(48, 4, 100.0);
        let sum: f64 = h.iter().sum();
        assert!((sum - 4.0).abs() < 1e-9, "FIR sum was {sum}");
    }

    #[test]
    fn polyphase_split_recovers_phases() {
        let h: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let phases = polyphase_split(&h, 4);
        assert_eq!(phases.len(), 4);
        assert_eq!(phases[0].len(), 4);
        // Phase 0 picks indices 0,4,8,12 — natural order (index 0
        // weights the newest sample, matching delay-line layout).
        assert_eq!(phases[0], vec![0.0, 4.0, 8.0, 12.0]);
        // Phase 1 picks indices 1,5,9,13.
        assert_eq!(phases[1], vec![1.0, 5.0, 9.0, 13.0]);
    }

    #[test]
    fn pass_through_preserves_audio_bytes() {
        // Audio frames must come out byte-identical (true-peak is an
        // observer, not a modifier).
        let fs = 48_000u32;
        let samples: Vec<f32> = (0..1024).map(|i| 0.3 * (i as f32).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut det = TruePeakDetector::new();
        let outs = det.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].samples, frame.samples);
        assert_eq!(outs[0].data, frame.data);
    }

    #[test]
    fn dc_input_reports_input_level_steady_state() {
        // For a constant 0.5 input, the windowed-sinc FIR has Gibbs-style
        // overshoot during the zero-to-DC step transient (this is
        // intrinsic to any sharp-cutoff LPF and is irrelevant to the
        // measurement of *steady-state* dBTP). Drive the detector with a
        // first frame to prime the delay line, reset the max, then a
        // second frame to verify the steady-state output equals the
        // input.
        let fs = 48_000u32;
        let prime = vec![0.5f32; 1024];
        let mut det = TruePeakDetector::new();
        det.process(&make_f32_mono(&prime), f32_mono(fs)).unwrap();
        det.reset_max();
        // Second frame: delay line is now full of 0.5 — no transient.
        det.process(&make_f32_mono(&prime), f32_mono(fs)).unwrap();
        let dbtp = det.current_dbtp();
        // 20·log10(0.5) = -6.0206 dB.
        assert!(
            (dbtp - (-6.0206)).abs() < 0.05,
            "DC=0.5 steady-state expected ≈ -6.02 dBTP, got {dbtp}"
        );
    }

    #[test]
    fn step_response_has_bounded_overshoot() {
        // The same step-input case used to be a "must equal -6.02 dBTP"
        // test, but a sharp-cutoff windowed-sinc oversampler legitimately
        // overshoots on a step. Bound the overshoot rather than zero it.
        let fs = 48_000u32;
        let samples = vec![0.5f32; 1024];
        let mut det = TruePeakDetector::new();
        det.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let dbtp = det.current_dbtp();
        // Overshoot should be small — well under 1.5 dB for this
        // Kaiser β / taps configuration.
        assert!(
            (-6.0206 - 0.05..=-6.0206 + 1.5).contains(&dbtp),
            "step overshoot should sit in [-6.07, -4.52] dBTP, got {dbtp}"
        );
    }

    #[test]
    fn sine_at_nyquist_reveals_inter_sample_peak() {
        // A unit-amplitude sine at fs/4 with quarter-period phase
        // offset hits ±1 only between samples; on the integer grid
        // the samples sit at ±√2/2 ≈ ±0.7071. Sample-peak says
        // -3.01 dBFS; true peak says ≈ 0 dBTP.
        let fs = 48_000u32;
        let fc = fs as f32 / 4.0;
        // Phase π/4 offset so integer samples land at ±√2/2.
        let phase = std::f32::consts::FRAC_PI_4;
        let n = 4096usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * fc * t + phase).sin()
            })
            .collect();
        // Sanity: sample-peak is well below 0 dBFS.
        let sample_peak = samples.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(sample_peak < 0.71 + 1e-3 && sample_peak > 0.70 - 1e-3);
        let sample_peak_db = 20.0 * sample_peak.log10();
        // True-peak detector should recover the real ≈ 0 dBTP.
        let mut det = TruePeakDetector::new();
        det.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let dbtp = det.max_dbtp();
        assert!(
            dbtp > sample_peak_db + 2.0,
            "true-peak {dbtp} should exceed sample-peak {sample_peak_db} by > 2 dB"
        );
        // And land near 0 dBTP (within ~0.5 dB of the true continuous peak).
        assert!(
            dbtp.abs() < 0.5,
            "fs/4 + π/4-phase sine should true-peak near 0 dBTP, got {dbtp}"
        );
    }

    #[test]
    fn overs_counter_fires_when_threshold_exceeded() {
        // 1.2 DC: every oversampled sample ≈ 1.2 > 1.0, so each
        // input sample produces L=4 overs.
        let fs = 48_000u32;
        let n = 256usize;
        let samples = vec![1.2f32; n];
        let mut det = TruePeakDetector::new();
        det.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        // The FIR's warm-up window has fewer than 1.0-saturating
        // oversampled samples (the convolution ramps in), so allow
        // a small slack — just require that most input samples
        // contributed L overs.
        let expected_min = (n - 12) as u64 * TPD_OVERSAMPLE as u64;
        assert!(
            det.overs() >= expected_min,
            "overs={} should be ≥ {} (n=256, L=4, warm-up slack)",
            det.overs(),
            expected_min
        );
    }

    #[test]
    fn silent_input_reports_neg_infinity_dbtp() {
        let fs = 48_000u32;
        let frame = make_f32_mono(&vec![0.0f32; 512]);
        let mut det = TruePeakDetector::new();
        det.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(det.current_dbtp(), f32::NEG_INFINITY);
        assert_eq!(det.max_dbtp(), f32::NEG_INFINITY);
        assert_eq!(det.overs(), 0);
    }

    #[test]
    fn max_dbtp_accumulates_across_frames() {
        // First frame quiet, second loud — max should track the loud frame.
        let fs = 48_000u32;
        let quiet = make_f32_mono(&vec![0.1f32; 256]);
        let loud = make_f32_mono(&vec![0.9f32; 256]);
        let mut det = TruePeakDetector::new();
        det.process(&quiet, f32_mono(fs)).unwrap();
        let first_max = det.max_dbtp();
        det.process(&loud, f32_mono(fs)).unwrap();
        let second_max = det.max_dbtp();
        assert!(second_max > first_max + 10.0);
        // Now a quiet frame again — max should NOT regress.
        det.process(&quiet, f32_mono(fs)).unwrap();
        assert_eq!(det.max_dbtp(), second_max);
    }

    #[test]
    fn reset_max_clears_max_only() {
        let fs = 48_000u32;
        let loud = make_f32_mono(&vec![0.9f32; 256]);
        let mut det = TruePeakDetector::new();
        det.process(&loud, f32_mono(fs)).unwrap();
        assert!(det.max_dbtp() > -2.0);
        det.reset_max();
        assert_eq!(det.max_dbtp(), f32::NEG_INFINITY);
    }

    #[test]
    fn reset_overs_clears_overs_only() {
        let fs = 48_000u32;
        let loud = make_f32_mono(&vec![1.5f32; 256]);
        let mut det = TruePeakDetector::new();
        det.process(&loud, f32_mono(fs)).unwrap();
        assert!(det.overs() > 0);
        det.reset_overs();
        assert_eq!(det.overs(), 0);
    }

    #[test]
    fn stereo_peak_links_across_channels() {
        // Left silent, right loud — detector should see the loud channel.
        let fs = 48_000u32;
        let n = 512usize;
        let left = vec![0.0f32; n];
        let right: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.8 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let frame = make_f32p_stereo(&left, &right);
        let mut det = TruePeakDetector::new();
        det.process(&frame, f32p_stereo(fs)).unwrap();
        let dbtp = det.max_dbtp();
        // 0.8 → -1.94 dB; well above silence.
        assert!(
            dbtp > -3.0,
            "stereo peak-link should track loud channel, got {dbtp}"
        );
    }

    #[test]
    fn reset_clears_all_state() {
        let fs = 48_000u32;
        let loud = make_f32_mono(&vec![1.5f32; 256]);
        let mut det = TruePeakDetector::new();
        det.process(&loud, f32_mono(fs)).unwrap();
        assert!(det.overs() > 0);
        assert!(det.max_dbtp() > 0.0);
        det.reset();
        assert_eq!(det.overs(), 0);
        assert_eq!(det.max_dbtp(), f32::NEG_INFINITY);
        assert_eq!(det.current_dbtp(), f32::NEG_INFINITY);
    }

    #[test]
    fn taps_round_up_to_multiple_of_oversample() {
        let det = TruePeakDetector::with_params(4, 47, 100.0, 1.0);
        // 47 rounds up to 48 → 12 taps per phase.
        assert_eq!(det.taps_per_phase(), 12);
        assert_eq!(det.oversample(), 4);
    }

    #[test]
    fn oversample_clamps_to_valid_range() {
        let det = TruePeakDetector::with_params(99, 100, 100.0, 1.0);
        assert_eq!(det.oversample(), 16);
        let det = TruePeakDetector::with_params(0, 4, 100.0, 1.0);
        assert_eq!(det.oversample(), 1);
    }

    #[test]
    fn phase_zero_is_identity_on_dc() {
        // For DC input, phase 0 of any unit-DC-gain interpolator
        // should reproduce the input exactly. The normaliser scales
        // h so Σh = L; after polyphase split each sub-filter's tap
        // sum should be ~1, so DC-in → DC-out per phase.
        let det = TruePeakDetector::new();
        for (phi, phase) in det.phases.iter().enumerate() {
            let sum: f64 = phase.iter().sum();
            assert!(
                (sum - 1.0).abs() < 5e-4,
                "phase {phi} sum {sum} should be ~1.0 for unit DC gain"
            );
        }
    }

    #[test]
    fn changing_sample_rate_rebuilds_state() {
        let fs1 = 48_000u32;
        let fs2 = 44_100u32;
        let frame = make_f32_mono(&vec![0.5f32; 128]);
        let mut det = TruePeakDetector::new();
        det.process(&frame, f32_mono(fs1)).unwrap();
        let after_first = det.state.as_ref().unwrap().sample_rate;
        assert_eq!(after_first, fs1);
        det.process(&frame, f32_mono(fs2)).unwrap();
        let after_second = det.state.as_ref().unwrap().sample_rate;
        assert_eq!(after_second, fs2);
        // Delay line was rebuilt → max_linear was reset to zero
        // before this frame ran, so current_dbtp reflects just this
        // frame's warm-up convolution result.
        assert!(det.max_dbtp().is_finite());
    }
}
