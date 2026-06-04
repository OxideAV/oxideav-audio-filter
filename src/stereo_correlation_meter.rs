//! Stereo correlation meter — pass-through observer reporting the
//! Pearson correlation coefficient between the left and right channels
//! over a sliding rectangular window.
//!
//! The Pearson correlation coefficient
//!
//! ```text
//!         Σ (x_i − x̄)(y_i − ȳ)
//!   ρ = ─────────────────────────
//!       √(Σ (x_i − x̄)² · Σ (y_i − ȳ)²)
//! ```
//!
//! is a unit-free scalar in `[-1, +1]` that classifies the stereo
//! image at a glance:
//!
//! * `ρ = +1` — both channels carry the same signal (mono content
//!   panned dead-centre).
//! * `ρ ≈ 0` — channels are uncorrelated (wide stereo bed, ambient
//!   reverb tails, decorrelated chorus voices).
//! * `ρ = -1` — channels are exact phase-inversions; a mono fold-down
//!   collapses to silence. This is the canonical broadcast hazard the
//!   meter is designed to flag — most TV / radio chains downstream
//!   from a mastering bus still emit a mono sum, and a programme that
//!   correlates strongly negative dies on the way out.
//!
//! Within this crate's observer family the stereo correlation meter
//! sits orthogonal to:
//!
//! * [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter) —
//!   peak-to-RMS scalar; says nothing about inter-channel phase.
//! * [`TruePeakDetector`](crate::true_peak_detector::TruePeakDetector) —
//!   absolute inter-sample peak (dBTP); single-channel summary, not a
//!   stereo relationship.
//! * [`LoudnessITU`](crate::loudness::LoudnessITU) — K-weighted
//!   integrated LUFS; uses the channel sum, drops the phase term.
//! * [`StereoWidener`](crate::stereo_widener::StereoWidener) and
//!   [`StereoImager`](crate::stereo_imager::StereoImager) — stereo
//!   width *processors* (mutate the signal); the correlation meter is
//!   observation-only.
//!
//! # Algorithm
//!
//! Five per-window incremental running sums are kept:
//!
//! ```text
//!   Sx  = Σ x       Sy  = Σ y
//!   Sxx = Σ x²      Syy = Σ y²
//!   Sxy = Σ x·y
//! ```
//!
//! On each new sample pair `(x_new, y_new)` entering the window and
//! `(x_old, y_old)` leaving it, every sum updates in `O(1)`:
//!
//! ```text
//!   Sx  ← Sx  + x_new − x_old
//!   Sy  ← Sy  + y_new − y_old
//!   Sxx ← Sxx + x_new² − x_old²
//!   Syy ← Syy + y_new² − y_old²
//!   Sxy ← Sxy + x_new·y_new − x_old·y_old
//! ```
//!
//! With `N = window_samples` and the convention that all sums are
//! computed in `f64`, the windowed Pearson correlation falls out by
//! algebraic identity:
//!
//! ```text
//!         N·Sxy − Sx·Sy
//!   ρ = ─────────────────────────────────────
//!       √((N·Sxx − Sx²) · (N·Syy − Sy²))
//! ```
//!
//! (The numerator is `N · Σ (x − x̄)(y − ȳ)` once the bias terms are
//! expanded out; the two factors in the denominator are likewise
//! `N · Σ (x − x̄)²` and `N · Σ (y − ȳ)²`. The expansion is purely
//! arithmetic — no statistical assumption is involved.)
//!
//! To keep `f64` round-off bounded on long streams, the filter rebuilds
//! all five sums from the active ring contents once per full window
//! (the same drift-bounding device used by
//! [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)).
//! The rebuild is `O(N)` once every `N` samples, i.e. `O(1)` amortised
//! per sample, and resets any cumulative subtraction error to zero.
//!
//! # Channel handling
//!
//! The meter requires exactly two channels (L = channel 0, R = channel
//! 1); the `process()` call is a pass-through identity but the
//! correlation update only runs when `channels == 2`. For mono or
//! multichannel input (channel count not equal to two) the audio
//! passes through unchanged, the pair-correlation update is skipped,
//! and `current()` keeps its previous value (or stays at `0.0` if no
//! stereo input has been seen). Mono is a degenerate case for the
//! metric anyway: the single-channel "self-correlation" is
//! identically `+1`.
//!
//! # Warm-up
//!
//! Until the window is first full, [`current`](StereoCorrelationMeter::current)
//! returns `0.0` and [`current_degrees`](StereoCorrelationMeter::current_degrees)
//! returns `90.0` (the "neutral" reading). The
//! [`samples_seen`](StereoCorrelationMeter::samples_seen) accessor exposes the
//! warm-up count explicitly so consumers can branch on "not yet ready"
//! before relying on the readout.
//!
//! # Polar mapping
//!
//! Mastering engineers traditionally read correlation as an angular
//! position on a goniometer display:
//!
//! ```text
//!   θ_deg = acos(ρ) · 180/π
//! ```
//!
//! so a perfectly-correlated mono signal sits at `0°`, an
//! uncorrelated stereo bed at `90°`, and a phase-inverted pair at
//! `180°`. [`current_degrees`](StereoCorrelationMeter::current_degrees) returns
//! this representation for direct UI display; the raw signed
//! correlation is available via [`current`](StereoCorrelationMeter::current).
//!
//! # Running minimum
//!
//! The phase-cancellation hazard is asymmetric — a programme that
//! transiently swings negative on an isolated frame is the failure
//! mode worth surfacing. The meter therefore tracks the running
//! *minimum* correlation since construction or last
//! [`reset_min`](StereoCorrelationMeter::reset_min), so a worst-case
//! readout survives a quieter trailing tail.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Default measurement window in milliseconds. Matches the EBU R128
/// short-term loudness window — the same window length the
/// [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
/// defaults to, so the two meters can be displayed side-by-side
/// without time-axis confusion.
pub const SCM_DEFAULT_WINDOW_MS: f32 = 400.0;

/// Maximum measurement window in samples (`= 192_000`, i.e. 4 s at
/// 48 kHz or 2 s at 96 kHz). Defends against pathological allocations
/// without rejecting any realistic broadcast / mastering window.
pub const SCM_MAX_WINDOW_SAMPLES: usize = 192_000;

/// Streaming stereo correlation meter.
///
/// Pass-through audio observer — bytes out are byte-for-byte equal to
/// bytes in. Consumers read the windowed Pearson correlation via the
/// accessor methods below.
#[derive(Debug, Clone)]
pub struct StereoCorrelationMeter {
    window_ms: f32,
    state: Option<MeterState>,
}

#[derive(Debug, Clone)]
struct MeterState {
    sample_rate: u32,
    channels: usize,
    /// Window length in samples (`N`), derived from `window_ms` and
    /// `sample_rate`, clamped to `[1, SCM_MAX_WINDOW_SAMPLES]`.
    window_samples: usize,
    /// Left-channel ring buffer of the most recent `N` `f32` samples.
    ring_l: Vec<f32>,
    /// Right-channel ring buffer of the most recent `N` `f32` samples.
    ring_r: Vec<f32>,
    /// Five incremental sums (Σx, Σy, Σx², Σy², Σxy) for the windowed
    /// Pearson correlation closed-form.
    sx: f64,
    sy: f64,
    sxx: f64,
    syy: f64,
    sxy: f64,
    /// Shared write cursor (`samples_seen` modulo `window_samples`).
    write: usize,
    /// Monotonically-increasing sample-pair count.
    samples_seen: u64,
    /// Last correlation computed at the close of the most recent
    /// `process()` call. `0.0` until the window is first full.
    last: f32,
    /// Running minimum correlation since construction or last
    /// [`reset_min`](StereoCorrelationMeter::reset_min). `f32::INFINITY`
    /// before the window first fills, so the first valid reading
    /// always lowers it.
    min: f32,
}

impl StereoCorrelationMeter {
    /// New meter with the default 400 ms window.
    pub fn new() -> Self {
        Self::with_window_ms(SCM_DEFAULT_WINDOW_MS)
    }

    /// New meter with explicit window length in milliseconds. Clamped
    /// to `[0.1, 10_000]` ms; the sample-count form is derived from
    /// the input stream's `sample_rate` at first frame and additionally
    /// clamped to `[1, SCM_MAX_WINDOW_SAMPLES]`.
    pub fn with_window_ms(window_ms: f32) -> Self {
        Self {
            window_ms: window_ms.clamp(0.1, 10_000.0),
            state: None,
        }
    }

    /// Configured window length in ms.
    pub fn window_ms(&self) -> f32 {
        self.window_ms
    }

    /// Resolved window length in samples (after first `process()` call).
    /// Returns `0` before the meter has seen its first stream.
    pub fn window_samples(&self) -> usize {
        self.state.as_ref().map(|s| s.window_samples).unwrap_or(0)
    }

    /// Number of stereo sample pairs observed since construction or
    /// last [`reset`](Self::reset). Before this reaches
    /// `window_samples` the [`current`](Self::current) readout stays
    /// at `0.0`.
    pub fn samples_seen(&self) -> u64 {
        self.state.as_ref().map(|s| s.samples_seen).unwrap_or(0)
    }

    /// Latest windowed Pearson correlation `ρ ∈ [-1, +1]` at the close
    /// of the most recent `process()` call. Returns `0.0` before the
    /// window is full or when the active window is bit-exact silent
    /// (either channel has zero variance — correlation undefined).
    pub fn current(&self) -> f32 {
        self.state.as_ref().map(|s| s.last).unwrap_or(0.0)
    }

    /// Polar / goniometer reading in degrees: `acos(ρ) · 180/π`. A
    /// perfectly-correlated mono signal reads `0°`, an uncorrelated
    /// stereo bed `90°`, and a phase-inverted pair `180°`. Returns
    /// `90.0` (the neutral reading) before the window is full or when
    /// the correlation is undefined.
    pub fn current_degrees(&self) -> f32 {
        let r = self.current();
        // `current()` returns 0.0 in the warm-up / undefined case,
        // which maps to acos(0) = 90°. The clamp guards against
        // round-off slightly exceeding ±1.
        (r.clamp(-1.0, 1.0).acos()) * std::f32::consts::FRAC_1_PI * 180.0
    }

    /// Running minimum correlation since construction or last
    /// [`reset_min`](Self::reset_min). Returns `f32::INFINITY` before
    /// the window first fills (so the first valid reading is always
    /// recorded). The minimum is the worst-case phase-cancellation
    /// reading the meter has observed.
    pub fn min(&self) -> f32 {
        self.state.as_ref().map(|s| s.min).unwrap_or(f32::INFINITY)
    }

    /// Wipe all per-channel state (rings, sums, counters). The
    /// configured `window_ms` survives; the resolved `window_samples`
    /// is re-derived on the next `process()` call.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Clear only the running minimum. The window contents,
    /// incremental sums, and the latest reading are unchanged.
    pub fn reset_min(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.min = f32::INFINITY;
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.channels != channels,
            None => true,
        };
        if rebuild {
            let n = ((self.window_ms as f64 * sample_rate as f64 / 1000.0).round() as usize).max(1);
            let window_samples = n.min(SCM_MAX_WINDOW_SAMPLES);
            self.state = Some(MeterState {
                sample_rate,
                channels,
                window_samples,
                ring_l: vec![0.0; window_samples],
                ring_r: vec![0.0; window_samples],
                sx: 0.0,
                sy: 0.0,
                sxx: 0.0,
                syy: 0.0,
                sxy: 0.0,
                write: 0,
                samples_seen: 0,
                last: 0.0,
                min: f32::INFINITY,
            });
        }
    }
}

impl Default for StereoCorrelationMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for StereoCorrelationMeter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate, params.channels as usize);
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n_in = channels.first().map(|c| c.len()).unwrap_or(0);

        // Only run the pair-correlation update for stereo input. Mono
        // and >2-channel layouts pass through unchanged with the meter
        // state untouched, so the readout keeps its previous value.
        if params.channels == 2 {
            let state = self.state.as_mut().expect("state ensured above");
            let nwin = state.window_samples;
            let left = &channels[0];
            let right = &channels[1];

            for i in 0..n_in {
                let x = left[i];
                let y = right[i];
                let xf = x as f64;
                let yf = y as f64;
                let x_old = state.ring_l[state.write];
                let y_old = state.ring_r[state.write];
                state.ring_l[state.write] = x;
                state.ring_r[state.write] = y;

                // Incremental sum update. While the window is still
                // warming up `x_old` / `y_old` are the zero-initialised
                // ring slots, so the "leaving" terms are zero and the
                // sums grow correctly.
                if state.samples_seen < nwin as u64 {
                    state.sx += xf;
                    state.sy += yf;
                    state.sxx += xf * xf;
                    state.syy += yf * yf;
                    state.sxy += xf * yf;
                } else {
                    let xof = x_old as f64;
                    let yof = y_old as f64;
                    state.sx += xf - xof;
                    state.sy += yf - yof;
                    state.sxx += xf * xf - xof * xof;
                    state.syy += yf * yf - yof * yof;
                    state.sxy += xf * yf - xof * yof;
                }

                state.write = (state.write + 1) % nwin;
                state.samples_seen = state.samples_seen.saturating_add(1);

                // Periodic rebuild of all five sums from the ring
                // contents once per full window. Bounds f64 round-off
                // drift on streams long enough to matter, at O(N)
                // every N samples = O(1) amortised per sample.
                if state.samples_seen >= nwin as u64 && state.samples_seen % (nwin as u64) == 0 {
                    let mut sx: f64 = 0.0;
                    let mut sy: f64 = 0.0;
                    let mut sxx: f64 = 0.0;
                    let mut syy: f64 = 0.0;
                    let mut sxy: f64 = 0.0;
                    for k in 0..nwin {
                        let lf = state.ring_l[k] as f64;
                        let rf = state.ring_r[k] as f64;
                        sx += lf;
                        sy += rf;
                        sxx += lf * lf;
                        syy += rf * rf;
                        sxy += lf * rf;
                    }
                    state.sx = sx;
                    state.sy = sy;
                    state.sxx = sxx;
                    state.syy = syy;
                    state.sxy = sxy;
                }
            }

            // Compute the windowed Pearson correlation at frame close.
            if state.samples_seen >= nwin as u64 {
                let nf = nwin as f64;
                let num = nf * state.sxy - state.sx * state.sy;
                let denom_x = (nf * state.sxx - state.sx * state.sx).max(0.0);
                let denom_y = (nf * state.syy - state.sy * state.sy).max(0.0);
                let denom = (denom_x * denom_y).sqrt();
                let rho = if denom > 0.0 {
                    (num / denom).clamp(-1.0, 1.0) as f32
                } else {
                    // Either channel has zero variance — correlation
                    // undefined. Match the convention that the
                    // observer reports the neutral reading.
                    0.0
                };
                state.last = rho;
                if rho < state.min {
                    state.min = rho;
                }
            }
        }

        // Pass-through: re-encode the decoded channels unchanged.
        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::SampleFormat;

    fn f32p_stereo(rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32P,
            channels: 2,
            sample_rate: rate,
        }
    }

    fn f32_mono(rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 1,
            sample_rate: rate,
        }
    }

    fn make_f32p_stereo(left: &[f32], right: &[f32]) -> AudioFrame {
        assert_eq!(left.len(), right.len());
        let mut lb = Vec::with_capacity(left.len() * 4);
        let mut rb = Vec::with_capacity(right.len() * 4);
        for (&l, &r) in left.iter().zip(right.iter()) {
            lb.extend_from_slice(&l.to_le_bytes());
            rb.extend_from_slice(&r.to_le_bytes());
        }
        AudioFrame {
            samples: left.len() as u32,
            pts: None,
            data: vec![lb, rb],
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

    #[test]
    fn pass_through_preserves_audio_bytes() {
        // Observation-only: bytes out must equal bytes in, regardless
        // of how the meter updates internally.
        let fs = 48_000u32;
        let left: Vec<f32> = (0..2048).map(|i| 0.4 * (i as f32 * 0.01).sin()).collect();
        let right: Vec<f32> = (0..2048).map(|i| 0.4 * (i as f32 * 0.02).cos()).collect();
        let frame = make_f32p_stereo(&left, &right);
        let mut m = StereoCorrelationMeter::new();
        let outs = m.process(&frame, f32p_stereo(fs)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].samples, frame.samples);
        assert_eq!(outs[0].data, frame.data);
    }

    #[test]
    fn before_window_full_returns_zero() {
        // 50 ms window @ 48 kHz = 2400 sample pairs; feed only 1024.
        let fs = 48_000u32;
        let mut m = StereoCorrelationMeter::with_window_ms(50.0);
        let left = vec![0.5f32; 1024];
        let right = vec![0.5f32; 1024];
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        assert_eq!(m.current(), 0.0);
        // Neutral degrees reading for an undefined correlation.
        assert!((m.current_degrees() - 90.0).abs() < 1e-3);
    }

    #[test]
    fn identical_channels_correlate_to_plus_one() {
        // L == R sample-for-sample → ρ = +1 exactly.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let left: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let right = left.clone();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let r = m.current();
        assert!(
            (r - 1.0).abs() < 1e-4,
            "identical channels should correlate to +1, got {r}"
        );
        // 0° on the goniometer.
        let deg = m.current_degrees();
        assert!(deg < 1.0, "identical → 0° on goniometer, got {deg}");
    }

    #[test]
    fn phase_inverted_channels_correlate_to_minus_one() {
        // R = -L sample-for-sample → ρ = -1; canonical mono-fold-down
        // cancellation hazard.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let left: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let right: Vec<f32> = left.iter().map(|&x| -x).collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let r = m.current();
        assert!(
            (r + 1.0).abs() < 1e-4,
            "phase-inverted channels should correlate to -1, got {r}"
        );
        // 180° on the goniometer.
        let deg = m.current_degrees();
        assert!(
            (deg - 180.0).abs() < 0.5,
            "phase-inverted → 180° on goniometer, got {deg}"
        );
    }

    #[test]
    fn quadrature_channels_correlate_to_zero() {
        // L = sin(ωt), R = cos(ωt) are orthogonal: their windowed
        // correlation tends to zero over a window holding an integer
        // number of periods. Allow some tolerance because the window
        // width is also bounded by f32 precision and the window need
        // not contain *exactly* an integer number of periods.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let period = 48usize; // 48000 / 1000 = 48 samples
        let n_window = period * 20; // 20 full periods inside the window
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let left: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let right: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).cos()
            })
            .collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let r = m.current();
        assert!(
            r.abs() < 0.05,
            "quadrature channels should correlate near 0, got {r}"
        );
        // ~90° on the goniometer.
        let deg = m.current_degrees();
        assert!(
            (deg - 90.0).abs() < 3.0,
            "quadrature → ~90° on goniometer, got {deg}"
        );
    }

    #[test]
    fn silence_returns_neutral_reading() {
        // Both channels at zero → both have zero variance → denominator
        // collapses → correlation undefined → meter reports 0.0
        // (neutral on the [-1, +1] axis).
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let left = vec![0.0f32; n_window * 2];
        let right = vec![0.0f32; n_window * 2];
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        assert_eq!(m.current(), 0.0);
    }

    #[test]
    fn one_silent_channel_returns_neutral_reading() {
        // Asymmetric zero variance also lands on the undefined branch.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let left = vec![0.0f32; n];
        let right: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        assert_eq!(m.current(), 0.0);
    }

    #[test]
    fn dc_offsets_do_not_bias_the_metric() {
        // Pearson is mean-centred, so adding a constant offset to both
        // channels must leave the correlation unchanged. Probe with
        // L = sin + 0.3, R = sin + 0.7; correlation should still be +1.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.5 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let left: Vec<f32> = base.iter().map(|&x| x + 0.3).collect();
        let right: Vec<f32> = base.iter().map(|&x| x + 0.7).collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let r = m.current();
        assert!(
            (r - 1.0).abs() < 1e-3,
            "DC-offset shifts should not bias Pearson; got {r}"
        );
    }

    #[test]
    fn scaling_does_not_bias_the_metric() {
        // Pearson is scale-invariant: scaling either channel by a
        // positive constant leaves ρ unchanged.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let left: Vec<f32> = base.iter().map(|&x| 0.2 * x).collect();
        let right: Vec<f32> = base.iter().map(|&x| 0.9 * x).collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let r = m.current();
        assert!(
            (r - 1.0).abs() < 1e-3,
            "scale invariance should hold; got {r}"
        );
    }

    #[test]
    fn min_tracks_worst_phase_cancellation() {
        // First frame: identical (ρ = +1). Second frame: phase-inverted
        // (ρ = -1). Third frame: identical again. min() should latch on
        // the inverted frame.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let neg: Vec<f32> = base.iter().map(|&x| -x).collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&base, &base), f32p_stereo(fs))
            .unwrap();
        let min_after_pos = m.min();
        m.process(&make_f32p_stereo(&base, &neg), f32p_stereo(fs))
            .unwrap();
        let min_after_neg = m.min();
        m.process(&make_f32p_stereo(&base, &base), f32p_stereo(fs))
            .unwrap();
        let min_after_back = m.min();
        assert!(min_after_neg < min_after_pos);
        assert_eq!(min_after_back, min_after_neg, "min must not regress");
    }

    #[test]
    fn reset_min_clears_min_only() {
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let neg: Vec<f32> = base.iter().map(|&x| -x).collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&base, &neg), f32p_stereo(fs))
            .unwrap();
        let min_before = m.min();
        assert!(min_before < -0.5);
        m.reset_min();
        // current is unchanged by reset_min.
        let cur_after = m.current();
        assert!((cur_after + 1.0).abs() < 1e-3);
        // min reverts to +INFINITY until the next valid reading.
        assert_eq!(m.min(), f32::INFINITY);
    }

    #[test]
    fn reset_clears_all_state() {
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 2;
        let base = vec![0.5f32; n];
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&base, &base), f32p_stereo(fs))
            .unwrap();
        assert!(m.samples_seen() > 0);
        m.reset();
        assert_eq!(m.samples_seen(), 0);
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.window_samples(), 0);
        assert_eq!(m.min(), f32::INFINITY);
    }

    #[test]
    fn streaming_continuity_split_equals_whole() {
        // A single 4·N-pair call should yield the same final reading
        // as four N-pair calls fed sequentially.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let total = n_window * 4;
        let left: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.6 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
            })
            .collect();
        let right: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.4 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 1300.0 * t).cos()
            })
            .collect();
        let mut m_whole = StereoCorrelationMeter::with_window_ms(window_ms);
        m_whole
            .process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let whole = m_whole.current();
        let mut m_split = StereoCorrelationMeter::with_window_ms(window_ms);
        for chunk_start in (0..total).step_by(n_window) {
            let chunk_end = chunk_start + n_window;
            m_split
                .process(
                    &make_f32p_stereo(
                        &left[chunk_start..chunk_end],
                        &right[chunk_start..chunk_end],
                    ),
                    f32p_stereo(fs),
                )
                .unwrap();
        }
        let split = m_split.current();
        assert!(
            (whole - split).abs() < 1e-3,
            "split ({split}) should match whole ({whole}) within 1e-3"
        );
    }

    #[test]
    fn mono_passes_through_without_updating() {
        // Mono input: pair-correlation update skipped, audio pass-through.
        let fs = 48_000u32;
        let samples: Vec<f32> = (0..2048).map(|i| 0.4 * (i as f32 * 0.01).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut m = StereoCorrelationMeter::new();
        let outs = m.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(outs[0].data, frame.data);
        // No stereo update — meter stays at warm-up reading.
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.samples_seen(), 0);
    }

    #[test]
    fn window_ms_clamps_to_bounds() {
        let m_low = StereoCorrelationMeter::with_window_ms(-100.0);
        assert!(m_low.window_ms() > 0.0);
        let m_high = StereoCorrelationMeter::with_window_ms(1.0e9);
        assert!(m_high.window_ms() <= 10_000.0);
    }

    #[test]
    fn window_samples_clamps_to_max() {
        // A 10_000 ms window at 192 kHz would request 1.92M samples;
        // the SCM_MAX_WINDOW_SAMPLES guard clamps that to 192_000.
        let fs = 192_000u32;
        let mut m = StereoCorrelationMeter::with_window_ms(10_000.0);
        let left = vec![0.0f32; 64];
        let right = vec![0.0f32; 64];
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        assert!(m.window_samples() <= SCM_MAX_WINDOW_SAMPLES);
    }

    #[test]
    fn sample_rate_change_rebuilds_window() {
        // Reconfiguring fs between calls should re-derive
        // `window_samples` from the new rate.
        let mut m = StereoCorrelationMeter::with_window_ms(100.0);
        let left = vec![0.5f32; 64];
        let right = vec![0.5f32; 64];
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(48_000))
            .unwrap();
        let n_at_48k = m.window_samples();
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(96_000))
            .unwrap();
        let n_at_96k = m.window_samples();
        assert!(
            n_at_96k > n_at_48k,
            "96 kHz window ({n_at_96k}) should exceed 48 kHz window ({n_at_48k})"
        );
        assert!(
            (n_at_96k as f32 / n_at_48k as f32 - 2.0).abs() < 0.1,
            "ratio {} should be ~2.0",
            n_at_96k as f32 / n_at_48k as f32
        );
    }

    #[test]
    fn long_stream_sums_do_not_drift() {
        // Process many windows of correlated DC and confirm the
        // correlation stays bit-stable (the periodic-rebuild safeguard
        // keeps round-off from accumulating on streams long enough to
        // matter).
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 100;
        let base: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&base, &base), f32p_stereo(fs))
            .unwrap();
        assert!((m.current() - 1.0).abs() < 1e-3);
        // Quiet trailing frame to verify continuity.
        let tail: Vec<f32> = (0..n_window)
            .map(|i| {
                let t = (n + i) as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        m.process(&make_f32p_stereo(&tail, &tail), f32p_stereo(fs))
            .unwrap();
        assert!((m.current() - 1.0).abs() < 1e-3);
    }

    #[test]
    fn current_clamped_into_canonical_range() {
        // Defensive: even a high-magnitude correlated pair should
        // produce a reading inside [-1, +1] after the clamp in the
        // closed-form path.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 4;
        let left: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.95 * (2.0 * std::f32::consts::PI * 1_000.0 * t).sin()
            })
            .collect();
        let right = left.clone();
        let mut m = StereoCorrelationMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let r = m.current();
        assert!((-1.0..=1.0).contains(&r));
        // Degrees representation respects the same range mapping.
        let deg = m.current_degrees();
        assert!((0.0..=180.0).contains(&deg));
    }
}
