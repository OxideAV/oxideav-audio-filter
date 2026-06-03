//! Crest-factor meter — pass-through observer reporting the peak-to-RMS
//! ratio in dB over a sliding rectangular window.
//!
//! The crest factor `CF = peak / rms` (or `20·log10(peak/rms)` in dB) is
//! a textbook scalar summarising how "spiky" or "transient-rich" a
//! signal is relative to its average power. A sine wave has
//! `CF = √2 ≈ 3.01 dB`; a square wave has `CF = 1 = 0 dB`; broadband
//! noise lands near `≈ 11 dB`; modern broadcast pop typically sits in
//! the `5..8 dB` range after heavy dynamic compression; sparse drum
//! transients can push above `20 dB`. The metric is rate-independent
//! and channel-link-by-max (a stereo programme reports the louder
//! channel's crest, not an L-vs-R average).
//!
//! Within this crate's observation family, the crest-factor meter sits
//! orthogonal to:
//! * [`TruePeakDetector`](crate::true_peak_detector::TruePeakDetector) —
//!   reports absolute true peak (dBTP) via 4× oversampling; says
//!   nothing about average power.
//! * [`LoudnessITU`](crate::loudness::LoudnessITU) — reports K-weighted
//!   integrated loudness (LUFS); says nothing about transient peaks.
//! * [`EnvelopeFollower`](crate::envelope_follower::EnvelopeFollower) —
//!   reports the *current* one-pole peak or RMS envelope; not a ratio.
//! * [`SilenceDetector`](crate::silence_detector::SilenceDetector) —
//!   binary above/below RMS threshold flag.
//!
//! # Algorithm
//!
//! The filter maintains a per-channel rectangular sliding window of
//! `window_samples` recent samples. Two running statistics are kept:
//!
//! * **Running sum-of-squares** `S = Σ x²`. On each new sample `x_new`
//!   entering the window and old sample `x_old` leaving it the running
//!   sum is updated incrementally:
//!
//!   ```text
//!   S ← S + x_new² - x_old²
//!   rms = √(S / N)
//!   ```
//!
//!   To bound `f64` round-off drift on long streams, `S` is periodically
//!   rebuilt from the ring contents every `window_samples` updates
//!   (cheap relative to the per-sample work, eliminates any unbounded
//!   accumulation of subtraction error).
//!
//! * **Sliding maximum of `|x|`** via the classical monotonic-deque
//!   trick: each entry stores `(|x|, index)`; on each new sample
//!   `(|x_new|, n)`, indices with `|x|` ≤ `|x_new|` are popped from the
//!   back, the new entry is pushed; then any front entries whose
//!   index expired (`n - window_samples`) are popped. The deque front
//!   always holds the maximum `|x|` over the active window, in `O(1)`
//!   amortised time per sample (each sample is pushed and popped at
//!   most once). This is the standard sliding-maximum primitive (cf.
//!   "monotonic deque" in any algorithms text); it is exact, has no
//!   round-off accumulation issue, and avoids the `O(N)` cost of a
//!   naive rescan on each window slide.
//!
//! Per channel the meter then reports `peak_ch = front-of-deque`,
//! `rms_ch = √(S/N)`, and the channel-linked
//! `peak / rms = max_ch(peak_ch) / max_ch(rms_ch)` is the crest factor.
//! (Channel-linked by `max` so a loud transient on one channel of a
//! split stereo bed is not masked by a quieter average on the other.)
//!
//! # Warm-up
//!
//! Until the window is full, the meter returns
//! [`f32::NEG_INFINITY`] for the dB readout (and `0.0` for the linear
//! readout) so that callers can branch on "not yet ready". The
//! `samples_seen()` accessor exposes the count to make this explicit.
//!
//! # Parameters
//!
//! * `window_ms` — measurement window in milliseconds (default `400`,
//!   matching the EBU R128 short-term loudness window). Internally
//!   converted to sample count `N = round(window_ms · fs / 1000)`,
//!   clamped to `[1, MAX_WINDOW_SAMPLES = 192_000]` (4 s at 48 kHz).
//!
//! # API surface
//!
//! The filter is observation-only — [`AudioFilter::process`] returns
//! the input frame unchanged. Consumers call:
//!
//! * [`CrestFactorMeter::current_db`] — peak-to-RMS ratio, dB, over the
//!   current window. `NEG_INFINITY` until the window is full or when
//!   the window is silent (rms = 0).
//! * [`CrestFactorMeter::current_linear`] — peak / rms, linear.
//! * [`CrestFactorMeter::current_peak`] /
//!   [`CrestFactorMeter::current_rms`] — the two raw inputs, in case
//!   the consumer wants them directly.
//! * [`CrestFactorMeter::max_db`] / [`CrestFactorMeter::reset_max`] —
//!   running max of the crest-factor dB over the meter's history.
//! * [`CrestFactorMeter::samples_seen`] — total input samples processed
//!   per channel since construction or last
//!   [`CrestFactorMeter::reset`].
//! * [`CrestFactorMeter::reset`] — wipe all per-channel state.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};
use std::collections::VecDeque;

/// Default measurement window in milliseconds (EBU R128 short-term).
pub const CFM_DEFAULT_WINDOW_MS: f32 = 400.0;

/// Maximum measurement window in samples (`= 192_000`, i.e. 4 s at
/// 48 kHz or 2 s at 96 kHz). Defends against pathological allocations
/// without rejecting any realistic broadcast / mastering window.
pub const CFM_MAX_WINDOW_SAMPLES: usize = 192_000;

/// Streaming crest-factor meter.
#[derive(Debug, Clone)]
pub struct CrestFactorMeter {
    window_ms: f32,
    state: Option<MeterState>,
}

#[derive(Debug, Clone)]
struct MeterState {
    sample_rate: u32,
    /// Window length in samples (`N`), derived from `window_ms` and
    /// `sample_rate`, clamped to `[1, CFM_MAX_WINDOW_SAMPLES]`.
    window_samples: usize,
    /// Per-channel ring buffer of the most recent `N` `f32` samples.
    /// `rings[ch].len() == N` when the window is full; before that,
    /// only `samples_seen` slots are populated.
    rings: Vec<Vec<f32>>,
    /// Per-channel running sum of `x²` over the active window.
    sum_sq: Vec<f64>,
    /// Per-channel monotonic-decreasing deque of `(|x|, write_index)`
    /// for the sliding maximum.
    peak_deque: Vec<VecDeque<(f32, u64)>>,
    /// Shared write cursor (`samples_seen` modulo `window_samples`).
    write: usize,
    /// Monotonically-increasing sample count, used as the index field
    /// in `peak_deque` so we can detect when entries fall out of the
    /// active window.
    samples_seen: u64,
    /// Last linear `peak / rms` ratio computed at the close of the most
    /// recent `process()` call. `0.0` until the window is first full or
    /// when the window is silent.
    last_linear: f32,
    /// Running max linear `peak / rms` since construction or last
    /// [`reset_max`](CrestFactorMeter::reset_max).
    max_linear: f32,
}

impl CrestFactorMeter {
    /// New meter with the default 400 ms window.
    pub fn new() -> Self {
        Self::with_window_ms(CFM_DEFAULT_WINDOW_MS)
    }

    /// New meter with explicit window length in milliseconds. Clamped
    /// to `[0.1, 10_000]` ms; the sample-count form is derived from
    /// the input stream's `sample_rate` at first frame and additionally
    /// clamped to `[1, CFM_MAX_WINDOW_SAMPLES]`.
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

    /// Number of samples observed per channel since construction or
    /// last [`reset`](Self::reset). Before this reaches `window_samples`
    /// the readouts return `NEG_INFINITY` / `0.0` — the window isn't
    /// full yet.
    pub fn samples_seen(&self) -> u64 {
        self.state.as_ref().map(|s| s.samples_seen).unwrap_or(0)
    }

    /// Peak-to-RMS ratio in dB at the end of the most recent
    /// `process()` call. Returns [`f32::NEG_INFINITY`] before the
    /// window is full or when the active window is bit-exact silent
    /// (rms = 0).
    pub fn current_db(&self) -> f32 {
        linear_to_db(self.state.as_ref().map(|s| s.last_linear).unwrap_or(0.0))
    }

    /// Peak-to-RMS ratio, linear. Returns `0.0` before the window is
    /// full or when the active window is bit-exact silent.
    pub fn current_linear(&self) -> f32 {
        self.state.as_ref().map(|s| s.last_linear).unwrap_or(0.0)
    }

    /// Channel-linked peak `max |x|` over the current window. Returns
    /// `0.0` before the window is full.
    pub fn current_peak(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return 0.0;
        };
        if s.samples_seen < s.window_samples as u64 {
            return 0.0;
        }
        s.peak_deque
            .iter()
            .filter_map(|dq| dq.front().map(|&(p, _)| p))
            .fold(0.0f32, f32::max)
    }

    /// Channel-linked RMS `sqrt(Σx² / N)` over the current window.
    /// Returns `0.0` before the window is full.
    pub fn current_rms(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return 0.0;
        };
        if s.samples_seen < s.window_samples as u64 {
            return 0.0;
        }
        let n = s.window_samples as f64;
        s.sum_sq
            .iter()
            .map(|&ss| (ss / n).max(0.0).sqrt() as f32)
            .fold(0.0f32, f32::max)
    }

    /// Running maximum of the crest-factor dB since construction or
    /// last [`reset_max`](Self::reset_max). Returns `NEG_INFINITY`
    /// before the window is first full.
    pub fn max_db(&self) -> f32 {
        linear_to_db(self.state.as_ref().map(|s| s.max_linear).unwrap_or(0.0))
    }

    /// Wipe all per-channel state (rings, sum-of-squares, deques,
    /// counters). The configured `window_ms` survives; the resolved
    /// `window_samples` is re-derived on the next `process()` call.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Clear only the running max, leaving the window contents and
    /// counters intact.
    pub fn reset_max(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.max_linear = 0.0;
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.rings.len() != channels,
            None => true,
        };
        if rebuild {
            let n = ((self.window_ms as f64 * sample_rate as f64 / 1000.0).round() as usize).max(1);
            let window_samples = n.min(CFM_MAX_WINDOW_SAMPLES);
            self.state = Some(MeterState {
                sample_rate,
                window_samples,
                rings: vec![vec![0.0; window_samples]; channels],
                sum_sq: vec![0.0; channels],
                peak_deque: vec![VecDeque::with_capacity(window_samples); channels],
                write: 0,
                samples_seen: 0,
                last_linear: 0.0,
                max_linear: 0.0,
            });
        }
    }
}

impl Default for CrestFactorMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for CrestFactorMeter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate, params.channels as usize);
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);

        let state = self.state.as_mut().expect("state ensured above");
        let nwin = state.window_samples;

        for i in 0..n {
            for (ch_idx, ch) in channels.iter().enumerate() {
                let x = ch[i];
                let xf = x as f64;
                let ring = &mut state.rings[ch_idx];
                let old = ring[state.write];
                ring[state.write] = x;

                // Update sum-of-squares incrementally. Once the window
                // is full, `old` is the sample just rotated out.
                if state.samples_seen < nwin as u64 {
                    state.sum_sq[ch_idx] += xf * xf;
                } else {
                    let oldf = old as f64;
                    state.sum_sq[ch_idx] += xf * xf - oldf * oldf;
                }

                // Update monotonic deque for sliding `|x|` max.
                let absx = x.abs();
                let n_idx = state.samples_seen;
                let dq = &mut state.peak_deque[ch_idx];
                while let Some(&(back_v, _)) = dq.back() {
                    if back_v <= absx {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((absx, n_idx));
                // Pop fronts whose index has expired (more than
                // `window_samples` ago).
                let cutoff = n_idx.saturating_sub(nwin as u64 - 1);
                while let Some(&(_, idx)) = dq.front() {
                    if idx < cutoff {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }
            }
            state.write = (state.write + 1) % nwin;
            state.samples_seen = state.samples_seen.saturating_add(1);

            // Periodic rebuild of sum_sq from ring contents once per
            // full window to bound f64 round-off drift on long streams.
            if state.samples_seen >= nwin as u64 && state.samples_seen % (nwin as u64) == 0 {
                for (ch_idx, ring) in state.rings.iter().enumerate() {
                    let mut acc: f64 = 0.0;
                    for &v in ring.iter() {
                        let vf = v as f64;
                        acc += vf * vf;
                    }
                    state.sum_sq[ch_idx] = acc;
                }
            }
        }

        // Compute current linear crest factor once at frame close.
        if state.samples_seen >= nwin as u64 {
            let mut peak: f32 = 0.0;
            let mut rms: f32 = 0.0;
            let nf = nwin as f64;
            for (ch_idx, dq) in state.peak_deque.iter().enumerate() {
                if let Some(&(p, _)) = dq.front() {
                    if p > peak {
                        peak = p;
                    }
                }
                let r = (state.sum_sq[ch_idx] / nf).max(0.0).sqrt() as f32;
                if r > rms {
                    rms = r;
                }
            }
            state.last_linear = if rms > 0.0 { peak / rms } else { 0.0 };
            if state.last_linear > state.max_linear {
                state.max_linear = state.last_linear;
            }
        }

        // Pass-through: re-encode the decoded channels unchanged.
        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

/// Convert a linear amplitude ratio to dB; ratios `≤ 0` map to
/// `-INFINITY` (matches the convention used by the rest of the crate's
/// observation filters).
fn linear_to_db(linear: f32) -> f32 {
    if linear <= 0.0 {
        f32::NEG_INFINITY
    } else {
        20.0 * linear.log10()
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

    fn f32p_stereo(rate: u32) -> AudioStreamParams {
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

    #[test]
    fn pass_through_preserves_audio_bytes() {
        // Observation-only: input bytes must come out byte-identical.
        let fs = 48_000u32;
        let samples: Vec<f32> = (0..2048).map(|i| 0.4 * (i as f32 * 0.01).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut m = CrestFactorMeter::new();
        let outs = m.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].samples, frame.samples);
        assert_eq!(outs[0].data, frame.data);
    }

    #[test]
    fn before_window_full_returns_neg_infinity() {
        // 50 ms window @ 48 kHz = 2400 samples; feed only 1024 first.
        let fs = 48_000u32;
        let mut m = CrestFactorMeter::with_window_ms(50.0);
        let samples = vec![0.5f32; 1024];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current_db(), f32::NEG_INFINITY);
        assert_eq!(m.current_linear(), 0.0);
        assert_eq!(m.current_peak(), 0.0);
        assert_eq!(m.current_rms(), 0.0);
    }

    #[test]
    fn dc_input_yields_zero_db_crest() {
        // DC has peak == rms so crest factor is exactly 1.0 = 0 dB.
        let fs = 48_000u32;
        let mut m = CrestFactorMeter::with_window_ms(20.0);
        let n = (fs as f32 * 0.020) as usize * 4; // 4 windows worth
        let samples = vec![0.7f32; n];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let cf_db = m.current_db();
        assert!(
            cf_db.abs() < 0.01,
            "DC crest factor should be 0 dB, got {cf_db}"
        );
        assert!(
            (m.current_linear() - 1.0).abs() < 1e-4,
            "DC linear crest factor should be 1.0, got {}",
            m.current_linear()
        );
    }

    #[test]
    fn full_scale_sine_crest_factor_is_sqrt2() {
        // A sine wave has rms = amplitude / √2, so peak/rms = √2 ≈ 3.0103 dB
        // — regardless of amplitude.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        // Window must hold an integer number of periods so the rectangular
        // window doesn't bias the RMS calculation. 1 kHz at 48 kHz = 48
        // samples / period; 480-sample window = exactly 10 periods.
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let n = n_window * 8;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let cf_db = m.current_db();
        // 20·log10(√2) = 3.0103.
        assert!(
            (cf_db - 3.0103).abs() < 0.15,
            "full-scale sine crest factor should be ≈ 3.01 dB, got {cf_db}"
        );
    }

    #[test]
    fn square_wave_crest_factor_is_zero_db() {
        // A symmetric square wave has |x| = constant so peak == rms,
        // crest factor = 0 dB.
        let fs = 48_000u32;
        let period = 96usize; // 500 Hz
        let n = period * 16;
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                if (i / (period / 2)) % 2 == 0 {
                    0.5
                } else {
                    -0.5
                }
            })
            .collect();
        // Choose window = integer periods so rms is exact 0.5.
        let n_window = period * 4;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let cf_db = m.current_db();
        assert!(
            cf_db.abs() < 0.05,
            "square wave crest factor should be 0 dB, got {cf_db}"
        );
    }

    #[test]
    fn transient_spike_lifts_crest_factor() {
        // A single full-scale spike against a quiet baseline gives an
        // arbitrarily high crest factor — the canonical "drum transient"
        // case the metric is designed to flag.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let mut samples = vec![0.01f32; n_window * 2];
        samples[n_window] = 1.0; // spike inside the second window
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        // RMS over a window holding one spike of 1.0 and (N-1) values
        // of 0.01: rms ≈ sqrt((1 + (N-1)·0.01²)/N) ≈ sqrt(1/N) for N=480
        // → ≈ 0.0456. Peak = 1.0 → CF ≈ 21.9 dB. Allow a wide band.
        let cf_db = m.current_db();
        assert!(cf_db > 15.0, "spike should yield CF > 15 dB, got {cf_db}");
    }

    #[test]
    fn silent_window_reports_neg_infinity() {
        // rms = 0 → log10 undefined → meter reports NEG_INFINITY.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let samples = vec![0.0f32; n_window * 2];
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current_db(), f32::NEG_INFINITY);
        assert_eq!(m.current_linear(), 0.0);
    }

    #[test]
    fn stereo_peak_links_across_channels() {
        // Left silent, right with a sine wave — meter should report the
        // right channel's CF, not an attenuated mix.
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
        let frame = make_f32p_stereo(&left, &right);
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&frame, f32p_stereo(fs)).unwrap();
        let cf_db = m.current_db();
        // Sine on the right → 3.01 dB; the link-by-max picks the right
        // channel's (peak, rms) so the result is the sine's crest factor,
        // not an L+R averaged figure.
        assert!(
            (cf_db - 3.0103).abs() < 0.5,
            "stereo peak-link should track loud channel, got {cf_db}"
        );
    }

    #[test]
    fn max_db_accumulates_across_frames() {
        // First frame: sine (3 dB). Second frame: spike (20 + dB).
        // Third frame: sine again. max_db should latch on the spike.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let f = 1_000.0f32;
        let sine: Vec<f32> = (0..n_window * 2)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let mut spike = vec![0.01f32; n_window * 2];
        spike[n_window] = 1.0;
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&sine), f32_mono(fs)).unwrap();
        let after_sine = m.max_db();
        m.process(&make_f32_mono(&spike), f32_mono(fs)).unwrap();
        let after_spike = m.max_db();
        m.process(&make_f32_mono(&sine), f32_mono(fs)).unwrap();
        let after_back = m.max_db();
        assert!(after_spike > after_sine + 10.0);
        assert_eq!(after_back, after_spike, "max should not regress");
    }

    #[test]
    fn reset_max_clears_max_only() {
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let mut spike = vec![0.01f32; n_window * 2];
        spike[n_window] = 1.0;
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&spike), f32_mono(fs)).unwrap();
        let max_before = m.max_db();
        assert!(max_before > 10.0);
        m.reset_max();
        assert_eq!(m.max_db(), f32::NEG_INFINITY);
        // current_db is unchanged by reset_max.
        assert!(m.current_db() > 10.0);
    }

    #[test]
    fn reset_clears_all_state() {
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let samples = vec![0.5f32; n_window * 2];
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        // DC is well above empty.
        assert!(m.current_linear() > 0.0);
        assert!(m.samples_seen() > 0);
        m.reset();
        assert_eq!(m.samples_seen(), 0);
        assert_eq!(m.current_db(), f32::NEG_INFINITY);
        assert_eq!(m.current_linear(), 0.0);
        assert_eq!(m.window_samples(), 0);
    }

    #[test]
    fn streaming_continuity_split_equals_whole() {
        // One single 4·N-sample call should produce the same final
        // current_linear as four N-sample calls of the same stream.
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let total = n_window * 4;
        let samples: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.6 * (2.0 * std::f32::consts::PI * 700.0 * t).sin()
                    + 0.05 * (i as f32 * 0.137).sin()
            })
            .collect();
        // Whole.
        let mut m_whole = CrestFactorMeter::with_window_ms(window_ms);
        m_whole
            .process(&make_f32_mono(&samples), f32_mono(fs))
            .unwrap();
        let whole_cf = m_whole.current_linear();
        // Split into four equal chunks.
        let mut m_split = CrestFactorMeter::with_window_ms(window_ms);
        for chunk in samples.chunks(n_window) {
            m_split
                .process(&make_f32_mono(chunk), f32_mono(fs))
                .unwrap();
        }
        let split_cf = m_split.current_linear();
        assert!(
            (whole_cf - split_cf).abs() < 1e-3,
            "split call ({split_cf}) should match whole call ({whole_cf}) within 1e-3"
        );
    }

    #[test]
    fn window_ms_clamps_to_bounds() {
        let m_low = CrestFactorMeter::with_window_ms(-100.0);
        assert!(m_low.window_ms() > 0.0);
        let m_high = CrestFactorMeter::with_window_ms(1.0e9);
        assert!(m_high.window_ms() <= 10_000.0);
    }

    #[test]
    fn window_samples_clamps_to_max() {
        // A 10_000 ms window at 192 kHz would request 1.92M samples;
        // the CFM_MAX_WINDOW_SAMPLES guard clamps that to 192_000.
        let fs = 192_000u32;
        let mut m = CrestFactorMeter::with_window_ms(10_000.0);
        let samples = vec![0.0f32; 64];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert!(m.window_samples() <= CFM_MAX_WINDOW_SAMPLES);
    }

    #[test]
    fn sample_rate_change_rebuilds_window() {
        // Reconfiguring fs between calls should re-derive
        // `window_samples` from the new rate.
        let mut m = CrestFactorMeter::with_window_ms(100.0);
        let samples = vec![0.5f32; 64];
        m.process(&make_f32_mono(&samples), f32_mono(48_000))
            .unwrap();
        let n_at_48k = m.window_samples();
        m.process(&make_f32_mono(&samples), f32_mono(96_000))
            .unwrap();
        let n_at_96k = m.window_samples();
        assert!(
            n_at_96k > n_at_48k,
            "96 kHz window ({n_at_96k}) should exceed 48 kHz window ({n_at_48k})"
        );
        // Window ratio should be ~2× (96/48).
        assert!(
            (n_at_96k as f32 / n_at_48k as f32 - 2.0).abs() < 0.1,
            "ratio {} should be ~2.0",
            n_at_96k as f32 / n_at_48k as f32
        );
    }

    #[test]
    fn linear_to_db_handles_zero_and_negative() {
        // Defensive: ratios `<= 0` should map to NEG_INFINITY rather
        // than NaN or +INFINITY, consistent with the rest of the
        // crate's observation filters.
        assert_eq!(linear_to_db(0.0), f32::NEG_INFINITY);
        assert_eq!(linear_to_db(-1.0), f32::NEG_INFINITY);
        assert!((linear_to_db(1.0) - 0.0).abs() < 1e-6);
        // sqrt(2) → 3.0103 dB.
        assert!((linear_to_db(std::f32::consts::SQRT_2) - 3.0103).abs() < 0.001);
    }

    #[test]
    fn long_stream_sum_sq_does_not_drift() {
        // Process many windows of identical DC and confirm rms stays
        // bit-stable (the periodic-rebuild safeguard keeps round-off
        // from accumulating on streams long enough to matter).
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        // 100 windows of 0.5 DC.
        let samples = vec![0.5f32; n_window * 100];
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        // DC: peak == rms → CF = 1.0 exactly (within rebuild tolerance).
        assert!((m.current_linear() - 1.0).abs() < 1e-4);
        // Then a quiet trailing frame to verify continuity.
        let quiet = vec![0.5f32; n_window];
        m.process(&make_f32_mono(&quiet), f32_mono(fs)).unwrap();
        assert!((m.current_linear() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn sliding_max_drops_expired_peak() {
        // Inject a single loud sample, then feed enough quiet samples
        // that the loud sample slides out of the window. The peak
        // should drop accordingly (the monotonic-deque pop_front
        // expired-index branch must fire).
        let fs = 48_000u32;
        let n_window = 480usize;
        let window_ms = (n_window as f32 / fs as f32) * 1000.0;
        let mut samples = vec![0.1f32; n_window * 3];
        samples[10] = 0.95; // loud sample early in the stream
        let mut m = CrestFactorMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        // After 3× the window, the loud sample at index 10 has long
        // since aged out; the current peak should reflect only the
        // baseline ±0.1.
        let peak = m.current_peak();
        assert!(
            peak < 0.2,
            "expired loud sample should not contribute to current peak; got {peak}"
        );
    }
}
