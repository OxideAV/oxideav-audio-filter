//! DC-offset meter — pass-through observer reporting the per-channel
//! running mean (DC component) of the signal over a sliding
//! rectangular window.
//!
//! The DC offset of a signal is the textbook scalar
//!
//! ```text
//! mean = (1/N) · Σ x[n]    over the active window of N samples
//! ```
//!
//! On AC-coupled audio the long-term mean is `0`; any deviation
//! reveals a low-frequency bias — a microphone preamp's bias trim
//! drifting, an ADC's quantiser midpoint sitting off zero, a
//! battery-powered field recorder whose op-amp rail is sagging, or a
//! synthesis chain that accidentally pushed a unipolar oscillator
//! through to the output bus. Any of those leaves the speaker cone
//! parked off centre, wastes a chunk of the available headroom on a
//! constant — inaudible — push, and starves transient peaks of the
//! linear range they would otherwise have used.
//!
//! Where [`DcBlocker`](crate::dc_blocker::DcBlocker) *removes* the DC
//! component with a single-pole high-pass, this meter *reports* it
//! without altering the signal — pass-through, channel-linked by the
//! absolute-largest per-channel mean so a single offset channel
//! survives a quiet sibling.
//!
//! Within the crate's observation family the DC-offset meter sits
//! orthogonal to:
//! * [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
//!   — peak-to-RMS ratio; insensitive to a constant offset because
//!   peak grows by the same amount the RMS does.
//! * [`TruePeakDetector`](crate::true_peak_detector::TruePeakDetector)
//!   — absolute inter-sample peak; a DC bias adds directly to the
//!   peak reading but the meter can't tell the bias apart from a
//!   genuine transient.
//! * [`LoudnessITU`](crate::loudness::LoudnessITU) — K-weighted
//!   loudness; the BS.1770 RLB pre-filter is itself a 38 Hz HPF, so
//!   any DC bias is filtered away before the loudness sum and is
//!   invisible to LUFS.
//! * [`EnvelopeFollower`](crate::envelope_follower::EnvelopeFollower) /
//!   [`SilenceDetector`](crate::silence_detector::SilenceDetector) —
//!   neither carries the signed mean; both rectify to `|x|` early.
//! * [`StereoCorrelationMeter`](crate::stereo_correlation_meter::StereoCorrelationMeter)
//!   — Pearson coefficient is mean-centred by construction, so a DC
//!   bias is invisible to it.
//! * [`ZeroCrossingRateMeter`](crate::zero_crossing_rate::ZeroCrossingRateMeter)
//!   — a DC-biased signal makes fewer sign changes (the offset shifts
//!   the zero line), so ZCR is *indirectly* sensitive to DC but can't
//!   quantify the bias.
//!
//! # Algorithm
//!
//! Per channel the meter keeps a ring buffer of the most recent `N`
//! samples plus a running sum `S = Σ x[n]` over the active window.
//! For every incoming sample `x_new`:
//!
//! 1. Look up the sample about to be overwritten, `x_old =
//!    ring[write]`.
//! 2. Once the window is full, update incrementally: `S ← S + x_new -
//!    x_old`. Before it fills, just accumulate `S ← S + x_new`.
//! 3. Write `x_new` into the ring, advance the shared write cursor.
//!
//! Per-window mean is then `mean = S / N`. The cost is `O(1)` per
//! sample with no branchy sort or deque; the only running cost is the
//! ring rotation.
//!
//! To bound `f64` round-off drift on long streams, `S` is periodically
//! rebuilt from the ring contents once per full window (cheap relative
//! to per-sample work, eliminates any unbounded subtraction-error
//! accumulation). This mirrors the same rebuild cadence used by
//! [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
//! and [`StereoCorrelationMeter`](crate::stereo_correlation_meter::StereoCorrelationMeter).
//!
//! Channel-link is by *signed* mean with largest absolute value: a
//! channel sitting at `+0.05` and a channel sitting at `-0.02` should
//! report `+0.05`, not `+0.015`. (Two equal-and-opposite offsets on a
//! split bed thus do *not* cancel in the readout — each channel's
//! sign-correct bias still reaches the speaker for that channel.)
//!
//! # Warm-up
//!
//! Until the window is full the readout returns `0.0` for the linear
//! form and `f32::NEG_INFINITY` for the dB form — the meter doesn't
//! have a full window's worth of samples to average yet. The
//! [`samples_seen`] accessor exposes the count to make this explicit.
//!
//! # Parameters
//!
//! * `window_ms` — measurement window in milliseconds (default `400`,
//!   matching the EBU R128 short-term loudness window — same default
//!   used by [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
//!   and [`StereoCorrelationMeter`](crate::stereo_correlation_meter::StereoCorrelationMeter)
//!   so all three meters share a time axis on a display).
//!   Internally converted to sample count `N = round(window_ms · fs /
//!   1000)`, clamped to `[1, DCM_MAX_WINDOW_SAMPLES = 192_000]` (4 s
//!   at 48 kHz).
//!
//! # API surface
//!
//! The filter is observation-only — [`AudioFilter::process`] returns
//! the input frame unchanged. Consumers call:
//!
//! * [`DcOffsetMeter::current`] — signed channel-linked mean over the
//!   current window. `0.0` until the window is full.
//! * [`DcOffsetMeter::current_db`] — `20·log10(|mean|)` over the
//!   current window. `NEG_INFINITY` when the mean is bit-exact zero
//!   or the window isn't full.
//! * [`DcOffsetMeter::per_channel`] — slice of per-channel signed
//!   means at the close of the most recent `process()` call.
//! * [`DcOffsetMeter::max_abs`] /
//!   [`DcOffsetMeter::reset_max`] — running max `|mean|` over the
//!   meter's history (linear amplitude).
//! * [`DcOffsetMeter::samples_seen`] — total input samples processed
//!   per channel since construction or last [`reset`].
//! * [`DcOffsetMeter::reset`] — wipe all per-channel state.
//!
//! [`reset`]: DcOffsetMeter::reset
//! [`samples_seen`]: DcOffsetMeter::samples_seen

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Default measurement window in milliseconds (EBU R128 short-term;
/// matches `CrestFactorMeter` / `StereoCorrelationMeter` defaults).
pub const DCM_DEFAULT_WINDOW_MS: f32 = 400.0;

/// Maximum measurement window in samples (`= 192_000`, i.e. 4 s at
/// 48 kHz or 2 s at 96 kHz). Defends against pathological allocations
/// without rejecting any realistic broadcast / mastering window.
pub const DCM_MAX_WINDOW_SAMPLES: usize = 192_000;

/// Streaming DC-offset meter.
#[derive(Debug, Clone)]
pub struct DcOffsetMeter {
    window_ms: f32,
    state: Option<MeterState>,
}

#[derive(Debug, Clone)]
struct MeterState {
    sample_rate: u32,
    /// Window length in samples (`N`), derived from `window_ms` and
    /// `sample_rate`, clamped to `[1, DCM_MAX_WINDOW_SAMPLES]`.
    window_samples: usize,
    /// Per-channel ring buffer of the most recent `N` `f32` samples.
    /// `rings[ch].len() == N` always; only `samples_seen` slots are
    /// "populated" before the window first fills, but the buffer
    /// itself is sized to `N` from construction.
    rings: Vec<Vec<f32>>,
    /// Per-channel running sum `S = Σ x` over the active window, in
    /// `f64` to bound round-off on long streams.
    sum: Vec<f64>,
    /// Shared write cursor (`samples_seen` modulo `window_samples`).
    write: usize,
    /// Monotonically-increasing per-channel sample count.
    samples_seen: u64,
    /// Per-channel signed mean at the close of the most recent
    /// `process()` call. `0.0` until the window is first full.
    last_means: Vec<f32>,
    /// Channel-linked signed mean (the per-channel mean with largest
    /// `|·|`, sign-preserved) at the close of the most recent
    /// `process()` call. `0.0` until the window is first full.
    last_linked: f32,
    /// Running max `|mean|` since construction or last
    /// [`reset_max`](DcOffsetMeter::reset_max).
    max_abs: f32,
}

impl DcOffsetMeter {
    /// New meter with the default 400 ms window.
    pub fn new() -> Self {
        Self::with_window_ms(DCM_DEFAULT_WINDOW_MS)
    }

    /// New meter with explicit window length in milliseconds. Clamped
    /// to `[0.1, 10_000]` ms at construction; the sample-count form
    /// is derived from the input stream's `sample_rate` at first
    /// frame and additionally clamped to `[1,
    /// DCM_MAX_WINDOW_SAMPLES]`.
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

    /// Resolved window length in samples (after first `process()`
    /// call). Returns `0` before the meter has seen its first stream.
    pub fn window_samples(&self) -> usize {
        self.state.as_ref().map(|s| s.window_samples).unwrap_or(0)
    }

    /// Number of samples observed per channel since construction or
    /// last [`reset`](Self::reset). Before this reaches
    /// `window_samples` the readouts return `0.0` / `NEG_INFINITY`
    /// — the window isn't full yet.
    pub fn samples_seen(&self) -> u64 {
        self.state.as_ref().map(|s| s.samples_seen).unwrap_or(0)
    }

    /// Channel-linked signed mean at the end of the most recent
    /// `process()` call. The link picks the per-channel mean with
    /// the largest `|·|` and preserves its sign. Returns `0.0`
    /// before the window is full.
    pub fn current(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return 0.0;
        };
        if s.samples_seen < s.window_samples as u64 {
            return 0.0;
        }
        s.last_linked
    }

    /// Channel-linked `|mean|` in dB: `20·log10(|mean|)`. Returns
    /// [`f32::NEG_INFINITY`] when the mean is bit-exact zero or
    /// before the window is full.
    pub fn current_db(&self) -> f32 {
        let m = self.current().abs();
        if m == 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * m.log10()
        }
    }

    /// Per-channel signed means at the end of the most recent
    /// `process()` call. Returns an empty slice before the meter has
    /// seen its first stream; all entries `0.0` before the window is
    /// full.
    pub fn per_channel(&self) -> &[f32] {
        match self.state.as_ref() {
            Some(s) => &s.last_means,
            None => &[],
        }
    }

    /// Running max `|mean|` (linear amplitude) since construction or
    /// last [`reset_max`](Self::reset_max). Returns `0.0` before the
    /// window is first full.
    pub fn max_abs(&self) -> f32 {
        self.state.as_ref().map(|s| s.max_abs).unwrap_or(0.0)
    }

    /// Wipe all per-channel state (rings, running sums, counters,
    /// last means, running max). The configured `window_ms`
    /// survives; the resolved `window_samples` is re-derived on the
    /// next `process()` call.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Clear only the running max, leaving the window contents and
    /// counters intact.
    pub fn reset_max(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.max_abs = 0.0;
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.rings.len() != channels,
            None => true,
        };
        if rebuild {
            let n = ((self.window_ms as f64 * sample_rate as f64 / 1000.0).round() as usize).max(1);
            let window_samples = n.min(DCM_MAX_WINDOW_SAMPLES);
            self.state = Some(MeterState {
                sample_rate,
                window_samples,
                rings: vec![vec![0.0; window_samples]; channels],
                sum: vec![0.0; channels],
                write: 0,
                samples_seen: 0,
                last_means: vec![0.0; channels],
                last_linked: 0.0,
                max_abs: 0.0,
            });
        }
    }
}

impl Default for DcOffsetMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for DcOffsetMeter {
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

                // Update running sum incrementally. Before the window
                // fills, `old` is still the zero we initialised the
                // ring with — accumulating `x_new - 0` is the same as
                // accumulating `x_new`, but we make the branch
                // explicit for clarity.
                if state.samples_seen < nwin as u64 {
                    state.sum[ch_idx] += xf;
                } else {
                    state.sum[ch_idx] += xf - old as f64;
                }
            }
            state.write = (state.write + 1) % nwin;
            state.samples_seen = state.samples_seen.saturating_add(1);

            // Periodic rebuild of `sum` from ring contents once per
            // full window to bound f64 round-off drift on long
            // streams. Same cadence as CrestFactorMeter.
            if state.samples_seen >= nwin as u64 && state.samples_seen % (nwin as u64) == 0 {
                for (ch_idx, ring) in state.rings.iter().enumerate() {
                    let mut acc: f64 = 0.0;
                    for &v in ring.iter() {
                        acc += v as f64;
                    }
                    state.sum[ch_idx] = acc;
                }
            }
        }

        // Compute per-channel means and channel-linked mean once at
        // frame close.
        if state.samples_seen >= nwin as u64 {
            let nf = nwin as f64;
            let mut linked: f32 = 0.0;
            let mut linked_abs: f32 = 0.0;
            for (ch_idx, &s) in state.sum.iter().enumerate() {
                let m = (s / nf) as f32;
                state.last_means[ch_idx] = m;
                let ma = m.abs();
                if ma > linked_abs {
                    linked_abs = ma;
                    linked = m;
                }
            }
            state.last_linked = linked;
            if linked_abs > state.max_abs {
                state.max_abs = linked_abs;
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
        let mut m = DcOffsetMeter::new();
        let outs = m.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].samples, frame.samples);
        assert_eq!(outs[0].data, frame.data);
    }

    #[test]
    fn before_window_full_returns_zero() {
        // 50 ms window @ 48 kHz = 2400 samples; feed only 1024 first.
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(50.0);
        let samples = vec![0.25f32; 1024];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current(), 0.0);
        assert!(m.current_db().is_infinite() && m.current_db() < 0.0);
        assert_eq!(m.max_abs(), 0.0);
    }

    #[test]
    fn constant_dc_yields_mean_equal_to_dc() {
        // A constant DC signal has mean equal to the DC value.
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        // 10 ms @ 48 kHz = 480 samples → feed 2400 (5 windows).
        let samples = vec![0.5f32; 2400];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert!(
            (m.current() - 0.5).abs() < 1e-5,
            "constant DC 0.5 should read 0.5, got {}",
            m.current()
        );
    }

    #[test]
    fn negative_dc_preserves_sign() {
        // A negative DC bias must be reported with its sign intact.
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let samples = vec![-0.25f32; 2400];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert!((m.current() - -0.25).abs() < 1e-5);
        // |mean| in dB: 20·log10(0.25) = -12.04 dB
        assert!((m.current_db() - (-12.041_2)).abs() < 0.01);
    }

    #[test]
    fn silence_yields_zero_mean() {
        // Bit-exact zeros: mean is identically zero.
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let samples = vec![0.0f32; 2400];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current(), 0.0);
        assert!(m.current_db().is_infinite() && m.current_db() < 0.0);
    }

    #[test]
    fn zero_mean_sine_reads_near_zero() {
        // An integer-period sine averages to (numerically) zero over
        // the window. Tolerance is loose to absorb the f32 rounding
        // accumulated across thousands of samples.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let nwin = 480usize; // exactly 10 periods at 1 kHz / 48 kHz.
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 6;
        let samples: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let mut m = DcOffsetMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let mean = m.current();
        assert!(
            mean.abs() < 1e-4,
            "integer-period sine should average to ~0, got {mean}"
        );
    }

    #[test]
    fn biased_sine_reads_the_bias() {
        // sin(2πf t) + 0.1 averages to 0.1 over an integer-period
        // window — the sine cancels, the DC stays.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 6;
        let bias = 0.1f32;
        let samples: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                bias + (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let mut m = DcOffsetMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let mean = m.current();
        assert!(
            (mean - bias).abs() < 1e-3,
            "biased sine should reveal the 0.1 DC, got {mean}"
        );
    }

    #[test]
    fn mean_independent_of_window_position_steady_state() {
        // After the running sum has stabilised, the reported mean
        // should be invariant to where in the input stream we look.
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let n_total = 4800; // 10 windows.
        let samples = vec![0.3f32; n_total];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let mean1 = m.current();
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let mean2 = m.current();
        assert!((mean1 - 0.3).abs() < 1e-5);
        assert!((mean2 - 0.3).abs() < 1e-5);
        assert!((mean1 - mean2).abs() < 1e-5);
    }

    #[test]
    fn reset_wipes_state() {
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let samples = vec![0.5f32; 2400];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert!(m.current() > 0.0);
        m.reset();
        assert_eq!(m.samples_seen(), 0);
        assert_eq!(m.window_samples(), 0);
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.max_abs(), 0.0);
    }

    #[test]
    fn reset_max_clears_running_peak_only() {
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let biased = vec![0.5f32; 2400];
        m.process(&make_f32_mono(&biased), f32_mono(fs)).unwrap();
        let peak = m.max_abs();
        assert!((peak - 0.5).abs() < 1e-5);
        m.reset_max();
        // Now feed a smaller bias; max_abs should track *that* lower
        // value because we wiped the previous high water mark.
        let low_bias = vec![0.1f32; 2400];
        m.process(&make_f32_mono(&low_bias), f32_mono(fs)).unwrap();
        let new_peak = m.max_abs();
        assert!(
            new_peak < 0.5,
            "max_abs should re-track from below after reset_max, got {new_peak}"
        );
    }

    #[test]
    fn stereo_channel_link_picks_largest_abs_with_sign() {
        // Left channel: +0.05 DC; right channel: -0.20 DC.
        // Channel-link by |·| picks the right channel's -0.20, sign
        // preserved.
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 4;
        let left: Vec<f32> = vec![0.05f32; total];
        let right: Vec<f32> = vec![-0.20f32; total];
        let mut m = DcOffsetMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let linked = m.current();
        assert!(
            (linked - -0.20).abs() < 1e-5,
            "channel-link should pick the larger |·| with sign intact, got {linked}"
        );
        // Per-channel readout should expose both.
        let per = m.per_channel();
        assert_eq!(per.len(), 2);
        assert!((per[0] - 0.05).abs() < 1e-5);
        assert!((per[1] - -0.20).abs() < 1e-5);
    }

    #[test]
    fn stereo_equal_and_opposite_biases_do_not_cancel() {
        // Left channel: +0.1 DC; right channel: -0.1 DC.
        // The link picks |·| = 0.1 from one channel (sign of that
        // channel) rather than cancelling to zero — each channel's
        // bias still reaches its own speaker.
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 4;
        let left: Vec<f32> = vec![0.1f32; total];
        let right: Vec<f32> = vec![-0.1f32; total];
        let mut m = DcOffsetMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        let linked = m.current();
        assert!(
            (linked.abs() - 0.1).abs() < 1e-5,
            "equal-and-opposite biases should NOT cancel; got |{linked}|"
        );
    }

    #[test]
    fn streaming_continuity_one_call_equals_two_halves() {
        // Splitting one process() into two halves on the same input
        // must give the identical mean since the algorithm is
        // per-sample and stateless across frame boundaries.
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 4;
        let samples: Vec<f32> = (0..total)
            .map(|i| 0.2 + 0.05 * (i as f32 * 0.013).sin())
            .collect();

        let mut m1 = DcOffsetMeter::with_window_ms(window_ms);
        m1.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();

        let mut m2 = DcOffsetMeter::with_window_ms(window_ms);
        let half = total / 2;
        m2.process(&make_f32_mono(&samples[..half]), f32_mono(fs))
            .unwrap();
        m2.process(&make_f32_mono(&samples[half..]), f32_mono(fs))
            .unwrap();

        let d = (m1.current() - m2.current()).abs();
        assert!(
            d < 1e-5,
            "streamed-in-halves mean ({}) should match one-call mean ({}), Δ = {d}",
            m2.current(),
            m1.current()
        );
        assert_eq!(m1.samples_seen(), m2.samples_seen());
    }

    #[test]
    fn periodic_rebuild_bounds_f64_drift_on_long_stream() {
        // 200 windows of a constant bias; the rebuild path should
        // keep the reading within f32 precision of the truth.
        let fs = 48_000u32;
        let nwin = 480usize; // 10 ms
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 200;
        let samples = vec![0.12345_f32; total];
        let mut m = DcOffsetMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let mean = m.current();
        // Tighter than the per-sample test because the rebuild keeps
        // the running sum honest.
        assert!(
            (mean - 0.12345).abs() < 1e-5,
            "long-stream constant bias should still read 0.12345 within f32 ε, got {mean}"
        );
    }

    #[test]
    fn window_clamped_at_construction() {
        // Out-of-range window_ms is clamped to [0.1, 10_000].
        let m = DcOffsetMeter::with_window_ms(-50.0);
        assert!((m.window_ms() - 0.1).abs() < 1e-6);
        let m = DcOffsetMeter::with_window_ms(1_000_000.0);
        assert!((m.window_ms() - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn window_samples_resolved_at_first_process() {
        // 400 ms @ 48 kHz = 19200 samples; @ 96 kHz = 38400; @ 16 kHz = 6400.
        let fs_cases = [(48_000u32, 19_200usize), (96_000, 38_400), (16_000, 6_400)];
        for (fs, expected) in fs_cases {
            let mut m = DcOffsetMeter::with_window_ms(400.0);
            let dummy = vec![0.0f32; 16];
            m.process(&make_f32_mono(&dummy), f32_mono(fs)).unwrap();
            assert_eq!(
                m.window_samples(),
                expected,
                "expected {expected} samples for 400 ms @ {fs} Hz"
            );
        }
    }

    #[test]
    fn window_resizes_on_sample_rate_change() {
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let dummy = vec![0.0f32; 16];
        m.process(&make_f32_mono(&dummy), f32_mono(48_000)).unwrap();
        let n48 = m.window_samples();
        m.process(&make_f32_mono(&dummy), f32_mono(96_000)).unwrap();
        let n96 = m.window_samples();
        assert_eq!(n48, 480);
        assert_eq!(n96, 960);
    }

    #[test]
    fn db_form_matches_log_of_abs_linear() {
        // 20·log10(|mean|) — spot-check at 0.5 (≈ -6.02 dB).
        let fs = 48_000u32;
        let mut m = DcOffsetMeter::with_window_ms(10.0);
        let samples = vec![0.5f32; 2400];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let db = m.current_db();
        assert!(
            (db - (-6.020_6)).abs() < 0.01,
            "0.5 linear → -6.02 dB, got {db}"
        );
    }
}
