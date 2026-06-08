//! Zero-crossing rate meter — pass-through observer reporting how
//! often the signal crosses zero per unit time over a sliding
//! rectangular window.
//!
//! The zero-crossing rate (`ZCR`) is the textbook scalar that counts
//! the number of times `sign(x[n]) != sign(x[n-1])` over a window of
//! `N` samples, expressed either as a raw per-sample fraction
//! `crossings / N`  ∈ `[0, 1]` or — more usefully on an audio meter
//! — in crossings per second (Hz) as `crossings · fs / N`.  It is a
//! cheap proxy for the spectral centroid (a tone whose frequency is
//! `f_0` produces `2·f_0` zero-crossings per second; a broadband
//! noise process produces a ZCR proportional to its high-frequency
//! energy) and is used widely in:
//!
//! * **Voiced / unvoiced classification in speech** — voiced phonemes
//!   (vowels, nasals) sit at low ZCR (≤ 1500 Hz typical); unvoiced
//!   fricatives ('s', 'f', 'sh') push the ZCR up into the multiple
//!   kHz range.
//! * **Tone pitch proxy** for clean monophonic tones — a sine at
//!   `f_0` has ZCR `≈ 2 · f_0`, so a 1 kHz sine reads `2000`.
//! * **Percussion vs harmonic separation** — a sustained pitched
//!   instrument has a stable low-to-medium ZCR; a noise burst /
//!   transient produces a transient ZCR excursion towards `fs / 2`.
//! * **Silence / noise-floor gating** — when the input is bit-exact
//!   zero the ZCR collapses to `0` (no sign changes); a noise floor
//!   well above zero gives a high but stable ZCR.
//!
//! Within this crate's observation family, the zero-crossing meter
//! sits orthogonal to:
//! * [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
//!   — reports peak-to-RMS ratio, says nothing about how often the
//!   signal crosses zero.
//! * [`TruePeakDetector`](crate::true_peak_detector::TruePeakDetector)
//!   — reports the inter-sample maximum amplitude, not its frequency.
//! * [`EnvelopeFollower`](crate::envelope_follower::EnvelopeFollower)
//!   — smoothed peak / RMS envelope, no crossing-rate information.
//! * [`LoudnessITU`](crate::loudness::LoudnessITU) — K-weighted
//!   integrated loudness; insensitive to the per-sample sign sequence.
//! * [`SilenceDetector`](crate::silence_detector::SilenceDetector) —
//!   single binary above/below RMS-threshold flag.
//!
//! # Algorithm
//!
//! Per channel the meter keeps a ring of `window_samples` boolean
//! "crossing flags" plus a one-sample latch holding the most recent
//! previously-seen sample.  For every incoming sample `x[t]`:
//!
//! 1. Form the pair `(prev, x[t])` where `prev` is the sample that
//!    arrived in the previous call (the latch).
//! 2. Compute `crossed = sign(prev) != sign(x[t])`.
//! 3. If the flag-ring is full, the flag about to be overwritten is
//!    the pair that just rolled out of the window — subtract it
//!    from the running `count`.
//! 4. Write the new flag into the ring and add it to `count`.
//! 5. Update the latch to `x[t]`.
//!
//! That gives an `O(1)` per-sample running count of crossings inside
//! a window of `N` adjacent-sample pairs (equivalently: the last
//! `N + 1` samples seen).  Sign is reduced to `{-1, +1}` with the
//! convention `sign(0.0) = +1` so a run of bit-exact zeros doesn't
//! manufacture phantom crossings.
//!
//! Channel-link is by `max`: a transiently noisy channel of a
//! split stereo bed isn't masked by a quieter average on the other.
//! (The crate's convention across all observation filters.)
//!
//! # Warm-up
//!
//! Until the window is full the readouts return `0.0` for the linear
//! form and `f32::NAN` for the rate — the meter doesn't have enough
//! history to bound the crossing count yet.  The [`samples_seen`] accessor
//! exposes the count to make this explicit so callers can branch on
//! "not yet ready".
//!
//! # Parameters
//!
//! * `window_ms` — measurement window in milliseconds (default `25`,
//!   the canonical frame length for short-time speech classification).
//!   Internally converted to sample count `N = round(window_ms · fs /
//!   1000)`, clamped to `[1, MAX_WINDOW_SAMPLES = 192_000]` (4 s at
//!   48 kHz).
//!
//! # API surface
//!
//! The filter is observation-only — [`AudioFilter::process`] returns
//! the input frame unchanged.  Consumers call:
//!
//! * [`ZeroCrossingRateMeter::current_rate_hz`] — crossings per second
//!   (Hz) over the current window. `NAN` until the window is full.
//! * [`ZeroCrossingRateMeter::current_fraction`] — crossings per
//!   sample, `∈ [0, 1]`. `0.0` until the window is full.
//! * [`ZeroCrossingRateMeter::current_count`] — raw crossing count
//!   inside the active window. `0` until the window is full.
//! * [`ZeroCrossingRateMeter::max_rate_hz`] /
//!   [`ZeroCrossingRateMeter::reset_max`] — running max rate over
//!   the meter's history.
//! * [`ZeroCrossingRateMeter::samples_seen`] — total input samples
//!   processed per channel since construction or last [`reset`].
//! * [`ZeroCrossingRateMeter::reset`] — wipe all per-channel state.
//!
//! [`reset`]: ZeroCrossingRateMeter::reset
//! [`samples_seen`]: ZeroCrossingRateMeter::samples_seen

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Default measurement window in milliseconds (25 ms, the canonical
/// short-time speech-analysis frame length).
pub const ZCR_DEFAULT_WINDOW_MS: f32 = 25.0;

/// Maximum measurement window in samples (`= 192_000`, i.e. 4 s at
/// 48 kHz or 2 s at 96 kHz). Defends against pathological allocations
/// without rejecting any realistic broadcast / mastering window.
pub const ZCR_MAX_WINDOW_SAMPLES: usize = 192_000;

/// Streaming zero-crossing-rate meter.
#[derive(Debug, Clone)]
pub struct ZeroCrossingRateMeter {
    window_ms: f32,
    state: Option<MeterState>,
}

#[derive(Debug, Clone)]
struct MeterState {
    sample_rate: u32,
    /// Window length in samples / adjacent-pair flags (`N`), derived
    /// from `window_ms` and `sample_rate`, clamped to
    /// `[1, ZCR_MAX_WINDOW_SAMPLES]`.
    window_samples: usize,
    /// Per-channel ring of length `N` holding the most recent `N`
    /// crossing flags (`true` if `sign(x[t-1]) != sign(x[t])`).
    flag_rings: Vec<Vec<bool>>,
    /// Per-channel latch holding the previously-arrived sample, used
    /// to form the next `(prev, x[t])` pair. `None` until the very
    /// first sample arrives on a given channel.
    prev_samples: Vec<Option<f32>>,
    /// Per-channel running count of `true` flags inside the active
    /// window.
    counts: Vec<u32>,
    /// Shared write cursor into `flag_rings` (`pairs_seen` modulo `N`).
    write: usize,
    /// Monotonically-increasing per-channel sample count.
    samples_seen: u64,
    /// Monotonically-increasing per-channel adjacent-pair count
    /// (`= samples_seen - 1` once at least one pair has been formed).
    pairs_seen: u64,
    /// Channel-linked crossing count at the close of the most recent
    /// `process()` call. `0` until the window is full.
    last_count: u32,
    /// Running max linear fraction (`count / N`) since construction
    /// or last [`reset_max`](ZeroCrossingRateMeter::reset_max).
    max_fraction: f32,
}

impl ZeroCrossingRateMeter {
    /// New meter with the default 25 ms window.
    pub fn new() -> Self {
        Self::with_window_ms(ZCR_DEFAULT_WINDOW_MS)
    }

    /// New meter with explicit window length in milliseconds. The
    /// argument is clamped to `[0.1, 10_000]` ms at construction; the
    /// sample-count form is derived from the input stream's
    /// `sample_rate` at first frame and additionally clamped to
    /// `[1, ZCR_MAX_WINDOW_SAMPLES]`.
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
    /// last [`reset`](Self::reset). Before this reaches
    /// `window_samples + 1` the readouts return `NAN` / `0.0` /
    /// `0` — the window isn't full yet.
    pub fn samples_seen(&self) -> u64 {
        self.state.as_ref().map(|s| s.samples_seen).unwrap_or(0)
    }

    /// Crossing rate in Hertz (crossings per second) at the end of
    /// the most recent `process()` call. Returns
    /// [`f32::NAN`] before the flag-ring's first fill (`pairs_seen <
    /// window_samples`).
    pub fn current_rate_hz(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return f32::NAN;
        };
        if s.pairs_seen < s.window_samples as u64 {
            return f32::NAN;
        }
        s.last_count as f32 * s.sample_rate as f32 / s.window_samples as f32
    }

    /// Crossing fraction `count / N` at the end of the most recent
    /// `process()` call. `∈ [0, 1]`. Returns `0.0` before the
    /// flag-ring's first fill.
    pub fn current_fraction(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return 0.0;
        };
        if s.pairs_seen < s.window_samples as u64 {
            return 0.0;
        }
        s.last_count as f32 / s.window_samples as f32
    }

    /// Raw channel-linked crossing count inside the active window
    /// at the end of the most recent `process()` call. `0` before
    /// the flag-ring's first fill.
    pub fn current_count(&self) -> u32 {
        let Some(s) = self.state.as_ref() else {
            return 0;
        };
        if s.pairs_seen < s.window_samples as u64 {
            return 0;
        }
        s.last_count
    }

    /// Running max crossing rate in Hz since construction or last
    /// [`reset_max`](Self::reset_max). Returns [`f32::NAN`] before
    /// the flag-ring is first full.
    pub fn max_rate_hz(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return f32::NAN;
        };
        if s.pairs_seen < s.window_samples as u64 {
            return f32::NAN;
        }
        s.max_fraction * s.sample_rate as f32
    }

    /// Wipe all per-channel state (rings, counts, counters). The
    /// configured `window_ms` survives; the resolved `window_samples`
    /// is re-derived on the next `process()` call.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Clear only the running max, leaving the window contents and
    /// counters intact.
    pub fn reset_max(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.max_fraction = 0.0;
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.flag_rings.len() != channels,
            None => true,
        };
        if rebuild {
            let n = ((self.window_ms as f64 * sample_rate as f64 / 1000.0).round() as usize).max(1);
            let window_samples = n.min(ZCR_MAX_WINDOW_SAMPLES);
            self.state = Some(MeterState {
                sample_rate,
                window_samples,
                flag_rings: vec![vec![false; window_samples]; channels],
                prev_samples: vec![None; channels],
                counts: vec![0; channels],
                write: 0,
                samples_seen: 0,
                pairs_seen: 0,
                last_count: 0,
                max_fraction: 0.0,
            });
        }
    }
}

impl Default for ZeroCrossingRateMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// IEEE-754-style sign convention reduced to two values. Zero is
/// counted as `+1` so a run of bit-exact zeros doesn't manufacture
/// phantom crossings. (`f32::signum` returns `-0.0` for `-0.0` which
/// would be exactly that bug.)
#[inline]
fn sign_pn(x: f32) -> i8 {
    if x < 0.0 {
        -1
    } else {
        1
    }
}

impl AudioFilter for ZeroCrossingRateMeter {
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

                // Form the (prev, x) pair only once we have a real
                // prev. On the very first sample of a channel there
                // isn't one yet — latch x and continue.
                let prev = state.prev_samples[ch_idx];
                if let Some(prev_x) = prev {
                    let crossed = sign_pn(prev_x) != sign_pn(x);
                    let flags = &mut state.flag_rings[ch_idx];
                    // Subtract the flag about to be overwritten if the
                    // flag ring is already full.
                    if state.pairs_seen >= nwin as u64 {
                        let old = flags[state.write];
                        if old {
                            state.counts[ch_idx] = state.counts[ch_idx].saturating_sub(1);
                        }
                    }
                    flags[state.write] = crossed;
                    if crossed {
                        state.counts[ch_idx] = state.counts[ch_idx].saturating_add(1);
                    }
                }
                state.prev_samples[ch_idx] = Some(x);
            }

            // Cursor + pair-count advance once per sample (only after
            // a real pair was formed, i.e. samples_seen >= 1 going in).
            if state.samples_seen >= 1 {
                state.write = (state.write + 1) % nwin;
                state.pairs_seen = state.pairs_seen.saturating_add(1);
            }
            state.samples_seen = state.samples_seen.saturating_add(1);
        }

        // Channel-link by max once the flag-ring has filled at least once.
        if state.pairs_seen >= nwin as u64 {
            let max_c = state.counts.iter().copied().max().unwrap_or(0);
            state.last_count = max_c;
            let frac = max_c as f32 / nwin as f32;
            if frac > state.max_fraction {
                state.max_fraction = frac;
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
        let mut m = ZeroCrossingRateMeter::new();
        let outs = m.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].samples, frame.samples);
        assert_eq!(outs[0].data, frame.data);
    }

    #[test]
    fn before_window_full_returns_nan_and_zero() {
        // 50 ms window @ 48 kHz = 2400 samples; feed only 1024 first.
        let fs = 48_000u32;
        let mut m = ZeroCrossingRateMeter::with_window_ms(50.0);
        // Use a non-zero DC offset so samples are present but won't
        // contribute any crossings.
        let samples = vec![0.25f32; 1024];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert!(m.current_rate_hz().is_nan());
        assert_eq!(m.current_fraction(), 0.0);
        assert_eq!(m.current_count(), 0);
    }

    #[test]
    fn constant_dc_yields_zero_crossings() {
        // A constant DC signal has no sign changes; ZCR is zero.
        let fs = 48_000u32;
        let mut m = ZeroCrossingRateMeter::with_window_ms(10.0);
        // Several windows worth: 10 ms @ 48 kHz = 480 samples → 2000.
        let samples = vec![0.5f32; 2000];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current_count(), 0);
        assert_eq!(m.current_fraction(), 0.0);
        assert_eq!(m.current_rate_hz(), 0.0);
    }

    #[test]
    fn silence_yields_zero_crossings() {
        // Bit-exact zeros: signum convention says +1 for 0.0 so no
        // crossings should be counted.
        let fs = 48_000u32;
        let mut m = ZeroCrossingRateMeter::with_window_ms(10.0);
        let samples = vec![0.0f32; 2000];
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current_count(), 0);
        assert_eq!(m.current_rate_hz(), 0.0);
    }

    #[test]
    fn alternating_signal_saturates_zcr_to_one() {
        // x[n] = (-1)^n alternates sign every sample, so every
        // adjacent pair is a crossing — fraction = (N - 0) / N → 1.
        // (Within the active window of N samples there are N
        // adjacent-pair comparisons because we also compare against
        // the sample immediately preceding the window.)
        let fs = 48_000u32;
        let nwin = 480usize; // 10 ms
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
        // Provide several windows of alternating ±1.
        let total = nwin * 4;
        let samples: Vec<f32> = (0..total)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.window_samples(), nwin);
        // Every adjacent pair within the window is a crossing.
        assert_eq!(m.current_count() as usize, nwin);
        assert!((m.current_fraction() - 1.0).abs() < 1e-6);
        // Rate is N * fs / N = fs.
        assert!((m.current_rate_hz() - fs as f32).abs() < 1.0);
    }

    #[test]
    fn pure_sine_zcr_matches_twice_frequency() {
        // A sine at f0 produces 2*f0 zero-crossings per second
        // (one positive-going and one negative-going per period).
        // Choose a window holding an integer number of periods so
        // the count is exact: 1 kHz at 48 kHz = 48 samples/period,
        // 480-sample window = exactly 10 periods → exactly 20
        // crossings inside the window.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 6;
        let samples: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        // 10 periods → 20 sign changes. Allow ±1 for the very first
        // sample (sin(0) = 0 → sign +1; if the boundary alignment
        // contributes one extra crossing at the window edge).
        let count = m.current_count();
        assert!(
            (count as i32 - 20).abs() <= 2,
            "sine ZCR count {count} should be ≈ 20 for a 480-sample window of 1 kHz @ 48 kHz"
        );
        // Rate ≈ 2 · f0 = 2000 Hz.
        let rate = m.current_rate_hz();
        assert!(
            (rate - 2.0 * f).abs() < 200.0,
            "sine rate {rate} Hz should be ≈ 2000 Hz"
        );
    }

    #[test]
    fn sine_rate_scales_with_frequency() {
        // Doubling f0 should ~double the reported rate.
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 6;
        let mut rates = Vec::new();
        for &f in &[500.0f32, 1_000.0, 2_000.0, 4_000.0] {
            let samples: Vec<f32> = (0..total)
                .map(|i| {
                    let t = i as f32 / fs as f32;
                    (2.0 * std::f32::consts::PI * f * t).sin()
                })
                .collect();
            let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
            m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
            rates.push(m.current_rate_hz());
        }
        // Each successive doubling produces ~2× rate (≤ 25 %
        // tolerance to absorb the integer crossing count's
        // quantisation on a finite window).
        for i in 0..rates.len() - 1 {
            let ratio = rates[i + 1] / rates[i];
            assert!(
                (ratio - 2.0).abs() < 0.5,
                "doubling f0 should ~double ZCR; got ratio {ratio} at index {i}"
            );
        }
    }

    #[test]
    fn rate_independent_of_amplitude() {
        // The crossing count depends only on the sign sequence, so
        // amplitude scaling should leave the readout unchanged.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 6;
        let mut rates = Vec::new();
        for &amp in &[0.001f32, 0.1, 1.0] {
            let samples: Vec<f32> = (0..total)
                .map(|i| {
                    let t = i as f32 / fs as f32;
                    amp * (2.0 * std::f32::consts::PI * f * t).sin()
                })
                .collect();
            let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
            m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
            rates.push(m.current_count());
        }
        // All amplitudes should give the identical crossing count.
        assert_eq!(rates[0], rates[1]);
        assert_eq!(rates[1], rates[2]);
    }

    #[test]
    fn reset_wipes_state() {
        let fs = 48_000u32;
        let mut m = ZeroCrossingRateMeter::with_window_ms(10.0);
        let samples: Vec<f32> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert!(m.current_count() > 0);
        m.reset();
        assert_eq!(m.samples_seen(), 0);
        assert_eq!(m.window_samples(), 0);
        assert!(m.current_rate_hz().is_nan());
    }

    #[test]
    fn reset_max_clears_running_peak_only() {
        let fs = 48_000u32;
        let mut m = ZeroCrossingRateMeter::with_window_ms(10.0);
        let samples: Vec<f32> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let peak = m.max_rate_hz();
        assert!(peak > 0.0);
        m.reset_max();
        // Feed silence so the current rate drops to 0; max also
        // resets toward zero on the way through.
        let silence = vec![0.5f32; 2000];
        m.process(&make_f32_mono(&silence), f32_mono(fs)).unwrap();
        // The current count is 0, so max_fraction stayed at 0 since
        // we wiped it.
        assert_eq!(m.max_rate_hz(), 0.0);
    }

    #[test]
    fn stereo_channel_link_picks_max() {
        // Left channel: alternating ±1 (saturates ZCR to 1.0).
        // Right channel: DC (no crossings).
        // Channel-link by max should report the left channel's rate.
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 4;
        let left: Vec<f32> = (0..total)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let right: Vec<f32> = vec![0.25f32; total];
        let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
        m.process(&make_f32p_stereo(&left, &right), f32p_stereo(fs))
            .unwrap();
        // Linked count is the left-channel count (≈ N), not the right's 0.
        assert!(m.current_count() as usize >= nwin - 1);
    }

    #[test]
    fn streaming_continuity_one_call_equals_two_halves() {
        // Splitting one process() into two halves on the same input
        // must give bit-identical state (same crossing count) since
        // the algorithm is per-sample and stateless across frames.
        let fs = 48_000u32;
        let f = 1_000.0f32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 3;
        let samples: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();

        let mut m1 = ZeroCrossingRateMeter::with_window_ms(window_ms);
        m1.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();

        let mut m2 = ZeroCrossingRateMeter::with_window_ms(window_ms);
        let half = total / 2;
        m2.process(&make_f32_mono(&samples[..half]), f32_mono(fs))
            .unwrap();
        m2.process(&make_f32_mono(&samples[half..]), f32_mono(fs))
            .unwrap();

        assert_eq!(m1.current_count(), m2.current_count());
        assert_eq!(m1.samples_seen(), m2.samples_seen());
    }

    #[test]
    fn window_clamped_at_construction() {
        // Out-of-range window_ms is clamped to [0.1, 10_000].
        let m = ZeroCrossingRateMeter::with_window_ms(-50.0);
        assert!((m.window_ms() - 0.1).abs() < 1e-6);
        let m = ZeroCrossingRateMeter::with_window_ms(1_000_000.0);
        assert!((m.window_ms() - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn window_samples_resolved_at_first_process() {
        // 25 ms @ 48 kHz = 1200 samples; @ 96 kHz = 2400; @ 16 kHz = 400.
        let fs_cases = [(48_000u32, 1200usize), (96_000, 2400), (16_000, 400)];
        for (fs, expected) in fs_cases {
            let mut m = ZeroCrossingRateMeter::with_window_ms(25.0);
            let dummy = vec![0.5f32; 16];
            m.process(&make_f32_mono(&dummy), f32_mono(fs)).unwrap();
            assert_eq!(
                m.window_samples(),
                expected,
                "expected {expected} samples for 25 ms @ {fs} Hz"
            );
        }
    }

    #[test]
    fn window_resizes_on_sample_rate_change() {
        let mut m = ZeroCrossingRateMeter::with_window_ms(10.0);
        let dummy = vec![0.5f32; 16];
        m.process(&make_f32_mono(&dummy), f32_mono(48_000)).unwrap();
        let n48 = m.window_samples();
        // Stream-rate change → state rebuild at the new sample rate.
        m.process(&make_f32_mono(&dummy), f32_mono(96_000)).unwrap();
        let n96 = m.window_samples();
        assert_eq!(n48, 480);
        assert_eq!(n96, 960);
    }

    #[test]
    fn fraction_in_unit_interval() {
        // Random-ish run shouldn't ever produce a fraction > 1.
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 4;
        // Deterministic xorshift32 to avoid pulling rand.
        let mut s: u32 = 0xdead_beef;
        let samples: Vec<f32> = (0..total)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let u = (s >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
                2.0 * u - 1.0 // [-1, 1)
            })
            .collect();
        let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        let frac = m.current_fraction();
        assert!(
            (0.0..=1.0).contains(&frac),
            "ZCR fraction {frac} outside [0, 1]"
        );
        let rate = m.current_rate_hz();
        assert!(
            (0.0..=fs as f32).contains(&rate),
            "ZCR rate {rate} outside [0, fs]"
        );
    }

    #[test]
    fn count_is_window_linked_after_long_stream() {
        // Feed many windows of an alternating-sign signal; the
        // running count must remain exactly N (no drift, no
        // overflow, no double-counting).
        let fs = 48_000u32;
        let nwin = 480usize;
        let window_ms = (nwin as f32 / fs as f32) * 1000.0;
        let total = nwin * 20; // 20 windows
        let samples: Vec<f32> = (0..total)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let mut m = ZeroCrossingRateMeter::with_window_ms(window_ms);
        m.process(&make_f32_mono(&samples), f32_mono(fs)).unwrap();
        assert_eq!(m.current_count() as usize, nwin);
    }
}
