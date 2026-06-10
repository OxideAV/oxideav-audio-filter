//! Stereo-balance meter — pass-through observer reporting the left /
//! right *energy* balance of a stereo signal over a sliding
//! rectangular window.
//!
//! The balance is the textbook normalised level-difference scalar
//!
//! ```text
//! B = (R_rms - L_rms) / (R_rms + L_rms)    ∈ [-1, +1]
//! ```
//!
//! where `L_rms` / `R_rms` are the windowed root-mean-square levels of
//! the two channels. `B = 0` for a centred image (both channels carry
//! equal energy — mono content panned dead-centre, or a symmetric
//! stereo bed); `B = -1` when all the energy sits on the left
//! (`R_rms = 0`, hard-left pan or a dead right channel); `B = +1` when
//! all the energy sits on the right (`L_rms = 0`). A right channel
//! twice as loud as the left reads `+1/3`; a left channel twice as
//! loud as the right reads `-1/3`.
//!
//! This is the *level* complement to
//! [`StereoCorrelationMeter`](crate::stereo_correlation_meter::StereoCorrelationMeter):
//! correlation reports the inter-channel *phase* relationship (are L
//! and R the same waveform?) and is mean-centred and scale-invariant,
//! so it is blind to a level imbalance — two perfectly correlated
//! channels at `+12 dB` / `-12 dB` still read `ρ = +1`. Balance
//! reports exactly the dimension correlation throws away: *where the
//! energy sits across the stereo field*. The two meters together pin
//! down both axes of a stereo image (phase + level) and share a time
//! axis when configured with the same window.
//!
//! A persistent non-zero balance flags an accidental pan offset, a
//! channel-trim mismatch in the capture chain, one dead or
//! intermittent channel, or a mono source mis-routed to a single leg
//! of a stereo bus — each of which leaves the apparent image pulled
//! off centre.
//!
//! Within the crate's observation family the balance meter sits
//! orthogonal to:
//! * [`StereoCorrelationMeter`](crate::stereo_correlation_meter::StereoCorrelationMeter)
//!   — inter-channel phase (Pearson `ρ`); blind to level imbalance.
//! * [`DcOffsetMeter`](crate::dc_offset_meter::DcOffsetMeter) — signed
//!   per-channel mean; a balance shift is a *level* asymmetry that
//!   carries no DC.
//! * [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
//!   — peak-to-RMS ratio; a single-channel scalar that says nothing
//!   about how energy splits across the pair.
//! * [`LoudnessITU`](crate::loudness::LoudnessITU) — K-weighted
//!   *summed* loudness across channels; a left/right swap leaves the
//!   integrated loudness unchanged but flips the balance sign.
//!
//! # Algorithm
//!
//! Per channel the meter keeps a ring buffer of the most recent `N`
//! samples plus a running sum-of-squares `Q = Σ x²` over the active
//! window. For every incoming `(left, right)` pair:
//!
//! 1. Look up the squared samples about to be overwritten.
//! 2. Once the window is full, update incrementally:
//!    `Q ← Q + x_new² - x_old²`. Before it fills, just accumulate.
//! 3. Write the new samples into the rings, advance the write cursor.
//!
//! Windowed RMS is then `rms = sqrt(Q / N)` per channel, and the
//! balance follows in closed form. Cost is `O(1)` per sample.
//!
//! To bound `f64` round-off drift on long streams, both `Q` sums are
//! periodically rebuilt from the ring contents once per full window —
//! the same rebuild cadence used by
//! [`CrestFactorMeter`](crate::crest_factor_meter::CrestFactorMeter)
//! and [`StereoCorrelationMeter`](crate::stereo_correlation_meter::StereoCorrelationMeter).
//!
//! # Stereo only
//!
//! The meter only updates on stereo (`channels == 2`) input. Mono and
//! multichannel (channel count not equal to two) layouts pass through
//! unchanged with the meter state untouched, so the readout keeps its
//! previous value.
//!
//! # Warm-up
//!
//! Until the window is full the readout returns `0.0` (the neutral
//! centred reading) — the meter does not yet have a full window's
//! worth of samples. [`samples_seen`](StereoBalanceMeter::samples_seen)
//! exposes the count.
//!
//! # Bit-exact silence
//!
//! When both channels are bit-exact silent over the window
//! (`L_rms + R_rms == 0`) the balance is undefined; the meter reports
//! the neutral `0.0` (centred).
//!
//! # Parameters
//!
//! * `window_ms` — measurement window in milliseconds (default `400`,
//!   matching the EBU R128 short-term loudness window that the other
//!   R128-aligned meters default to, so they share a time axis on a
//!   display). Internally converted to `N = round(window_ms · fs /
//!   1000)`, clamped to `[1, SBM_MAX_WINDOW_SAMPLES = 192_000]`.
//!
//! # API surface
//!
//! Observation-only — [`AudioFilter::process`] returns the input frame
//! unchanged. Consumers call:
//!
//! * [`StereoBalanceMeter::current`] — balance `∈ [-1, +1]` over the
//!   current window. `0.0` until the window is full.
//! * [`StereoBalanceMeter::rms_left`] /
//!   [`StereoBalanceMeter::rms_right`] — per-channel windowed RMS.
//! * [`StereoBalanceMeter::max_abs`] /
//!   [`StereoBalanceMeter::reset_max`] — running max `|balance|` over
//!   the meter's history.
//! * [`StereoBalanceMeter::samples_seen`] — total input samples
//!   processed per channel since construction or last [`reset`].
//! * [`StereoBalanceMeter::reset`] — wipe all state.
//!
//! [`reset`]: StereoBalanceMeter::reset

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Default measurement window in milliseconds (EBU R128 short-term;
/// matches `StereoCorrelationMeter` / `CrestFactorMeter` defaults).
pub const SBM_DEFAULT_WINDOW_MS: f32 = 400.0;

/// Maximum measurement window in samples (`= 192_000`, i.e. 4 s at
/// 48 kHz or 2 s at 96 kHz).
pub const SBM_MAX_WINDOW_SAMPLES: usize = 192_000;

/// Streaming stereo-balance meter.
#[derive(Debug, Clone)]
pub struct StereoBalanceMeter {
    window_ms: f32,
    state: Option<MeterState>,
}

#[derive(Debug, Clone)]
struct MeterState {
    sample_rate: u32,
    /// Window length in samples (`N`).
    window_samples: usize,
    /// Left / right rings of the most recent `N` samples.
    ring_l: Vec<f32>,
    ring_r: Vec<f32>,
    /// Running sums-of-squares over the active window (`f64` to bound
    /// round-off on long streams).
    qx: f64,
    qy: f64,
    /// Shared write cursor.
    write: usize,
    /// Monotonically-increasing per-channel sample count.
    samples_seen: u64,
    /// Balance at the close of the most recent `process()` call. `0.0`
    /// until the window is first full.
    last: f32,
    /// Per-channel windowed RMS at the close of the most recent
    /// `process()` call.
    last_rms_l: f32,
    last_rms_r: f32,
    /// Running max `|balance|` since construction or last `reset_max`.
    max_abs: f32,
}

impl StereoBalanceMeter {
    /// New meter with the default 400 ms window.
    pub fn new() -> Self {
        Self::with_window_ms(SBM_DEFAULT_WINDOW_MS)
    }

    /// New meter with explicit window length in milliseconds. Clamped
    /// to `[0.1, 10_000]` ms at construction; the sample-count form is
    /// derived from the input stream's `sample_rate` at first frame
    /// and additionally clamped to `[1, SBM_MAX_WINDOW_SAMPLES]`.
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
    /// call). `0` before the meter has seen its first stream.
    pub fn window_samples(&self) -> usize {
        self.state.as_ref().map(|s| s.window_samples).unwrap_or(0)
    }

    /// Samples observed per channel since construction / last
    /// [`reset`](Self::reset).
    pub fn samples_seen(&self) -> u64 {
        self.state.as_ref().map(|s| s.samples_seen).unwrap_or(0)
    }

    /// Balance `∈ [-1, +1]` at the end of the most recent `process()`
    /// call: `-1` = all energy on the left, `0` = centred, `+1` = all
    /// energy on the right. Returns `0.0` before the window is full.
    pub fn current(&self) -> f32 {
        let Some(s) = self.state.as_ref() else {
            return 0.0;
        };
        if s.samples_seen < s.window_samples as u64 {
            return 0.0;
        }
        s.last
    }

    /// Windowed RMS of the left channel at the end of the most recent
    /// `process()` call. `0.0` before the window is full.
    pub fn rms_left(&self) -> f32 {
        self.ready().map(|s| s.last_rms_l).unwrap_or(0.0)
    }

    /// Windowed RMS of the right channel at the end of the most recent
    /// `process()` call. `0.0` before the window is full.
    pub fn rms_right(&self) -> f32 {
        self.ready().map(|s| s.last_rms_r).unwrap_or(0.0)
    }

    /// Running max `|balance|` since construction / last
    /// [`reset_max`](Self::reset_max). `0.0` before the window is full.
    pub fn max_abs(&self) -> f32 {
        self.state.as_ref().map(|s| s.max_abs).unwrap_or(0.0)
    }

    /// Wipe all state. The configured `window_ms` survives; the
    /// resolved `window_samples` is re-derived on the next `process()`.
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

    fn ready(&self) -> Option<&MeterState> {
        self.state
            .as_ref()
            .filter(|s| s.samples_seen >= s.window_samples as u64)
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        // The meter only meaningfully tracks stereo, but it must still
        // allocate ring storage sized to the resolved window so that a
        // later stereo frame at the same rate finds state ready. The
        // channel count itself doesn't change the allocation (the
        // rings are always L/R), so it isn't part of the rebuild test.
        let _ = channels;
        if rebuild {
            let n = ((self.window_ms as f64 * sample_rate as f64 / 1000.0).round() as usize).max(1);
            let window_samples = n.min(SBM_MAX_WINDOW_SAMPLES);
            self.state = Some(MeterState {
                sample_rate,
                window_samples,
                ring_l: vec![0.0; window_samples],
                ring_r: vec![0.0; window_samples],
                qx: 0.0,
                qy: 0.0,
                write: 0,
                samples_seen: 0,
                last: 0.0,
                last_rms_l: 0.0,
                last_rms_r: 0.0,
                max_abs: 0.0,
            });
        }
    }
}

impl Default for StereoBalanceMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for StereoBalanceMeter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate, params.channels as usize);
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n_in = channels.first().map(|c| c.len()).unwrap_or(0);

        // Only run the balance update for stereo input. Mono and
        // >2-channel layouts pass through unchanged with the meter
        // state untouched, so the readout keeps its previous value.
        if params.channels == 2 {
            let state = self.state.as_mut().expect("state ensured above");
            let nwin = state.window_samples;
            let left = &channels[0];
            let right = &channels[1];

            for i in 0..n_in {
                let xf = left[i] as f64;
                let yf = right[i] as f64;
                let x_old = state.ring_l[state.write] as f64;
                let y_old = state.ring_r[state.write] as f64;
                state.ring_l[state.write] = left[i];
                state.ring_r[state.write] = right[i];

                if state.samples_seen < nwin as u64 {
                    state.qx += xf * xf;
                    state.qy += yf * yf;
                } else {
                    state.qx += xf * xf - x_old * x_old;
                    state.qy += yf * yf - y_old * y_old;
                }

                state.write = (state.write + 1) % nwin;
                state.samples_seen = state.samples_seen.saturating_add(1);

                // Periodic rebuild of both sums from the ring contents
                // once per full window. Bounds f64 round-off drift on
                // long streams at O(1) amortised per sample.
                if state.samples_seen >= nwin as u64 && state.samples_seen % (nwin as u64) == 0 {
                    let mut qx: f64 = 0.0;
                    let mut qy: f64 = 0.0;
                    for k in 0..nwin {
                        let lf = state.ring_l[k] as f64;
                        let rf = state.ring_r[k] as f64;
                        qx += lf * lf;
                        qy += rf * rf;
                    }
                    state.qx = qx;
                    state.qy = qy;
                }
            }

            // Compute the windowed balance at frame close.
            if state.samples_seen >= nwin as u64 {
                let nf = nwin as f64;
                let rms_l = (state.qx / nf).max(0.0).sqrt();
                let rms_r = (state.qy / nf).max(0.0).sqrt();
                let sum = rms_l + rms_r;
                let bal = if sum > 0.0 {
                    ((rms_r - rms_l) / sum).clamp(-1.0, 1.0) as f32
                } else {
                    // Both channels silent over the window — balance
                    // undefined; report the neutral centred reading.
                    0.0
                };
                state.last = bal;
                state.last_rms_l = rms_l as f32;
                state.last_rms_r = rms_r as f32;
                let ba = bal.abs();
                if ba > state.max_abs {
                    state.max_abs = ba;
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

    // Window helper: N samples → window_ms at fs.
    fn win_ms(nwin: usize, fs: u32) -> f32 {
        (nwin as f32 / fs as f32) * 1000.0
    }

    #[test]
    fn pass_through_preserves_audio_bytes() {
        let fs = 48_000u32;
        let l: Vec<f32> = (0..2048).map(|i| 0.4 * (i as f32 * 0.01).sin()).collect();
        let r: Vec<f32> = (0..2048).map(|i| 0.3 * (i as f32 * 0.02).cos()).collect();
        let frame = make_f32p_stereo(&l, &r);
        let mut m = StereoBalanceMeter::new();
        let outs = m.process(&frame, f32p_stereo(fs)).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].samples, frame.samples);
        assert_eq!(outs[0].data, frame.data);
    }

    #[test]
    fn before_window_full_returns_zero() {
        let fs = 48_000u32;
        let mut m = StereoBalanceMeter::with_window_ms(50.0); // 2400 samples
        let l = vec![0.5f32; 1024];
        let r = vec![0.0f32; 1024];
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.max_abs(), 0.0);
    }

    #[test]
    fn equal_channels_read_centred() {
        // Identical L/R energy → balance 0.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.5f32; total];
        let r = vec![0.5f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            m.current().abs() < 1e-5,
            "equal channels should read centred 0, got {}",
            m.current()
        );
    }

    #[test]
    fn hard_left_reads_minus_one() {
        // All energy on the left, right silent → -1.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.6f32; total];
        let r = vec![0.0f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m.current() - -1.0).abs() < 1e-5,
            "hard-left should read -1, got {}",
            m.current()
        );
    }

    #[test]
    fn hard_right_reads_plus_one() {
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.0f32; total];
        let r = vec![0.6f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m.current() - 1.0).abs() < 1e-5,
            "hard-right should read +1, got {}",
            m.current()
        );
    }

    #[test]
    fn right_twice_as_loud_reads_one_third() {
        // R_rms = 2·L_rms → (2-1)/(2+1) = +1/3.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.25f32; total];
        let r = vec![0.5f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m.current() - (1.0 / 3.0)).abs() < 1e-5,
            "R twice L should read +1/3, got {}",
            m.current()
        );
    }

    #[test]
    fn left_twice_as_loud_reads_minus_one_third() {
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.5f32; total];
        let r = vec![0.25f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m.current() - (-1.0 / 3.0)).abs() < 1e-5,
            "L twice R should read -1/3, got {}",
            m.current()
        );
    }

    #[test]
    fn per_channel_rms_matches_constant_level() {
        // Constant level c has RMS = |c|.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.3f32; total];
        let r = vec![0.4f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!((m.rms_left() - 0.3).abs() < 1e-5, "L rms {}", m.rms_left());
        assert!(
            (m.rms_right() - 0.4).abs() < 1e-5,
            "R rms {}",
            m.rms_right()
        );
        // Balance from those RMS values: (0.4-0.3)/(0.7) = 1/7.
        assert!((m.current() - (0.1 / 0.7)).abs() < 1e-5);
    }

    #[test]
    fn silence_reads_neutral_zero() {
        // Both channels bit-exact silent → balance undefined → 0.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.0f32; total];
        let r = vec![0.0f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.rms_left(), 0.0);
        assert_eq!(m.rms_right(), 0.0);
    }

    #[test]
    fn correlated_channels_at_unequal_levels_still_show_balance() {
        // The whole point vs StereoCorrelationMeter: two perfectly
        // correlated channels (ρ = +1) at different levels still
        // register a non-zero balance. L = x, R = 2x → +1/3.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let f = 1_000.0f32;
        let l: Vec<f32> = (0..total)
            .map(|i| {
                let t = i as f32 / fs as f32;
                0.3 * (2.0 * std::f32::consts::PI * f * t).sin()
            })
            .collect();
        let r: Vec<f32> = l.iter().map(|&x| 2.0 * x).collect();
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m.current() - (1.0 / 3.0)).abs() < 1e-3,
            "correlated-but-louder-right should read +1/3, got {}",
            m.current()
        );
    }

    #[test]
    fn sign_flips_on_channel_swap() {
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let a = vec![0.5f32; total];
        let b = vec![0.2f32; total];
        let mut m1 = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m1.process(&make_f32p_stereo(&a, &b), f32p_stereo(fs))
            .unwrap();
        let mut m2 = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m2.process(&make_f32p_stereo(&b, &a), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m1.current() + m2.current()).abs() < 1e-5,
            "swapping channels should flip the balance sign: {} vs {}",
            m1.current(),
            m2.current()
        );
    }

    #[test]
    fn mono_input_passes_through_untouched() {
        // Mono should not update the meter; readout stays at warm-up 0.
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let samples = vec![0.5f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        let frame = make_f32_mono(&samples);
        let outs = m.process(&frame, f32_mono(fs)).unwrap();
        assert_eq!(outs[0].data, frame.data);
        // No stereo update ever ran, so samples_seen stays 0.
        assert_eq!(m.samples_seen(), 0);
        assert_eq!(m.current(), 0.0);
    }

    #[test]
    fn reset_wipes_state() {
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l = vec![0.6f32; total];
        let r = vec![0.0f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!((m.current() - -1.0).abs() < 1e-5);
        m.reset();
        assert_eq!(m.samples_seen(), 0);
        assert_eq!(m.window_samples(), 0);
        assert_eq!(m.current(), 0.0);
        assert_eq!(m.max_abs(), 0.0);
    }

    #[test]
    fn reset_max_clears_running_peak_only() {
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        // Hard-left → |balance| = 1.
        let l = vec![0.6f32; total];
        let r = vec![0.0f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!((m.max_abs() - 1.0).abs() < 1e-5);
        m.reset_max();
        // Feed a gentler imbalance; max_abs re-tracks from below.
        let l2 = vec![0.5f32; total];
        let r2 = vec![0.4f32; total];
        m.process(&make_f32p_stereo(&l2, &r2), f32p_stereo(fs))
            .unwrap();
        assert!(
            m.max_abs() < 1.0,
            "max_abs should re-track from below after reset_max, got {}",
            m.max_abs()
        );
    }

    #[test]
    fn streaming_continuity_one_call_equals_two_halves() {
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 4;
        let l: Vec<f32> = (0..total)
            .map(|i| 0.3 + 0.05 * (i as f32 * 0.013).sin())
            .collect();
        let r: Vec<f32> = (0..total)
            .map(|i| 0.5 + 0.05 * (i as f32 * 0.017).cos())
            .collect();

        let mut m1 = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m1.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();

        let mut m2 = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        let half = total / 2;
        m2.process(&make_f32p_stereo(&l[..half], &r[..half]), f32p_stereo(fs))
            .unwrap();
        m2.process(&make_f32p_stereo(&l[half..], &r[half..]), f32p_stereo(fs))
            .unwrap();

        let d = (m1.current() - m2.current()).abs();
        assert!(
            d < 1e-5,
            "streamed-in-halves balance ({}) should match one-call ({}), Δ = {d}",
            m2.current(),
            m1.current()
        );
        assert_eq!(m1.samples_seen(), m2.samples_seen());
    }

    #[test]
    fn periodic_rebuild_bounds_f64_drift_on_long_stream() {
        // 200 windows of a constant imbalance; the rebuild keeps the
        // reading within f32 precision of the truth (+1/3).
        let fs = 48_000u32;
        let nwin = 480usize;
        let total = nwin * 200;
        let l = vec![0.25f32; total];
        let r = vec![0.5f32; total];
        let mut m = StereoBalanceMeter::with_window_ms(win_ms(nwin, fs));
        m.process(&make_f32p_stereo(&l, &r), f32p_stereo(fs))
            .unwrap();
        assert!(
            (m.current() - (1.0 / 3.0)).abs() < 1e-5,
            "long-stream constant imbalance should still read +1/3, got {}",
            m.current()
        );
    }

    #[test]
    fn window_clamped_at_construction() {
        let m = StereoBalanceMeter::with_window_ms(-50.0);
        assert!((m.window_ms() - 0.1).abs() < 1e-6);
        let m = StereoBalanceMeter::with_window_ms(1_000_000.0);
        assert!((m.window_ms() - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn window_samples_resolved_at_first_process() {
        let fs_cases = [(48_000u32, 19_200usize), (96_000, 38_400), (16_000, 6_400)];
        for (fs, expected) in fs_cases {
            let mut m = StereoBalanceMeter::with_window_ms(400.0);
            let dummy = vec![0.0f32; 16];
            m.process(&make_f32p_stereo(&dummy, &dummy), f32p_stereo(fs))
                .unwrap();
            assert_eq!(m.window_samples(), expected, "400 ms @ {fs} Hz");
        }
    }

    #[test]
    fn window_resizes_on_sample_rate_change() {
        let mut m = StereoBalanceMeter::with_window_ms(10.0);
        let dummy = vec![0.0f32; 16];
        m.process(&make_f32p_stereo(&dummy, &dummy), f32p_stereo(48_000))
            .unwrap();
        assert_eq!(m.window_samples(), 480);
        m.process(&make_f32p_stereo(&dummy, &dummy), f32p_stereo(96_000))
            .unwrap();
        assert_eq!(m.window_samples(), 960);
    }
}
