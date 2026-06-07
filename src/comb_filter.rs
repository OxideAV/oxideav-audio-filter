//! Tunable comb filter — feedforward (FIR) and feedback (IIR) forms.
//!
//! The comb filter is one of the fundamental DSP primitives: a single
//! delay line wired around an adder produces a transfer function whose
//! magnitude response is a set of evenly-spaced peaks (or notches),
//! i.e. the "teeth of a comb" in the frequency domain.  This module
//! exposes both standard forms as a single tunable filter so callers
//! can build resonators, flange/chorus building blocks, plucked-string
//! synth voices, comb-EQ effects, and dereverberation pre-filters
//! without re-deriving the delay-line plumbing.
//!
//! Within this crate the comb filter is distinct from the existing
//! delay-line filters:
//!
//! * [`Echo`](crate::echo::Echo) — single-tap delay (`echo` style):
//!   bigger delays (tens to hundreds of ms), wet/dry mix knob,
//!   tail-into-silence presentation.  Algorithmically the feedback
//!   comb is *the same recurrence* with `dry = wet = 1`, but the
//!   intended use-case is different — `Echo` is a time-domain
//!   *audible* repetition effect, the feedback comb is a *resonator*
//!   tuned via short delays in the millisecond / sub-millisecond range
//!   for ringing tones, vowel formant resonances, or Karplus-Strong
//!   plucked-string synthesis.
//! * [`Flanger`](crate::flanger::Flanger) — feedback comb with an LFO
//!   *modulating* the delay length and a positive feedback path
//!   audible as a "jet whoosh".  The comb filter here has a fixed
//!   delay; the flanger is the LFO-driven version.
//! * [`Reverb`](crate::reverb::Reverb) — four parallel comb lines
//!   *summed* and then run through two serial all-passes.  The reverb
//!   uses combs internally as building blocks but presents them
//!   collectively as a room simulator.  This filter exposes a single
//!   tunable comb directly.
//!
//! # Forms
//!
//! [`CombFilter`] supports two modes selected by [`CombMode`]:
//!
//! * **Feedforward** ([`CombMode::Feedforward`]): a finite-impulse-response
//!   recurrence
//!
//!   ```text
//!   y[n] = x[n] + g · x[n - D]
//!   ```
//!
//!   The transfer function is `H(z) = 1 + g · z^{-D}`.  The magnitude
//!   response is `|H(e^{jω})| = √(1 + g² + 2g·cos(ω·D))`, which has
//!   `D + 1` evenly-spaced extrema in `[0, π]` (i.e. `[0, fs/2]`).
//!   With `g > 0` the extrema at `ω = 2πk/D` are *peaks* of `1 + g`
//!   and the extrema at `ω = (2k+1)π/D` are *troughs* of `|1 − g|`;
//!   with `g < 0` the roles swap.  This is the canonical "single-tap
//!   FIR comb" used in stereo widener side-paths, decorrelation
//!   networks, and frequency-domain dereverberation prefilters
//!   (whose peaks/notches deliberately mask a room's modal harmonic
//!   series).  The filter is unconditionally stable (FIR — no
//!   poles); `g` may be any real number, though `|g| ≤ 1` is the
//!   normal range so the trough at the notch frequency doesn't lift
//!   the pass-band above unity gain.
//!
//! * **Feedback** ([`CombMode::Feedback`]): an infinite-impulse-response
//!   recurrence
//!
//!   ```text
//!   y[n] = x[n] + g · y[n - D]
//!   ```
//!
//!   The transfer function is `H(z) = 1 / (1 − g · z^{-D})`.  The
//!   `D` poles sit on a circle of radius `|g|^{1/D}` in the `z`
//!   plane, evenly distributed at `arg = 2πk/D + arg(g)/D`.  The
//!   magnitude response is `|H(e^{jω})| = 1 / √(1 + g² − 2g·cos(ωD))`
//!   — *peaks* of `1 / (1 − g)` at `ω = 2πk/D` for `g > 0`
//!   (resonance frequencies `f_k = k·fs/D`) with a `−3 dB`
//!   bandwidth `≈ (1 − g) · fs / (πD)`.  Stable iff `|g| < 1`;
//!   the constructors clamp `|g|` to `0.999` to enforce a safety
//!   margin (`g = 1` is a marginally-stable ideal integrator that
//!   self-oscillates on any tiny denormal).
//!
//!   The feedback comb's resonance at `f_0 = fs/D` is the canonical
//!   *Karplus-Strong plucked-string* recurrence; a short noise burst
//!   loaded into the delay line then circulates and decays.  The
//!   [`CombFilter::karplus_strong`] convenience constructor picks
//!   the delay for a target pitch (`D = round(fs / freq_hz)`) and
//!   sets a damping factor that gives the classic decaying-overtone
//!   string tone.
//!
//!   An optional **damping** factor in [`CombMode::Feedback`] inserts
//!   a one-pole low-pass in the feedback path:
//!
//!   ```text
//!   s[n] = (1 − a) · y[n − D] + a · s[n − 1]
//!   y[n] = x[n] + g · s[n]
//!   ```
//!
//!   with `a ∈ [0, 1)`.  `a = 0` is the bare feedback comb
//!   (frequency-independent decay).  Larger `a` causes high-frequency
//!   harmonics to decay faster than low-frequency ones — the natural
//!   behaviour of a damped string, where bowing or plucking energy
//!   in the upper partials radiates away faster than in the
//!   fundamental.  This is the same damping role the
//!   [`Reverb`](crate::reverb::Reverb) module's combs use.
//!
//! # Delay specification
//!
//! Callers may set the delay in two equivalent ways:
//!
//! * **Samples** ([`CombFilter::with_delay_samples`]): `D` is the
//!   exact integer length of the ring buffer.  The resonance
//!   frequencies fall at `k · fs / D`, sample-rate-dependent.
//! * **Milliseconds** ([`CombFilter::with_delay_ms`]): the integer
//!   sample delay `D = round(delay_ms · fs / 1000)` is derived on
//!   the first `process()` call against the input stream's
//!   `sample_rate`.  This form is rate-portable — feeding the same
//!   `delay_ms = 5.0 ms` to a 44.1 kHz and a 48 kHz stream resonates
//!   at the same audible pitch on both.
//!
//! `D` is clamped to `[1, MAX_DELAY_SAMPLES] = [1, 192_000]`
//! (4 s at 48 kHz, defending against pathological allocations
//! without rejecting any realistic configuration).
//!
//! # Per-channel state
//!
//! Each channel carries its own ring buffer and (in [`CombMode::Feedback`]
//! with non-zero damping) its own one-pole LP state.  Channels do
//! not cross-talk through the filter.  [`CombFilter::reset`] zeros
//! every ring without changing the configured mode or delay.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Upper bound on the delay-line length in samples (`= 192_000`, i.e.
/// 4 s at 48 kHz or 2 s at 96 kHz).  Defends against pathological
/// per-sample allocations without rejecting any realistic configuration.
pub const MAX_DELAY_SAMPLES: usize = 192_000;

/// Feedback-gain safety clamp (`|g| ≤ 0.999`).  `g = 1.0` is a
/// marginally-stable ideal integrator that will self-oscillate on any
/// tiny denormal; the safety margin guarantees a strictly decaying
/// impulse response on any practical input.
const FEEDBACK_GAIN_CLAMP: f32 = 0.999;

/// Damping clamp.  `a = 1.0` would make the LP integrator infinite —
/// the filter would never release stored energy.
const DAMPING_CLAMP: f32 = 0.999;

/// Comb filter topology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombMode {
    /// Feedforward FIR comb: `y[n] = x[n] + g · x[n − D]`.
    ///
    /// `gain` is the feedforward coefficient; positive values
    /// produce peaks at `0 / fs/D / 2·fs/D / …` and troughs in
    /// between, negative values swap the roles.  Unconditionally
    /// stable for any finite `gain`; the constructors don't clamp,
    /// the user is trusted to pick a sensible magnitude (typical
    /// values lie in `[−1, +1]`).
    Feedforward { gain: f32 },

    /// Feedback IIR comb: `y[n] = x[n] + g · damped(y[n − D])`,
    /// optionally with a one-pole LP `damped(·)` in the feedback
    /// path.  `gain` is clamped into `[−FEEDBACK_GAIN_CLAMP,
    /// +FEEDBACK_GAIN_CLAMP]` (= `±0.999`) at construction so the
    /// recurrence stays strictly stable; `damping` is clamped into
    /// `[0.0, DAMPING_CLAMP]` (= `[0.0, 0.999]`).
    ///
    /// * `damping = 0` is the bare feedback comb.
    /// * Larger `damping` makes the high-frequency overtones decay
    ///   faster than the fundamental — the natural plucked-string
    ///   behaviour the [`CombFilter::karplus_strong`] convenience
    ///   constructor uses.
    Feedback { gain: f32, damping: f32 },
}

impl CombMode {
    fn clamp(self) -> Self {
        match self {
            CombMode::Feedforward { gain } => CombMode::Feedforward { gain },
            CombMode::Feedback { gain, damping } => CombMode::Feedback {
                gain: gain.clamp(-FEEDBACK_GAIN_CLAMP, FEEDBACK_GAIN_CLAMP),
                damping: damping.clamp(0.0, DAMPING_CLAMP),
            },
        }
    }
}

/// How the delay length is specified.
#[derive(Debug, Clone, Copy)]
enum DelaySpec {
    /// Exact integer sample delay; sample-rate-dependent resonance.
    Samples(usize),
    /// Delay in milliseconds; resolved to an integer sample count
    /// against the input stream's `sample_rate` on first
    /// `process()`.  Rate-portable.
    Millis(f32),
}

/// Per-channel ring buffer + one-pole LP state for the feedback path.
#[derive(Debug, Clone)]
struct ChState {
    /// Ring buffer; `ring.len() == window_samples`.
    ring: Vec<f32>,
    /// Per-channel one-pole LP state used only by the feedback
    /// damping path; ignored by the feedforward mode.
    lp_state: f32,
    /// Next slot to overwrite (write cursor).
    write_idx: usize,
}

impl ChState {
    fn new(d: usize) -> Self {
        Self {
            ring: vec![0.0; d.max(1)],
            lp_state: 0.0,
            write_idx: 0,
        }
    }
    fn reset(&mut self) {
        for v in self.ring.iter_mut() {
            *v = 0.0;
        }
        self.lp_state = 0.0;
        self.write_idx = 0;
    }
}

/// Tunable comb filter (feedforward or feedback) — see the
/// [module docs](self) for the algorithm.
#[derive(Debug, Clone)]
pub struct CombFilter {
    mode: CombMode,
    delay: DelaySpec,
    /// Resolved sample delay (`D`) on the most recent `process()` call.
    /// `0` before the first call.
    resolved_d: usize,
    /// Sample rate observed on the most recent `process()` call.
    /// `0` before the first call.  Used to detect a rate change and
    /// rebuild the ring buffer.
    sample_rate: u32,
    state: Vec<ChState>,
}

impl CombFilter {
    /// Build a comb filter with the given mode and an integer sample
    /// delay.  `delay_samples` is clamped into
    /// `[1, MAX_DELAY_SAMPLES]`.
    pub fn with_delay_samples(mode: CombMode, delay_samples: usize) -> Self {
        let d = delay_samples.clamp(1, MAX_DELAY_SAMPLES);
        Self {
            mode: mode.clamp(),
            delay: DelaySpec::Samples(d),
            resolved_d: 0,
            sample_rate: 0,
            state: Vec::new(),
        }
    }

    /// Build a comb filter with the given mode and a millisecond
    /// delay.  The integer sample count is derived against the input
    /// stream's `sample_rate` on the first `process()` call and
    /// clamped to `[1, MAX_DELAY_SAMPLES]`.
    pub fn with_delay_ms(mode: CombMode, delay_ms: f32) -> Self {
        let dm = delay_ms.clamp(0.0, 10_000.0);
        Self {
            mode: mode.clamp(),
            delay: DelaySpec::Millis(dm),
            resolved_d: 0,
            sample_rate: 0,
            state: Vec::new(),
        }
    }

    /// Feedforward FIR comb at the given millisecond delay and gain.
    /// Convenience shortcut for `with_delay_ms(Feedforward { gain }, ms)`.
    pub fn feedforward(delay_ms: f32, gain: f32) -> Self {
        Self::with_delay_ms(CombMode::Feedforward { gain }, delay_ms)
    }

    /// Feedback IIR comb at the given millisecond delay and gain (no
    /// damping).  Convenience shortcut.
    pub fn feedback(delay_ms: f32, gain: f32) -> Self {
        Self::with_delay_ms(CombMode::Feedback { gain, damping: 0.0 }, delay_ms)
    }

    /// Karplus-Strong plucked-string voice tuned to `freq_hz`.  The
    /// integer sample delay is `round(fs / freq_hz)` (resolved on the
    /// first `process()` call against the input stream's
    /// `sample_rate`); the feedback gain is `decay ∈ (0, 0.999]`
    /// (clamped); a small damping (`0.5`) is built in to give the
    /// classic decaying-overtone string tone.
    ///
    /// The filter does not synthesise the pluck excitation — feed a
    /// short white-noise burst (or any broadband impulse) of about
    /// `D` samples and the comb's loop circulates and decays into
    /// the audible string tone.  Combine with
    /// [`crate::WhiteNoise`] for the noise source.
    pub fn karplus_strong(freq_hz: f32, decay: f32) -> Self {
        // Special-case freq_hz <= 0 → very long delay (sub-audible);
        // the karplus_strong API conventionally interprets a positive
        // frequency.  We map ≤ 0 to MAX_DELAY_SAMPLES so the filter
        // produces a sub-audible drone rather than an out-of-range
        // panic.
        let f = freq_hz.max(0.001);
        // Sample delay is derived against the actual stream rate in
        // ensure_state().  We stash the target frequency by encoding
        // it as the "delay_ms" form: D = round(fs / f) <=> delay_ms
        // = 1000 / f.
        let ms = 1000.0 / f;
        Self::with_delay_ms(
            CombMode::Feedback {
                gain: decay,
                damping: 0.5,
            },
            ms,
        )
    }

    /// Current mode (clamped).
    pub fn mode(&self) -> CombMode {
        self.mode
    }

    /// Replace the filter mode.  Per-channel state is preserved
    /// (delay line + LP state); only the recurrence selection
    /// changes.
    pub fn set_mode(&mut self, mode: CombMode) {
        self.mode = mode.clamp();
    }

    /// Resolved integer sample delay `D` after the first `process()`
    /// call.  Returns `0` before then.
    pub fn delay_samples(&self) -> usize {
        self.resolved_d
    }

    /// Wipe per-channel state without changing the configured mode
    /// or delay.
    pub fn reset(&mut self) {
        for ch in self.state.iter_mut() {
            ch.reset();
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let want_d = match self.delay {
            DelaySpec::Samples(d) => d,
            DelaySpec::Millis(ms) => {
                let raw = (ms as f64 * sample_rate as f64 / 1000.0).round() as usize;
                raw.clamp(1, MAX_DELAY_SAMPLES)
            }
        };
        let rebuild = self.sample_rate != sample_rate
            || self.state.len() != channels
            || self.resolved_d != want_d;
        if rebuild {
            self.state = (0..channels).map(|_| ChState::new(want_d)).collect();
            self.resolved_d = want_d;
            self.sample_rate = sample_rate;
        }
    }
}

impl AudioFilter for CombFilter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate, params.channels as usize);
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);
        let d = self.resolved_d;

        match self.mode {
            CombMode::Feedforward { gain } => {
                for i in 0..n {
                    for (ch_idx, ch) in channels.iter_mut().enumerate() {
                        let x = ch[i];
                        let state = &mut self.state[ch_idx];
                        let delayed = state.ring[state.write_idx];
                        let y = x + gain * delayed;
                        // Feedforward stores the *input* sample, not
                        // the output — that's the FIR recurrence
                        // y[n] = x[n] + g·x[n−D].
                        state.ring[state.write_idx] = x;
                        state.write_idx += 1;
                        if state.write_idx >= d {
                            state.write_idx = 0;
                        }
                        ch[i] = y.clamp(-1.0, 1.0);
                    }
                }
            }
            CombMode::Feedback { gain, damping } => {
                for i in 0..n {
                    for (ch_idx, ch) in channels.iter_mut().enumerate() {
                        let x = ch[i];
                        let state = &mut self.state[ch_idx];
                        let delayed = state.ring[state.write_idx];
                        // One-pole low-pass on the feedback path.
                        // damping = 0 is bare feedback (lp_state =
                        // delayed for that path); larger damping
                        // attenuates HF overtones faster than the
                        // fundamental — the natural plucked-string
                        // decay shape.
                        state.lp_state = (1.0 - damping) * delayed + damping * state.lp_state;
                        let y = x + gain * state.lp_state;
                        // Feedback stores the *output* sample —
                        // y[n] = x[n] + g·damped(y[n−D]).
                        state.ring[state.write_idx] = y;
                        state.write_idx += 1;
                        if state.write_idx >= d {
                            state.write_idx = 0;
                        }
                        ch[i] = y.clamp(-1.0, 1.0);
                    }
                }
            }
        }

        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::SampleFormat;

    fn params(sr: u32, ch: u16) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: ch,
            sample_rate: sr,
        }
    }

    fn f32_frame_mono(samples: &[f32]) -> AudioFrame {
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

    fn f32_frame_stereo(left: &[f32], right: &[f32]) -> AudioFrame {
        assert_eq!(left.len(), right.len());
        // F32 is interleaved (one plane) per the AudioFrame contract.
        let mut bytes = Vec::with_capacity(left.len() * 8);
        for (l, r) in left.iter().zip(right.iter()) {
            bytes.extend_from_slice(&l.to_le_bytes());
            bytes.extend_from_slice(&r.to_le_bytes());
        }
        AudioFrame {
            samples: left.len() as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    fn read_f32_mono(frame: &AudioFrame) -> Vec<f32> {
        frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_f32_stereo(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for c in frame.data[0].chunks_exact(8) {
            left.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            right.push(f32::from_le_bytes([c[4], c[5], c[6], c[7]]));
        }
        (left, right)
    }

    /// Feedforward `g = 0.5`, `D = 3`.  An impulse at n = 0 should
    /// emerge at the output as `1.0` at n = 0 and `0.5` at n = 3 with
    /// every other sample exactly 0 — the closed-form FIR comb
    /// response.
    #[test]
    fn feedforward_impulse_response_is_two_tap_fir() {
        let mut x = vec![0.0f32; 16];
        x[0] = 1.0;
        let frame = f32_frame_mono(&x);
        let mut c = CombFilter::with_delay_samples(CombMode::Feedforward { gain: 0.5 }, 3);
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        let mut expected = [0.0f32; 16];
        expected[0] = 1.0;
        expected[3] = 0.5;
        for (i, (a, b)) in y.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {i}: got {a}, expected {b}");
        }
    }

    /// Feedback `g = 0.5`, `D = 4`.  Impulse → geometric decay every
    /// `D` samples: `1.0` at n = 0, `0.5` at n = 4, `0.25` at n = 8,
    /// `0.125` at n = 12 (no damping → bare exponential).
    #[test]
    fn feedback_impulse_response_is_geometric_decay() {
        let mut x = vec![0.0f32; 17];
        x[0] = 1.0;
        let frame = f32_frame_mono(&x);
        let mut c = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.5,
                damping: 0.0,
            },
            4,
        );
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        let expected = [(0, 1.0f32), (4, 0.5), (8, 0.25), (12, 0.125), (16, 0.0625)];
        for (idx, val) in expected.iter() {
            assert!(
                (y[*idx] - val).abs() < 1e-5,
                "n = {idx}: got {}, expected {val}",
                y[*idx]
            );
        }
        // Every other sample is exactly zero (no input, no echo).
        for n in (1..17).filter(|n| n % 4 != 0) {
            assert!(y[n].abs() < 1e-6, "n = {n} should be 0, got {}", y[n]);
        }
    }

    /// Feedforward `g = 1` at `D` samples: at the resonance frequency
    /// `fs / (2D)` (a "trough" frequency at `g = +1` since the two
    /// taps are exactly 180° apart) the output should cancel to
    /// near-zero.
    #[test]
    fn feedforward_notch_at_trough_frequency() {
        // D = 8, fs = 48 kHz → notch at fs / (2D) = 3000 Hz.
        let d = 8usize;
        let fs = 48_000.0f32;
        let f_notch = fs / (2.0 * d as f32);
        let n = 2048;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * f_notch * i as f32 / fs).sin())
            .collect();
        let frame = f32_frame_mono(&x);
        let mut c = CombFilter::with_delay_samples(CombMode::Feedforward { gain: 1.0 }, d);
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        // Skip first 2·D samples (filter warm-up).
        let tail: Vec<f32> = y.iter().skip(2 * d).copied().collect();
        let max = tail.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // True analytic notch is zero; allow generous slack for
        // numeric round-off and the finite-window leakage.
        assert!(max < 0.01, "notch should cancel; max remaining = {max}");
    }

    /// Feedforward `g = 1` at the *peak* frequency `f = 0` (DC) gets
    /// gain 2 (`1 + g`).  A DC input should emerge doubled (after the
    /// warm-up: the ring is initially zero so the first `D` samples
    /// pass through at gain 1, then the comb engages).
    #[test]
    fn feedforward_dc_passes_at_one_plus_g() {
        let d = 5usize;
        let x = vec![0.5f32; 64];
        let frame = f32_frame_mono(&x);
        let mut c = CombFilter::with_delay_samples(CombMode::Feedforward { gain: 1.0 }, d);
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        // After warm-up the output sits at 0.5·(1+1) = 1.0 (clamped).
        for s in y.iter().skip(d) {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "DC × (1 + g) should be 1.0, got {s}"
            );
        }
    }

    /// Feedback `g = 0` (identity).
    #[test]
    fn feedback_zero_gain_is_identity() {
        let x: Vec<f32> = (0..32).map(|i| 0.1 * (i as f32).sin()).collect();
        let frame = f32_frame_mono(&x);
        let mut c = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.0,
                damping: 0.0,
            },
            10,
        );
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        for (i, (a, b)) in y.iter().zip(x.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "sample {i}: identity broken — got {a}, expected {b}"
            );
        }
    }

    /// `with_delay_ms` resolves to the right sample count at the
    /// input rate.  At 48 kHz, `delay_ms = 10` ⇒ `D = 480` samples.
    #[test]
    fn delay_ms_resolves_against_sample_rate() {
        let mut c = CombFilter::with_delay_ms(CombMode::Feedforward { gain: 0.5 }, 10.0);
        // Need at least one process call to resolve.
        let frame = f32_frame_mono(&[0.0; 1]);
        c.process(&frame, params(48_000, 1)).unwrap();
        assert_eq!(c.delay_samples(), 480);
        // Rate change rebuilds the delay length.
        c.process(&frame, params(96_000, 1)).unwrap();
        assert_eq!(c.delay_samples(), 960);
    }

    /// Karplus-Strong: feeding a short noise burst should produce a
    /// tone whose dominant period is `D = round(fs / freq_hz)`.  We
    /// check this by measuring the autocorrelation peak after the
    /// noise burst has fully circulated.
    #[test]
    fn karplus_strong_resonates_at_target_frequency() {
        let fs = 48_000.0f32;
        let target_hz = 440.0;
        // Build a 10 ms burst, then 50 ms of silence; let the loop ring.
        let burst = (fs as usize) * 10 / 1000;
        let total = (fs as usize) * 60 / 1000;
        let mut x = vec![0.0f32; total];
        // Deterministic xorshift32 burst (matches the crate's noise
        // generators' philosophy).
        let mut s: u32 = 0xcafe_f00d;
        for v in x.iter_mut().take(burst) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        let frame = f32_frame_mono(&x);
        let mut c = CombFilter::karplus_strong(target_hz, 0.99);
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        // The resolved delay should match round(fs / target_hz).
        let expected_d = (fs / target_hz).round() as usize;
        assert_eq!(c.delay_samples(), expected_d);
        // Energy must persist after the burst (the loop is ringing).
        let tail_energy: f32 = y[burst + expected_d..].iter().map(|v| v * v).sum();
        assert!(
            tail_energy > 1e-3,
            "loop should be ringing post-burst; tail_energy = {tail_energy}"
        );
    }

    /// Per-channel state isolation: feeding an impulse on the left
    /// channel only must leave the right channel silent.
    #[test]
    fn stereo_channels_do_not_cross_talk() {
        let mut left = vec![0.0f32; 32];
        left[0] = 1.0;
        let right = vec![0.0f32; 32];
        let frame = f32_frame_stereo(&left, &right);
        let mut c = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.7,
                damping: 0.0,
            },
            5,
        );
        let out = c.process(&frame, params(48_000, 2)).unwrap();
        let (yl, yr) = read_f32_stereo(&out[0]);
        // Right channel should remain bit-exact silent.
        assert!(
            yr.iter().all(|v| v.abs() < 1e-7),
            "right channel should be silent; max = {}",
            yr.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        );
        // Left channel should have the impulse + delayed echoes.
        assert!((yl[0] - 1.0).abs() < 1e-6);
        assert!((yl[5] - 0.7).abs() < 1e-5);
        assert!((yl[10] - 0.49).abs() < 1e-5);
    }

    /// Streaming continuity: a single call on a `2·N`-sample frame
    /// must give the same output as two calls on `N`-sample halves.
    #[test]
    fn streaming_continuity_is_bit_identical() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let frame = f32_frame_mono(&x);
        let mut c1 = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.6,
                damping: 0.2,
            },
            7,
        );
        let single = c1.process(&frame, params(48_000, 1)).unwrap();
        let y1 = read_f32_mono(&single[0]);

        let mut c2 = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.6,
                damping: 0.2,
            },
            7,
        );
        let frame_a = f32_frame_mono(&x[..64]);
        let frame_b = f32_frame_mono(&x[64..]);
        let out_a = c2.process(&frame_a, params(48_000, 1)).unwrap();
        let out_b = c2.process(&frame_b, params(48_000, 1)).unwrap();
        let mut y2 = read_f32_mono(&out_a[0]);
        y2.extend(read_f32_mono(&out_b[0]));
        for (i, (a, b)) in y1.iter().zip(y2.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "streaming continuity broken at sample {i}: single = {a}, split = {b}"
            );
        }
    }

    /// Feedback gain is clamped at construction so callers can't
    /// accidentally request a self-oscillating recurrence.
    #[test]
    fn feedback_gain_is_clamped_for_stability() {
        let c = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 2.0,
                damping: 0.0,
            },
            5,
        );
        match c.mode() {
            CombMode::Feedback { gain, .. } => {
                assert!(gain.abs() <= FEEDBACK_GAIN_CLAMP + 1e-6, "gain not clamped");
            }
            _ => panic!("wrong mode"),
        }
        let c2 = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: -3.0,
                damping: 1.5,
            },
            5,
        );
        match c2.mode() {
            CombMode::Feedback { gain, damping } => {
                assert!(gain.abs() <= FEEDBACK_GAIN_CLAMP + 1e-6);
                assert!((0.0..=DAMPING_CLAMP).contains(&damping));
            }
            _ => panic!("wrong mode"),
        }
    }

    /// Delay-length clamps: zero is bumped to one, and the upper
    /// bound caps requests above `MAX_DELAY_SAMPLES`.
    #[test]
    fn delay_samples_clamp_at_both_ends() {
        let c0 = CombFilter::with_delay_samples(CombMode::Feedforward { gain: 0.5 }, 0);
        // Resolved on process; here just confirm the configured form
        // didn't store an invalid zero.
        let frame = f32_frame_mono(&[0.0; 1]);
        let mut c = c0;
        c.process(&frame, params(48_000, 1)).unwrap();
        assert!(c.delay_samples() >= 1);

        let cbig = CombFilter::with_delay_samples(
            CombMode::Feedforward { gain: 0.5 },
            MAX_DELAY_SAMPLES * 10,
        );
        let mut c = cbig;
        c.process(&frame, params(48_000, 1)).unwrap();
        assert_eq!(c.delay_samples(), MAX_DELAY_SAMPLES);
    }

    /// `reset` zeros the ring without changing the configured delay
    /// or mode.
    #[test]
    fn reset_clears_state() {
        let mut c = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.8,
                damping: 0.0,
            },
            4,
        );
        let mut x = vec![0.0f32; 32];
        x[0] = 1.0;
        let frame = f32_frame_mono(&x);
        let _ = c.process(&frame, params(48_000, 1)).unwrap();
        c.reset();
        // After reset, a zero input must produce a zero output (no
        // residual decay).
        let zero = f32_frame_mono(&[0.0f32; 32]);
        let out = c.process(&zero, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        assert!(
            y.iter().all(|v| v.abs() < 1e-7),
            "ring not zeroed; max residual = {}",
            y.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        );
    }

    /// `set_mode` flips the recurrence while preserving the delay
    /// buffer.  We drive an impulse through a feedforward comb, then
    /// switch to feedback with the same gain — the next impulse
    /// should now ring instead of producing a single echo.
    #[test]
    fn set_mode_preserves_delay_buffer() {
        let mut c = CombFilter::with_delay_samples(CombMode::Feedforward { gain: 0.5 }, 3);
        let frame = f32_frame_mono(&[0.0; 4]);
        c.process(&frame, params(48_000, 1)).unwrap();
        c.set_mode(CombMode::Feedback {
            gain: 0.5,
            damping: 0.0,
        });
        let mut x = vec![0.0f32; 12];
        x[0] = 1.0;
        let frame = f32_frame_mono(&x);
        let out = c.process(&frame, params(48_000, 1)).unwrap();
        let y = read_f32_mono(&out[0]);
        // After mode-switch the recurrence is feedback; impulse →
        // geometric decay every D=3 samples.
        assert!((y[0] - 1.0).abs() < 1e-6);
        assert!((y[3] - 0.5).abs() < 1e-5);
        assert!((y[6] - 0.25).abs() < 1e-5);
    }

    /// Feedback comb with damping has a smaller resonance peak than
    /// the un-damped version (a one-pole LP in the loop drains
    /// energy faster).  Compare the two impulse-response energies.
    #[test]
    fn feedback_damping_reduces_resonance_peak() {
        let mut x = vec![0.0f32; 256];
        x[0] = 1.0;
        let frame = f32_frame_mono(&x);

        let mut bare = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.9,
                damping: 0.0,
            },
            8,
        );
        let mut damped = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 0.9,
                damping: 0.8,
            },
            8,
        );
        let yb = read_f32_mono(&bare.process(&frame, params(48_000, 1)).unwrap()[0]);
        let yd = read_f32_mono(&damped.process(&frame, params(48_000, 1)).unwrap()[0]);
        let eb: f32 = yb.iter().skip(64).map(|v| v * v).sum();
        let ed: f32 = yd.iter().skip(64).map(|v| v * v).sum();
        assert!(
            ed < eb,
            "damping should reduce ring-out energy; bare = {eb}, damped = {ed}"
        );
    }

    /// Mode accessor returns the clamped value, not the raw user
    /// argument — guard against the "set, read back, mutate" round-trip
    /// silently widening the gain.
    #[test]
    fn mode_accessor_returns_clamped_value() {
        let c = CombFilter::with_delay_samples(
            CombMode::Feedback {
                gain: 5.0,
                damping: 2.0,
            },
            5,
        );
        let m = c.mode();
        if let CombMode::Feedback { gain, damping } = m {
            assert!((gain - FEEDBACK_GAIN_CLAMP).abs() < 1e-6);
            assert!((damping - DAMPING_CLAMP).abs() < 1e-6);
        } else {
            panic!("wrong mode");
        }
    }
}
