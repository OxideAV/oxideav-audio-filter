//! Slew limiter — bounds the per-sample change in amplitude.
//!
//! The classic "slope limiter" / "anti-zipper" smoother: the output
//! tracks the input verbatim **except** that the absolute change from
//! one output sample to the next can never exceed a configurable cap.
//! Sharp jumps in the input get rate-limited into ramps; smooth
//! signals already inside the slope budget are an exact pass-through.
//!
//! # Recurrence
//!
//! Per channel, per sample, given a current input `x[n]`, the previous
//! output `y[n-1]`, and a per-sample step cap `s ≥ 0`:
//!
//! ```text
//! Δ      = x[n] - y[n-1]
//! y[n]   = y[n-1] + clamp(Δ, -s, +s)
//! ```
//!
//! Equivalently:
//!
//! * If `|Δ| ≤ s` the filter is a bit-exact pass-through for sample
//!   `n`: `y[n] = x[n]`.
//! * If `Δ > +s` the output ramps **upward** at the maximum rate:
//!   `y[n] = y[n-1] + s`.
//! * If `Δ < -s` the output ramps **downward** at the maximum rate:
//!   `y[n] = y[n-1] - s`.
//!
//! The bound `s` is derived from the user's slope spec
//! (`max_slew_per_sec`, in amplitude-units per second) divided by the
//! stream's sample rate:
//!
//! ```text
//! s = max_slew_per_sec / fs
//! ```
//!
//! So a setting of `2.0 / s` at `fs = 48000` Hz means the output can
//! traverse the full `[-1, +1]` amplitude window in `(1 - -1)/2 = 1`
//! second — i.e. `1.0/48000 ≈ 2.08 × 10⁻⁵` per sample.
//!
//! # Asymmetric slew (separate rise / fall)
//!
//! A common variant lets the upward rate differ from the downward
//! rate (e.g. envelope-follower-style fast attack, slow release).
//! [`SlewLimiter::with_asymmetric`] exposes that:
//!
//! ```text
//! Δ      = x[n] - y[n-1]
//! Δ_lim  = if Δ ≥ 0 { min(Δ, +s_up) } else { max(Δ, -s_dn) }
//! y[n]   = y[n-1] + Δ_lim
//! ```
//!
//! `s_up = s_dn` gives the symmetric case above; setting one of them
//! to a very large number gives a one-sided limiter (e.g. limit only
//! the rise, leave falls instantaneous).
//!
//! # Why not just low-pass?
//!
//! A low-pass filter (one-pole / biquad LPF) also smooths sharp
//! jumps, but its response is **time-domain exponential** — the
//! per-sample change shrinks as the residual error shrinks, so the
//! output asymptotically approaches the input rather than reaching
//! it. A slew limiter is **time-domain linear** — it ramps at exactly
//! the cap rate until the input is reached, then snaps to
//! pass-through. The result is a piecewise-linear trajectory with
//! discontinuous slope (at the corners where ramp meets pass-through),
//! useful for modulation smoothing on control-rate parameters and for
//! anti-pop on hard mute/unmute events.
//!
//! # Per-channel state
//!
//! Each channel keeps its own `y[n-1]` history scalar so stereo input
//! does not cross-talk through the limiter. The first sample of each
//! channel uses `y[n-1] = 0`, so a non-zero leading sample will
//! itself be ramped from zero — see
//! [`SlewLimiter::with_initial_value`] to seed the held value
//! explicitly (useful when this filter is splicing onto an existing
//! signal segment).
//!
//! # General DSP literature
//!
//! Slew limiting is a textbook control-rate / audio-rate smoothing
//! primitive — used in analog-synth envelope smoothers, "portamento"
//! glide between MIDI notes, and anti-zipper smoothers on
//! click-prone parameter changes (volume, pan, EQ Q-factor). The
//! closed-form recurrence above is the standard implementation;
//! implemented here from first principles, no external source
//! consulted.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Per-channel slew-limiter state: just the last output sample.
#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    y_prev: f32,
}

/// Streaming slew-rate limiter.
///
/// Bounds the per-sample change in the output. The per-sample cap is
/// derived from `max_slew_per_sec / sample_rate` (re-derived on every
/// call against the stream-level [`AudioStreamParams`], so the same
/// filter instance behaves identically at 44.1 / 48 / 96 kHz in
/// **amplitude-units per second**).
#[derive(Debug, Clone)]
pub struct SlewLimiter {
    slew_up: f32,
    slew_dn: f32,
    initial: f32,
    state: Vec<ChState>,
    initialised: bool,
}

impl SlewLimiter {
    /// Symmetric slew cap. `max_slew_per_sec` is the maximum absolute
    /// change in amplitude (in `[-1, +1]`-normalised units) the output
    /// is allowed to traverse **per second**. Internally translated to
    /// a per-sample step `s = max_slew_per_sec / fs` at process time.
    ///
    /// `max_slew_per_sec` is clamped to `[0, 1e9]`. A value of `0`
    /// freezes the output at the held value (useful for an
    /// "infinite-glide" portamento that never finishes); a value of
    /// `1e9` effectively disables the limiter (the cap will exceed
    /// any realistic per-sample delta even at very low sample rates).
    pub fn new(max_slew_per_sec: f32) -> Self {
        let s = max_slew_per_sec.clamp(0.0, 1e9);
        Self {
            slew_up: s,
            slew_dn: s,
            initial: 0.0,
            state: Vec::new(),
            initialised: false,
        }
    }

    /// Asymmetric slew caps: `up_per_sec` for positive jumps
    /// (`x > y_prev`), `dn_per_sec` for negative jumps
    /// (`x < y_prev`). Each clamped to `[0, 1e9]` independently —
    /// setting either to `0` freezes the corresponding direction (the
    /// output can move only the other way until the input crosses
    /// back across the held value).
    pub fn with_asymmetric(up_per_sec: f32, dn_per_sec: f32) -> Self {
        Self {
            slew_up: up_per_sec.clamp(0.0, 1e9),
            slew_dn: dn_per_sec.clamp(0.0, 1e9),
            initial: 0.0,
            state: Vec::new(),
            initialised: false,
        }
    }

    /// Seed the held value (`y[n-1]` for the very first sample of every
    /// channel). Defaults to `0.0`, which means a non-zero leading
    /// input sample will itself be ramped up from zero. Setting this
    /// explicitly is useful when splicing the slew-limited segment
    /// onto a pre-existing signal at a known amplitude.
    pub fn with_initial_value(mut self, v: f32) -> Self {
        self.initial = v;
        // Reset any previously-recorded state so the next process()
        // call re-seeds from `initial`.
        self.initialised = false;
        self.state.clear();
        self
    }

    /// Currently-configured upward slew cap (per-second).
    pub fn slew_up_per_sec(&self) -> f32 {
        self.slew_up
    }

    /// Currently-configured downward slew cap (per-second).
    pub fn slew_dn_per_sec(&self) -> f32 {
        self.slew_dn
    }

    /// Currently-configured initial held value.
    pub fn initial_value(&self) -> f32 {
        self.initial
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.state.len() != channels {
            self.state = vec![
                ChState {
                    y_prev: self.initial
                };
                channels
            ];
            self.initialised = true;
        } else if !self.initialised {
            for ch in self.state.iter_mut() {
                ch.y_prev = self.initial;
            }
            self.initialised = true;
        }
    }
}

impl Default for SlewLimiter {
    /// 2.0 amplitude-units per second — a 1-second traverse across the
    /// full `[-1, +1]` window. A reasonable starting point for
    /// anti-zipper smoothing of slow control changes; for anti-pop
    /// hard-mute work you typically want a much faster cap.
    fn default() -> Self {
        Self::new(2.0)
    }
}

impl AudioFilter for SlewLimiter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let channels = params.channels as usize;
        self.ensure_state(channels);
        let fs = params.sample_rate.max(1) as f32;
        // Per-sample step caps. Re-derived per call so the same filter
        // instance is stream-rate-agnostic.
        let step_up = self.slew_up / fs;
        let step_dn = self.slew_dn / fs;

        let mut decoded = decode_to_f32(input, params.format, params.channels)?;
        for (ch, buf) in decoded.iter_mut().enumerate() {
            let mut y_prev = self.state[ch].y_prev;
            for s in buf.iter_mut() {
                let delta = *s - y_prev;
                let limited = if delta >= 0.0 {
                    delta.min(step_up)
                } else {
                    // delta < 0; clamp downward magnitude
                    delta.max(-step_dn)
                };
                y_prev += limited;
                *s = y_prev;
            }
            self.state[ch].y_prev = y_prev;
        }
        let out = encode_from_f32(params.format, params.channels, input, &decoded)?;
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

    fn read_f32(frame: &AudioFrame) -> Vec<f32> {
        frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn step_input_ramps_at_cap_rate() {
        // fs = 10 Hz, slew = 1.0/s ⇒ per-sample cap = 0.1.
        // Held value starts at 0. Input is a unit step (10 samples of 1.0).
        // Expected output (closed form): y[n] = min(1.0, (n+1)·0.1)
        //   n=0: 0 + clamp(1-0, ±0.1) = 0.1
        //   n=1: 0.2
        //   ...
        //   n=9: 1.0 (reaches the target on the 10th sample)
        let input = [1.0f32; 10];
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::new(1.0);
        let out = sl.process(&frame, f32_mono(10)).unwrap();
        let got = read_f32(&out[0]);
        let expected: Vec<f32> = (0..10).map(|n| ((n + 1) as f32 * 0.1).min(1.0)).collect();
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-6, "sample {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn signal_within_budget_is_passthrough() {
        // slew = 1000/s, fs = 1000 Hz ⇒ per-sample cap = 1.0. Every
        // sample's delta is well within ±1.0 since the signal is
        // already bounded to ±0.5, so output == input exactly.
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::new(1000.0);
        let out = sl.process(&frame, f32_mono(1000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
            assert!((g - x).abs() < 1e-6, "sample {i}: got {g}, expected {x}");
        }
    }

    #[test]
    fn downward_step_ramps_at_cap_rate() {
        // Seed held value at +1.0, then push −1.0 input. fs = 10,
        // slew = 1.0/s ⇒ cap = 0.1/sample. Output ramps DOWNWARD:
        //   n=0: 1.0 + clamp(−2, ±0.1) = 1.0 − 0.1 = 0.9
        //   n=1: 0.8 … n=19: −1.0
        let input = [-1.0f32; 30];
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::new(1.0).with_initial_value(1.0);
        let out = sl.process(&frame, f32_mono(10)).unwrap();
        let got = read_f32(&out[0]);
        for (n, &g) in got.iter().enumerate() {
            let want = (1.0f32 - (n as f32 + 1.0) * 0.1).max(-1.0);
            assert!((g - want).abs() < 1e-6, "sample {n}: got {g}, want {want}");
        }
    }

    #[test]
    fn asymmetric_caps_act_independently() {
        // Fast attack (slew_up = 10/s ⇒ cap=1.0/sample at fs=10),
        // slow release (slew_dn = 1/s ⇒ cap=0.1/sample). Input is a
        // unit step UP (instantaneous, since cap_up ≥ |Δ|) followed
        // by a step DOWN (must ramp at 0.1/sample).
        let mut input = vec![1.0f32; 5]; // step up
        input.extend(vec![0.0f32; 20]); // step down
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::with_asymmetric(10.0, 1.0);
        let out = sl.process(&frame, f32_mono(10)).unwrap();
        let got = read_f32(&out[0]);
        // n=0: 0 + clamp(1, ±cap_up=1.0) = 1.0 (instant attack)
        // n=1..4: input still 1.0 ⇒ delta=0 ⇒ stay at 1.0
        for (i, &g) in got.iter().take(5).enumerate() {
            assert!((g - 1.0).abs() < 1e-6, "attack sample {i}: got {g}");
        }
        // n=5: 1.0 + clamp(−1, ±cap_dn=0.1) = 1.0 − 0.1 = 0.9
        // n=6: 0.8 … n=14: 0.0 (reaches target on the 10th release sample)
        for n in 0..20 {
            let g = got[5 + n];
            let want = (1.0 - (n as f32 + 1.0) * 0.1).max(0.0);
            assert!((g - want).abs() < 1e-6, "release {n}: got {g}, want {want}");
        }
    }

    #[test]
    fn streaming_continuity_across_calls() {
        // Splitting a long step into two calls must produce the same
        // output as one big call — state has to carry through.
        let total = 30;
        let input: Vec<f32> = vec![1.0; total];

        let mut sl1 = SlewLimiter::new(1.0);
        let frame_a = make_f32_mono(&input[..10]);
        let frame_b = make_f32_mono(&input[10..]);
        let out_a = sl1.process(&frame_a, f32_mono(10)).unwrap();
        let out_b = sl1.process(&frame_b, f32_mono(10)).unwrap();
        let mut concat = read_f32(&out_a[0]);
        concat.extend(read_f32(&out_b[0]));

        let mut sl2 = SlewLimiter::new(1.0);
        let one_shot = sl2.process(&make_f32_mono(&input), f32_mono(10)).unwrap();
        let one_shot = read_f32(&one_shot[0]);

        assert_eq!(concat.len(), one_shot.len());
        for (i, (&a, &b)) in concat.iter().zip(one_shot.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "sample {i}: split={a}, one-shot={b}");
        }
    }

    #[test]
    fn stereo_channels_independent() {
        // L: step UP from 0 to +1. R: step DOWN from initial 0 to −1.
        // Each channel must ramp at its own pace with no cross-talk
        // through the held value.
        let n = 10usize;
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&1.0f32.to_le_bytes()); // L
            bytes.extend_from_slice(&(-1.0f32).to_le_bytes()); // R
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut sl = SlewLimiter::new(1.0); // cap = 0.1/sample @ fs=10
        let out = sl
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 2,
                    sample_rate: 10,
                },
            )
            .unwrap();
        let bytes = &out[0].data[0];
        let rd = |s: usize, c: usize| {
            let off = (s * 2 + c) * 4;
            f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
        };
        for i in 0..n {
            let want_l = ((i as f32 + 1.0) * 0.1).min(1.0);
            let want_r = (-(i as f32 + 1.0) * 0.1).max(-1.0);
            assert!(
                (rd(i, 0) - want_l).abs() < 1e-6,
                "L[{i}]: got {}, want {want_l}",
                rd(i, 0)
            );
            assert!(
                (rd(i, 1) - want_r).abs() < 1e-6,
                "R[{i}]: got {}, want {want_r}",
                rd(i, 1)
            );
        }
    }

    #[test]
    fn zero_slew_freezes_output() {
        // slew = 0 ⇒ the output never moves; stays at the initial
        // held value (0) regardless of input.
        let input: Vec<f32> = (0..16).map(|i| (i as f32 * 0.5).sin()).collect();
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::new(0.0);
        let out = sl.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, &g) in got.iter().enumerate() {
            assert!(g.abs() < 1e-9, "sample {i} should be 0 (frozen): got {g}");
        }
    }

    #[test]
    fn very_high_slew_is_passthrough() {
        // slew = 1e9 ⇒ cap ≫ any realistic delta; the limiter never
        // engages and the output equals the input.
        let input: Vec<f32> = (0..32)
            .map(|i| (i as f32 * 0.3).sin() * 0.8 + (i as f32 * 0.1).cos() * 0.1)
            .collect();
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::new(1e9);
        let out = sl.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
            assert!((g - x).abs() < 1e-6, "sample {i}: got {g}, expected {x}");
        }
    }

    #[test]
    fn rate_scaling_keeps_per_second_slope_invariant() {
        // The same SlewLimiter at different sample rates must reach
        // the same amplitude after the same wall-clock duration.
        //
        // slew = 4.0/s. After 1 second of input=1.0 (starting from 0):
        //   * fs=10 ⇒ cap=0.4/sample; reaches 1.0 after ceil(1/0.4)=3 samples
        //     (samples 0,1,2 → 0.4, 0.8, 1.0)
        //   * fs=100 ⇒ cap=0.04/sample; reaches 1.0 after 25 samples
        //   * fs=1000 ⇒ cap=0.004/sample; reaches 1.0 after 250 samples
        // After 1 second of wall time, all three must equal 1.0.
        for (fs, n_per_sec) in [(10u32, 10usize), (100, 100), (1000, 1000)] {
            let input = vec![1.0f32; n_per_sec];
            let frame = make_f32_mono(&input);
            let mut sl = SlewLimiter::new(4.0);
            let out = sl.process(&frame, f32_mono(fs)).unwrap();
            let got = read_f32(&out[0]);
            let last = *got.last().unwrap();
            assert!(
                (last - 1.0).abs() < 1e-6,
                "fs={fs}: after 1s, output should reach 1.0; got {last}"
            );
        }
    }

    #[test]
    fn initial_value_seeds_first_sample_correctly() {
        // Seed at 0.5, then input is 0.5 (constant). With any cap > 0
        // the output is a bit-exact pass-through at 0.5 from sample 0
        // because the delta is always zero.
        let input = [0.5f32; 8];
        let frame = make_f32_mono(&input);
        let mut sl = SlewLimiter::new(10.0).with_initial_value(0.5);
        let out = sl.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, &g) in got.iter().enumerate() {
            assert!((g - 0.5).abs() < 1e-6, "sample {i}: got {g}");
        }
    }

    #[test]
    fn parameters_clamped() {
        // Negative slew clamped up to 0 (which freezes the filter).
        let sl = SlewLimiter::new(-5.0);
        assert_eq!(sl.slew_up_per_sec(), 0.0);
        assert_eq!(sl.slew_dn_per_sec(), 0.0);

        // Huge slew clamped to 1e9.
        let sl = SlewLimiter::new(1e20);
        assert_eq!(sl.slew_up_per_sec(), 1e9);
        assert_eq!(sl.slew_dn_per_sec(), 1e9);

        // Asymmetric: each axis clamped independently.
        let sl = SlewLimiter::with_asymmetric(-1.0, 1e20);
        assert_eq!(sl.slew_up_per_sec(), 0.0);
        assert_eq!(sl.slew_dn_per_sec(), 1e9);

        // Initial value getter mirrors the seed.
        let sl = SlewLimiter::new(1.0).with_initial_value(-0.3);
        assert_eq!(sl.initial_value(), -0.3);
    }
}
