//! Brickwall peak limiter with optional look-ahead.
//!
//! # How it works
//!
//! Maintains two collaborating structures per channel:
//!
//! 1. A look-ahead **sample delay line** of length `L`. Incoming
//!    samples are written into the line; the limiter emits the sample
//!    that's `L` slots old, so the gain envelope (which is computed
//!    from the *new* input peak) effectively "sees the future" of the
//!    delayed sample.
//! 2. A look-ahead **peak-hold window** of the same length `L`. We
//!    track the maximum `|x|` over a sliding window of the next `L`
//!    samples so the envelope is the louder of (current envelope,
//!    peak in the window).
//!
//! # Gain envelope
//!
//! Given the look-ahead peak `p` and the ceiling linear `C`,
//!
//! ```text
//! target_gain = min(1.0, C / max(p, ε))
//! ```
//!
//! is the **instantaneous required attenuation**. We then track that
//! target with a release-only smoother (attack is instant — we always
//! drop gain immediately when needed):
//!
//! ```text
//! if target_gain < env_gain:   env_gain ← target_gain
//! else:                        env_gain ← α_rel · env_gain + (1-α_rel) · target_gain
//! ```
//!
//! where `α_rel = exp(-1 / (τ · fs))` is the standard one-pole IIR
//! time-constant coefficient.
//!
//! With `look_ahead = 0` the limiter is causal/post-peak (gain comes
//! down *after* the peak arrives, producing a brief overshoot). With
//! `look_ahead > 0` the envelope is ramped down *during* the
//! look-ahead window so transients land at the ceiling exactly.
//!
//! Multi-channel: peak-linked envelope (max over channels). All
//! channels are then scaled by the same per-sample gain.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};
use std::collections::VecDeque;

/// Brickwall peak limiter.
#[derive(Debug, Clone)]
pub struct Limiter {
    ceiling_db: f32,
    release_ms: f32,
    /// Look-ahead in samples. `0` ⇒ causal, no delay.
    look_ahead_samples: usize,
    state: Option<LimiterState>,
}

#[derive(Debug, Clone)]
struct LimiterState {
    sample_rate: u32,
    channels: usize,
    ceiling_lin: f32,
    alpha_rel: f32,
    /// Per-channel ring of (peak, sample) so we can both delay the
    /// emitted sample and compute the peak in the window. We use a
    /// `VecDeque<f32>` per channel.
    delays: Vec<VecDeque<f32>>,
    /// Current smoothed gain envelope (linear).
    env_gain: f32,
}

impl Limiter {
    /// Build a new limiter.
    ///
    /// * `ceiling_db` — output peak ceiling in dBFS (≤ 0).
    /// * `release_ms` — gain-recovery time constant.
    /// * `look_ahead_samples` — 0..=2048. With 0 the limiter is causal.
    pub fn new(ceiling_db: f32, release_ms: f32, look_ahead_samples: usize) -> Self {
        Self {
            ceiling_db: crate::clamp_param(ceiling_db, 0.0, -144.0, 24.0),
            release_ms: release_ms.max(0.0),
            look_ahead_samples: look_ahead_samples.min(2048),
            state: None,
        }
    }

    /// Number of samples the limiter delays its output relative to its
    /// input, equal to the configured look-ahead length.
    pub fn latency_samples(&self) -> usize {
        self.look_ahead_samples
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.channels != channels,
            None => true,
        };
        if needs_rebuild {
            let ceiling_lin = 10.0f32.powf(self.ceiling_db / 20.0);
            let alpha_rel = if self.release_ms <= 1.0e-6 {
                0.0
            } else {
                let n = self.release_ms / 1000.0 * sample_rate as f32;
                (-1.0 / n.max(1.0)).exp()
            };
            self.state = Some(LimiterState {
                sample_rate,
                channels,
                ceiling_lin,
                alpha_rel,
                delays: (0..channels)
                    .map(|_| {
                        let mut q = VecDeque::with_capacity(self.look_ahead_samples + 1);
                        for _ in 0..self.look_ahead_samples {
                            q.push_back(0.0);
                        }
                        q
                    })
                    .collect(),
                env_gain: 1.0,
            });
        }
    }

    /// Reset the gain envelope and delay lines.
    pub fn reset(&mut self) {
        if let Some(s) = self.state.as_mut() {
            s.env_gain = 1.0;
            for d in s.delays.iter_mut() {
                d.clear();
                for _ in 0..self.look_ahead_samples {
                    d.push_back(0.0);
                }
            }
        }
    }

    /// Run the limiter on a per-channel buffer set in place.
    /// `channels.len()` ≥ 1.
    #[allow(clippy::needless_range_loop)]
    fn process_channels(&mut self, channels: &mut [Vec<f32>]) {
        let n_chan = channels.len();
        if n_chan == 0 {
            return;
        }
        let n_samples = channels[0].len();
        let state = self.state.as_mut().expect("ensure_state ran");

        // We process sample-by-sample so the peak-link is exact. For
        // look-ahead > 0 we push the incoming sample, compute the peak
        // across the (new) window, drive the envelope, then pop the
        // oldest sample, scale it by the current envelope, and emit.
        for s in 0..n_samples {
            // Push the new input sample into each channel's delay.
            for ch in 0..n_chan {
                state.delays[ch].push_back(channels[ch][s]);
            }

            // Compute peak across all channels' entire delay windows.
            let mut peak = 0.0f32;
            for ch in 0..n_chan {
                for &v in state.delays[ch].iter() {
                    let a = v.abs();
                    if a > peak {
                        peak = a;
                    }
                }
            }

            // Instantaneous target gain.
            let target = if peak <= state.ceiling_lin {
                1.0
            } else {
                state.ceiling_lin / peak.max(1.0e-12)
            };

            // Attack instant, release smoothed.
            state.env_gain = if target < state.env_gain {
                target
            } else {
                state.alpha_rel * state.env_gain + (1.0 - state.alpha_rel) * target
            };

            // Emit the delayed sample scaled by env_gain.
            for ch in 0..n_chan {
                let delayed = state.delays[ch].pop_front().unwrap_or(0.0);
                channels[ch][s] = delayed * state.env_gain;
            }
        }
    }
}

impl AudioFilter for Limiter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(params.sample_rate, channels.len());
        self.process_channels(&mut channels);
        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_peak_bounded_by_ceiling() {
        let fs = 48_000u32;
        let mut lim = Limiter::new(-3.0, 50.0, 32);
        lim.ensure_state(fs, 1);
        // Mixed amplitude signal — sine + transients.
        let mut x = Vec::with_capacity(4_096);
        for i in 0..4_096 {
            let s = 0.9 * ((i as f32) * 0.05).sin();
            x.push(s + if i == 1_000 { 0.8 } else { 0.0 });
        }
        let mut channels = vec![x];
        lim.process_channels(&mut channels);
        let ceiling_lin = 10.0f32.powf(-3.0 / 20.0);
        // Skip the warm-up window where the delay line is still being
        // filled with zeros (no overshoot, but the output before the
        // first emitted sample is 0).
        let tail = &channels[0][lim.latency_samples() + 8..];
        let peak = tail.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // With look-ahead the ceiling should be respected to high
        // precision; allow a tiny ε for floating-point error.
        assert!(
            peak <= ceiling_lin * 1.001,
            "peak {} > ceiling {}",
            peak,
            ceiling_lin
        );
    }

    #[test]
    fn zero_db_ceiling_clips_oversized_input() {
        let fs = 48_000u32;
        let mut lim = Limiter::new(0.0, 30.0, 0); // causal, 0 dBFS
        lim.ensure_state(fs, 1);
        let x = vec![1.5f32; 1_024];
        let mut channels = vec![x];
        lim.process_channels(&mut channels);
        // After the first sample (causal attack) the output must be ≈ 1.
        let tail = &channels[0][2..];
        let peak = tail.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            (peak - 1.0).abs() < 0.05,
            "0-dB ceiling output peak = {}",
            peak
        );
    }

    #[test]
    fn lookahead_eliminates_impulse_overshoot() {
        let fs = 48_000u32;
        let look = 128usize;
        let mut lim = Limiter::new(0.0, 50.0, look);
        lim.ensure_state(fs, 1);
        // An impulse of amplitude 2.0 at sample N somewhere safely
        // inside the buffer.
        let n = 1_024usize;
        let mut x = vec![0.0f32; 4_096];
        x[n] = 2.0;
        let mut channels = vec![x];
        lim.process_channels(&mut channels);
        // The impulse should appear in the output at sample `n` but
        // scaled to ≤ 1.0 (ceiling). No overshoot.
        let peak = channels[0].iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(peak <= 1.001, "look-ahead overshoot, peak={}", peak);
        // And the *output* impulse value at position `n` (delayed by
        // look-ahead inside our buffer, since we emit a delayed sample
        // each step) should be near the ceiling. The impulse arrives
        // at output position `n` because the input at that step is
        // already in the look-ahead window when env_gain is set.
        let nonzero: Vec<usize> = channels[0]
            .iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > 0.5)
            .map(|(i, _)| i)
            .collect();
        assert!(!nonzero.is_empty(), "impulse vanished entirely from output");
        let any_emitted = nonzero
            .iter()
            .any(|&i| (channels[0][i].abs() - 1.0).abs() < 0.05);
        assert!(any_emitted, "no sample landed near ceiling");
    }

    #[test]
    fn release_governs_recovery_rate() {
        let fs = 48_000u32;
        // Short release.
        let mut lim_fast = Limiter::new(0.0, 5.0, 0);
        lim_fast.ensure_state(fs, 1);
        // Long release.
        let mut lim_slow = Limiter::new(0.0, 500.0, 0);
        lim_slow.ensure_state(fs, 1);

        // Loud transient followed by quiet tail.
        let mut x = vec![0.1f32; fs as usize / 4];
        x[0] = 4.0;
        let mut a = vec![x.clone()];
        let mut b = vec![x.clone()];
        lim_fast.process_channels(&mut a);
        lim_slow.process_channels(&mut b);

        // After 2× the fast-release time constant, gain should be
        // most of the way back to unity; the slow limiter still
        // suppresses the 0.1 input substantially.
        let probe = (fs as f32 * 0.020) as usize; // 20 ms in
        let g_fast = a[0][probe] / 0.1;
        let g_slow = b[0][probe] / 0.1;
        assert!(
            g_fast > g_slow + 0.05,
            "release ordering failed: fast={} slow={}",
            g_fast,
            g_slow
        );
        // And the gain values are bounded in [0, 1].
        assert!(g_fast > 0.0 && g_fast <= 1.0001, "fast g={}", g_fast);
        assert!(g_slow > 0.0 && g_slow <= 1.0001, "slow g={}", g_slow);
    }
}
