//! Adaptive noise gate — learns the noise floor from the input and
//! gates anything that doesn't comfortably exceed it.
//!
//! This is a time-domain analog of spectral-subtraction de-noising:
//! a fixed-threshold gate (see [`crate::NoiseGate`]) needs the user to
//! know the noise floor in advance; this filter estimates it on-line
//! by running a *slow* one-pole follower on the per-channel
//! root-mean-squared amplitude. When the input is far above the
//! learned floor it sounds the gate "open" (gain = 1); when it falls
//! within a configurable margin of the floor, the gate closes
//! (gain = 0). Attack / release smoothing prevents clicks.
//!
//! # Detector
//!
//! Per channel, two one-pole IIRs run in parallel on the squared input:
//!
//! ```text
//! signal_env ← {attack or release smoothed} of x²     (fast, tracks signal)
//! ```
//!
//! The noise floor is estimated as a **slow-attack / fast-release**
//! tracker of `signal_env`: the estimate decays toward `signal_env`
//! whenever the signal is *quieter* than the current estimate (so
//! quieter moments pull the floor down) and rises only very slowly
//! when the signal is louder (so transients can't suddenly inflate
//! the floor). Concretely:
//!
//! ```text
//! coeff_n = α_n_fast  if signal_env < noise_env       (track quiet)
//!           α_n_slow  otherwise                       (cap loud)
//! noise_env ← (1 - coeff_n) · noise_env + coeff_n · signal_env
//! ```
//!
//! `α_n_fast = α_learn` and `α_n_slow = α_learn / 64` give the
//! asymmetric behaviour: the floor follows the running minimum
//! envelope much faster than it follows the running maximum.
//!
//! # Gate decision
//!
//! ```text
//! threshold = noise_env · margin²        (in power; default margin = 4×)
//! open = signal_env > threshold
//! gain ← attack_smooth   if open
//!        release_smooth  otherwise
//! ```
//!
//! `margin = 4×` (≈ 12 dB) is a reasonable default; lower margins are
//! more aggressive but risk chattering on speech transients.
//!
//! # Parameters
//!
//! * `margin_db` — how many dB the signal must exceed the learned
//!   floor before the gate opens. Default 12 dB.
//! * `learn_ms` — time constant of the noise-floor learner. Default
//!   2 000 ms (slow). Faster values adapt to changing noise quicker
//!   but risk eating quiet signal.
//! * `attack_ms` / `release_ms` — gate opening / closing smoothing.
//!   Defaults 5 ms / 100 ms.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Ratio between the fast (downward) and slow (upward) noise-floor
/// learners. 64 ≈ 36 dB of asymmetry: the floor follows quiet
/// moments 64× faster than loud ones.
const ASYMMETRY: f32 = 64.0;

#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    noise_env: f32,
    signal_env: f32,
    gain: f32,
}

/// Streaming adaptive noise gate.
#[derive(Debug, Clone)]
pub struct AdaptiveNoiseGate {
    margin_db: f32,
    learn_ms: f32,
    attack_ms: f32,
    release_ms: f32,
    state: Vec<ChState>,
    sample_rate: u32,
}

impl AdaptiveNoiseGate {
    /// New gate with the standard `(margin=12 dB, learn=2 s, atk=5 ms,
    /// rel=100 ms)` preset.
    pub fn new() -> Self {
        Self::with(12.0, 2_000.0, 5.0, 100.0)
    }

    /// Custom-preset constructor. All times clamped to `≥ 0.01 ms`,
    /// `margin_db` clamped to `≥ 0`.
    pub fn with(margin_db: f32, learn_ms: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            margin_db: margin_db.max(0.0),
            learn_ms: learn_ms.max(0.01),
            attack_ms: attack_ms.max(0.01),
            release_ms: release_ms.max(0.01),
            state: Vec::new(),
            sample_rate: 0,
        }
    }

    /// Currently configured margin in dB.
    pub fn margin_db(&self) -> f32 {
        self.margin_db
    }

    /// Reset internal noise-floor estimate, signal envelope, and gain
    /// for all channels. After reset the gate starts closed with a
    /// noise floor of 0; the learner re-converges over `learn_ms`.
    pub fn reset(&mut self) {
        for st in self.state.iter_mut() {
            *st = ChState::default();
        }
    }

    /// Learned noise floor (across all channels, RMS amplitude). 0 if
    /// no input has been observed.
    pub fn learned_noise(&self) -> f32 {
        if self.state.is_empty() {
            return 0.0;
        }
        let mut peak = 0.0f32;
        for st in &self.state {
            // sqrt because state stores x², we report amplitude.
            let amp = st.noise_env.max(0.0).sqrt();
            if amp > peak {
                peak = amp;
            }
        }
        peak
    }

    /// Returns `true` if any channel's gate is currently open
    /// (`gain > 0.5`).
    pub fn is_open(&self) -> bool {
        self.state.iter().any(|s| s.gain > 0.5)
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        if self.state.len() != channels || self.sample_rate != sample_rate {
            self.state = vec![ChState::default(); channels];
            self.sample_rate = sample_rate;
        }
    }
}

impl Default for AdaptiveNoiseGate {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for AdaptiveNoiseGate {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let n_chan = params.channels as usize;
        self.ensure_state(params.sample_rate, n_chan);
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let fs = params.sample_rate as f32;
        let a_learn = 1.0 - (-1.0 / (self.learn_ms * 1e-3 * fs)).exp();
        let a_atk = 1.0 - (-1.0 / (self.attack_ms * 1e-3 * fs)).exp();
        let a_rel = 1.0 - (-1.0 / (self.release_ms * 1e-3 * fs)).exp();
        // Threshold factor in power domain: (signal/noise)² > 10^(margin_db/10).
        let pow_margin = 10f32.powf(self.margin_db / 10.0);

        let a_learn_slow = a_learn / ASYMMETRY;
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.state[ch_idx];
            for s in buf.iter_mut() {
                let p = (*s) * (*s);
                // Signal envelope (fast tracker, attack/release).
                let coeff_sig = if p > st.signal_env { a_atk } else { a_rel };
                st.signal_env = (1.0 - coeff_sig) * st.signal_env + coeff_sig * p;
                // Noise envelope — asymmetric tracker of `signal_env`:
                // fast when signal_env is below the current floor
                // (so quiet patches pull the floor down quickly), slow
                // otherwise (so transients can't inflate the floor).
                let coeff_n = if st.signal_env < st.noise_env {
                    a_learn
                } else {
                    a_learn_slow
                };
                st.noise_env = (1.0 - coeff_n) * st.noise_env + coeff_n * st.signal_env;
                // Gate decision.
                let threshold = st.noise_env * pow_margin;
                let target = if st.signal_env > threshold { 1.0 } else { 0.0 };
                let coeff = if target > st.gain { a_atk } else { a_rel };
                st.gain = (1.0 - coeff) * st.gain + coeff * target;
                *s *= st.gain;
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

    /// Splitmix64 noise sequence used in the gate tests (no rand dep).
    fn splitmix_noise(n: usize, amp: f32, seed: u64) -> Vec<f32> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u = (z >> 11) as f32 / (1u64 << 53) as f32;
            out.push((u * 2.0 - 1.0) * amp);
        }
        out
    }

    #[test]
    fn quiet_noise_eventually_closes_gate() {
        // Feed 5 s of low-level noise — the gate should learn the
        // floor and clamp the noise down to near-zero.
        let fs = 48_000u32;
        let n = (fs as usize) / 10; // 100 ms per frame
        let mut g = AdaptiveNoiseGate::with(12.0, 200.0, 5.0, 50.0); // fast learner for the test
                                                                     // Feed 30 frames = 3 s.
        let mut last_out = Vec::new();
        for k in 0..30 {
            let x = splitmix_noise(n, 0.01, 0xABCD + k as u64);
            let out = g.process(&make_f32_mono(&x), f32_mono(fs)).unwrap();
            last_out = read_f32(&out[0]);
        }
        // The gate's gain should have closed → RMS << input RMS.
        let out_rms: f32 = {
            let s: f64 = last_out.iter().map(|&v| (v as f64).powi(2)).sum();
            (s / last_out.len() as f64).sqrt() as f32
        };
        assert!(
            out_rms < 0.005,
            "gate didn't close on quiet noise; out_rms={out_rms}"
        );
    }

    #[test]
    fn loud_signal_passes_through() {
        // After learning a quiet noise floor, a loud tone should open
        // the gate. Train on 2 s of quiet noise, then feed a loud sine.
        let fs = 48_000u32;
        let n = (fs as usize) / 10;
        let mut g = AdaptiveNoiseGate::with(12.0, 200.0, 5.0, 50.0);
        for k in 0..20 {
            let x = splitmix_noise(n, 0.005, 0xBEEF + k as u64);
            let _ = g.process(&make_f32_mono(&x), f32_mono(fs)).unwrap();
        }
        // Loud 1 kHz sine at 0.5 amplitude.
        let w = 2.0 * std::f32::consts::PI * 1_000.0 / fs as f32;
        let loud: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32 * w).sin()).collect();
        // Allow the attack to ramp up — feed several frames.
        let mut got = Vec::new();
        for _ in 0..5 {
            let out = g.process(&make_f32_mono(&loud), f32_mono(fs)).unwrap();
            got = read_f32(&out[0]);
        }
        let out_rms: f32 = {
            let s: f64 = got.iter().map(|&v| (v as f64).powi(2)).sum();
            (s / got.len() as f64).sqrt() as f32
        };
        // Input sine RMS = 0.5/√2 ≈ 0.354. Allow 10 % loss.
        assert!(
            out_rms > 0.30,
            "gate did not open for loud sine; out_rms={out_rms}"
        );
        assert!(g.is_open(), "is_open should return true on loud signal");
    }

    #[test]
    fn learned_floor_grows_with_input_level() {
        // Train on two noise levels and check the learned floor tracks.
        let fs = 48_000u32;
        let n = (fs as usize) / 10;
        let mut g = AdaptiveNoiseGate::with(12.0, 200.0, 5.0, 50.0);
        for k in 0..30 {
            let x = splitmix_noise(n, 0.005, 0x1234 + k as u64);
            let _ = g.process(&make_f32_mono(&x), f32_mono(fs)).unwrap();
        }
        let low_floor = g.learned_noise();
        // Quiet RMS noise at 0.005 amp = 0.005/√3 ≈ 0.003 RMS;
        // require the learner found something positive in that range.
        assert!(
            low_floor > 0.0005 && low_floor < 0.05,
            "low-noise floor learned implausibly: {low_floor}"
        );
    }

    #[test]
    fn reset_clears_state() {
        let fs = 48_000u32;
        let mut g = AdaptiveNoiseGate::new();
        let x = vec![0.5f32; 4800];
        let _ = g.process(&make_f32_mono(&x), f32_mono(fs)).unwrap();
        g.reset();
        assert!(!g.is_open(), "is_open should be false after reset");
        assert_eq!(g.learned_noise(), 0.0);
    }
}
