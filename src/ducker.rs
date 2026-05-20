//! Ducker — internally-keyed sidechain compressor.
//!
//! A "ducker" is the broadcast-radio classic: voice-over fades the
//! music down whenever the announcer speaks. Structurally it's a
//! downward compressor whose detector input is **not** the audio
//! being processed, but a separate key signal.
//!
//! This crate's [`AudioFilter`] contract is single-frame in / single-
//! frame out — we don't have a second port for the key. The standard
//! workaround is the **self-keyed ducker**: the key is derived from
//! the audio itself, but with a deliberately slow attack so the
//! compressor only reacts when the input has been loud for a while.
//! For "music vs voice" mixing where the announcer signal happens to
//! be the input, this is exactly the desired behaviour.
//!
//! Alternatively the caller can pre-attenuate one channel of a
//! stereo pair as the "key" and let the ducker react to that — see
//! [`Ducker::with_key_channel`]. With `key_channel = Some(0)` the
//! detector only watches L; both channels are then ducked together.
//!
//! # Topology
//!
//! ```text
//!                ┌────── one-pole peak follower ─────┐
//!     key  ──►  │  env[n] = α · |key[n]| + (1-α)·env  │
//!                └──────────────┬────────────────────┘
//!                               ▼
//!                     |env| > threshold ?
//!                               │
//!                          gain_db = (threshold_db - env_db) · slope
//!                               │
//!                          (slope = (ratio - 1) / ratio,
//!                           ratio ∈ [1, ∞))
//!                               ▼
//!     x  ──────────────►  ×  10^(gain_db/20) · max_gain_reduction_clip  ──► y
//! ```
//!
//! When the key envelope is above `threshold_db`, the static curve
//! computes an attenuation in dB and applies it (sample-by-sample
//! linear gain) to **all** input channels. When the key is quiet,
//! gain returns to unity at the `release_ms` time constant. The
//! attack / release on the *gain trajectory* is separate from the
//! envelope detector's smoothing — the detector chases the key
//! quickly (1 ms attack) while the gain trajectory follows
//! `attack_ms` / `release_ms` for the audible duck.
//!
//! # Parameters
//!
//! * `threshold_db` — key level above which ducking begins (default
//!   `-20 dBFS`, range `[-100, 0]`).
//! * `ratio` — duck slope, `[1, 20]` (default 8.0). `1` = no
//!   ducking; `∞` makes the duck a hard gate.
//! * `attack_ms` — gain-trajectory attack (default 5 ms, ≥ 0.1 ms).
//! * `release_ms` — gain-trajectory release (default 250 ms,
//!   ≥ 0.1 ms).
//! * `max_reduction_db` — safety floor on the gain reduction so the
//!   output never goes below this many dB (default `-30 dB`).
//! * `key_channel` — `None` (peak-link over all channels) or
//!   `Some(c)` to key only off channel `c`.
//!
//! # References
//!
//! Static compressor curve + dual-time-constant detector are
//! textbook (Zölzer DAFX, "Dynamic Range Processing"). No external
//! ducker source was consulted.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming ducker.
#[derive(Debug, Clone)]
pub struct Ducker {
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    max_reduction_db: f32,
    key_channel: Option<usize>,
    coeffs: Option<Coeffs>,
    /// Smoothed key envelope (amplitude domain).
    env: f32,
    /// Current applied gain (linear), 1.0 = unity.
    gain: f32,
}

#[derive(Debug, Clone, Copy)]
struct Coeffs {
    sample_rate: u32,
    alpha_det: f32,
    alpha_atk: f32,
    alpha_rel: f32,
}

impl Ducker {
    /// New ducker with broadcast-style defaults.
    pub fn new() -> Self {
        Self::with(-20.0, 8.0, 5.0, 250.0)
    }

    /// Custom-parameter ducker. Other fields take their defaults.
    pub fn with(threshold_db: f32, ratio: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            threshold_db: threshold_db.clamp(-100.0, 0.0),
            ratio: ratio.clamp(1.0, 20.0),
            attack_ms: attack_ms.max(0.1),
            release_ms: release_ms.max(0.1),
            max_reduction_db: -30.0,
            key_channel: None,
            coeffs: None,
            env: 0.0,
            gain: 1.0,
        }
    }

    /// Builder: clamp the maximum gain reduction (default −30 dB).
    pub fn with_max_reduction_db(mut self, db: f32) -> Self {
        self.max_reduction_db = db.clamp(-100.0, 0.0);
        self
    }

    /// Builder: pick a single channel as the detector key. `None`
    /// reverts to "peak-link over all input channels".
    pub fn with_key_channel(mut self, ch: Option<usize>) -> Self {
        self.key_channel = ch;
        self
    }

    /// Current threshold (dBFS).
    pub fn threshold_db(&self) -> f32 {
        self.threshold_db
    }
    /// Current ratio.
    pub fn ratio(&self) -> f32 {
        self.ratio
    }
    /// Current attack (ms).
    pub fn attack_ms(&self) -> f32 {
        self.attack_ms
    }
    /// Current release (ms).
    pub fn release_ms(&self) -> f32 {
        self.release_ms
    }
    /// Latest applied gain (linear). 1.0 = no duck, 0.0 = silent.
    pub fn current_gain(&self) -> f32 {
        self.gain
    }
    /// Latest applied gain in dB (0 = no duck, negative = ducked).
    pub fn current_gain_db(&self) -> f32 {
        20.0 * self.gain.max(1e-6).log10()
    }

    /// Reset envelope + gain state.
    pub fn reset(&mut self) {
        self.env = 0.0;
        self.gain = 1.0;
    }

    fn ensure_coeffs(&mut self, sample_rate: u32) {
        let need = !matches!(self.coeffs, Some(c) if c.sample_rate == sample_rate);
        if !need {
            return;
        }
        let fs = sample_rate as f32;
        let alpha = |ms: f32| 1.0 - (-1.0 / (ms.max(0.1) * 1.0e-3 * fs)).exp();
        // Detector is fast (1 ms) so it tracks transients in the key.
        self.coeffs = Some(Coeffs {
            sample_rate,
            alpha_det: alpha(1.0),
            alpha_atk: alpha(self.attack_ms),
            alpha_rel: alpha(self.release_ms),
        });
    }
}

impl Default for Ducker {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for Ducker {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = channels.len();
        if n_chan == 0 {
            let out = encode_from_f32(params.format, params.channels, input, &channels)?;
            return Ok(vec![out]);
        }
        let n_samples = channels[0].len();
        self.ensure_coeffs(params.sample_rate);
        let c = self.coeffs.expect("coeffs initialised in ensure_coeffs()");

        let threshold_lin = 10f32.powf(self.threshold_db / 20.0);
        let max_red_lin = 10f32.powf(self.max_reduction_db / 20.0);
        // The compressor slope in dB-per-dB above threshold is
        // `(ratio - 1) / ratio`. At ratio = 1 the slope is 0 (no
        // ducking); at ratio = ∞ it is 1 (full reduction).
        let slope = (self.ratio - 1.0) / self.ratio;

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            // Compose key sample. peak-link → max(|x_ch|); single
            // channel → `|channels[ch][i]|`.
            let key = match self.key_channel {
                Some(ch) if ch < n_chan => channels[ch][i].abs(),
                _ => {
                    let mut m = 0.0f32;
                    for c_buf in channels.iter().take(n_chan) {
                        let a = c_buf[i].abs();
                        if a > m {
                            m = a;
                        }
                    }
                    m
                }
            };

            // Detector: one-pole peak follower with fast α_det in
            // BOTH directions. The compressor's "attack vs release"
            // distinction lives in the gain trajectory below; the
            // detector's job is just to give a stable instantaneous
            // estimate of the key amplitude.
            self.env += c.alpha_det * (key - self.env);

            // Static curve: dB-domain. Above the threshold, target
            // gain is `(thr_db - env_db) · slope`; below threshold,
            // target gain is 0 dB.
            let target_gain = if self.env > threshold_lin {
                let env_db = 20.0 * self.env.max(1e-9).log10();
                let reduction_db = (self.threshold_db - env_db) * slope; // negative
                let target_lin = 10f32.powf(reduction_db / 20.0);
                target_lin.max(max_red_lin)
            } else {
                1.0
            };

            // Gain trajectory: attack when going down, release when
            // coming back up.
            let alpha = if target_gain < self.gain {
                c.alpha_atk
            } else {
                c.alpha_rel
            };
            self.gain += alpha * (target_gain - self.gain);

            for c_buf in channels.iter_mut().take(n_chan) {
                c_buf[i] *= self.gain;
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

    fn rms(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        (s / x.len() as f64).sqrt() as f32
    }

    fn sine(amp: f32, freq: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| amp * (i as f32 * w).sin()).collect()
    }

    #[test]
    fn quiet_input_is_passthrough() {
        let fs = 48_000u32;
        let n = 4_800usize;
        let amp = 0.05f32; // -26 dBFS — below the default -20 dB threshold
        let samples = sine(amp, 500.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut d = Ducker::new();
        let out = d.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // -26 dBFS sine never crosses -20 dB → gain stays at 1.0 →
        // output ≈ input.
        let in_r = rms(&samples);
        let out_r = rms(&got);
        let delta_db = 20.0 * (out_r / in_r).log10();
        assert!(
            delta_db.abs() < 0.1,
            "below-threshold drift = {delta_db} dB"
        );
    }

    #[test]
    fn loud_input_gets_ducked() {
        let fs = 48_000u32;
        let n = fs as usize; // 1 s — let the gain reach steady state
        let samples = sine(0.7, 500.0, fs, n); // -3 dBFS, well above -20 dB
        let frame = make_f32_mono(&samples);
        let mut d = Ducker::with(-20.0, 8.0, 5.0, 250.0);
        let out = d.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // After 0.5 s the ducker should be near its steady-state
        // reduction. Compare RMS of the second half.
        let half = n / 2;
        let in_r = rms(&samples[half..]);
        let out_r = rms(&got[half..]);
        let delta_db = 20.0 * (out_r / in_r).log10();
        // Expected steady state: env ≈ amp/√2 (sine RMS), env_db ≈
        // -6 dBFS. Reduction = (-20 - (-6)) · (7/8) = -12.25 dB.
        // Allow ± 4 dB for detector-vs-gain dynamics.
        assert!(
            delta_db < -8.0 && delta_db > -16.0,
            "steady-state ducking out of range: {delta_db} dB (want roughly -12)"
        );
    }

    #[test]
    fn gain_reduction_clamped() {
        let fs = 48_000u32;
        let n = fs as usize;
        let samples = sine(1.0, 500.0, fs, n); // 0 dBFS peak
        let frame = make_f32_mono(&samples);
        // Insanely aggressive: ratio 20, threshold -60 → would push
        // reduction to ~-50 dB. Floor it at -10 dB and verify the
        // output never goes below that.
        let mut d = Ducker::with(-60.0, 20.0, 5.0, 250.0).with_max_reduction_db(-10.0);
        let out = d.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let half = n / 2;
        let delta_db = 20.0 * (rms(&got[half..]) / rms(&samples[half..])).log10();
        // Allow a little tail because the ratio-clamped target is
        // -10 dB exactly; gain should not exceed that.
        assert!(
            delta_db > -10.5 && delta_db < -8.0,
            "max-reduction clamp not honoured: delta = {delta_db} dB"
        );
    }

    #[test]
    fn ratio_one_is_passthrough_even_for_loud_input() {
        let fs = 48_000u32;
        let n = fs as usize;
        let samples = sine(0.7, 500.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut d = Ducker::with(-30.0, 1.0, 5.0, 100.0);
        let out = d.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let in_r = rms(&samples[n / 2..]);
        let out_r = rms(&got[n / 2..]);
        let delta_db = 20.0 * (out_r / in_r).log10();
        assert!(
            delta_db.abs() < 0.1,
            "ratio=1 should be bypass: delta = {delta_db} dB"
        );
    }
}
