//! Transient Designer — attack / sustain envelope shaping.
//!
//! A transient shaper detects per-sample whether
//! the instantaneous signal is in an **attack** phase (rising) or in a
//! **sustain** phase (decaying), and applies independent gain
//! adjustments to each. Boosting attack adds punch to drums; cutting
//! attack tames percussive transients without the side-effects of
//! compression (no threshold, no make-up gain, no level dependence).
//! Sustain shaping changes the perceived room/decay tail.
//!
//! # Two-envelope detector
//!
//! The standard topology runs two parallel envelope followers on
//! `|x[n]|`: a **fast** one (a few ms) that tracks the actual peak
//! shape, and a **slow** one (tens to hundreds of ms) that lags
//! behind. Their ratio yields the shape information:
//!
//! ```text
//! env_fast[n] = LP_fast(|x[n]|)        // chases transients quickly
//! env_slow[n] = LP_slow(|x[n]|)        // long-term level
//!
//! attack_factor  = max(0, env_fast - env_slow) / env_slow  // > 0 during onsets
//! sustain_factor = max(0, env_slow - env_fast) / env_slow  // > 0 during decay
//! ```
//!
//! A positive `attack` knob amplifies the attack factor; a negative
//! `attack` attenuates onsets. `sustain` does the same to the
//! decay-side factor. The output gain per sample is therefore
//!
//! ```text
//! g[n] = 1 + attack · attack_factor[n] - (1 - sustain_gain) · sustain_factor[n]
//! ```
//!
//! with `attack ∈ [-1, +1]` and `sustain ∈ [-1, +1]` clamped at the
//! constructor and gain clamped at runtime to `[0, 8]` to avoid
//! pathological boosts when `env_slow` collapses to zero.
//!
//! Both detectors use the one-pole exponential coefficient
//! `α = 1 − exp(−1 / (τ · f_s))` from the [`EnvelopeFollower`]
//! reference. The fast attack defaults to 1 ms, the slow attack to
//! 35 ms; the release times trail by an order of magnitude — the
//! conventional operating ranges for this effect class.
//!
//! # Parameters
//!
//! * `attack` — onset gain in `[-1, +1]`. `0` = no change, `+1` =
//!   strong onset boost, `-1` = transient softening.
//! * `sustain` — tail gain in `[-1, +1]`. `0` = no change, `+1` = tail
//!   lift (more "room"), `-1` = drier decay.
//! * `attack_ms_fast` / `attack_ms_slow` — detector time-constants
//!   (default 1.0 / 35.0 ms; clamped ≥ 0.1 ms).
//!
//! # Design notes
//!
//! The two-envelope difference detector is standard dynamics-processing
//! practice; the default fast/slow time constants (1 ms / 35 ms,
//! releases trailing ~10x) are the conventional operating ranges for
//! this effect class.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming transient designer.
#[derive(Debug, Clone)]
pub struct TransientDesigner {
    attack: f32,
    sustain: f32,
    attack_ms_fast: f32,
    attack_ms_slow: f32,
    /// Set lazily on the first frame (sample rate is unknown until then).
    coeffs: Option<Coeffs>,
    /// Per-channel state for the two detectors. Lazily sized on first frame.
    state: Vec<DetectorState>,
}

#[derive(Debug, Clone, Copy)]
struct Coeffs {
    sample_rate: u32,
    alpha_fast_atk: f32,
    alpha_fast_rel: f32,
    alpha_slow_atk: f32,
    alpha_slow_rel: f32,
}

#[derive(Debug, Clone, Copy, Default)]
struct DetectorState {
    env_fast: f32,
    env_slow: f32,
}

impl TransientDesigner {
    /// New designer with neutral defaults (`attack=0, sustain=0`).
    pub fn new() -> Self {
        Self::with(0.0, 0.0, 1.0, 35.0)
    }

    /// Custom-parameter constructor.
    pub fn with(attack: f32, sustain: f32, attack_ms_fast: f32, attack_ms_slow: f32) -> Self {
        Self {
            attack: crate::clamp_param(attack, 0.0, -1.0, 1.0),
            sustain: crate::clamp_param(sustain, 0.0, -1.0, 1.0),
            attack_ms_fast: attack_ms_fast.max(0.1),
            attack_ms_slow: attack_ms_slow.max(0.1),
            coeffs: None,
            state: Vec::new(),
        }
    }

    /// Current attack knob.
    pub fn attack(&self) -> f32 {
        self.attack
    }
    /// Current sustain knob.
    pub fn sustain(&self) -> f32 {
        self.sustain
    }
    /// Current fast time constant (ms).
    pub fn attack_ms_fast(&self) -> f32 {
        self.attack_ms_fast
    }
    /// Current slow time constant (ms).
    pub fn attack_ms_slow(&self) -> f32 {
        self.attack_ms_slow
    }

    /// Reset all per-channel detector state.
    pub fn reset(&mut self) {
        for s in &mut self.state {
            *s = DetectorState::default();
        }
    }

    fn ensure_coeffs(&mut self, sample_rate: u32) {
        let need = !matches!(self.coeffs, Some(c) if c.sample_rate == sample_rate);
        if !need {
            return;
        }
        let fs = sample_rate as f32;
        // Release trails attack by an order of magnitude, giving the
        // conventional exponential-decay tail for this effect class.
        let alpha = |ms: f32| 1.0 - (-1.0 / (ms.max(0.1) * 1.0e-3 * fs)).exp();
        let af_a = alpha(self.attack_ms_fast);
        let af_r = alpha(self.attack_ms_fast * 10.0);
        let as_a = alpha(self.attack_ms_slow);
        let as_r = alpha(self.attack_ms_slow * 10.0);
        self.coeffs = Some(Coeffs {
            sample_rate,
            alpha_fast_atk: af_a,
            alpha_fast_rel: af_r,
            alpha_slow_atk: as_a,
            alpha_slow_rel: as_r,
        });
    }
}

impl Default for TransientDesigner {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for TransientDesigner {
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
        // Grow per-channel state lazily.
        if self.state.len() < n_chan {
            self.state.resize(n_chan, DetectorState::default());
        }
        let c = self.coeffs.expect("coeffs initialised in ensure_coeffs()");

        // Sustain knob in `[-1, +1]` maps to a multiplier on the
        // (slow - fast) factor:
        //   sustain = 0    → 0   (sustain factor ignored)
        //   sustain = +1   → +1  (full sustain lift)
        //   sustain = -1   → -1  (subtract the tail)
        // The attack knob has the same scaling on the (fast - slow) factor.
        let k_atk = self.attack;
        let k_sus = self.sustain;

        for (ch, buf) in channels.iter_mut().enumerate().take(n_chan) {
            let st = &mut self.state[ch];
            for s in buf.iter_mut().take(n_samples) {
                let x = *s;
                let abs = x.abs();
                // Fast detector (rapid chase).
                let af = if abs > st.env_fast {
                    c.alpha_fast_atk
                } else {
                    c.alpha_fast_rel
                };
                st.env_fast += af * (abs - st.env_fast);
                // Slow detector (lazy reference).
                let asw = if abs > st.env_slow {
                    c.alpha_slow_atk
                } else {
                    c.alpha_slow_rel
                };
                st.env_slow += asw * (abs - st.env_slow);

                // Normalise relative to the slow envelope. A small
                // floor prevents division explosion during silence.
                let denom = st.env_slow.max(1e-6);
                let attack_factor = ((st.env_fast - st.env_slow) / denom).max(0.0);
                let sustain_factor = ((st.env_slow - st.env_fast) / denom).max(0.0);

                let mut g = 1.0 + k_atk * attack_factor + k_sus * sustain_factor;
                // Clamp to keep accidental boost finite.
                g = g.clamp(0.0, 8.0);
                *s = x * g;
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

    fn drum_envelope(fs: u32, n: usize) -> Vec<f32> {
        // Three short percussive bursts: a half-cycle 200-Hz sine
        // shaped by a fast attack + exponential decay envelope.
        let mut out = vec![0.0f32; n];
        let bursts_at_s = [0.10f32, 0.40, 0.70];
        for &t in &bursts_at_s {
            let onset = (t * fs as f32) as usize;
            for k in 0..(fs as usize / 8) {
                if onset + k >= n {
                    break;
                }
                let env = (-(k as f32) / (fs as f32 * 0.04)).exp();
                let phase = 2.0 * std::f32::consts::PI * 200.0 * (k as f32) / fs as f32;
                out[onset + k] = 0.7 * env * phase.sin();
            }
        }
        out
    }

    fn peak(x: &[f32]) -> f32 {
        x.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
    }

    fn sum_sq(x: &[f32]) -> f64 {
        x.iter().map(|&v| (v as f64).powi(2)).sum()
    }

    #[test]
    fn neutral_settings_are_identity() {
        let fs = 48_000u32;
        let samples = drum_envelope(fs, fs as usize);
        let frame = make_f32_mono(&samples);
        let mut td = TransientDesigner::new();
        let out = td.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Tolerance: with attack=sustain=0 the gain reduces to 1.0
        // up to floating-point error from the multiply-after-detector.
        for i in 0..samples.len() {
            assert!(
                (got[i] - samples[i]).abs() < 1.0e-4,
                "neutral not identity at i={i}: got={} want={}",
                got[i],
                samples[i]
            );
        }
    }

    #[test]
    fn positive_attack_boosts_drum_peak() {
        let fs = 48_000u32;
        let samples = drum_envelope(fs, fs as usize);
        let dry = peak(&samples);
        let frame = make_f32_mono(&samples);
        let mut td = TransientDesigner::with(0.8, 0.0, 1.0, 50.0);
        let out = td.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let wet = peak(&got);
        assert!(
            wet > dry * 1.05,
            "attack boost did not raise peak: dry={dry} wet={wet}"
        );
    }

    #[test]
    fn negative_attack_softens_drum_peak() {
        let fs = 48_000u32;
        let samples = drum_envelope(fs, fs as usize);
        let dry = peak(&samples);
        let frame = make_f32_mono(&samples);
        let mut td = TransientDesigner::with(-0.8, 0.0, 1.0, 50.0);
        let out = td.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let wet = peak(&got);
        assert!(
            wet < dry * 0.95,
            "attack cut did not lower peak: dry={dry} wet={wet}"
        );
    }

    #[test]
    fn positive_sustain_lifts_decay_energy() {
        let fs = 48_000u32;
        let samples = drum_envelope(fs, fs as usize);
        let frame = make_f32_mono(&samples);
        let mut td = TransientDesigner::with(0.0, 0.8, 1.0, 50.0);
        let out = td.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Measure energy in the decay tail of the first burst — say
        // 50 ms onwards from the first onset.
        let onset = (0.10 * fs as f32) as usize + (fs as usize / 50);
        let tail_end = (0.10 * fs as f32) as usize + (fs as usize / 8);
        let dry_e = sum_sq(&samples[onset..tail_end.min(samples.len())]);
        let wet_e = sum_sq(&got[onset..tail_end.min(got.len())]);
        assert!(
            wet_e > dry_e * 1.02,
            "sustain lift did not raise tail energy: dry={dry_e} wet={wet_e}"
        );
    }

    #[test]
    fn output_stays_bounded() {
        let fs = 48_000u32;
        let samples = drum_envelope(fs, fs as usize);
        let frame = make_f32_mono(&samples);
        // Max-out both knobs to stress the saturator.
        let mut td = TransientDesigner::with(1.0, 1.0, 0.5, 40.0);
        let out = td.process(&frame, f32_mono(fs)).unwrap();
        for v in read_f32(&out[0]) {
            assert!(v.is_finite(), "non-finite output");
            // The internal clamp caps gain at 8 → with a 0.7-peak
            // input the output should never exceed 5.6.
            assert!(v.abs() < 6.0, "transient designer sample {v} out of bounds");
        }
    }
}
