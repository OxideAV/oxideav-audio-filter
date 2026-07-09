//! Gain Normalizer (AGC) — slow automatic level control.
//!
//! Aims a long-window RMS estimate at a user-selected target dBFS and
//! applies the resulting linear gain to all channels. Differs from a
//! [`Compressor`](crate::Compressor) in three ways:
//!
//! * The detector is a **long-window RMS** (default 500 ms time
//!   constant), not a peak follower with millisecond attack/release.
//!   The aim is to track the song's *programme level*, not its
//!   individual transients.
//! * There is no threshold — every sample contributes to the
//!   target-tracking integrator.
//! * The gain is **multiplied** straight onto the audio (no static
//!   curve, no knee). The smoothness of the gain trajectory comes
//!   entirely from the slow detector + a separate gain time
//!   constant.
//!
//! # Topology
//!
//! ```text
//!     x[n]  ──►  env_sq = LP(x²)            (slow RMS²)
//!                env    = √env_sq           (linear amplitude)
//!                env_db = 20·log10(env)
//!                err_db = target_db − env_db
//!
//!                gain_db[n] = LP_g(err_db)  (further smoothed)
//!                gain_db    = clamp(gain_db, [-max_atten_db, +max_gain_db])
//!
//!     y[n]  = x[n] · 10^(gain_db[n] / 20)
//! ```
//!
//! Both low-passes use one-pole exponential coefficients
//! `α = 1 − exp(−1 / (τ · f_s))`. The gain trajectory smoother
//! prevents audible "pumping" — a fast detector reacting to a brief
//! transient would otherwise step the gain in 5 ms windows. The
//! defaults bake in a ~1 s overall response, suitable for podcast
//! / streaming-mix levelling.
//!
//! # Silence handling
//!
//! In parallel to the long-window RMS detector, a **fast** peak
//! follower (10 ms time constant) tracks `|x|`. When the fast
//! envelope falls below `silence_threshold_db` (default `−60 dBFS`)
//! the gain integrator freezes — no adjustment is made during
//! pauses, so the next loud passage doesn't get blasted by an
//! integrator that wound up during silence. The gain stays at its
//! pre-silence value.
//!
//! Using a fast detector for the silence check (rather than the slow
//! programme-RMS detector) lets the freeze engage promptly when the
//! input goes quiet — the slow detector takes seconds to decay below
//! the threshold and during that decay the gain would chase the
//! falling level upward.
//!
//! # Parameters
//!
//! * `target_db` — desired programme RMS in dBFS (default `−16`,
//!   range `[-60, 0]`).
//! * `detector_ms` — RMS averaging time constant (default 500 ms,
//!   ≥ 10 ms).
//! * `gain_ms` — gain-trajectory smoothing (default 200 ms,
//!   ≥ 10 ms).
//! * `max_gain_db` — upper limit on applied gain (default `+24`,
//!   range `[0, 60]`).
//! * `max_atten_db` — lower limit on applied attenuation (default
//!   `-24`, range `[-60, 0]`).
//! * `silence_threshold_db` — below this RMS the integrator freezes
//!   (default `-60`).
//!
//! # Design notes
//!
//! Long-window RMS + gain-domain smoothing is the standard automatic
//! gain-control topology.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming gain normaliser / AGC.
#[derive(Debug, Clone)]
pub struct GainNormalizer {
    target_db: f32,
    detector_ms: f32,
    gain_ms: f32,
    max_gain_db: f32,
    max_atten_db: f32,
    silence_threshold_db: f32,
    coeffs: Option<Coeffs>,
    env_sq: f32,
    /// Fast peak-amplitude envelope used only for silence detection.
    fast_env: f32,
    gain_db: f32,
}

#[derive(Debug, Clone, Copy)]
struct Coeffs {
    sample_rate: u32,
    alpha_det: f32,
    alpha_gain: f32,
    /// 10 ms fast attack/release pole for the silence detector.
    alpha_fast: f32,
}

impl GainNormalizer {
    /// New AGC with podcast-style defaults (-16 LUFS-ish target,
    /// 500 ms detector, 200 ms gain smoother).
    pub fn new() -> Self {
        Self::with(-16.0, 500.0, 200.0)
    }

    /// Custom-parameter constructor.
    pub fn with(target_db: f32, detector_ms: f32, gain_ms: f32) -> Self {
        Self {
            target_db: crate::clamp_param(target_db, -16.0, -60.0, 0.0),
            detector_ms: detector_ms.max(10.0),
            gain_ms: gain_ms.max(10.0),
            max_gain_db: 24.0,
            max_atten_db: -24.0,
            silence_threshold_db: -60.0,
            coeffs: None,
            env_sq: 0.0,
            fast_env: 0.0,
            gain_db: 0.0,
        }
    }

    /// Builder: cap the upward gain (default `+24 dB`).
    pub fn with_max_gain_db(mut self, db: f32) -> Self {
        self.max_gain_db = crate::clamp_param(db, 24.0, 0.0, 60.0);
        self
    }

    /// Builder: cap the downward attenuation (default `-24 dB`).
    pub fn with_max_atten_db(mut self, db: f32) -> Self {
        self.max_atten_db = crate::clamp_param(db, -24.0, -60.0, 0.0);
        self
    }

    /// Builder: silence threshold below which the integrator freezes
    /// (default `-60 dBFS`).
    pub fn with_silence_threshold_db(mut self, db: f32) -> Self {
        self.silence_threshold_db = db.clamp(-120.0, 0.0);
        self
    }

    /// Current target RMS in dBFS.
    pub fn target_db(&self) -> f32 {
        self.target_db
    }

    /// Most recently applied gain in dB. `0` = bypass.
    pub fn current_gain_db(&self) -> f32 {
        self.gain_db
    }

    /// Most recently observed smoothed RMS in dBFS (`-Inf` if silent).
    pub fn current_level_db(&self) -> f32 {
        if self.env_sq <= 0.0 {
            -120.0
        } else {
            10.0 * self.env_sq.log10()
        }
    }

    /// Reset detector + gain state.
    pub fn reset(&mut self) {
        self.env_sq = 0.0;
        self.fast_env = 0.0;
        self.gain_db = 0.0;
    }

    fn ensure_coeffs(&mut self, sample_rate: u32) {
        let need = !matches!(self.coeffs, Some(c) if c.sample_rate == sample_rate);
        if !need {
            return;
        }
        let fs = sample_rate as f32;
        let alpha = |ms: f32| 1.0 - (-1.0 / (ms.max(0.1) * 1.0e-3 * fs)).exp();
        self.coeffs = Some(Coeffs {
            sample_rate,
            alpha_det: alpha(self.detector_ms),
            alpha_gain: alpha(self.gain_ms),
            alpha_fast: alpha(10.0),
        });
    }
}

impl Default for GainNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for GainNormalizer {
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

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_samples {
            // Channel-mean of x² is the standard programme-RMS² we
            // want to track (peak-link would over-react to a hot
            // single channel). Track |x| in parallel for the silence
            // detector — that's a fast envelope so the freeze
            // engages promptly when input goes quiet.
            let mut s = 0.0f32;
            let mut a = 0.0f32;
            for c_buf in channels.iter().take(n_chan) {
                let v = c_buf[i];
                s += v * v;
                let av = v.abs();
                if av > a {
                    a = av;
                }
            }
            let inst_sq = s / n_chan as f32;
            self.env_sq += c.alpha_det * (inst_sq - self.env_sq);
            self.fast_env += c.alpha_fast * (a - self.fast_env);

            // Convert RMS² → level dB; convert fast peak → fast dB
            // for the silence comparison.
            let level_db = if self.env_sq <= 1.0e-12 {
                -120.0
            } else {
                10.0 * self.env_sq.log10()
            };
            let fast_db = 20.0 * self.fast_env.max(1e-6).log10();
            if fast_db > self.silence_threshold_db {
                let target = (self.target_db - level_db).clamp(self.max_atten_db, self.max_gain_db);
                self.gain_db += c.alpha_gain * (target - self.gain_db);
            }
            // Apply gain to every channel.
            let g = 10f32.powf(self.gain_db / 20.0);
            for c_buf in channels.iter_mut().take(n_chan) {
                c_buf[i] *= g;
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

    fn rms_db(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        let r = (s / x.len() as f64).sqrt() as f32;
        20.0 * r.max(1e-6).log10()
    }

    fn sine(amp: f32, freq: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| amp * (i as f32 * w).sin()).collect()
    }

    #[test]
    fn quiet_signal_gets_pushed_up_to_target() {
        let fs = 48_000u32;
        // 4 s lets a 500 ms detector + 200 ms gain smoother fully settle.
        let n = (fs as f32 * 4.0) as usize;
        // -40 dBFS sine (RMS = amp/√2 → -43 dBFS-ish).
        let amp = 10f32.powf(-40.0 / 20.0);
        let samples = sine(amp, 500.0, fs, n);
        let frame = make_f32_mono(&samples);
        // Target -16, max_gain +24 → asks for ~+27 dB but ceiling at +24.
        let mut a = GainNormalizer::with(-16.0, 500.0, 200.0);
        let out = a.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Look at the steady-state tail.
        let tail = &got[(fs as usize * 3)..];
        let out_db = rms_db(tail);
        // Without the cap the AGC would hit -16; with the +24 dB cap
        // it can only lift -43 dBFS up to ~-19 dBFS.
        assert!(
            out_db > -21.5 && out_db < -15.0,
            "AGC steady-state level out of expected band: {out_db} dBFS"
        );
    }

    #[test]
    fn loud_signal_gets_pushed_down_to_target() {
        let fs = 48_000u32;
        let n = (fs as f32 * 4.0) as usize;
        // -3 dBFS sine (RMS ≈ -6 dBFS).
        let amp = 10f32.powf(-3.0 / 20.0);
        let samples = sine(amp, 500.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut a = GainNormalizer::with(-16.0, 500.0, 200.0);
        let out = a.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[(fs as usize * 3)..];
        let out_db = rms_db(tail);
        assert!(
            (out_db - (-16.0)).abs() < 1.5,
            "AGC didn't pull loud sine to target: {out_db} dBFS"
        );
    }

    #[test]
    fn at_target_is_passthrough() {
        let fs = 48_000u32;
        let n = (fs as f32 * 3.0) as usize;
        // -16 dBFS sine RMS already at the target.
        let amp = 10f32.powf(-16.0 / 20.0) * (2.0_f32.sqrt());
        let samples = sine(amp, 500.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut a = GainNormalizer::new();
        let out = a.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let in_db = rms_db(&samples[(fs as usize * 2)..]);
        let out_db = rms_db(&got[(fs as usize * 2)..]);
        assert!(
            (in_db - out_db).abs() < 1.0,
            "at-target signal drifted: in={in_db} out={out_db}"
        );
    }

    #[test]
    fn silence_freezes_gain() {
        let fs = 48_000u32;
        let n = (fs as f32 * 2.0) as usize;
        // First, feed a steady-state signal so the AGC commits to a
        // gain. Then feed silence and confirm the gain doesn't drift
        // upwards on its own (no integrator wind-up).
        let amp = 10f32.powf(-30.0 / 20.0);
        let samples1 = sine(amp, 500.0, fs, n);
        let frame1 = make_f32_mono(&samples1);
        let mut a = GainNormalizer::new();
        a.process(&frame1, f32_mono(fs)).unwrap();
        let gain_after = a.current_gain_db();

        // Two seconds of silence.
        let samples2 = vec![0.0f32; n];
        let frame2 = make_f32_mono(&samples2);
        a.process(&frame2, f32_mono(fs)).unwrap();
        let gain_post = a.current_gain_db();
        assert!(
            (gain_post - gain_after).abs() < 1.0,
            "silence triggered integrator wind-up: before={gain_after} after={gain_post}"
        );
    }

    #[test]
    fn max_atten_cap_honoured() {
        let fs = 48_000u32;
        let n = (fs as f32 * 4.0) as usize;
        // Very loud sine (0 dBFS-ish) with a target far below + a tight
        // -6 dB attenuation ceiling.
        let samples = sine(1.0, 500.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut a = GainNormalizer::with(-30.0, 500.0, 200.0).with_max_atten_db(-6.0);
        a.process(&frame, f32_mono(fs)).unwrap();
        let g = a.current_gain_db();
        assert!(
            (-6.1..=-5.0).contains(&g),
            "AGC ignored -6 dB attenuation cap: gain = {g} dB"
        );
    }
}
