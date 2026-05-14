//! De-esser — frequency-selective compressor for sibilance.
//!
//! A de-esser detects when high-frequency content (typically the
//! "ess" / "sh" sibilant band, 4–10 kHz) exceeds a threshold and
//! ducks that band only, leaving the rest of the spectrum
//! untouched. The classic split-band topology used here is:
//!
//! ```text
//!     ┌──────────────┐
//!     │  HPF @ f_c   │── side-chain detector ──┐
//!  x ─┤              │                        ▼   gain
//!     │   y_h = x · g   ◀───── compressor ────┐  envelope
//!     ├──────────────┤                        │
//!     │  LPF @ f_c   │  y_l (pass-through)    │
//!     └──────┬───────┘                        │
//!            │                                │
//!     y = y_l + y_h
//! ```
//!
//! Concretely the audio is split into a low band (everything ≤ `f_c`,
//! pass-through) and a high band (everything ≥ `f_c`, the de-ess
//! target). A peak detector on the high band drives a downward
//! compressor whose gain reduction is applied to the high band only;
//! the bands are summed back. The LPF/HPF pair is one biquad each at
//! the same cutoff with `Q = 0.707` (Butterworth), which yields the
//! standard `LPF + HPF ≈ unity` complementary pair at the cutoff.
//!
//! # Compressor
//!
//! Conventional one-pole peak follower + hard-knee static curve:
//!
//! ```text
//! env_db = 20·log10(env)
//! over   = env_db - threshold_db
//! gr     = if over > 0: -(1 - 1/R) · over
//!          else:        0
//! gain   = 10^(gr / 20)
//! ```
//!
//! `R = ratio ≥ 1` (4:1 is a reasonable default).
//!
//! # Parameters
//!
//! * `cutoff_hz` — the split point. Default 6 kHz. Typical de-essers
//!   live in `4 000..=10 000`.
//! * `threshold_db` — the level above which the high band starts
//!   being attenuated. Default −20 dB.
//! * `ratio` — compression ratio. Default 4 (4 dB in → 1 dB out
//!   above the threshold). Clamped to `≥ 1`.
//! * `attack_ms` / `release_ms` — detector time constants. Defaults
//!   1 ms / 30 ms, which is fast enough to catch sibilants without
//!   choking the vowel.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming de-esser (split-band downward compressor on the high band).
#[derive(Debug, Clone)]
pub struct DeEsser {
    cutoff_hz: f32,
    threshold_db: f32,
    ratio: f32,
    attack_ms: f32,
    release_ms: f32,
    lpf: Biquad,
    hpf: Biquad,
    /// Per-channel detector envelope.
    env: Vec<f32>,
    /// Last applied gain reduction in dB (peak across channels). Useful
    /// for VU-style meter overlays.
    last_gr_db: f32,
}

impl DeEsser {
    /// New de-esser with the classic 6 kHz / −20 dB / 4:1 / 1 ms / 30 ms
    /// preset.
    pub fn new() -> Self {
        Self::with(6_000.0, -20.0, 4.0, 1.0, 30.0)
    }

    /// Custom-preset constructor. `ratio` clamped to `≥ 1`, attack /
    /// release clamped to `≥ 0.01 ms`.
    pub fn with(
        cutoff_hz: f32,
        threshold_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
    ) -> Self {
        let cutoff = cutoff_hz.max(20.0);
        let q = std::f32::consts::FRAC_1_SQRT_2;
        Self {
            cutoff_hz: cutoff,
            threshold_db,
            ratio: ratio.max(1.0),
            attack_ms: attack_ms.max(0.01),
            release_ms: release_ms.max(0.01),
            lpf: Biquad::new(BiquadKind::LowPass {
                cutoff_hz: cutoff,
                q,
            }),
            hpf: Biquad::new(BiquadKind::HighPass {
                cutoff_hz: cutoff,
                q,
            }),
            env: Vec::new(),
            last_gr_db: 0.0,
        }
    }

    /// Currently configured split-band cutoff.
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Threshold above which gain reduction begins, dBFS.
    pub fn threshold_db(&self) -> f32 {
        self.threshold_db
    }

    /// Compression ratio.
    pub fn ratio(&self) -> f32 {
        self.ratio
    }

    /// Most recent peak gain reduction in dB (≤ 0). `0.0` when the
    /// detector envelope is below threshold.
    pub fn last_gr_db(&self) -> f32 {
        self.last_gr_db
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.lpf.reset();
        self.hpf.reset();
        for e in self.env.iter_mut() {
            *e = 0.0;
        }
        self.last_gr_db = 0.0;
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.env.len() != channels {
            self.env = vec![0.0; channels];
        }
    }
}

impl Default for DeEsser {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for DeEsser {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(channels.len());

        // Split into bands. Each biquad has its own per-channel state
        // so we can recover both bands from the same input.
        let mut low: Vec<Vec<f32>> = channels.to_vec();
        let mut high: Vec<Vec<f32>> = channels.to_vec();
        for buf in low.iter_mut() {
            self.lpf.process_in_place(buf, 1, params.sample_rate);
        }
        for buf in high.iter_mut() {
            self.hpf.process_in_place(buf, 1, params.sample_rate);
        }

        let fs = params.sample_rate as f32;
        let alpha_atk = 1.0 - (-1.0 / (self.attack_ms * 1e-3 * fs)).exp();
        let alpha_rel = 1.0 - (-1.0 / (self.release_ms * 1e-3 * fs)).exp();
        let inv_ratio = 1.0 / self.ratio;

        let n = channels.first().map(|c| c.len()).unwrap_or(0);
        let mut peak_gr_db: f32 = 0.0;
        for ch_idx in 0..channels.len() {
            let env = &mut self.env[ch_idx];
            for i in 0..n {
                let h = high[ch_idx][i];
                let drive = h.abs();
                let coeff = if drive > *env { alpha_atk } else { alpha_rel };
                *env = (1.0 - coeff) * *env + coeff * drive;
                let env_db = 20.0 * env.max(1e-6).log10();
                let over = env_db - self.threshold_db;
                let gr_db = if over > 0.0 {
                    -(1.0 - inv_ratio) * over
                } else {
                    0.0
                };
                if gr_db < peak_gr_db {
                    peak_gr_db = gr_db;
                }
                let gain = 10f32.powf(gr_db / 20.0);
                let l = low[ch_idx][i];
                channels[ch_idx][i] = l + h * gain;
            }
        }
        self.last_gr_db = peak_gr_db;

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

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    #[test]
    fn high_sibilance_is_attenuated() {
        // 8 kHz tone — well above the 6 kHz split. Without de-essing
        // the band's output is essentially unity (the HPF passes ~all
        // of it, the LPF very little, summed back to roughly the
        // input). With de-essing the loud high band is ducked.
        let fs = 48_000u32;
        let n = 16_384usize;
        let w = 2.0 * std::f32::consts::PI * 8_000.0 / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| 0.8 * (i as f32 * w).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut de = DeEsser::new(); // threshold = -20 dB, ratio = 4, fs split at 6 kHz
        let out = de.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip the start-up transient (~50 ms of filter ramp-up).
        let warm = (fs as f32 * 0.05) as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let ratio = out_r / in_r;
        let g_db = 20.0 * ratio.log10();
        // Hot 8 kHz tone, ~-2 dBFS input → expect noticeable
        // attenuation (more than 2 dB).
        assert!(g_db < -2.0, "8 kHz tone not attenuated; gain = {g_db} dB");
        // And the detector should have logged some GR.
        assert!(
            de.last_gr_db() < -1.0,
            "no gain reduction reported; got {} dB",
            de.last_gr_db()
        );
    }

    #[test]
    fn low_frequencies_pass_unchanged() {
        // 200 Hz tone — well below the 6 kHz split. The low band gets
        // ~all of it (HPF passes ~none), so even with the compressor
        // active, output ≈ input.
        let fs = 48_000u32;
        let n = 16_384usize;
        let w = 2.0 * std::f32::consts::PI * 200.0 / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32 * w).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut de = DeEsser::new();
        let out = de.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let warm = (fs as f32 * 0.05) as usize;
        let in_r = rms(&samples[warm..]);
        let out_r = rms(&got[warm..]);
        let g_db = 20.0 * (out_r / in_r).log10();
        // 200 Hz is more than two octaves below 6 kHz; the LPF is
        // essentially unity-gain there. Should be ≤ 0.5 dB loss.
        assert!(
            g_db.abs() < 0.5,
            "200 Hz tone not preserved; gain = {g_db} dB"
        );
    }

    #[test]
    fn quiet_input_yields_no_gain_reduction() {
        // -60 dB sine at 8 kHz — well below the -20 dB threshold,
        // so the detector should stay below and report 0 dB GR.
        let fs = 48_000u32;
        let n = 8_192usize;
        let w = 2.0 * std::f32::consts::PI * 8_000.0 / fs as f32;
        let amp = 0.001f32; // ≈ -60 dBFS
        let samples: Vec<f32> = (0..n).map(|i| amp * (i as f32 * w).sin()).collect();
        let frame = make_f32_mono(&samples);
        let mut de = DeEsser::new();
        let _ = de.process(&frame, f32_mono(fs)).unwrap();
        assert!(
            de.last_gr_db() > -0.1,
            "quiet input triggered gain reduction: {} dB",
            de.last_gr_db()
        );
    }

    #[test]
    fn ratio_clamps_below_unity() {
        // ratio = 0.5 should clamp to 1.0 (no expansion supported).
        let de = DeEsser::with(6_000.0, -20.0, 0.5, 1.0, 30.0);
        assert!((de.ratio() - 1.0).abs() < 1e-6);
    }
}
