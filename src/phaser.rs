//! Phaser — N all-pass filter stages with LFO-modulated cutoffs.
//!
//! The classic phaser sound comes from cascading several first-order
//! all-pass sections whose cutoffs sweep up and down with an LFO. Summing
//! the resulting phase-shifted signal back with the dry input creates moving
//! comb-like notches in the spectrum.
//!
//! # First-order all-pass design
//!
//! For a normalised cutoff `w = 2π · f_c / fs`, the first-order all-pass
//! has the difference equation
//!
//! ```text
//! a    = (1 - tan(w/2)) / (1 + tan(w/2))
//! y[n] = a · x[n] + x[n-1] - a · y[n-1]
//! ```
//!
//! `|H(jω)| = 1` for every ω (hence "all-pass"); the filter affects only
//! phase. Cascading N such sections gives `N · 90°` of phase shift at the
//! corner frequency, so an N-stage phaser produces N/2 notches in the
//! dry+wet sum.
//!
//! # Recurrence
//!
//! ```text
//! lfo[n]  = sin(2π · rate · n / fs)
//! fc[n]   = depth_hz · (1 + lfo[n]) / 2     ∈ [0, depth_hz]
//! wet[n]  = AP_chain(x[n])                  (N cascaded all-pass)
//! line    = x[n] + feedback · wet[n]        (single-stage feedback)
//! out[n]  = (1 - mix) · dry + mix · wet     wet/dry blend
//! ```
//!
//! # Parameters
//!
//! * `n_stages` — clamped to `2..=12`; default 4 (classic phaser depth).
//! * `rate_hz` — LFO speed; 0..10 Hz typical; `0` freezes the cutoff.
//! * `depth_hz` — peak LFO cutoff (sweep upper bound). Default 1000 Hz.
//! * `feedback` — wet → input fed back; `0..=0.95` clamp.
//! * `mix` — wet/dry blend in `[0, 1]`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

#[derive(Debug, Clone)]
pub struct Phaser {
    n_stages: u8,
    rate_hz: f32,
    depth_hz: f32,
    feedback: f32,
    mix: f32,
    state: Option<PhaserState>,
}

#[derive(Debug, Clone)]
struct PhaserState {
    sample_rate: u32,
    channels: usize,
    /// `[channel][stage]` previous (x[n-1], y[n-1]) pair for each AP.
    sections: Vec<Vec<(f32, f32)>>,
    /// `[channel]` last wet sample used for feedback.
    last_wet: Vec<f32>,
    /// Shared LFO phase across channels.
    phase: f64,
}

impl Phaser {
    /// New phaser.
    ///
    /// `n_stages` is clamped to `2..=12`. `feedback` to `[0, 0.95]`,
    /// `mix` to `[0, 1]`. `depth_hz` is the LFO upper-bound cutoff (clamped
    /// to `> 0`).
    pub fn new(n_stages: u8, rate_hz: f32, depth_hz: f32, feedback: f32, mix: f32) -> Self {
        Self {
            n_stages: n_stages.clamp(2, 12),
            rate_hz: crate::clamp_param(rate_hz, 0.0, 0.0, f32::MAX),
            depth_hz: crate::clamp_param(depth_hz, 1.0, 1.0, f32::MAX),
            feedback: crate::clamp_param(feedback, 0.0, 0.0, 0.95),
            mix: crate::clamp_param(mix, 0.0, 0.0, 1.0),
            state: None,
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => {
                s.sample_rate != sample_rate
                    || s.channels != channels
                    || s.sections.first().map(|s| s.len()).unwrap_or(0) != self.n_stages as usize
            }
            None => true,
        };
        if needs_rebuild {
            let sections = (0..channels)
                .map(|_| vec![(0.0, 0.0); self.n_stages as usize])
                .collect();
            self.state = Some(PhaserState {
                sample_rate,
                channels,
                sections,
                last_wet: vec![0.0; channels],
                phase: 0.0,
            });
        }
    }
}

impl AudioFilter for Phaser {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let n_chan = params.channels as usize;
        self.ensure_state(params.sample_rate, n_chan);
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n_samples = channels.first().map(|c| c.len()).unwrap_or(0);

        let state = self.state.as_mut().expect("state ensured above");
        let fs = state.sample_rate as f32;
        let dphase = 2.0 * std::f64::consts::PI * (self.rate_hz as f64) / (fs as f64);
        let n_stages = self.n_stages as usize;
        let total_channels = channels.len();

        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let sections = &mut state.sections[ch_idx];
            let mut local_phase = state.phase;
            let mut last_wet = state.last_wet[ch_idx];
            for sample in buf.iter_mut().take(n_samples) {
                let dry = *sample;
                let lfo = 0.5 + 0.5 * (local_phase.sin() as f32);
                // Sweep cutoff from ~near-0 to depth_hz. Avoid cutoffs
                // greater than fs/2.
                let fc = (self.depth_hz * lfo).clamp(1.0, fs * 0.49);
                let w = 2.0 * std::f32::consts::PI * fc / fs;
                let t = (w * 0.5).tan();
                let a = (1.0 - t) / (1.0 + t);
                // Drive the chain with dry + feedback·last_wet
                let mut x = dry + self.feedback * last_wet;
                for sec in sections.iter_mut().take(n_stages) {
                    // y = a·x + x_prev - a·y_prev
                    let y = a * x + sec.0 - a * sec.1;
                    sec.0 = x;
                    sec.1 = y;
                    x = y;
                }
                let wet = x;
                last_wet = wet;
                *sample = dry * (1.0 - self.mix) + wet * self.mix;
                local_phase += dphase;
                if local_phase >= 2.0 * std::f64::consts::PI {
                    local_phase -= 2.0 * std::f64::consts::PI;
                }
            }
            state.last_wet[ch_idx] = last_wet;
            if ch_idx + 1 == total_channels {
                state.phase = local_phase;
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

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let s: f64 = samples.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / samples.len() as f64).sqrt() as f32
    }

    #[test]
    fn mix_zero_is_bypass() {
        let in_samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.08).sin() * 0.4).collect();
        let frame = make_f32_mono(&in_samples);
        let mut p = Phaser::new(4, 1.0, 1000.0, 0.0, 0.0);
        let out = p.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..in_samples.len() {
            assert!(
                (got[i] - in_samples[i]).abs() < 1.0e-6,
                "mix=0 not bypass at {i}: got={} want={}",
                got[i],
                in_samples[i]
            );
        }
    }

    #[test]
    fn all_pass_preserves_energy_at_mix_one() {
        // Wet path is N cascaded all-passes → |H| = 1 → RMS preserved
        // (for steady-state, after the start-up transient).
        let fs = 48_000u32;
        let n = 8192usize;
        // Test tone at 1 kHz.
        let in_samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / fs as f32).sin() * 0.5)
            .collect();
        let frame = make_f32_mono(&in_samples);
        let mut p = Phaser::new(4, 0.0, 1500.0, 0.0, 1.0); // wet only, no fb, no LFO
        let out = p.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip transient.
        let in_rms = rms(&in_samples[2000..]);
        let out_rms = rms(&got[2000..]);
        let ratio = out_rms / in_rms;
        assert!(
            (0.9..1.1).contains(&ratio),
            "AP cascade did not preserve energy: in_rms={in_rms} out_rms={out_rms} ratio={ratio}"
        );
    }

    #[test]
    fn rate_zero_is_stable_lti() {
        // rate=0 means the AP cutoffs never change. Processing two
        // identical inputs back-to-back through the same Phaser should
        // produce identical *steady-state* outputs (with feedback off).
        let fs = 48_000u32;
        let n = 1024usize;
        let mut p = Phaser::new(4, 0.0, 1000.0, 0.0, 1.0);
        let in_samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 200.0 * i as f32 / fs as f32).sin() * 0.4)
            .collect();
        let frame1 = make_f32_mono(&in_samples);
        let frame2 = make_f32_mono(&in_samples);
        let o1 = p.process(&frame1, f32_mono(fs)).unwrap();
        let o2 = p.process(&frame2, f32_mono(fs)).unwrap();
        let s1 = read_f32(&o1[0]);
        let s2 = read_f32(&o2[0]);
        // Compare last 256 samples of frame1 to last 256 of frame2 — they
        // should match within numeric noise once steady state is reached.
        // Note: the filter state persists across frames so the second
        // frame's tail follows naturally from the first frame's tail.
        // We just need it to not diverge.
        let tail_rms_diff = {
            let last1 = &s1[(n - 64)..];
            let last2 = &s2[(n - 64)..];
            rms(&last1
                .iter()
                .zip(last2)
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>())
        };
        assert!(
            tail_rms_diff < 1e-3,
            "rate=0 phaser tails diverge: rms diff = {tail_rms_diff}"
        );
    }

    #[test]
    fn notch_introduces_attenuation_in_wet_dry_sum() {
        // With mix=0.5 + a moving LFO, the dry+wet sum forms a swept
        // comb filter. Driving white noise through the phaser and
        // comparing the FFT against the input's FFT should reveal
        // at least one bin where the output is ≥ 1 dB below the input.
        use crate::fft::real_fft;
        let fs = 48_000u32;
        let n = 4096usize;
        // White-noise input via splitmix64 (same algorithm as
        // crate::WhiteNoise; inlined to keep the test self-contained).
        let mut s: u64 = 0xC0FF_EE12_3456_7890;
        let in_samples: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 11) as f64 / (1u64 << 53) as f64;
                ((2.0 * u - 1.0) as f32) * 0.5
            })
            .collect();
        let frame = make_f32_mono(&in_samples);
        // Wide-sweep phaser: rate=2 Hz, depth_hz=5000, 6 stages.
        let mut p = Phaser::new(6, 2.0, 5_000.0, 0.0, 0.5);
        let out = p.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);

        // Compare FFT magnitudes bin-by-bin, looking for any band
        // attenuated by ≥ 1 dB. Skip the lowest few bins (transient).
        let in_bins = real_fft(&in_samples);
        let out_bins = real_fft(&got);
        let mut found_atten = false;
        let mut max_atten_db = 0.0f32;
        // Group into 32-bin chunks (~375 Hz each) to average out noise.
        for chunk_start in (16..(in_bins.len() - 32)).step_by(32) {
            let in_mag: f64 = (chunk_start..chunk_start + 32)
                .map(|i| (in_bins[i].magnitude() as f64).powi(2))
                .sum();
            let out_mag: f64 = (chunk_start..chunk_start + 32)
                .map(|i| (out_bins[i].magnitude() as f64).powi(2))
                .sum();
            let db = 10.0 * (out_mag / in_mag.max(1e-12)).log10() as f32;
            if -db > max_atten_db {
                max_atten_db = -db;
            }
            if db <= -1.0 {
                found_atten = true;
            }
        }
        assert!(
            found_atten,
            "phaser swept comb did not produce ≥ 1 dB attenuation in any band: max_atten_db={max_atten_db}"
        );
    }
}
