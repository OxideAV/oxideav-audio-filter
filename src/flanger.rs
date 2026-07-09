//! Flanger — short LFO-modulated delay (1–15 ms) with positive feedback for
//! the classic comb-filter "swept resonance" sound.
//!
//! Differs from [`crate::Chorus`] in two ways:
//!
//! 1. Delay times are tens of samples to ~700 samples (1–15 ms) rather than
//!    20–40 ms — short enough that the dry+wet sum forms an audible *comb
//!    filter* (interference notches every `fs / delay` Hz).
//! 2. Has explicit positive feedback (`y[n] → line[n]`) so the comb's
//!    notches sharpen into resonant peaks as feedback approaches 1.0.
//!
//! # Recurrence
//!
//! ```text
//! lfo[n] = sin(2π · rate · n / fs)
//! d[n]   = depth_samples · (1 + lfo[n]) / 2    (i.e. 0..depth)
//! tap    = line.read_fractional(d[n])
//! line.write(dry + feedback · tap)
//! out    = (1 - mix) · dry + mix · tap
//! ```
//!
//! When `feedback = 0` and `rate = 0` the filter collapses to a fixed-tap
//! delay, and at `mix = 0` it is bit-exact bypass.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

#[derive(Debug, Clone)]
pub struct Flanger {
    rate_hz: f32,
    depth_ms: f32,
    feedback: f32,
    mix: f32,
    state: Option<FlangerState>,
}

#[derive(Debug, Clone)]
struct FlangerState {
    sample_rate: u32,
    channels: usize,
    lines: Vec<Vec<f32>>,
    write_idx: Vec<usize>,
    phase: f64,
}

impl Flanger {
    /// New flanger.
    ///
    /// `depth_ms` is clamped to `[1, 15]` (short-delay regime).
    /// `feedback` is clamped to `[0, 0.95]` for stability.
    /// `mix` is clamped to `[0, 1]`.
    pub fn new(rate_hz: f32, depth_ms: f32, feedback: f32, mix: f32) -> Self {
        Self {
            rate_hz: crate::clamp_param(rate_hz, 0.0, 0.0, f32::MAX),
            depth_ms: crate::clamp_param(depth_ms, 1.0, 1.0, 15.0),
            feedback: crate::clamp_param(feedback, 0.0, 0.0, 0.95),
            mix: crate::clamp_param(mix, 0.0, 0.0, 1.0),
            state: None,
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.channels != channels,
            None => true,
        };
        if needs_rebuild {
            // Line length = max delay + headroom. depth_ms is also the
            // maximum delay (LFO swings 0..depth).
            let max_samples = (self.depth_ms / 1000.0 * sample_rate as f32) as usize + 4;
            let line_len = max_samples.max(4);
            self.state = Some(FlangerState {
                sample_rate,
                channels,
                lines: (0..channels).map(|_| vec![0.0; line_len]).collect(),
                write_idx: vec![0; channels],
                phase: 0.0,
            });
        }
    }
}

impl AudioFilter for Flanger {
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
        let depth_d = self.depth_ms / 1000.0 * fs;
        let dphase = 2.0 * std::f64::consts::PI * (self.rate_hz as f64) / (fs as f64);
        let total_channels = channels.len();

        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let line = &mut state.lines[ch_idx];
            let line_len = line.len();
            let mut widx = state.write_idx[ch_idx];
            let mut local_phase = state.phase;
            for sample in buf.iter_mut().take(n_samples) {
                let dry = *sample;
                // LFO swings 0..1, scaled to 0..depth_d
                let lfo = 0.5 + 0.5 * (local_phase.sin() as f32);
                let d = (depth_d * lfo).max(1.0);
                let rd_pos = widx as f32 - d;
                let rd_int = rd_pos.floor() as i64;
                let frac = rd_pos - rd_int as f32;
                let i0 = rd_int.rem_euclid(line_len as i64) as usize;
                let i1 = (i0 + 1) % line_len;
                let tap = line[i0] * (1.0 - frac) + line[i1] * frac;
                line[widx] = dry + self.feedback * tap;
                widx = (widx + 1) % line_len;
                let out = dry * (1.0 - self.mix) + tap * self.mix;
                *sample = out;
                local_phase += dphase;
                if local_phase >= 2.0 * std::f64::consts::PI {
                    local_phase -= 2.0 * std::f64::consts::PI;
                }
            }
            state.write_idx[ch_idx] = widx;
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

    fn impulse_f32(n: usize) -> AudioFrame {
        let mut samples = vec![0.0f32; n];
        samples[0] = 1.0;
        make_f32_mono(&samples)
    }

    fn read_f32(frame: &AudioFrame) -> Vec<f32> {
        frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn mix_zero_is_bypass() {
        let in_samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let frame = make_f32_mono(&in_samples);
        let mut f = Flanger::new(0.5, 5.0, 0.3, 0.0);
        let out = f.process(&frame, f32_mono(48_000)).unwrap();
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
    fn rate_zero_is_constant_delay() {
        // rate=0 freezes the LFO at phase=0 → sin(0)=0 → d = depth · 0.5
        let fs = 48_000u32;
        let depth_ms = 8.0f32;
        let mut f = Flanger::new(0.0, depth_ms, 0.0, 1.0); // wet only, no fb
        let frame = impulse_f32(2000);
        let out = f.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // d = depth · 0.5 = 4 ms = 192 samples
        let expected = (depth_ms * 0.5 * 1e-3 * fs as f32) as usize;
        let near = (0..3)
            .map(|o| {
                got[expected + o]
                    .abs()
                    .max(got[expected - o.min(expected)].abs())
            })
            .fold(0.0f32, f32::max);
        assert!(
            near > 0.5,
            "no impulse at delay sample {expected}: max nearby = {near}"
        );
    }

    #[test]
    fn feedback_creates_resonant_peaks() {
        // With feedback > 0 and a long impulse output, the comb's tail
        // should contain multiple decaying repeats.
        let fs = 48_000u32;
        let mut f = Flanger::new(0.0, 4.0, 0.6, 1.0);
        let frame = impulse_f32(4096);
        let out = f.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // With rate=0, d = depth · 0.5 = 2 ms = 96 samples.
        let d = (4.0 * 0.5 * 1e-3 * fs as f32) as usize;
        // First echo at d, second at 2d, third at 3d (scaled by feedback).
        let r1 = got[d - 1].abs().max(got[d].abs()).max(got[d + 1].abs());
        let r2 = got[2 * d - 1]
            .abs()
            .max(got[2 * d].abs())
            .max(got[2 * d + 1].abs());
        assert!(r1 > 0.3, "first echo too weak: {r1}");
        assert!(r2 > 0.05, "feedback echo missing: {r2}");
        // Feedback echo must be smaller than the first echo.
        assert!(r2 < r1 + 0.1, "feedback echo {r2} not less than first {r1}");
    }

    #[test]
    fn feedback_zero_no_second_echo() {
        let fs = 48_000u32;
        let mut f = Flanger::new(0.0, 6.0, 0.0, 1.0);
        let frame = impulse_f32(4096);
        let out = f.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let d = (6.0 * 0.5 * 1e-3 * fs as f32) as usize;
        // Second echo region should be ~0.
        let r2_region: f32 = got[(2 * d - 2)..(2 * d + 3)]
            .iter()
            .cloned()
            .fold(0.0, |a, b| a.max(b.abs()));
        assert!(r2_region < 1.0e-3, "fb=0 produced 2nd echo: {r2_region}");
    }
}
