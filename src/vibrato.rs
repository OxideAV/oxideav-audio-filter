//! Vibrato — LFO-modulated pitch via fractional-delay line read.
//!
//! Companion to [`Tremolo`](crate::Tremolo): tremolo modulates *amplitude*,
//! vibrato modulates *delay* and therefore *pitch* (because reading from a
//! moving point in a delay line stretches/compresses time).
//!
//! # Recurrence
//!
//! Per channel:
//!
//! ```text
//! lfo[n] = sin(2π · rate · n / fs)
//! d[n]   = base_delay + depth · lfo[n]               (samples)
//! y[n]   = read_fractional(line, write_idx - d[n])
//! line[write_idx] = x[n]
//! write_idx ← (write_idx + 1) mod line_len
//! ```
//!
//! `read_fractional` is linear interpolation between the two integer line
//! indices straddling the requested fractional position. The *output is
//! 100% wet* — vibrato is conventionally a pure-modulation effect; mix
//! with the dry path via the upstream [`AudioFilter`](crate::AudioFilter)
//! graph if a dry/wet blend is needed.
//!
//! # Pitch swing
//!
//! Frequency modulation is the time derivative of phase, so the
//! instantaneous frequency multiplier seen at the output is:
//!
//! ```text
//! 1 - d'[n]  =  1 - depth · (2π · rate / fs) · cos(2π · rate · n / fs)
//! ```
//!
//! For example `depth = 5 ms` at `rate = 5 Hz` and `fs = 48 kHz` gives a
//! peak deviation of `5e-3 · 2π · 5 ≈ 0.157` ⇒ the pitch swings within
//! ±15.7 % of the carrier frequency at the LFO peaks.
//!
//! # Parameters
//!
//! * `rate_hz` — LFO frequency (typical 4..7 Hz musically). Clamped to
//!   `[0, 20]`.
//! * `depth_ms` — delay-modulation amplitude (typical 0.5..3 ms).
//!   Clamped to `[0, 50]`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming vibrato.
#[derive(Debug, Clone)]
pub struct Vibrato {
    rate_hz: f32,
    depth_ms: f32,
    state: Option<VibratoState>,
}

#[derive(Debug, Clone)]
struct VibratoState {
    sample_rate: u32,
    channels: usize,
    /// One ring buffer per channel.
    lines: Vec<Vec<f32>>,
    /// Per-channel write index.
    write_idx: Vec<usize>,
    /// Shared LFO phase (radians).
    phase: f64,
}

impl Vibrato {
    /// New vibrato. `rate_hz` clamped to `[0, 20]`, `depth_ms` clamped to
    /// `[0, 50]`.
    pub fn new(rate_hz: f32, depth_ms: f32) -> Self {
        Self {
            rate_hz: rate_hz.clamp(0.0, 20.0),
            depth_ms: depth_ms.clamp(0.0, 50.0),
            state: None,
        }
    }

    /// Currently-configured LFO rate.
    pub fn rate_hz(&self) -> f32 {
        self.rate_hz
    }

    /// Currently-configured modulation depth.
    pub fn depth_ms(&self) -> f32 {
        self.depth_ms
    }

    /// Reset internal delay lines and LFO phase.
    pub fn reset(&mut self) {
        if let Some(s) = self.state.as_mut() {
            for line in s.lines.iter_mut() {
                for v in line.iter_mut() {
                    *v = 0.0;
                }
            }
            for w in s.write_idx.iter_mut() {
                *w = 0;
            }
            s.phase = 0.0;
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.channels != channels,
            None => true,
        };
        if needs_rebuild {
            // Centre the read tap at `depth_ms` so the LFO swing
            // `±depth_ms` always reads from valid line samples (delay 0
            // at peak, 2·depth_ms at trough). 4 samples of headroom
            // covers fractional-interpolation rounding.
            let max_delay_samples =
                ((2.0 * self.depth_ms) / 1000.0 * sample_rate as f32) as usize + 4;
            let line_len = max_delay_samples.max(8);
            self.state = Some(VibratoState {
                sample_rate,
                channels,
                lines: (0..channels).map(|_| vec![0.0; line_len]).collect(),
                write_idx: vec![0; channels],
                phase: 0.0,
            });
        }
    }
}

impl AudioFilter for Vibrato {
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
        // Centre the read tap at depth_ms so swings ±depth_ms stay valid.
        let base_d = (self.depth_ms / 1000.0) * fs;
        let depth_d = base_d; // peak swing equals base offset
        let dphase = 2.0 * std::f64::consts::PI * (self.rate_hz as f64) / (fs as f64);
        let total_channels = channels.len();

        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let line = &mut state.lines[ch_idx];
            let line_len = line.len();
            let mut widx = state.write_idx[ch_idx];
            let mut local_phase = state.phase;
            for sample in buf.iter_mut().take(n_samples) {
                let dry = *sample;
                line[widx] = dry;
                let lfo = local_phase.sin() as f32;
                let d = base_d + depth_d * lfo;
                let rd_pos = widx as f32 - d;
                let rd_int = rd_pos.floor() as i64;
                let frac = rd_pos - rd_int as f32;
                let i0 = rd_int.rem_euclid(line_len as i64) as usize;
                let i1 = (i0 + 1) % line_len;
                let tap = line[i0] * (1.0 - frac) + line[i1] * frac;
                widx = (widx + 1) % line_len;
                local_phase += dphase;
                if local_phase >= 2.0 * std::f64::consts::PI {
                    local_phase -= 2.0 * std::f64::consts::PI;
                }
                *sample = tap;
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

    fn read_f32(frame: &AudioFrame) -> Vec<f32> {
        frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn depth_zero_constant_delay_dc_passes() {
        // depth=0 → constant 0-sample delay → DC passes after one sample.
        let frame = make_f32_mono(&vec![0.5f32; 1024]);
        let mut v = Vibrato::new(5.0, 0.0);
        let out = v.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // After warmup the constant input should produce constant output.
        let tail = &got[100..];
        for &s in tail.iter() {
            assert!(
                (s - 0.5).abs() < 1e-5,
                "depth=0 DC drift: sample={}, expected 0.5",
                s
            );
        }
    }

    #[test]
    fn shifts_phase_of_sine() {
        // Apply vibrato to a 440 Hz sine. The output should still oscillate
        // with energy in the 440 Hz region, but with a time-varying phase.
        // We don't verify pitch shift bin-for-bin — that needs FFT — but we
        // do check the output is sinusoidal (bounded, non-trivial RMS).
        let fs = 48_000u32;
        let f = 440.0f32;
        let n = 4096;
        let w = 2.0 * std::f32::consts::PI * f / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect();
        let frame = make_f32_mono(&samples);
        let mut v = Vibrato::new(5.0, 2.0);
        let out = v.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Skip warmup.
        let warm = 1024;
        let rms: f32 =
            ((got[warm..].iter().map(|&s| s * s).sum::<f32>()) / (got.len() - warm) as f32).sqrt();
        assert!(rms > 0.1 && rms < 0.6, "vibrato output rms = {}", rms);
        // Output must be bounded.
        let peak = got.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak < 1.5, "vibrato output peak = {}", peak);
    }

    #[test]
    fn rate_zero_pure_delay() {
        // rate=0 freezes LFO at sin(0)=0, so the read tap sits on
        // base_delay = depth_ms. Verify an impulse appears at that delay.
        let fs = 48_000u32;
        let depth_ms = 2.0f32;
        let mut samples = vec![0.0f32; 2048];
        samples[0] = 1.0;
        let frame = make_f32_mono(&samples);
        let mut v = Vibrato::new(0.0, depth_ms);
        let out = v.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let delay = (depth_ms * 1e-3 * fs as f32) as usize;
        // Look in a small window for the impulse.
        let lo = delay.saturating_sub(2);
        let hi = (delay + 3).min(got.len());
        let peak: f32 = got[lo..hi].iter().cloned().fold(0.0, |a, b| a.max(b.abs()));
        assert!(
            peak > 0.5,
            "no impulse at expected delay {}: peak={}",
            delay,
            peak
        );
    }

    #[test]
    fn streaming_continuity() {
        // Splitting a long sine over multiple frames must give the same
        // sample-by-sample output as one big frame (state preserved).
        let fs = 48_000u32;
        let f = 200.0f32;
        let n = 4096;
        let w = 2.0 * std::f32::consts::PI * f / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect();

        let frame_one = make_f32_mono(&samples);
        let mut v_one = Vibrato::new(5.0, 1.5);
        let out_one = v_one.process(&frame_one, f32_mono(fs)).unwrap();
        let got_one = read_f32(&out_one[0]);

        let mut v_split = Vibrato::new(5.0, 1.5);
        let mut got_split: Vec<f32> = Vec::new();
        for chunk in samples.chunks(513) {
            let f = make_f32_mono(chunk);
            let out = v_split.process(&f, f32_mono(fs)).unwrap();
            got_split.extend(read_f32(&out[0]));
        }
        assert_eq!(got_one.len(), got_split.len());
        for i in 0..got_one.len() {
            assert!(
                (got_one[i] - got_split[i]).abs() < 1e-6,
                "mismatch at {}: monolithic={}, split={}",
                i,
                got_one[i],
                got_split[i]
            );
        }
    }
}
