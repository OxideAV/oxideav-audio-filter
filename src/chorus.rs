//! Chorus — `n_voices` short delayed taps with LFO-modulated delay times.
//!
//! Each voice runs its own sine LFO with a slightly different phase offset so
//! the resulting tap delays decorrelate over time, producing the classic
//! "multiple-singer" / "ensemble" effect. The voice outputs are summed,
//! scaled by `1/n_voices` to keep level constant across voice counts, and
//! blended with the dry signal by `mix`.
//!
//! # Recurrence (per voice `v`)
//!
//! ```text
//! lfo_v[n] = sin(2π · rate · n / fs + φ_v)
//! d_v[n]   = base_delay + depth · lfo_v[n]          (in samples)
//! tap_v    = line.read_fractional(d_v[n])
//! wet[n]   = (Σ tap_v) / n_voices
//! out[n]   = (1 - mix) · dry + mix · wet
//! line.write(dry)                                    (no feedback)
//! ```
//!
//! The phase offsets are `φ_v = 2π · v / n_voices` so voices are uniformly
//! distributed around one LFO cycle.
//!
//! Fractional-delay reads use linear interpolation between the two integer
//! ring-buffer indices nearest to the modulated delay.
//!
//! # Parameters
//!
//! * `n_voices` — clamped to `1..=4`.
//! * `base_delay_ms` — typical 15..40 ms for "chorus", > 50 ms drifts into
//!   slap-back echo territory.
//! * `depth_ms` — peak modulation depth; must be `< base_delay_ms` to keep
//!   the LFO swing inside the delay line.
//! * `rate_hz` — typical 0.1..3 Hz. `rate_hz = 0` collapses every voice to a
//!   constant `base_delay_ms` tap → multiple equal delays sum to a single
//!   echo at `mix=1`.
//! * `mix` — wet/dry blend in `[0, 1]`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

#[derive(Debug, Clone)]
pub struct Chorus {
    n_voices: u8,
    base_delay_ms: f32,
    depth_ms: f32,
    rate_hz: f32,
    mix: f32,
    state: Option<ChorusState>,
}

#[derive(Debug, Clone)]
struct ChorusState {
    sample_rate: u32,
    channels: usize,
    /// One ring buffer per channel.
    lines: Vec<Vec<f32>>,
    /// Per-channel write index.
    write_idx: Vec<usize>,
    /// Shared per-voice LFO phase (radians).
    phase: Vec<f64>,
}

/// Upper bound on the base voice delay (chorus is a short-delay effect;
/// 1 s is already far beyond the musical regime). Bounds the per-channel
/// ring allocation so a hostile / garbage `base_delay_ms` can never turn
/// into a multi-gigabyte `Vec` (allocation failure aborts the process).
pub const MAX_BASE_DELAY_MS: f32 = 1_000.0;

impl Chorus {
    /// New chorus.
    ///
    /// `n_voices` is clamped to `1..=4`. `base_delay_ms` is clamped to
    /// `[1, MAX_BASE_DELAY_MS]`. `depth_ms` is clamped to
    /// `[0, base_delay_ms - 1.0]` so the swing stays inside the line.
    /// `mix` is clamped to `[0, 1]`. Non-finite parameters fall back to
    /// their nearest neutral value (`f32::clamp` propagates NaN).
    pub fn new(n_voices: u8, base_delay_ms: f32, depth_ms: f32, rate_hz: f32, mix: f32) -> Self {
        let base_delay_ms = if base_delay_ms.is_finite() {
            base_delay_ms
        } else {
            1.0
        };
        let depth_ms = if depth_ms.is_finite() { depth_ms } else { 0.0 };
        let rate_hz = if rate_hz.is_finite() { rate_hz } else { 0.0 };
        let mix = if mix.is_finite() { mix } else { 0.0 };
        let n_voices = n_voices.clamp(1, 4);
        let base_delay_ms = base_delay_ms.clamp(1.0, MAX_BASE_DELAY_MS);
        let max_depth = (base_delay_ms - 1.0).max(0.0);
        Self {
            n_voices,
            base_delay_ms,
            depth_ms: depth_ms.clamp(0.0, max_depth),
            rate_hz: rate_hz.max(0.0),
            mix: mix.clamp(0.0, 1.0),
            state: None,
        }
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.channels != channels,
            None => true,
        };
        if needs_rebuild {
            // Worst-case delay = base + depth, plus 2 samples of fractional
            // interpolation headroom.
            let max_delay_samples =
                ((self.base_delay_ms + self.depth_ms) / 1000.0 * sample_rate as f32) as usize + 4;
            let line_len = max_delay_samples.max(4);
            let phase = (0..self.n_voices as usize)
                .map(|v| 2.0 * std::f64::consts::PI * (v as f64) / (self.n_voices as f64))
                .collect();
            self.state = Some(ChorusState {
                sample_rate,
                channels,
                lines: (0..channels).map(|_| vec![0.0; line_len]).collect(),
                write_idx: vec![0; channels],
                phase,
            });
        }
    }
}

impl AudioFilter for Chorus {
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
        let n_voices = self.n_voices as usize;
        let voice_scale = 1.0 / n_voices as f32;
        let base_d = self.base_delay_ms / 1000.0 * fs;
        let depth_d = self.depth_ms / 1000.0 * fs;
        let dphase = 2.0 * std::f64::consts::PI * (self.rate_hz as f64) / (fs as f64);
        let total_channels = channels.len();

        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let line = &mut state.lines[ch_idx];
            let line_len = line.len();
            let mut widx = state.write_idx[ch_idx];
            // Snapshot phases for this channel; share the same LFO across
            // channels (re-advanced once per output sample after the channel
            // loop). Channel 0 advances the phases; subsequent channels copy
            // the same final phase via a saved snapshot below.
            let mut local_phase: Vec<f64> = state.phase.clone();
            for sample in buf.iter_mut().take(n_samples) {
                let dry = *sample;
                let mut wet_sum = 0.0f32;
                for (v, ph) in local_phase.iter().enumerate().take(n_voices) {
                    let lfo = ph.sin() as f32;
                    let d = base_d + depth_d * lfo;
                    // Read at `widx - d` (with wrap-around). Use fractional
                    // linear interpolation.
                    let rd_pos = widx as f32 - d;
                    let rd_int = rd_pos.floor() as i64;
                    let frac = rd_pos - rd_int as f32;
                    let i0 = rd_int.rem_euclid(line_len as i64) as usize;
                    let i1 = (i0 + 1) % line_len;
                    let tap = line[i0] * (1.0 - frac) + line[i1] * frac;
                    wet_sum += tap;
                    // Silence unused-variable warning when v isn't read.
                    let _ = v;
                }
                let wet = wet_sum * voice_scale;
                let out = dry * (1.0 - self.mix) + wet * self.mix;
                line[widx] = dry;
                widx = (widx + 1) % line_len;
                for ph in local_phase.iter_mut().take(n_voices) {
                    *ph += dphase;
                    if *ph >= 2.0 * std::f64::consts::PI {
                        *ph -= 2.0 * std::f64::consts::PI;
                    }
                }
                *sample = out;
            }
            state.write_idx[ch_idx] = widx;
            // Last channel wins on phase update — fine because LFO phase is
            // a stream-level property, not a per-channel one.
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
    fn rate_zero_is_constant_delay() {
        // rate=0 freezes every LFO at its phase offset; for n_voices=1
        // (phase=0, sin(0)=0) the effective delay is base_delay_ms only.
        let fs = 48_000u32;
        let mut c = Chorus::new(1, 10.0, 5.0, 0.0, 1.0); // wet only
        let frame = impulse_f32(2000);
        let out = c.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // 10 ms at 48 kHz = 480 samples
        let delay_samples = (10e-3 * fs as f32) as usize;
        // wet=1, dry=0 → impulse at delay_samples is ~1.0 (linear interp loses a tiny bit if depth=0)
        let near = got[delay_samples - 1]
            .max(got[delay_samples])
            .max(got[delay_samples + 1]);
        assert!(
            (near - 1.0).abs() < 0.05,
            "constant-delay tap at {delay_samples} = {near}, expected ~1"
        );
    }

    #[test]
    fn mix_zero_is_bypass() {
        // mix=0 → output equals dry input
        let in_samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let frame = make_f32_mono(&in_samples);
        let mut c = Chorus::new(2, 20.0, 5.0, 1.0, 0.0);
        let out = c.process(&frame, f32_mono(48_000)).unwrap();
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
    fn impulse_produces_modulated_tap_around_base_delay() {
        // For a single impulse and mix=1, the wet tap should land near
        // `base_delay ± depth` and dominate that region.
        let fs = 48_000u32;
        let base_ms = 20.0f32;
        let depth_ms = 5.0f32;
        let mut c = Chorus::new(1, base_ms, depth_ms, 1.0, 1.0);
        let frame = impulse_f32(4096);
        let out = c.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let center = (base_ms * 1e-3 * fs as f32) as usize;
        let span = (depth_ms * 1e-3 * fs as f32) as usize + 4;
        let lo = center.saturating_sub(span);
        let hi = (center + span).min(got.len());
        let max_in_window: f32 = got[lo..hi].iter().cloned().fold(0.0, f32::max);
        // Should find a tap of size ~1.0 (single voice, mix=1)
        assert!(
            max_in_window > 0.5,
            "no chorus tap found in [{lo}..{hi}]: max={max_in_window}"
        );
    }

    #[test]
    fn n_voices_normalised_amplitude() {
        // Verify 1-voice and 4-voice impulse outputs have similar peak
        // amplitude (because we divide wet by n_voices).
        let fs = 48_000u32;
        let mut c1 = Chorus::new(1, 20.0, 0.0, 1.0, 1.0);
        let mut c4 = Chorus::new(4, 20.0, 0.0, 1.0, 1.0);
        let frame = impulse_f32(4096);
        let o1 = c1.process(&frame, f32_mono(fs)).unwrap();
        let o4 = c4.process(&frame, f32_mono(fs)).unwrap();
        let s1 = read_f32(&o1[0]);
        let s4 = read_f32(&o4[0]);
        let max1: f32 = s1.iter().cloned().fold(0.0, |a, b| a.max(b.abs()));
        let max4: f32 = s4.iter().cloned().fold(0.0, |a, b| a.max(b.abs()));
        // With depth=0 all voices coincide → both should peak at ~1.0.
        assert!(
            (max1 - max4).abs() < 0.1,
            "n=1 max={max1} vs n=4 max={max4} drift too large"
        );
    }
}
