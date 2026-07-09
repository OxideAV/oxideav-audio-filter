//! Wah — LFO-swept resonant band-pass.
//!
//! Models the classic "wah pedal" effect by sweeping the centre
//! frequency of a high-`Q` band-pass biquad with a sinusoidal LFO:
//!
//! ```text
//! lfo[n]  = (sin(2π · f_lfo · n / fs) + 1) / 2          ∈ [0, 1]
//! f_c[n]  = f_min · (f_max / f_min) ^ lfo[n]            log sweep
//! y[n]    = BPF(x[n], f_c[n], Q)
//! ```
//!
//! The centre-frequency sweep is logarithmic (geometric interpolation
//! between `f_min` and `f_max`) so musically equal swept intervals
//! correspond to equal LFO phase intervals.
//!
//! # Implementation notes
//!
//! * The biquad coefficients are recomputed every `update_period`
//!   samples (default 32) rather than every sample, which is plenty
//!   fast for the perceptual frequency rate (≤ 5 Hz typically) of a
//!   wah LFO at 48 kHz and avoids per-sample `sin/cos` cost.
//! * The biquad's internal `(s1, s2)` state is preserved across
//!   coefficient updates so the swept cutoff does not click.
//! * `Q` is fixed for the whole sweep — physical wah pedals do this
//!   too. Default `Q = 2.5` gives the characteristic "vowel" peak.
//!
//! # Parameters
//!
//! * `rate_hz` — LFO frequency. Default 0.8 Hz. Clamped to `(0, 20]`.
//! * `f_min` / `f_max` — sweep extremes. Defaults 400 Hz / 2 200 Hz
//!   (matches a Cry-Baby-style throw). Clamped so `f_min < f_max`
//!   and both `≥ 20 Hz`.
//! * `q` — band-pass `Q`. Default 2.5. Clamped to `(0.5, 20]`.
//! * `mix` — dry/wet mix `∈ [0, 1]`. Default 1.0 (fully wet).

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming LFO-driven wah.
#[derive(Debug, Clone)]
pub struct Wah {
    rate_hz: f32,
    f_min: f32,
    f_max: f32,
    q: f32,
    mix: f32,
    /// Recompile the biquad every this many samples.
    update_period: u32,
    bpf: Biquad,
    /// LFO phase accumulator in radians.
    phase: f32,
    /// Sample counter for `update_period` gating.
    counter: u32,
}

impl Wah {
    /// New wah with the Cry-Baby-style preset (0.8 Hz LFO, 400–2200 Hz
    /// sweep, Q = 2.5, fully wet).
    pub fn new() -> Self {
        Self::with(0.8, 400.0, 2_200.0, 2.5, 1.0)
    }

    /// Custom-preset constructor. All knobs clamped to musical ranges.
    pub fn with(rate_hz: f32, f_min: f32, f_max: f32, q: f32, mix: f32) -> Self {
        let rate = rate_hz.clamp(0.001, 20.0);
        let mut lo = f_min.max(20.0);
        let hi = f_max.max(20.0);
        if lo >= hi {
            // Defensive: collapse to a degenerate but valid range.
            lo = hi * 0.5;
        }
        let q = q.clamp(0.5, 20.0);
        let mix = mix.clamp(0.0, 1.0);
        let centre = (lo * hi).sqrt();
        Self {
            rate_hz: rate,
            f_min: lo,
            f_max: hi,
            q,
            mix,
            update_period: 32,
            bpf: Biquad::new(BiquadKind::BandPass {
                center_hz: centre,
                q,
            }),
            phase: 0.0,
            counter: 0,
        }
    }

    /// LFO rate in Hz.
    pub fn rate_hz(&self) -> f32 {
        self.rate_hz
    }

    /// Sweep range as `(low, high)` in Hz.
    pub fn sweep_range(&self) -> (f32, f32) {
        (self.f_min, self.f_max)
    }

    /// Band-pass `Q`.
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Dry/wet mix.
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Set the inner biquad update period (samples between coefficient
    /// recompiles). Lower = smoother sweep, more CPU. Clamped to
    /// `[1, 4096]`.
    pub fn with_update_period(mut self, n: u32) -> Self {
        self.update_period = n.clamp(1, 4_096);
        self
    }

    /// Reset LFO phase and inner biquad delay-line state.
    pub fn reset(&mut self) {
        self.bpf.reset();
        self.phase = 0.0;
        self.counter = 0;
    }
}

impl Default for Wah {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for Wah {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);
        let fs = params.sample_rate as f32;
        let dphase = 2.0 * std::f32::consts::PI * self.rate_hz / fs;
        let log_ratio = (self.f_max / self.f_min).ln();

        // Process sample-by-sample but only refresh coefficients every
        // `update_period` samples.
        // Strategy: collect dry copy of channels first, then run biquad
        // segment-by-segment (one segment per coefficient set).
        // `self.counter` carries the samples remaining until the next
        // refresh ACROSS process() calls, so the refresh grid is a
        // property of the sample stream, not of how the caller sliced
        // it into frames (chunk-size invariance: 1-sample frames and
        // one big frame produce bit-identical output).
        let dry: Vec<Vec<f32>> = channels.to_vec();
        let n_chan = channels.len();
        let two_pi = 2.0 * std::f32::consts::PI;
        let mut pos = 0usize;
        while pos < n {
            if self.counter == 0 {
                // Refresh the biquad cutoff for the next period.
                let lfo = 0.5 * (self.phase.sin() + 1.0);
                let f_c = self.f_min * (lfo * log_ratio).exp();
                self.bpf.set_kind(BiquadKind::BandPass {
                    center_hz: f_c,
                    q: self.q,
                });
                // The phase is only ever READ here, and refreshes land
                // on an absolute-sample grid (multiples of
                // `update_period`) that frame slicing cannot move — so
                // advance it by exactly one period per refresh. Any
                // per-segment accumulation would round differently for
                // different frame sizes and break bit-exact chunk-size
                // invariance.
                self.phase += dphase * self.update_period as f32;
                if self.phase >= two_pi {
                    self.phase -= two_pi * (self.phase / two_pi).floor();
                }
                self.counter = self.update_period;
            }

            let seg_end = (pos + self.counter as usize).min(n);
            for (ch_idx, ch) in channels.iter_mut().enumerate() {
                // Per-channel state slots: repeated single-channel
                // process_in_place would leak channel 0's delay-line
                // history into channel 1.
                self.bpf.process_channel_in_place(
                    &mut ch[pos..seg_end],
                    ch_idx,
                    n_chan,
                    params.sample_rate,
                );
            }

            self.counter -= (seg_end - pos) as u32;
            pos = seg_end;
        }

        // Apply dry/wet mix.
        if self.mix < 0.999 {
            for (ch_idx, wet) in channels.iter_mut().enumerate() {
                for i in 0..n {
                    wet[i] = self.mix * wet[i] + (1.0 - self.mix) * dry[ch_idx][i];
                }
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
    fn wet_signal_has_envelope_modulation() {
        // Feed wide-band noise; the wah's swept BPF should produce an
        // output whose short-term RMS *varies* over the LFO cycle.
        let fs = 48_000u32;
        let n = (fs as usize) * 2; // 2 s
        let mut x = vec![0.0f32; n];
        // Deterministic PRNG (splitmix64) so we don't depend on rand.
        let mut s: u64 = 0xC0FFEE;
        for v in x.iter_mut() {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            let u = (z >> 11) as f32 / (1u64 << 53) as f32;
            *v = (u * 2.0 - 1.0) * 0.5;
        }
        let frame = make_f32_mono(&x);
        let mut w = Wah::new(); // 0.8 Hz, sweep 400-2200 Hz
        let out = w.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Bucket the output into 50 ms windows and compute the RMS of
        // each bucket. With a 0.8 Hz LFO over 2 s the sweep covers
        // ~1.6 cycles; the bucket-RMS sequence should have a clear
        // min/max swing.
        let win = (fs as usize) / 20; // 50 ms
        let mut min_r = f32::MAX;
        let mut max_r = f32::MIN;
        let mut i = (fs as usize) / 4; // skip first 250 ms ramp
        while i + win < got.len() {
            let r = rms(&got[i..i + win]);
            if r < min_r {
                min_r = r;
            }
            if r > max_r {
                max_r = r;
            }
            i += win;
        }
        // Ratio of max-to-min should be substantial (≥ 2× → at least
        // 6 dB swing) for a sweep-modulated BPF.
        assert!(
            max_r > 2.0 * min_r,
            "wah envelope swing weak: max={max_r}, min={min_r}"
        );
    }

    #[test]
    fn dry_mix_is_pass_through() {
        // mix=0 → output should equal the dry input.
        let fs = 48_000u32;
        let n = 4096usize;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let frame = make_f32_mono(&samples);
        let mut w = Wah::with(1.0, 400.0, 2_200.0, 2.5, 0.0);
        let out = w.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..n {
            assert!(
                (got[i] - samples[i]).abs() < 1e-5,
                "dry mix differs at {i}: got={} want={}",
                got[i],
                samples[i]
            );
        }
    }

    #[test]
    fn defensive_clamps_apply() {
        // f_min ≥ f_max should be collapsed defensively.
        let w = Wah::with(0.0, 2000.0, 1000.0, 100.0, 5.0);
        let (lo, hi) = w.sweep_range();
        assert!(lo < hi, "swept range invalid: lo={lo} hi={hi}");
        assert!(w.q() <= 20.0, "Q not clamped: {}", w.q());
        assert!(w.mix() <= 1.0 && w.mix() >= 0.0, "mix out of range");
        assert!(w.rate_hz() > 0.0, "rate must be positive");
    }

    #[test]
    fn stereo_channels_are_independent() {
        // Channel 0 carries a tone, channel 1 is silent. With correct
        // per-channel biquad state slots, channel 1 must stay silent —
        // a shared state slot would ring channel 0's history into it.
        let fs = 48_000u32;
        let n = 4_096usize;
        let mut interleaved = Vec::with_capacity(n * 2);
        for i in 0..n {
            let w = 2.0 * std::f32::consts::PI * 660.0 / fs as f32;
            interleaved.push(0.5 * (i as f32 * w).sin()); // L
            interleaved.push(0.0); // R silent
        }
        let mut bytes = Vec::with_capacity(interleaved.len() * 4);
        for s in &interleaved {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut w = Wah::new();
        let out = w
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 2,
                    sample_rate: fs,
                },
            )
            .unwrap();
        let got = read_f32(&out[0]);
        let mut right_peak = 0.0f32;
        let mut left_peak = 0.0f32;
        for pair in got.chunks_exact(2) {
            left_peak = left_peak.max(pair[0].abs());
            right_peak = right_peak.max(pair[1].abs());
        }
        assert!(left_peak > 0.05, "left channel unexpectedly quiet");
        assert!(
            right_peak < 1.0e-6,
            "silent right channel leaked energy: peak={right_peak}"
        );
    }

    #[test]
    fn silence_in_yields_silence_out() {
        let fs = 48_000u32;
        let frame = make_f32_mono(&vec![0.0f32; 4096]);
        let mut w = Wah::new();
        let out = w.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let peak = got.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(peak < 1.0e-6, "silence-in produced output: peak={peak}");
    }
}
