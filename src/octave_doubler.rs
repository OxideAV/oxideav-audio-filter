//! Octave doubler — adds a pitch-shifted layer one octave above the
//! dry signal, blended back in.
//!
//! Implements the classic analog octave-up trick from fuzz pedals such
//! as the Tycobrahe Octavia: a **full-wave rectifier** maps `x → |x|`,
//! which doubles the fundamental's frequency. Combined with the dry
//! signal this produces the characteristic "octave-up" sound — every
//! cycle of the input becomes two cycles of the rectified output.
//!
//! ```text
//! ──────────       ╱╲          ╱╲
//!   sin(ωt) ──►   ╱  ╲   ╱╲   ╱  ╲   (period = 2π/ω)
//!
//!   |sin(ωt)|     ╱╲╱╲╱╲╱╲╱╲ (period = π/ω → frequency × 2)
//! ```
//!
//! Optionally the rectified path is high-pass-filtered to remove the
//! DC offset that rectification introduces (a half-wave-rectified
//! sine has a non-zero mean of `2/π`). With both the DC blocker and
//! the dry layer the output preserves the original timbre while adding
//! an octave-up harmonic.
//!
//! # Recurrence
//!
//! Per channel:
//!
//! ```text
//! oct = |x[n]| - dc_state                         (DC-blocked rectified)
//! dc_state = 0.995 · dc_state + 0.005 · |x[n]|    (one-pole DC estimator)
//! y[n] = dry · x[n] + wet · oct
//! ```
//!
//! # Mixing
//!
//! `dry ∈ [0, 1]` and `wet ∈ [0, 1]` are independent. The default
//! `(dry=1, wet=0.5)` is a tasteful "splash of octave"; `(dry=0,
//! wet=1)` is a fully-octave-doubled tone.
//!
//! # Parameters
//!
//! * `dry` — direct-signal mix (default 1.0).
//! * `wet` — rectified-signal mix (default 0.5).
//! * `dc_block` — whether to subtract the rectified DC (default true).

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// One-pole DC estimator pole. `0.995` ⇒ ~5–10 Hz settling at 48 kHz.
const DC_POLE: f32 = 0.995;

#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    /// Estimated DC level of the rectified signal.
    dc: f32,
}

/// Streaming octave doubler.
#[derive(Debug, Clone)]
pub struct OctaveDoubler {
    dry: f32,
    wet: f32,
    dc_block: bool,
    state: Vec<ChState>,
}

impl OctaveDoubler {
    /// New doubler with the default mix `(dry=1, wet=0.5)` and DC
    /// blocking enabled.
    pub fn new() -> Self {
        Self::with(1.0, 0.5, true)
    }

    /// Custom-mix constructor. `dry` / `wet` clamped to `[0, 1]`.
    pub fn with(dry: f32, wet: f32, dc_block: bool) -> Self {
        Self {
            dry: dry.clamp(0.0, 1.0),
            wet: wet.clamp(0.0, 1.0),
            dc_block,
            state: Vec::new(),
        }
    }

    /// Dry-signal mix coefficient.
    pub fn dry(&self) -> f32 {
        self.dry
    }
    /// Wet (rectified) mix coefficient.
    pub fn wet(&self) -> f32 {
        self.wet
    }
    /// Whether DC blocking is enabled on the rectified path.
    pub fn dc_block(&self) -> bool {
        self.dc_block
    }

    /// Reset internal per-channel DC estimator state.
    pub fn reset(&mut self) {
        for st in self.state.iter_mut() {
            *st = ChState::default();
        }
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.state.len() != channels {
            self.state = vec![ChState::default(); channels];
        }
    }
}

impl Default for OctaveDoubler {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for OctaveDoubler {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(channels.len());
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.state[ch_idx];
            for s in buf.iter_mut() {
                let dry = *s;
                let rectified = dry.abs();
                let oct = if self.dc_block {
                    let v = rectified - st.dc;
                    // Update DC estimator. Coefficient (1 - DC_POLE) so
                    // the estimator integrates slowly.
                    st.dc = DC_POLE * st.dc + (1.0 - DC_POLE) * rectified;
                    v
                } else {
                    rectified
                };
                *s = self.dry * dry + self.wet * oct;
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

    /// Count zero-crossings in a vector.
    fn zero_crossings(x: &[f32]) -> usize {
        let mut c = 0;
        for i in 1..x.len() {
            if x[i - 1] >= 0.0 && x[i] < 0.0 {
                c += 1;
            }
            if x[i - 1] < 0.0 && x[i] >= 0.0 {
                c += 1;
            }
        }
        c
    }

    #[test]
    fn dry_only_is_identity() {
        // dry=1, wet=0 should pass the input through.
        let samples: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin() * 0.4).collect();
        let frame = make_f32_mono(&samples);
        let mut od = OctaveDoubler::with(1.0, 0.0, true);
        let out = od.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..samples.len() {
            assert!(
                (got[i] - samples[i]).abs() < 1e-6,
                "dry-only differs at {i}: got={} want={}",
                got[i],
                samples[i]
            );
        }
    }

    #[test]
    fn wet_only_has_double_zero_crossings() {
        // Pure 1 kHz sine for 100 ms — should cross zero roughly
        // 200 times. The full-wave-rectified version (after DC block
        // converges) should cross zero ~400 times, i.e. double.
        let fs = 48_000u32;
        let n = 4800usize; // 100 ms
        let f = 1_000.0f32;
        let w = 2.0 * std::f32::consts::PI * f / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect();
        let frame = make_f32_mono(&samples);
        let mut od = OctaveDoubler::with(0.0, 1.0, true);
        // Run several frames so the DC estimator has converged.
        let mut out_all = Vec::new();
        for _ in 0..3 {
            let out = od.process(&frame, f32_mono(fs)).unwrap();
            out_all.extend(read_f32(&out[0]));
        }
        // Look at the last frame's worth of output (DC has fully settled).
        let tail = &out_all[out_all.len() - n..];
        let zc_in = zero_crossings(&samples);
        let zc_out = zero_crossings(tail);
        // Approximately doubled crossings (allow ±10 % slack).
        let expected = 2 * zc_in;
        let lo = expected * 9 / 10;
        let hi = expected * 11 / 10;
        assert!(
            (lo..=hi).contains(&zc_out),
            "octave-up zc check: got {zc_out}, expected ~{expected} (input zc = {zc_in})"
        );
    }

    #[test]
    fn dc_block_keeps_output_centred() {
        // After enough settling, the wet path must average ~zero.
        let fs = 48_000u32;
        let n = 4800usize;
        let f = 500.0f32;
        let w = 2.0 * std::f32::consts::PI * f / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect();
        let frame = make_f32_mono(&samples);
        let mut od = OctaveDoubler::with(0.0, 1.0, true);
        // Settle.
        for _ in 0..5 {
            let _ = od.process(&frame, f32_mono(fs)).unwrap();
        }
        // Now examine one more frame's DC.
        let out = od.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let mean: f64 = got.iter().map(|&v| v as f64).sum::<f64>() / got.len() as f64;
        assert!(
            mean.abs() < 0.02,
            "DC-blocked wet still has offset: mean={mean}"
        );
    }

    #[test]
    fn no_dc_block_retains_offset() {
        // With dc_block=false the rectified path has a strict positive
        // mean (≈ 2/π for a pure sine).
        let fs = 48_000u32;
        let n = 4800usize;
        let f = 500.0f32;
        let w = 2.0 * std::f32::consts::PI * f / fs as f32;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin() * 0.5).collect();
        let frame = make_f32_mono(&samples);
        let mut od = OctaveDoubler::with(0.0, 1.0, false);
        let out = od.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let mean: f64 = got.iter().map(|&v| v as f64).sum::<f64>() / got.len() as f64;
        // 2/π · 0.5 ≈ 0.318.
        assert!(
            mean > 0.25 && mean < 0.4,
            "no-DC-block path should retain offset; mean={mean}"
        );
    }
}
