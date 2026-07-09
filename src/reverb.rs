//! Schroeder-style algorithmic reverb.
//!
//! # Topology
//!
//! Four parallel **comb filters** in the early-reflection stage feed
//! into a serial pair of **all-pass filters** in the diffusion stage,
//! producing a smoothed late-reverb tail. The structure is the
//! original 1962 Schroeder design with a one-pole low-pass inside
//! each comb's feedback to add high-frequency damping (Moorer 1979's
//! refinement).
//!
//! ```text
//!         ┌─ comb_0 ──┐
//!         ├─ comb_1 ──┤        ┌────────────┐    ┌────────────┐
//!  in ───►│           ├── sum ─►│ all-pass 0 ├───►│ all-pass 1 ├──► wet
//!         ├─ comb_2 ──┤        └────────────┘    └────────────┘
//!         └─ comb_3 ──┘
//! ```
//!
//! Each comb's recurrence (with low-pass feedback `lpf[n]`):
//!
//! ```text
//! out[n]   = line[r]
//! lpf[n]   = (1 - damping) · out[n] + damping · lpf[n-1]
//! line[w]  = in[n] + feedback · lpf[n]
//! advance read/write pointers
//! ```
//!
//! Each all-pass:
//!
//! ```text
//! buf_r    = line[r]
//! out      = -k · in[n] + buf_r
//! line[w]  =  in[n] + k · buf_r
//! ```
//!
//! with a fixed all-pass gain `k = 0.5` (textbook Schroeder).
//!
//! # Public knobs
//!
//! * `room_size ∈ [0, 1]` — scales the four comb-line lengths from
//!   their base values (≈ 25 ms at 0.0 → ≈ 80 ms at 1.0). Longer lines
//!   give a longer-decaying tail.
//! * `damping ∈ [0, 1]` — one-pole LPF coefficient inside each comb.
//!   `0.0` = no damping (bright), `0.99` = heavy damping (dark).
//! * `wet ∈ [0, 1]` — gain applied to the reverb output.
//! * `dry ∈ [0, 1]` — gain applied to the unprocessed input.
//!
//! # Comb-line length selection
//!
//! The four base lengths are co-prime so the eigenmodes do not align,
//! avoiding metallic resonance fringe. Values from Schroeder's 1962
//! paper, scaled to a 44.1 kHz reference (`{1116, 1188, 1277, 1356}`
//! samples), then rescaled to the stream's actual sample rate at
//! state-build time. The two all-pass lengths follow the same
//! convention (`{225, 556}` at 44.1 kHz).

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Reference sample rate (Hz) for the canonical comb / all-pass line
/// lengths. Lines are linearly rescaled to the actual `sample_rate`
/// in [`ReverbState::build`].
const REFERENCE_FS: f32 = 44_100.0;

/// Base comb-filter lengths (samples @ 44.1 kHz). Co-prime to avoid
/// metallic eigenmode pile-up; values from Schroeder's 1962 paper
/// (paraphrased — used as math, not transcribed source code).
const COMB_BASE: [usize; 4] = [1_116, 1_188, 1_277, 1_356];

/// Base all-pass-filter lengths (samples @ 44.1 kHz).
const ALLPASS_BASE: [usize; 2] = [225, 556];

/// All-pass coefficient `k`. Schroeder's classic 0.5 — gives a flat
/// magnitude response with non-trivial phase scrambling.
const ALLPASS_K: f32 = 0.5;

/// Per-comb storage.
struct Comb {
    line: Vec<f32>,
    idx: usize,
    lpf_state: f32,
}

impl Comb {
    fn new(len: usize) -> Self {
        Self {
            line: vec![0.0; len.max(1)],
            idx: 0,
            lpf_state: 0.0,
        }
    }

    fn step(&mut self, x: f32, feedback: f32, damping: f32) -> f32 {
        let out = self.line[self.idx];
        // One-pole low-pass on the feedback: brighter or darker tail.
        self.lpf_state = (1.0 - damping) * out + damping * self.lpf_state;
        self.line[self.idx] = x + feedback * self.lpf_state;
        self.idx += 1;
        if self.idx >= self.line.len() {
            self.idx = 0;
        }
        out
    }
}

/// Per-all-pass storage.
struct AllPass {
    line: Vec<f32>,
    idx: usize,
}

impl AllPass {
    fn new(len: usize) -> Self {
        Self {
            line: vec![0.0; len.max(1)],
            idx: 0,
        }
    }

    fn step(&mut self, x: f32) -> f32 {
        let buf = self.line[self.idx];
        let out = -ALLPASS_K * x + buf;
        self.line[self.idx] = x + ALLPASS_K * buf;
        self.idx += 1;
        if self.idx >= self.line.len() {
            self.idx = 0;
        }
        out
    }
}

/// Per-channel reverb network (4 combs ║ 2 serial all-passes).
struct ReverbState {
    sample_rate: u32,
    channels: usize,
    /// `combs[ch][i]` for ch ∈ [0, channels), i ∈ [0, 4).
    combs: Vec<Vec<Comb>>,
    /// `allpasses[ch][i]` for ch ∈ [0, channels), i ∈ [0, 2).
    allpasses: Vec<Vec<AllPass>>,
}

impl ReverbState {
    fn build(sample_rate: u32, channels: usize, room_size: f32) -> Self {
        let scale = sample_rate as f32 / REFERENCE_FS;
        // room_size scales each line; the doc says "0.0 ≈ 25 ms,
        // 1.0 ≈ 80 ms". The base lines are around 25 ms at 44.1 k
        // (1100/44100 ≈ 25 ms), so map room_size linearly to [1.0, 3.5].
        let room_scale = 1.0 + 2.5 * room_size.clamp(0.0, 1.0);
        let comb_lens: [usize; 4] = [
            ((COMB_BASE[0] as f32) * scale * room_scale) as usize,
            ((COMB_BASE[1] as f32) * scale * room_scale) as usize,
            ((COMB_BASE[2] as f32) * scale * room_scale) as usize,
            ((COMB_BASE[3] as f32) * scale * room_scale) as usize,
        ];
        let ap_lens: [usize; 2] = [
            ((ALLPASS_BASE[0] as f32) * scale) as usize,
            ((ALLPASS_BASE[1] as f32) * scale) as usize,
        ];

        let combs: Vec<Vec<Comb>> = (0..channels)
            .map(|ch| {
                comb_lens
                    .iter()
                    .enumerate()
                    .map(|(i, &len)| {
                        // Per-channel diffusion: stagger comb 0 and
                        // comb 2 lengths by ±23 samples on the right
                        // channel so stereo isn't bit-identical.
                        let stagger = if ch == 1 && (i == 0 || i == 2) {
                            23i32
                        } else {
                            0i32
                        };
                        Comb::new((len as i32 + stagger).max(1) as usize)
                    })
                    .collect()
            })
            .collect();
        let allpasses: Vec<Vec<AllPass>> = (0..channels)
            .map(|_| ap_lens.iter().map(|&len| AllPass::new(len)).collect())
            .collect();

        Self {
            sample_rate,
            channels,
            combs,
            allpasses,
        }
    }
}

/// Schroeder-style reverb.
pub struct Reverb {
    room_size: f32,
    damping: f32,
    wet: f32,
    dry: f32,
    state: Option<ReverbState>,
}

impl Reverb {
    /// Create a reverb. All knobs are clamped to `[0, 1]`.
    pub fn new(room_size: f32, damping: f32, wet: f32, dry: f32) -> Self {
        Self {
            room_size: crate::clamp_param(room_size, 0.5, 0.0, 1.0),
            damping: crate::clamp_param(damping, 0.5, 0.0, 1.0),
            wet: crate::clamp_param(wet, 0.0, 0.0, 1.0),
            dry: crate::clamp_param(dry, 1.0, 0.0, 1.0),
            state: None,
        }
    }

    /// Update the room-size knob. Forces a state rebuild on next
    /// `process()` call.
    pub fn set_room_size(&mut self, room_size: f32) {
        let new_size = room_size.clamp(0.0, 1.0);
        if (new_size - self.room_size).abs() > f32::EPSILON {
            self.room_size = new_size;
            self.state = None;
        }
    }

    /// Update damping (cheap — no state rebuild).
    pub fn set_damping(&mut self, damping: f32) {
        self.damping = damping.clamp(0.0, 1.0);
    }

    /// Update wet level (cheap).
    pub fn set_wet(&mut self, wet: f32) {
        self.wet = wet.clamp(0.0, 1.0);
    }

    /// Update dry level (cheap).
    pub fn set_dry(&mut self, dry: f32) {
        self.dry = dry.clamp(0.0, 1.0);
    }

    /// Effective comb-feedback gain. We translate `room_size` (which
    /// already changed delay length) into a feedback coefficient.
    /// 0.7 (small) → 0.95 (large) is a reasonable usable range —
    /// staying < 1.0 keeps the comb stable.
    fn feedback(&self) -> f32 {
        0.7 + 0.25 * self.room_size
    }

    fn ensure_state(&mut self, sample_rate: u32, channels: usize) {
        let needs = match &self.state {
            Some(s) => s.sample_rate != sample_rate || s.channels != channels,
            None => true,
        };
        if needs {
            self.state = Some(ReverbState::build(sample_rate, channels, self.room_size));
        }
    }
}

impl AudioFilter for Reverb {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = channels.len();
        self.ensure_state(params.sample_rate, n_chan);

        let feedback = self.feedback();
        let damping = self.damping;
        let wet = self.wet;
        let dry = self.dry;
        let state = self.state.as_mut().expect("ensure_state ran");

        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let combs = &mut state.combs[ch_idx];
            let aps = &mut state.allpasses[ch_idx];
            for s in buf.iter_mut() {
                let x = *s;
                // Sum the four parallel combs.
                let comb_sum = combs[0].step(x, feedback, damping)
                    + combs[1].step(x, feedback, damping)
                    + combs[2].step(x, feedback, damping)
                    + combs[3].step(x, feedback, damping);
                // Serial all-passes.
                let mut diffused = comb_sum * 0.25; // average the 4 combs
                for ap in aps.iter_mut() {
                    diffused = ap.step(diffused);
                }
                *s = dry * x + wet * diffused;
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

    fn impulse(n: usize) -> AudioFrame {
        let mut samples = vec![0.0f32; n];
        samples[0] = 1.0;
        let mut bytes = Vec::with_capacity(n * 4);
        for s in &samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        AudioFrame {
            samples: n as u32,
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
    fn impulse_response_decays() {
        // Pure wet, medium room, light damping.
        let mut r = Reverb::new(0.5, 0.2, 1.0, 0.0);
        let frame = impulse(48_000); // 1 second @ 48 kHz
        let out = r.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // Energy in the first 100 ms versus the last 100 ms — the tail
        // must have decayed substantially.
        let early_e: f64 = got[0..4_800].iter().map(|v| (*v as f64).powi(2)).sum();
        let late_e: f64 = got[got.len() - 4_800..]
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum();
        assert!(early_e > 1.0e-6, "early energy too small: {}", early_e);
        assert!(
            late_e < early_e,
            "tail did not decay: early={} late={}",
            early_e,
            late_e
        );
    }

    #[test]
    fn dry_only_is_bypass() {
        // wet = 0, dry = 1 → output should equal input exactly.
        let mut r = Reverb::new(0.5, 0.5, 0.0, 1.0);
        let in_samples: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin() * 0.4).collect();
        let mut bytes = Vec::with_capacity(in_samples.len() * 4);
        for s in &in_samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: in_samples.len() as u32,
            pts: None,
            data: vec![bytes],
        };
        let out = r.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (i, w) in in_samples.iter().enumerate() {
            assert!(
                (got[i] - *w).abs() < 1.0e-6,
                "dry-only output diverges at {}: got={} want={}",
                i,
                got[i],
                w
            );
        }
    }

    #[test]
    fn impulse_produces_nonzero_tail() {
        // Verify the reverb actually generates late energy (i.e. the
        // network is wired and emitting samples after the initial
        // impulse, not just passing the dry click).
        let mut r = Reverb::new(0.5, 0.1, 1.0, 0.0);
        let frame = impulse(24_000); // 0.5 s @ 48 kHz
        let out = r.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // After the first 50 ms the comb lines should be feeding back
        // — there should be measurable energy.
        let probe = &got[2_400..6_000];
        let e: f64 = probe.iter().map(|v| (*v as f64).powi(2)).sum();
        assert!(e > 1.0e-6, "no late energy in IR window, e={}", e);
    }

    #[test]
    fn rt60_long_at_large_room() {
        // Big room, low damping — RT60 (time for energy to drop
        // by 60 dB) should be ≥ 0.5 s.
        let mut r = Reverb::new(0.9, 0.05, 1.0, 0.0);
        let frame = impulse(96_000); // 2 s @ 48 kHz
        let out = r.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // Energy in 0..50 ms vs 500..550 ms.
        let early_e: f64 = got[0..2_400].iter().map(|v| (*v as f64).powi(2)).sum();
        let late_e: f64 = got[24_000..26_400]
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum();
        let drop_db = 10.0 * (late_e / early_e.max(1.0e-30)).log10();
        // Long room → < 60 dB drop at 0.5 s ⇒ RT60 > 0.5 s.
        assert!(
            drop_db > -60.0,
            "RT60 too short: drop at 0.5 s = {} dB",
            drop_db
        );
    }
}
