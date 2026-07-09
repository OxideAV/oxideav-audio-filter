//! Ring modulator — sine-carrier amplitude multiplication.
//!
//! A ring modulator multiplies the input signal by a sinusoidal
//! carrier in the audible band (typically 20 Hz … 4 kHz):
//!
//! ```text
//! y[n] = (1 − mix) · x[n]  +  mix · x[n] · cos(2π · fc · n / fs)
//! ```
//!
//! ## How it differs from neighbouring effects
//!
//! * [`Tremolo`](crate::Tremolo) uses a *sub-audible* sine LFO (≈ 1–20 Hz)
//!   to make the amplitude itself audibly modulate. The DSP is the same
//!   multiplication, but `fc < 20 Hz` so the listener hears the amplitude
//!   wobble rather than new spectral content.
//! * [`FreqShifter`](crate::FreqShifter) is a *single-sideband* shifter
//!   that adds a constant `Δf` to every spectral component — only the
//!   upper sideband survives. The Hilbert FIR is what cancels the lower
//!   sideband.
//! * The ring modulator below is a **double-sideband suppressed-carrier**
//!   AM operator: a tone at frequency `f` in the input becomes two new
//!   tones at `|f − fc|` and `f + fc` of equal magnitude. The carrier
//!   itself is suppressed (no DC offset added) because the multiplier
//!   has zero mean over a full period of the carrier.
//!
//! ## Derivation (product-to-sum identity)
//!
//! Take a pure input tone `x(t) = sin(2π f t)` and a cosine carrier
//! `c(t) = cos(2π fc t)`. The output is:
//!
//! ```text
//! x(t) · c(t) = sin(2π f t) · cos(2π fc t)
//!             = ½ · [ sin(2π (f + fc) t) + sin(2π (f − fc) t) ]
//! ```
//!
//! using the identity `sin α · cos β = ½ (sin(α+β) + sin(α−β))`. Two
//! mirror sidebands at `f ± fc`, each at half amplitude — the classic
//! "Dalek voice" / metallic-bell sound when `fc` is in the speech band.
//!
//! The implementation accumulates the carrier phase in `f64` so that
//! long streams don't drift; the phase wraps to `[0, 2π)` per sample.
//! All channels share a single phase accumulator so the stereo image
//! is preserved.
//!
//! ## Parameters
//!
//! * `carrier_hz` — clamped to `[0, 20_000]`. `0` collapses the output
//!   to `x · 1 = x` (no modulation; cosine of 0 is 1).
//! * `mix` — clamped to `[0, 1]`. `0` → dry pass-through; `1` → fully
//!   wet (the carrier replaces the dry signal entirely).
//!
//! ## Special cases
//!
//! * `mix = 0` ⇒ bit-exact pass-through. The phase accumulator is not
//!   advanced; on a subsequent call with non-zero mix the carrier
//!   restarts at phase 0.
//! * `mix = 1`, `carrier_hz = 0` ⇒ output equals input (the carrier
//!   reduces to a constant `1`).

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Sine-carrier ring modulator.
///
/// `carrier_hz` is the carrier frequency in Hz; `mix` is the dry/wet
/// blend in `[0, 1]`. Carrier phase persists across `process` calls
/// (set `mix = 0` to disable; see module docs).
#[derive(Debug, Clone)]
pub struct RingModulator {
    carrier_hz: f32,
    mix: f32,
    /// Carrier phase accumulator (radians, `[0, 2π)`).
    phase: f64,
}

impl RingModulator {
    /// New ring modulator. `carrier_hz` is clamped to `[0, 20_000]`;
    /// `mix` is clamped to `[0, 1]`.
    pub fn new(carrier_hz: f32, mix: f32) -> Self {
        Self {
            carrier_hz: crate::clamp_param(carrier_hz, 0.0, 0.0, 20_000.0),
            mix: crate::clamp_param(mix, 0.0, 0.0, 1.0),
            phase: 0.0,
        }
    }

    /// Currently-configured carrier frequency.
    pub fn carrier_hz(&self) -> f32 {
        self.carrier_hz
    }

    /// Currently-configured dry/wet mix.
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Reset the carrier phase to 0.
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }
}

impl AudioFilter for RingModulator {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n = channels.first().map(|c| c.len()).unwrap_or(0);

        // `mix = 0` is a bit-exact bypass. We deliberately do NOT
        // advance the phase here — the carrier behaves as if it were
        // off, and a later non-zero-mix call restarts at the existing
        // phase value (which is whatever a prior wet pass left it).
        if self.mix == 0.0 {
            let out = encode_from_f32(params.format, params.channels, input, &channels)?;
            return Ok(vec![out]);
        }

        let dphase = TWO_PI * (self.carrier_hz as f64) / (params.sample_rate as f64);
        let mix = self.mix;
        let dry = 1.0 - mix;
        for i in 0..n {
            let carrier = self.phase.cos() as f32;
            // y = dry·x + wet·(x·carrier) = x·(dry + wet·carrier).
            let g = dry + mix * carrier;
            for ch in channels.iter_mut() {
                ch[i] *= g;
            }
            self.phase += dphase;
            if self.phase >= TWO_PI {
                self.phase -= TWO_PI;
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

    fn f32_stereo(rate: u32) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 2,
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

    fn make_f32_stereo_interleaved(samples: &[(f32, f32)]) -> AudioFrame {
        // SampleFormat::F32 is interleaved (planar variant is F32P), so
        // a stereo frame carries one data plane with L0,R0,L1,R1,...
        let mut bytes = Vec::with_capacity(samples.len() * 8);
        for (l, r) in samples {
            bytes.extend_from_slice(&l.to_le_bytes());
            bytes.extend_from_slice(&r.to_le_bytes());
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

    /// mix = 0 must pass the signal through unchanged, bit-for-bit.
    #[test]
    fn mix_zero_is_bypass() {
        let in_samples: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        let frame = make_f32_mono(&in_samples);
        let mut rm = RingModulator::new(440.0, 0.0);
        let out = rm.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..in_samples.len() {
            assert_eq!(got[i], in_samples[i], "mix=0 not bit-exact at {}", i);
        }
    }

    /// carrier_hz = 0 reduces the carrier to a constant `cos(0) = 1`
    /// so even with `mix = 1` the output is `x · 1 = x`.
    #[test]
    fn zero_carrier_is_identity() {
        let in_samples: Vec<f32> = (0..256).map(|i| 0.3 * (i as f32 * 0.1).sin()).collect();
        let frame = make_f32_mono(&in_samples);
        let mut rm = RingModulator::new(0.0, 1.0);
        let out = rm.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..in_samples.len() {
            assert!(
                (got[i] - in_samples[i]).abs() < 1.0e-6,
                "zero carrier should be identity at {}: got={} want={}",
                i,
                got[i],
                in_samples[i]
            );
        }
    }

    /// Hand-derive a few samples of `x[n] · cos(2π fc n / fs)` and
    /// verify the implementation matches to within float epsilon.
    ///
    /// fs = 8000 Hz, fc = 1000 Hz ⇒ dphase = 2π · 1000 / 8000 = π/4.
    /// Carrier phase at sample n is `n · π/4`. We pick `x[n] = 1.0`
    /// (DC unity) so output = cos(n · π/4):
    ///   n=0  → cos(0)       =  1.000000
    ///   n=1  → cos(π/4)     =  √2/2 ≈ 0.7071068
    ///   n=2  → cos(π/2)     =  0.000000
    ///   n=3  → cos(3π/4)    = -√2/2 ≈ -0.7071068
    ///   n=4  → cos(π)       = -1.000000
    ///   n=5  → cos(5π/4)    = -√2/2 ≈ -0.7071068
    ///   n=6  → cos(3π/2)    =  0.000000
    ///   n=7  → cos(7π/4)    =  √2/2 ≈ 0.7071068
    ///   n=8  → cos(2π)      =  1.000000 (after wrap)
    #[test]
    fn hand_derived_dc_input_matches_carrier_shape() {
        let in_samples = vec![1.0_f32; 9];
        let frame = make_f32_mono(&in_samples);
        let mut rm = RingModulator::new(1000.0, 1.0);
        let out = rm.process(&frame, f32_mono(8000)).unwrap();
        let got = read_f32(&out[0]);
        let sqrt2_half = std::f32::consts::FRAC_1_SQRT_2;
        let expected: [f32; 9] = [
            1.0,
            sqrt2_half,
            0.0,
            -sqrt2_half,
            -1.0,
            -sqrt2_half,
            0.0,
            sqrt2_half,
            1.0,
        ];
        for i in 0..expected.len() {
            assert!(
                (got[i] - expected[i]).abs() < 1.0e-5,
                "sample {}: got={} want={}",
                i,
                got[i],
                expected[i]
            );
        }
    }

    /// With a half-wet mix the dry signal contributes `0.5·x` and the
    /// wet path contributes `0.5·x·cos(0)=0.5·x` at the first sample,
    /// so `y[0] = x[0]`. At n = 2 the carrier is `cos(π/2) = 0` so
    /// `y[2] = 0.5·x[2]`.
    #[test]
    fn half_mix_matches_dry_plus_wet() {
        let in_samples = vec![1.0_f32, 1.0, 1.0, 1.0];
        let frame = make_f32_mono(&in_samples);
        let mut rm = RingModulator::new(1000.0, 0.5);
        let out = rm.process(&frame, f32_mono(8000)).unwrap();
        let got = read_f32(&out[0]);
        let half_sqrt2 = 0.5_f32 + 0.5 * std::f32::consts::FRAC_1_SQRT_2;
        // n=0: 0.5 + 0.5·cos(0)        = 1.0
        // n=1: 0.5 + 0.5·cos(π/4)      = 0.5 + 0.5·(√2/2) ≈ 0.853553
        // n=2: 0.5 + 0.5·cos(π/2)      = 0.5
        // n=3: 0.5 + 0.5·cos(3π/4)     = 0.5 − 0.5·(√2/2) ≈ 0.146447
        let expected = [1.0_f32, half_sqrt2, 0.5, 1.0 - half_sqrt2];
        for i in 0..expected.len() {
            assert!(
                (got[i] - expected[i]).abs() < 1.0e-5,
                "sample {}: got={} want={}",
                i,
                got[i],
                expected[i]
            );
        }
    }

    /// Phase must persist across successive `process` calls so the
    /// output of `[ frame_a, frame_b ]` equals processing `[a;b]` in
    /// one go. This is the streaming contract.
    #[test]
    fn phase_persists_across_calls() {
        let in_samples: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.01).sin() * 0.4).collect();

        // One-shot pass.
        let frame_full = make_f32_mono(&in_samples);
        let mut rm_a = RingModulator::new(750.0, 1.0);
        let out_a = rm_a.process(&frame_full, f32_mono(48_000)).unwrap();
        let got_a = read_f32(&out_a[0]);

        // Same data split into two halves, fed sequentially.
        let half = in_samples.len() / 2;
        let frame_lo = make_f32_mono(&in_samples[..half]);
        let frame_hi = make_f32_mono(&in_samples[half..]);
        let mut rm_b = RingModulator::new(750.0, 1.0);
        let out_lo = rm_b.process(&frame_lo, f32_mono(48_000)).unwrap();
        let out_hi = rm_b.process(&frame_hi, f32_mono(48_000)).unwrap();
        let mut got_b = read_f32(&out_lo[0]);
        got_b.extend(read_f32(&out_hi[0]));

        assert_eq!(got_a.len(), got_b.len());
        for i in 0..got_a.len() {
            assert!(
                (got_a[i] - got_b[i]).abs() < 1.0e-5,
                "streaming continuity broken at {}: one-shot={} split={}",
                i,
                got_a[i],
                got_b[i],
            );
        }
    }

    /// Stereo: both channels share a single phase, so the carrier
    /// applies identically to L and R. An input with `L = R` must
    /// still satisfy `L_out = R_out` for every sample.
    #[test]
    fn stereo_channels_track_each_other() {
        let pairs: Vec<(f32, f32)> = (0..256)
            .map(|i| {
                let v = 0.3 * (i as f32 * 0.05).sin();
                (v, v)
            })
            .collect();
        let frame = make_f32_stereo_interleaved(&pairs);
        let mut rm = RingModulator::new(600.0, 1.0);
        let out = rm.process(&frame, f32_stereo(48_000)).unwrap();
        // The output frame is interleaved L,R,L,R, ... 8 bytes/sample.
        let interleaved: Vec<f32> = out[0].data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(interleaved.len(), pairs.len() * 2);
        for i in 0..pairs.len() {
            let l = interleaved[2 * i];
            let r = interleaved[2 * i + 1];
            assert!(
                (l - r).abs() < 1.0e-6,
                "L/R diverge at sample {}: L={} R={}",
                i,
                l,
                r,
            );
        }
    }

    /// Sanity: a DC input ring-modulated by a non-zero carrier must
    /// produce zero mean over an integer number of carrier periods.
    /// fs = 8000, fc = 1000 ⇒ 8 samples per period. Sum over the
    /// first 8 samples of `1·cos(n·π/4)` is 0 (the discrete cosine
    /// over a full period sums to zero).
    #[test]
    fn dc_input_has_zero_mean_over_one_period() {
        let in_samples = vec![1.0_f32; 8];
        let frame = make_f32_mono(&in_samples);
        let mut rm = RingModulator::new(1000.0, 1.0);
        let out = rm.process(&frame, f32_mono(8000)).unwrap();
        let got = read_f32(&out[0]);
        let sum: f32 = got.iter().sum();
        assert!(
            sum.abs() < 1.0e-5,
            "expected zero mean over one carrier period, got sum={}",
            sum
        );
    }

    /// Out-of-range parameters are clamped — the constructor must not
    /// panic on weird inputs.
    #[test]
    fn parameters_are_clamped() {
        let rm = RingModulator::new(-100.0, -0.5);
        assert_eq!(rm.carrier_hz(), 0.0);
        assert_eq!(rm.mix(), 0.0);

        let rm = RingModulator::new(1.0e9, 2.0);
        assert_eq!(rm.carrier_hz(), 20_000.0);
        assert_eq!(rm.mix(), 1.0);
    }

    /// Empty input frame must produce a valid empty output frame
    /// (no panic, no carrier advance issues).
    #[test]
    fn empty_frame_passes_through() {
        let frame = make_f32_mono(&[]);
        let mut rm = RingModulator::new(1000.0, 1.0);
        let out = rm.process(&frame, f32_mono(48_000)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].samples, 0);
    }
}
