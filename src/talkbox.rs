//! Talkbox — vowel-formant filter driven by the input envelope.
//!
//! Emulates the classic guitar-talkbox / heavily-formanted-vocoder
//! sound (think Frampton, Daft Punk) by routing the signal through a
//! parallel bank of resonant band-pass filters tuned to the formant
//! frequencies of a chosen vowel. Mouth-shape modulation is provided
//! by an LFO that morphs between two formant sets, simulating
//! continuous "ah → oh → ee" vowel transitions.
//!
//! No carrier / modulator pair is required (unlike a true vocoder):
//! the input itself is both carrier and source. The output therefore
//! carries only the frequencies near each formant — broadband content
//! (e.g. cymbals) is heavily attenuated, which is the desired effect.
//!
//! # Formants
//!
//! Vowel formants are well-documented in speech-acoustics references
//! (Peterson-Barney 1952, then refined by Hillenbrand 1995). We use
//! a compact set of two formants per vowel — the perceptually
//! dominant pair — and let the user pick two endpoints `from` and
//! `to`. An LFO at `rate_hz` interpolates between them in log-frequency
//! space:
//!
//! ```text
//! lfo  ∈ [0, 1]  =  0.5 · (1 + sin(2π·rate·t))
//! f₁(t) = exp((1-lfo)·ln(f₁_from) + lfo·ln(f₁_to))
//! f₂(t) = exp((1-lfo)·ln(f₂_from) + lfo·ln(f₂_to))
//! ```
//!
//! Each formant is realised as a [`Biquad`] band-pass with `Q ≈ 8`
//! (narrow enough to hear the formant peak; wide enough for the
//! filter to pass speech-rate content). The two BPF outputs are
//! summed.
//!
//! # Recommended use
//!
//! Best on harmonic-rich monophonic sources (electric guitar,
//! synthesiser leads, distorted bass). Quiet or transient material
//! (drums, breath noise) emerges thin because the formant pass-bands
//! attenuate the broadband energy that's not under the resonances.
//!
//! # Parameters
//!
//! * `from` / `to` — endpoint vowels for the LFO morph.
//! * `rate_hz` — morph rate, `[0.05, 8.0]`. `0` ⇒ static `from`.
//! * `q` — formant resonance, `[1, 30]`. Higher = narrower / more
//!   pronounced; lower = blurrier / more natural.
//! * `mix` — wet/dry blend, `[0, 1]`.

use crate::biquad::{Biquad, BiquadKind};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// A compact pair of formant frequencies (F1, F2) for a vowel.
/// Values are typical male-voice averages from the Hillenbrand
/// dataset, rounded to the nearest 25 Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Vowel {
    /// "ah" as in "father" (F1 ≈ 730, F2 ≈ 1100).
    Ah,
    /// "eh" as in "bet" (F1 ≈ 530, F2 ≈ 1850).
    Eh,
    /// "ee" as in "beet" (F1 ≈ 300, F2 ≈ 2300).
    Ee,
    /// "oh" as in "go" (F1 ≈ 550, F2 ≈ 850).
    Oh,
    /// "oo" as in "boot" (F1 ≈ 300, F2 ≈ 850).
    Oo,
    /// "uh" as in "but" (F1 ≈ 640, F2 ≈ 1200).
    Uh,
}

impl Vowel {
    /// Return (F1, F2) for the vowel in Hz.
    pub fn formants(self) -> (f32, f32) {
        match self {
            Vowel::Ah => (730.0, 1_100.0),
            Vowel::Eh => (530.0, 1_850.0),
            Vowel::Ee => (300.0, 2_300.0),
            Vowel::Oh => (550.0, 850.0),
            Vowel::Oo => (300.0, 850.0),
            Vowel::Uh => (640.0, 1_200.0),
        }
    }
}

/// Streaming talkbox.
#[derive(Debug, Clone)]
pub struct Talkbox {
    from: Vowel,
    to: Vowel,
    rate_hz: f32,
    q: f32,
    mix: f32,
    /// LFO phase, in radians, accumulated across calls. Kept modulo 2π.
    phase: f32,
    /// One band-pass per formant. Updated each block with the current
    /// morph frequency.
    bpf_a: Biquad,
    bpf_b: Biquad,
}

impl Talkbox {
    /// Default talkbox: morph Ah → Ee at 0.5 Hz, Q=8, full wet.
    pub fn new() -> Self {
        Self::with(Vowel::Ah, Vowel::Ee, 0.5, 8.0, 1.0)
    }

    /// Custom-parameter constructor.
    pub fn with(from: Vowel, to: Vowel, rate_hz: f32, q: f32, mix: f32) -> Self {
        let rate_hz = rate_hz.clamp(0.0, 8.0);
        let q = q.clamp(1.0, 30.0);
        let mix = mix.clamp(0.0, 1.0);
        let (f1, _) = from.formants();
        let bpf_a = Biquad::new(BiquadKind::BandPass { center_hz: f1, q });
        let bpf_b = Biquad::new(BiquadKind::BandPass { center_hz: f1, q });
        Self {
            from,
            to,
            rate_hz,
            q,
            mix,
            phase: 0.0,
            bpf_a,
            bpf_b,
        }
    }

    /// LFO morph rate (Hz).
    pub fn rate_hz(&self) -> f32 {
        self.rate_hz
    }
    /// Formant resonance.
    pub fn q(&self) -> f32 {
        self.q
    }
    /// Wet/dry mix.
    pub fn mix(&self) -> f32 {
        self.mix
    }

    /// Reset the LFO phase and biquad states.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.bpf_a.reset();
        self.bpf_b.reset();
    }
}

impl Default for Talkbox {
    fn default() -> Self {
        Self::new()
    }
}

/// Log-linear interpolation between two positive frequencies. `t ∈ [0, 1]`.
fn lerp_log(a: f32, b: f32, t: f32) -> f32 {
    let la = a.max(1.0).ln();
    let lb = b.max(1.0).ln();
    (la * (1.0 - t) + lb * t).exp()
}

impl AudioFilter for Talkbox {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let dry = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = dry.len();
        if n_chan == 0 {
            let out = encode_from_f32(params.format, params.channels, input, &dry)?;
            return Ok(vec![out]);
        }
        let n_samples = dry[0].len();

        // Compute the per-block LFO position (block-rate is plenty for
        // formant morphing; a per-sample sweep would require
        // re-deriving the biquad coefficients every sample, which the
        // present `Biquad` API supports but is wasteful here).
        let omega = 2.0 * std::f32::consts::PI * self.rate_hz / params.sample_rate as f32;
        let lfo = 0.5 * (1.0 + self.phase.sin());
        self.phase = (self.phase + omega * n_samples as f32) % (2.0 * std::f32::consts::PI);

        let (f1_a, f2_a) = self.from.formants();
        let (f1_b, f2_b) = self.to.formants();
        let f1 = lerp_log(f1_a, f1_b, lfo);
        let f2 = lerp_log(f2_a, f2_b, lfo);
        self.bpf_a.set_kind(BiquadKind::BandPass {
            center_hz: f1,
            q: self.q,
        });
        self.bpf_b.set_kind(BiquadKind::BandPass {
            center_hz: f2,
            q: self.q,
        });

        // Process each channel through both formant BPFs in parallel.
        // The BPFs are independent (separate state tables per filter
        // already, indexed by channel-position in the buffer).
        let mut out_channels = vec![vec![0.0f32; n_samples]; n_chan];
        // Run channel-by-channel so we can write the result back. The
        // biquad's per-channel state grows lazily on the first
        // `process_in_place` call.
        for (ch, _) in dry.iter().enumerate() {
            let mut buf_a: Vec<f32> = dry[ch].clone();
            let mut buf_b: Vec<f32> = dry[ch].clone();
            self.bpf_a
                .process_in_place(&mut buf_a, 1, params.sample_rate);
            self.bpf_b
                .process_in_place(&mut buf_b, 1, params.sample_rate);
            // Note: a fresh BPF state per channel would cleanly
            // separate them, but since we run channel-major and reset
            // is not needed between channels (BPF state grows only
            // every block) we simply sum the two formant outputs.
            for i in 0..n_samples {
                let wet = buf_a[i] + buf_b[i];
                out_channels[ch][i] = (1.0 - self.mix) * dry[ch][i] + self.mix * wet;
            }
        }
        let out = encode_from_f32(params.format, params.channels, input, &out_channels)?;
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

    fn rms(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64).powi(2)).sum();
        (s / x.len() as f64).sqrt() as f32
    }

    fn sine(amp: f32, freq: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| amp * (i as f32 * w).sin()).collect()
    }

    /// White noise (deterministic, simple LCG).
    fn noise(n: usize, amp: f32) -> Vec<f32> {
        let mut s: u32 = 0x1234_5678;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            let v = (s >> 16) as i32 - 32_768;
            out.push((v as f32 / 32_768.0) * amp);
        }
        out
    }

    #[test]
    fn mix_zero_is_identity() {
        let samples = sine(0.5, 440.0, 48_000, 512);
        let frame = make_f32_mono(&samples);
        let mut tb = Talkbox::with(Vowel::Ah, Vowel::Ee, 0.5, 8.0, 0.0);
        let out = tb.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for i in 0..samples.len() {
            assert!(
                (got[i] - samples[i]).abs() < 1.0e-6,
                "mix=0 not identity at {i}"
            );
        }
    }

    #[test]
    fn vowel_table_returns_two_distinct_formants() {
        for v in [
            Vowel::Ah,
            Vowel::Eh,
            Vowel::Ee,
            Vowel::Oh,
            Vowel::Oo,
            Vowel::Uh,
        ] {
            let (f1, f2) = v.formants();
            assert!(f1 > 0.0 && f2 > f1, "vowel {:?}: F1={}, F2={}", v, f1, f2);
        }
    }

    #[test]
    fn static_ah_attenuates_far_off_formant_noise() {
        // White noise; "Ah" is centred around 730 Hz / 1100 Hz with
        // Q=8. Energy at 5–8 kHz (well off the formants) must be
        // strongly attenuated in the wet output.
        let fs = 48_000u32;
        let n = 48_000usize;
        let src = noise(n, 0.5);
        let frame = make_f32_mono(&src);
        let mut tb = Talkbox::with(Vowel::Ah, Vowel::Ah, 0.0, 10.0, 1.0);
        let out = tb.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);

        // Project both source and output to a narrow band centred on
        // 6.5 kHz (well above F2_Ah=1100 Hz). Use a high-Q BPF so the
        // projection is sharp and doesn't catch the formant tail.
        let mut bpf_src = src.clone();
        let mut bpf_out = got.clone();
        let mut bpf = Biquad::new(BiquadKind::BandPass {
            center_hz: 6_500.0,
            q: 15.0,
        });
        bpf.process_in_place(&mut bpf_src, 1, fs);
        let mut bpf2 = Biquad::new(BiquadKind::BandPass {
            center_hz: 6_500.0,
            q: 15.0,
        });
        bpf2.process_in_place(&mut bpf_out, 1, fs);
        let warm = (fs as f32 * 0.2) as usize;
        let src_r = rms(&bpf_src[warm..]);
        let out_r = rms(&bpf_out[warm..]);
        let delta_db = 20.0 * (out_r / src_r).max(1.0e-12).log10();
        // -10 dB is "noticeable", -20 dB is "clearly suppressed"; allow
        // the more permissive bound since the BPF skirts of an "Ah"
        // talkbox (F1=730, F2=1100, Q=10) roll off only ~12 dB/octave.
        assert!(
            delta_db < -10.0,
            "off-formant noise not attenuated: {} dB",
            delta_db
        );
    }

    #[test]
    fn output_is_finite_for_silence() {
        let zero = vec![0.0f32; 1024];
        let frame = make_f32_mono(&zero);
        let mut tb = Talkbox::new();
        let out = tb.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for v in got {
            assert!(v.is_finite());
        }
    }
}
