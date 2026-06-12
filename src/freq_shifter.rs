//! Frequency Shifter — single-sideband (Hilbert) frequency shifter.
//!
//! Unlike a [`PitchShift`](crate::PitchShift), which scales frequencies
//! by a ratio (so harmonic ratios are preserved), a **frequency
//! shifter** adds a constant `Δf` to every frequency component:
//!
//! ```text
//! sin(2π f t)  →  sin(2π (f + Δf) t)
//! ```
//!
//! Because each component shifts by the same number of Hz, harmonic
//! intervals are destroyed: a 100/200/300/400 Hz harmonic series
//! shifted by +50 Hz becomes 150/250/350/450 Hz — an inharmonic
//! bell-like tone. This is the classic "ring-mod / Bode 7720" sound.
//!
//! # Method — Hilbert SSB demodulation
//!
//! A true frequency shift is achieved by upper-sideband modulation:
//!
//! ```text
//! y(t) = x(t) · cos(2π Δf t)  −  hilbert(x(t)) · sin(2π Δf t)
//! ```
//!
//! where `hilbert(x)` is the +90° phase-shifted version of `x`. The
//! Hilbert transform is implemented here as a windowed-sinc FIR with
//! `taps = 2·half_taps + 1` (default 127) — the FIR with non-zero
//! coefficients only at odd indices, weighted by a Blackman window.
//! The standard derivation:
//!
//! ```text
//! h[n] = (2 / (π · n))    for n odd
//!        0                for n even or n = 0
//!
//! windowed: h_w[n] = h[n] · blackman[n + half_taps]
//! ```
//!
//! Each input sample is convolved against `h_w` to produce the
//! quadrature signal; the real channel is delayed by `half_taps`
//! samples (a circular delay buffer) so the two arms align. The
//! SSB combine then yields the shifted output.
//!
//! Latency: `half_taps` samples (default 63). At 48 kHz that's
//! ~1.3 ms — typically inaudible.
//!
//! Negative `delta_hz` shifts the spectrum downward (use the
//! conjugate carrier).
//!
//! # Parameters
//!
//! * `delta_hz` — shift amount in Hz, range `[-10_000, +10_000]`.
//! * `half_taps` — Hilbert FIR half-length (default 63, clamped to
//!   `[15, 255]`; total FIR length is `2·half_taps + 1`). Higher
//!   = sharper low-frequency response, more latency.
//!
//! # Limitations
//!
//! * The Hilbert FIR has zero response at DC and Nyquist, so very
//!   low (< 50 Hz at 48 kHz with half_taps=63) and very high (~
//!   Nyquist) components leak into the un-shifted sideband as
//!   distortion. Increase `half_taps` to extend the low-frequency
//!   response.
//! * Aliasing: components shifted past Nyquist fold back. The
//!   caller is responsible for low-passing the input if `delta_hz`
//!   could push content over `(sample_rate/2 − delta_hz)`.
//!
//! # References
//!
//! The Hilbert-FIR single-sideband modulator is the standard
//! frequency-shifter construction: a windowed-sinc Hilbert pair forms
//! the analytic signal, which a complex oscillator rotates before
//! taking the real part.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming Hilbert SSB frequency shifter.
#[derive(Debug, Clone)]
pub struct FreqShifter {
    delta_hz: f32,
    half_taps: usize,
    /// FIR coefficients, length `2·half_taps + 1`. Odd-indexed
    /// (relative to centre) only; even taps are zero by construction
    /// of the ideal Hilbert FIR.
    hilbert_fir: Vec<f32>,
    /// One ring buffer per channel — holds the last `2·half_taps + 1`
    /// input samples for the Hilbert convolution.
    history: Vec<Vec<f32>>,
    /// Carrier phase in radians, accumulated per sample.
    phase: f32,
    sample_rate: u32,
}

fn blackman(n: usize, len: usize) -> f32 {
    // Standard symmetric Blackman window of length `len`:
    //   w[n] = 0.42 - 0.5·cos(2π n / (len-1)) + 0.08·cos(4π n / (len-1))
    if len <= 1 {
        return 1.0;
    }
    let phi = std::f32::consts::PI * n as f32 / (len - 1) as f32;
    0.42 - 0.5 * (2.0 * phi).cos() + 0.08 * (4.0 * phi).cos()
}

fn hilbert_kernel(half_taps: usize) -> Vec<f32> {
    let len = 2 * half_taps + 1;
    let mut out = vec![0.0f32; len];
    for (k, slot) in out.iter_mut().enumerate() {
        let n_signed = k as isize - half_taps as isize;
        // n == 0 → 0; even n → 0 in the ideal Hilbert FIR.
        if n_signed == 0 || n_signed.unsigned_abs() % 2 == 0 {
            *slot = 0.0;
            continue;
        }
        let pi = std::f32::consts::PI;
        let raw = 2.0 / (pi * n_signed as f32);
        *slot = raw * blackman(k, len);
    }
    out
}

impl FreqShifter {
    /// New shifter with `delta_hz = +100, half_taps = 63`.
    pub fn new() -> Self {
        Self::with(100.0, 63)
    }

    /// Custom-parameter constructor.
    pub fn with(delta_hz: f32, half_taps: usize) -> Self {
        let delta_hz = delta_hz.clamp(-10_000.0, 10_000.0);
        let half_taps = half_taps.clamp(15, 255);
        let hilbert_fir = hilbert_kernel(half_taps);
        Self {
            delta_hz,
            half_taps,
            hilbert_fir,
            history: Vec::new(),
            phase: 0.0,
            sample_rate: 0,
        }
    }

    /// Current shift (Hz).
    pub fn delta_hz(&self) -> f32 {
        self.delta_hz
    }

    /// Current Hilbert-FIR half-length.
    pub fn half_taps(&self) -> usize {
        self.half_taps
    }

    /// Reset per-channel history + carrier phase.
    pub fn reset(&mut self) {
        for h in &mut self.history {
            for v in h.iter_mut() {
                *v = 0.0;
            }
        }
        self.phase = 0.0;
    }
}

impl Default for FreqShifter {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for FreqShifter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let channels_in = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = channels_in.len();
        if n_chan == 0 {
            let out = encode_from_f32(params.format, params.channels, input, &channels_in)?;
            return Ok(vec![out]);
        }
        let n_samples = channels_in[0].len();
        let fir_len = self.hilbert_fir.len();
        // Initialise / grow per-channel history. Each history holds
        // the last `fir_len` input samples (centre-tap aligned).
        if self.history.len() < n_chan {
            self.history.resize(n_chan, vec![0.0f32; fir_len]);
        } else {
            for h in &mut self.history {
                if h.len() != fir_len {
                    h.clear();
                    h.resize(fir_len, 0.0);
                }
            }
        }
        self.sample_rate = params.sample_rate;

        let mut out_channels: Vec<Vec<f32>> =
            (0..n_chan).map(|_| vec![0.0f32; n_samples]).collect();

        // Carrier phase increment per sample.
        let omega = 2.0 * std::f32::consts::PI * self.delta_hz / params.sample_rate as f32;

        for i in 0..n_samples {
            // Step the carrier first so every channel uses the same
            // phase at sample `i`.
            let c = self.phase.cos();
            let s = self.phase.sin();
            self.phase += omega;
            // Keep phase bounded so cos/sin remain accurate for long
            // streams.
            if self.phase > std::f32::consts::TAU {
                self.phase -= std::f32::consts::TAU;
            } else if self.phase < -std::f32::consts::TAU {
                self.phase += std::f32::consts::TAU;
            }

            for ch in 0..n_chan {
                let hist = &mut self.history[ch];
                // Shift older samples by one, append the new sample.
                hist.copy_within(1..fir_len, 0);
                hist[fir_len - 1] = channels_in[ch][i];

                // Convolve against the Hilbert FIR → quadrature signal.
                let mut q = 0.0f32;
                for (k, &h) in hist.iter().enumerate() {
                    q += h * self.hilbert_fir[fir_len - 1 - k];
                }
                // Real-channel value at the centre tap of the FIR.
                let r = hist[self.half_taps];
                // SSB upper-sideband combine: y = r·cos − q·sin.
                out_channels[ch][i] = r * c - q * s;
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

    fn sine(amp: f32, freq: f32, fs: u32, n: usize) -> Vec<f32> {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        (0..n).map(|i| amp * (i as f32 * w).sin()).collect()
    }

    /// Estimate the dominant frequency of a real signal using the
    /// crate's [`crate::fft`] module. Returns the bin centre in Hz.
    fn dominant_freq(x: &[f32], fs: u32) -> f32 {
        use crate::fft::real_fft;
        // Pad / trim to a power of two for the radix-2 FFT in this crate.
        let mut len = 1usize;
        while len * 2 <= x.len() {
            len *= 2;
        }
        let spec = real_fft(&x[..len]);
        let n_bins = spec.len() / 2;
        // Skip DC (bin 0) — Hilbert FIR has zero response there.
        let mut best = 1usize;
        let mut best_mag = 0.0f32;
        for (i, c) in spec.iter().take(n_bins).enumerate().skip(1) {
            let m = c.magnitude();
            if m > best_mag {
                best_mag = m;
                best = i;
            }
        }
        best as f32 * fs as f32 / len as f32
    }

    #[test]
    fn zero_shift_passes_through() {
        let fs = 48_000u32;
        let n = 4_096usize;
        let samples = sine(0.5, 1_000.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut fs_shift = FreqShifter::with(0.0, 63);
        let out = fs_shift.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // With Δf=0, y = r·1 − q·0 = r, i.e. the centre-tap delay of
        // the input. Verify dominant frequency is unchanged.
        let f_dom = dominant_freq(&got, fs);
        assert!(
            (f_dom - 1_000.0).abs() < (fs as f32 / n as f32) * 2.0,
            "zero shift altered dominant frequency: f_dom={f_dom}"
        );
    }

    #[test]
    fn positive_shift_moves_tone_up() {
        let fs = 48_000u32;
        let n = 16_384usize;
        let f_in = 1_000.0f32;
        let delta = 300.0f32;
        let samples = sine(0.5, f_in, fs, n);
        let frame = make_f32_mono(&samples);
        let mut fs_shift = FreqShifter::with(delta, 127);
        let out = fs_shift.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        // Drop the start so the FIR is fully primed.
        let tail = &got[1_024..];
        let f_dom = dominant_freq(tail, fs);
        let bin = fs as f32 / (1usize << 14) as f32;
        assert!(
            (f_dom - (f_in + delta)).abs() < bin * 3.0,
            "positive shift didn't move tone: dom={f_dom}, expected ~{}",
            f_in + delta
        );
    }

    #[test]
    fn negative_shift_moves_tone_down() {
        let fs = 48_000u32;
        let n = 16_384usize;
        let f_in = 2_000.0f32;
        let delta = -500.0f32;
        let samples = sine(0.5, f_in, fs, n);
        let frame = make_f32_mono(&samples);
        let mut fs_shift = FreqShifter::with(delta, 127);
        let out = fs_shift.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[1_024..];
        let f_dom = dominant_freq(tail, fs);
        let bin = fs as f32 / (1usize << 14) as f32;
        assert!(
            (f_dom - (f_in + delta)).abs() < bin * 3.0,
            "negative shift didn't move tone: dom={f_dom}, expected ~{}",
            f_in + delta
        );
    }

    #[test]
    fn output_stays_bounded() {
        let fs = 48_000u32;
        let n = 4_096usize;
        let samples = sine(0.9, 1_500.0, fs, n);
        let frame = make_f32_mono(&samples);
        let mut fs_shift = FreqShifter::with(750.0, 127);
        let out = fs_shift.process(&frame, f32_mono(fs)).unwrap();
        for v in read_f32(&out[0]) {
            assert!(v.is_finite(), "non-finite SSB output");
            // SSB combine of two normalised signals (|cos|, |sin| ≤ 1)
            // peaks at ≈ √2 × input peak. Allow some headroom.
            assert!(v.abs() < 2.0, "shifter sample {v} out of bounds");
        }
    }

    #[test]
    fn hilbert_kernel_has_zero_even_taps() {
        let k = hilbert_kernel(63);
        let centre = 63usize;
        for (i, v) in k.iter().enumerate() {
            let n = i as isize - centre as isize;
            if n == 0 || n.unsigned_abs() % 2 == 0 {
                assert!(
                    v.abs() < 1e-9,
                    "even tap {i} should be 0 in ideal Hilbert FIR, got {v}"
                );
            }
        }
    }
}
