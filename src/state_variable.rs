//! State Variable Filter (SVF) — Chamberlin two-integrator-loop topology.
//!
//! A second-order resonant filter built from a pair of integrators
//! arranged in a series feedback loop. Unlike the bilinear-transform
//! [`Biquad`](crate::biquad::Biquad) (Direct Form II Transposed, a
//! transfer-function realisation) the SVF is a *state-space* topology
//! whose internal variables are the band-pass and low-pass outputs.
//! All four canonical responses — low-pass, band-pass, high-pass, and
//! notch — fall out of the same recurrence "for free" and can be read
//! simultaneously, with no recoefficient computation when the cutoff or
//! resonance is modulated. That property makes the SVF the canonical
//! synthesiser filter (envelope-swept cutoff, LFO-swept Q) and a
//! standard choice for modulation-friendly EQ stages.
//!
//! # Topology and recurrence (Chamberlin form)
//!
//! Two integrators in series with two feedback paths produce the four
//! outputs in a single update:
//!
//! ```text
//! hp[n]    = x[n] - q · bp[n-1] - lp[n-1]
//! bp[n]    = f · hp[n] + bp[n-1]
//! lp[n]    = f · bp[n] + lp[n-1]
//! notch[n] = hp[n] + lp[n]
//! ```
//!
//! The two coefficients are
//!
//! ```text
//! f = 2 · sin(π · f_c / f_s)        (frequency parameter)
//! q = 1 / Q                          (damping coefficient)
//! ```
//!
//! and the per-channel state is the pair `(bp[n-1], lp[n-1])` — no
//! input history is retained. (The two-integrator structure makes the
//! filter `f`-tunable per-sample without pre-warping or coefficient
//! resolves; this is the synth-filter advantage over a bilinear
//! biquad.)
//!
//! # Stability bound
//!
//! The discrete two-integrator loop is conditionally stable: roughly,
//! `f · q + f² < 4`, which is satisfied as long as the cutoff stays
//! well below `f_s / 6` (≈ 8 kHz at 48 kHz) and `Q ≥ 0.5`. We clamp
//! the cutoff to `f_s / 6.5` and `Q` to `[0.5, 50]` at construction
//! and on `set_cutoff` / `set_q`, both to enforce stability and to
//! match the conservative range documented in the classical references.
//! Callers needing a higher cutoff should fall back to the bilinear
//! [`Biquad`](crate::biquad::Biquad).
//!
//! # Output mode
//!
//! The SVF runs all four taps internally; [`SvfMode`] selects which
//! tap is written into the frame. The cheap routing means a single
//! SVF instance can be reconfigured between LP/BP/HP/Notch without
//! touching the state. Construction `SvfFilter::low_pass(...)` is a
//! shorthand for `SvfFilter::new(SvfMode::LowPass, ...)`.
//!
//! Note that the [`SvfMode::Notch`] tap is implemented as
//! `hp + lp`. In the analog prototype this cancels at `f_c` for
//! every `Q`, but in the discrete two-integrator-loop form the
//! cancellation degrades as `Q` rises (the band-pass tap has peak
//! gain `Q` at centre and the discrete `f`-warp leaves a small
//! residual; at `Q = 25` the centre gain can even rise above
//! unity). The notch is therefore most useful at low `Q` (≤ 1);
//! callers wanting a sharp narrow-band reject at higher `Q` should
//! use the bilinear [`Biquad`](crate::biquad::Biquad) notch.
//!
//! # Per-channel state
//!
//! Each input channel keeps its own `(bp, lp)` integrator pair so
//! stereo inputs do not cross-couple through the resonant loop.
//!
//! # Algorithm provenance
//!
//! The two-integrator-loop topology is a classical analog-circuit
//! design (op-amp Tow–Thomas / KHN biquad) and the discrete-time form
//! used here is the Chamberlin difference-equation realisation
//! documented in *Hal Chamberlin, "Musical Applications of
//! Microprocessors" (Hayden Books, 2nd ed., 1985, ch. 19)* and
//! reproduced in subsequent academic DSP texts. The frequency
//! parameter `f = 2·sin(π·f_c/f_s)` and damping `q = 1/Q` are derived
//! from first principles by setting the analog
//! `H_lp(s) = 1 / (s² + s/Q + 1)` equal to the digital response of
//! the integrator loop and matching the two parameters.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Which of the four canonical SVF taps to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvfMode {
    /// Low-pass output `lp[n]`. −12 dB/oct rolloff above `f_c`.
    LowPass,
    /// Band-pass output `bp[n]`. Centered at `f_c`, bandwidth set by `Q`.
    BandPass,
    /// High-pass output `hp[n]`. −12 dB/oct rolloff below `f_c`.
    HighPass,
    /// Notch (band-stop) output `hp[n] + lp[n]`. Zero gain at `f_c`,
    /// unity gain at DC and Nyquist.
    Notch,
}

/// Per-channel integrator state.
#[derive(Debug, Clone, Copy, Default)]
struct ChState {
    /// Band-pass output of the previous sample.
    bp_prev: f32,
    /// Low-pass output of the previous sample.
    lp_prev: f32,
}

/// Streaming Chamberlin State Variable Filter.
///
/// Cutoff and Q are mutable post-construction without recomputing
/// coefficients (a single `sin` per `set_cutoff` call); see
/// [`SvfFilter::set_cutoff`] / [`SvfFilter::set_q`].
#[derive(Debug, Clone)]
pub struct SvfFilter {
    mode: SvfMode,
    cutoff_hz: f32,
    q: f32,
    /// Cached `f = 2·sin(π·f_c/f_s)`. Recomputed when the cached
    /// sample rate diverges from the per-call `params.sample_rate`,
    /// or when `set_cutoff` is called.
    f_coef: f32,
    /// Cached `q_coef = 1/Q`.
    q_coef: f32,
    /// Cached sample rate against which `f_coef` was last computed;
    /// `0` until first `process` call.
    cached_sr: u32,
    state: Vec<ChState>,
}

impl SvfFilter {
    /// Minimum permitted `Q` (any lower → numerical instability).
    pub const MIN_Q: f32 = 0.5;
    /// Maximum permitted `Q` (above ~50 the bandpass is so narrow that
    /// quantisation noise dominates; classical references cap at this).
    pub const MAX_Q: f32 = 50.0;
    /// Maximum permitted `f_c` as a fraction of the sample rate. The
    /// Chamberlin form's discrete stability bound `f·q + f² < 4`
    /// breaks down above roughly `f_s / 6`; we clamp slightly below
    /// that to preserve headroom across all in-range `Q` values.
    pub const MAX_CUTOFF_FRACTION: f32 = 1.0 / 6.5;

    /// Build an SVF with the given output tap, cutoff, and `Q`. The
    /// cutoff and `Q` are clamped to the documented stability range.
    pub fn new(mode: SvfMode, cutoff_hz: f32, q: f32) -> Self {
        Self {
            mode,
            cutoff_hz: crate::clamp_param(cutoff_hz, 1_000.0, 0.0, f32::MAX),
            q: crate::clamp_param(q, 0.707, Self::MIN_Q, Self::MAX_Q),
            f_coef: 0.0,
            q_coef: 1.0 / crate::clamp_param(q, 0.707, Self::MIN_Q, Self::MAX_Q),
            cached_sr: 0,
            state: Vec::new(),
        }
    }

    /// Shorthand for [`SvfFilter::new`] with [`SvfMode::LowPass`].
    pub fn low_pass(cutoff_hz: f32, q: f32) -> Self {
        Self::new(SvfMode::LowPass, cutoff_hz, q)
    }

    /// Shorthand for [`SvfFilter::new`] with [`SvfMode::BandPass`].
    pub fn band_pass(center_hz: f32, q: f32) -> Self {
        Self::new(SvfMode::BandPass, center_hz, q)
    }

    /// Shorthand for [`SvfFilter::new`] with [`SvfMode::HighPass`].
    pub fn high_pass(cutoff_hz: f32, q: f32) -> Self {
        Self::new(SvfMode::HighPass, cutoff_hz, q)
    }

    /// Shorthand for [`SvfFilter::new`] with [`SvfMode::Notch`].
    pub fn notch(center_hz: f32, q: f32) -> Self {
        Self::new(SvfMode::Notch, center_hz, q)
    }

    /// Currently selected output tap.
    pub fn mode(&self) -> SvfMode {
        self.mode
    }

    /// Reconfigure the output tap without touching integrator state.
    pub fn set_mode(&mut self, mode: SvfMode) {
        self.mode = mode;
    }

    /// Current cutoff in Hz (clamped value).
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Current `Q` (clamped value).
    pub fn q(&self) -> f32 {
        self.q
    }

    /// Set a new cutoff. Recomputes the `f` coefficient against the
    /// last known sample rate (or defers until first `process` if the
    /// filter has not yet seen one). Modulating per sample is cheap
    /// (single `sin`), so this is the synth-filter envelope-sweep API.
    pub fn set_cutoff(&mut self, cutoff_hz: f32) {
        self.cutoff_hz = cutoff_hz.max(0.0);
        if self.cached_sr > 0 {
            self.recompute_coefs(self.cached_sr);
        }
    }

    /// Set a new `Q`. Recomputes the damping coefficient `q_coef = 1/Q`.
    pub fn set_q(&mut self, q: f32) {
        self.q = q.clamp(Self::MIN_Q, Self::MAX_Q);
        self.q_coef = 1.0 / self.q;
    }

    /// Clear per-channel integrator state without rebuilding the cached
    /// `f` / `q` coefficients.
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

    fn recompute_coefs(&mut self, sample_rate: u32) {
        let fs = sample_rate as f32;
        let max_cutoff = fs * Self::MAX_CUTOFF_FRACTION;
        let fc = self.cutoff_hz.clamp(0.0, max_cutoff);
        self.f_coef = 2.0 * (std::f32::consts::PI * fc / fs).sin();
        self.q_coef = 1.0 / self.q;
        self.cached_sr = sample_rate;
    }
}

impl AudioFilter for SvfFilter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        if self.cached_sr != params.sample_rate {
            self.recompute_coefs(params.sample_rate);
        }
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(channels.len());
        let f = self.f_coef;
        let q = self.q_coef;
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.state[ch_idx];
            let mut bp = st.bp_prev;
            let mut lp = st.lp_prev;
            for s in buf.iter_mut() {
                let x = *s;
                let hp = x - q * bp - lp;
                bp += f * hp;
                lp += f * bp;
                // JOINT flush-to-zero on the integrator pair: an
                // impulse tail otherwise dwells in the f32 subnormal
                // range for seconds. Both components must flush
                // atomically — truncating one while the other is live
                // can sustain a limit cycle in the two-integrator loop
                // (same failure mode as the biquad state pair; see
                // biquad::State::flush_denormals).
                if bp.abs() < 1.0e-25 && lp.abs() < 1.0e-25 {
                    bp = 0.0;
                    lp = 0.0;
                }
                *s = match self.mode {
                    SvfMode::LowPass => lp,
                    SvfMode::BandPass => bp,
                    SvfMode::HighPass => hp,
                    SvfMode::Notch => hp + lp,
                };
            }
            st.bp_prev = bp;
            st.lp_prev = lp;
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

    /// Render a sine of `freq` Hz for `n_samples` at `fs`, push through
    /// the filter, and return the steady-state RMS gain (output_RMS
    /// divided by input_RMS) skipping a 10-ms warm-up.
    fn measure_rms_gain(filter: &mut SvfFilter, freq: f32, fs: u32, n_samples: usize) -> f32 {
        let w = 2.0 * std::f32::consts::PI * freq / fs as f32;
        let input: Vec<f32> = (0..n_samples).map(|i| (i as f32 * w).sin()).collect();
        let frame = make_f32_mono(&input);
        let out = filter.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let warm = (fs as f32 * 0.010) as usize;
        let in_rms = {
            let s: f64 = input[warm..].iter().map(|&v| (v as f64) * (v as f64)).sum();
            (s / (input.len() - warm) as f64).sqrt()
        };
        let out_rms = {
            let s: f64 = got[warm..].iter().map(|&v| (v as f64) * (v as f64)).sum();
            (s / (got.len() - warm) as f64).sqrt()
        };
        (out_rms / in_rms) as f32
    }

    #[test]
    fn low_pass_passes_below_cutoff_attenuates_above() {
        // 1 kHz LPF, Q = 0.707 (Butterworth-equivalent damping).
        // 100 Hz should pass at ~unity; 8 kHz should be attenuated
        // by well over 20 dB (the discrete SVF's −24 dB/oct asymptote
        // takes a couple of decades to settle but at 3 octaves above
        // cutoff we expect ≥ 20 dB rejection).
        let fs = 48_000u32;
        let mut lp = SvfFilter::low_pass(1_000.0, 0.707);
        let g_pass = measure_rms_gain(&mut lp, 100.0, fs, 8_192);
        let g_pass_db = 20.0 * g_pass.log10();
        assert!(
            g_pass_db.abs() < 0.5,
            "100 Hz pass-band gain = {} dB (expected ~0)",
            g_pass_db
        );

        let mut lp = SvfFilter::low_pass(1_000.0, 0.707);
        let g_stop = measure_rms_gain(&mut lp, 8_000.0, fs, 8_192);
        let g_stop_db = 20.0 * g_stop.log10();
        assert!(
            g_stop_db < -20.0,
            "8 kHz stop-band gain = {} dB (expected ≤ -20)",
            g_stop_db
        );
    }

    #[test]
    fn high_pass_attenuates_below_cutoff() {
        // 1 kHz HPF, Q = 0.707.
        let fs = 48_000u32;
        let mut hp = SvfFilter::high_pass(1_000.0, 0.707);
        let g_pass = measure_rms_gain(&mut hp, 8_000.0, fs, 8_192);
        let g_pass_db = 20.0 * g_pass.log10();
        // Even though the discrete SVF rolls off above f_s / 6 the
        // band 8 kHz @ 48 kHz is comfortably inside the pass-band; the
        // HPF should preserve it to within 1 dB.
        assert!(
            g_pass_db.abs() < 1.0,
            "8 kHz pass-band gain (HPF) = {} dB",
            g_pass_db
        );

        let mut hp = SvfFilter::high_pass(1_000.0, 0.707);
        let g_stop = measure_rms_gain(&mut hp, 100.0, fs, 8_192);
        let g_stop_db = 20.0 * g_stop.log10();
        assert!(
            g_stop_db < -20.0,
            "100 Hz stop-band (HPF) gain = {} dB",
            g_stop_db
        );
    }

    #[test]
    fn band_pass_peak_near_center() {
        // BPF centred at 1 kHz with moderate Q. Probe at the centre
        // and one decade either side; the centre tap should be the
        // loudest by a clear margin.
        let fs = 48_000u32;
        let mut bp = SvfFilter::band_pass(1_000.0, 4.0);
        let g_center = measure_rms_gain(&mut bp, 1_000.0, fs, 16_384);

        let mut bp = SvfFilter::band_pass(1_000.0, 4.0);
        let g_low = measure_rms_gain(&mut bp, 100.0, fs, 16_384);

        let mut bp = SvfFilter::band_pass(1_000.0, 4.0);
        let g_high = measure_rms_gain(&mut bp, 10_000.0, fs, 16_384);

        assert!(
            g_center > g_low * 5.0,
            "BPF centre gain {} not >> low-skirt gain {}",
            g_center,
            g_low
        );
        assert!(
            g_center > g_high * 3.0,
            "BPF centre gain {} not >> high-skirt gain {}",
            g_center,
            g_high
        );
    }

    #[test]
    fn notch_rejects_center_keeps_dc_and_high_band() {
        // Notch at 1 kHz with Q = 0.5 (the minimum permitted Q, which
        // gives the *deepest* notch in the Chamberlin topology — see
        // the module docs for why the discrete `hp + lp` notch is a
        // low-Q-only configuration, with depth shrinking as Q rises).
        // At Q = 0.5 the centre dip is ~24 dB while the DC band stays
        // within 0.5 dB of unity gain.
        let fs = 48_000u32;
        let mut nt = SvfFilter::notch(1_000.0, 0.5);
        let g_center = measure_rms_gain(&mut nt, 1_000.0, fs, 32_768);

        let mut nt = SvfFilter::notch(1_000.0, 0.5);
        let g_low = measure_rms_gain(&mut nt, 50.0, fs, 32_768);

        let g_center_db = 20.0 * g_center.log10();
        let g_low_db = 20.0 * g_low.log10();
        assert!(
            g_center_db < -15.0,
            "Notch centre gain = {} dB (expected deep cut)",
            g_center_db
        );
        assert!(
            g_low_db.abs() < 1.0,
            "Notch DC-band gain = {} dB (expected ~0)",
            g_low_db
        );
    }

    #[test]
    fn mode_switch_does_not_clobber_state() {
        // Run a few hundred samples in LPF mode then flip to BPF —
        // the integrator pair must persist (we read both internal
        // taps from the same recurrence).
        let fs = 48_000u32;
        let mut svf = SvfFilter::low_pass(1_000.0, 0.707);
        let w = 2.0 * std::f32::consts::PI * 1_000.0 / fs as f32;
        let input: Vec<f32> = (0..256).map(|i| (i as f32 * w).sin()).collect();
        let frame = make_f32_mono(&input);
        let _ = svf.process(&frame, f32_mono(fs)).unwrap();
        let bp_before = svf.state[0].bp_prev;
        let lp_before = svf.state[0].lp_prev;
        svf.set_mode(SvfMode::BandPass);
        assert_eq!(svf.state[0].bp_prev, bp_before);
        assert_eq!(svf.state[0].lp_prev, lp_before);
        assert_eq!(svf.mode(), SvfMode::BandPass);
    }

    #[test]
    fn channels_do_not_cross_talk() {
        // Stereo: L = unit step, R = silence. Through the BPF the L
        // channel rings — the R channel must remain silent.
        let fs = 48_000u32;
        let n = 4_096usize;
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&1.0f32.to_le_bytes()); // L
            bytes.extend_from_slice(&0.0f32.to_le_bytes()); // R
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut bp = SvfFilter::band_pass(1_000.0, 4.0);
        let out = bp
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 2,
                    sample_rate: fs,
                },
            )
            .unwrap();
        let bytes = &out[0].data[0];
        let mut r_peak = 0.0f32;
        for s in 0..n {
            let off = (s * 2 + 1) * 4;
            let v =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
            r_peak = r_peak.max(v.abs());
        }
        assert!(r_peak < 1.0e-6, "R-channel leaked, peak = {}", r_peak);
    }

    #[test]
    fn reset_clears_state_keeps_coefs() {
        let fs = 48_000u32;
        let mut svf = SvfFilter::low_pass(1_000.0, 0.707);
        // Prime it.
        let frame = make_f32_mono(&vec![1.0; 1024]);
        let _ = svf.process(&frame, f32_mono(fs)).unwrap();
        let f_before = svf.f_coef;
        let q_before = svf.q_coef;
        assert!(svf.state[0].lp_prev.abs() > 1e-6);
        svf.reset();
        assert_eq!(svf.state[0].bp_prev, 0.0);
        assert_eq!(svf.state[0].lp_prev, 0.0);
        assert_eq!(svf.f_coef, f_before);
        assert_eq!(svf.q_coef, q_before);
    }

    #[test]
    fn set_cutoff_recomputes_after_first_sample_rate_seen() {
        // Brand-new filter has not seen a sample rate yet; set_cutoff
        // must NOT panic and must NOT recompute. After one process()
        // call the cached SR is populated; further set_cutoff calls
        // recompute against that SR.
        let fs = 48_000u32;
        let mut svf = SvfFilter::low_pass(1_000.0, 0.707);
        assert_eq!(svf.f_coef, 0.0); // not yet sized
        svf.set_cutoff(2_000.0);
        assert_eq!(svf.f_coef, 0.0); // still deferred
        let frame = make_f32_mono(&vec![0.5; 256]);
        let _ = svf.process(&frame, f32_mono(fs)).unwrap();
        let f_at_2k = svf.f_coef;
        svf.set_cutoff(4_000.0);
        let f_at_4k = svf.f_coef;
        assert!(f_at_4k > f_at_2k, "f should grow with cutoff");
    }

    #[test]
    fn q_clamp_within_range() {
        let svf = SvfFilter::low_pass(1_000.0, 0.01);
        assert_eq!(svf.q(), SvfFilter::MIN_Q);
        let svf = SvfFilter::low_pass(1_000.0, 999.0);
        assert_eq!(svf.q(), SvfFilter::MAX_Q);
        let mut svf = SvfFilter::low_pass(1_000.0, 1.0);
        svf.set_q(0.0);
        assert_eq!(svf.q(), SvfFilter::MIN_Q);
    }

    #[test]
    fn cutoff_clamped_to_stability_bound() {
        // Push the cutoff well above fs/6 — the filter must clamp it
        // internally so the recurrence stays bounded. Probe with a
        // unit-impulse train and confirm the output never explodes.
        let fs = 48_000u32;
        let mut svf = SvfFilter::low_pass(20_000.0, 4.0);
        let input: Vec<f32> = (0..2_048)
            .map(|i| if i % 64 == 0 { 1.0 } else { 0.0 })
            .collect();
        let frame = make_f32_mono(&input);
        let out = svf.process(&frame, f32_mono(fs)).unwrap();
        let got = read_f32(&out[0]);
        let peak = got.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            peak.is_finite() && peak < 100.0,
            "SVF blew up: peak = {}",
            peak
        );
    }

    #[test]
    fn streaming_continuity_split_calls() {
        // Two equal-length input chunks processed sequentially must
        // produce the same output as the same input processed in one
        // shot — the integrator state must survive between calls.
        let fs = 48_000u32;
        let w = 2.0 * std::f32::consts::PI * 1_000.0 / fs as f32;
        let n = 1_024usize;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 * w).sin()).collect();

        let mut svf_one_shot = SvfFilter::low_pass(2_000.0, 0.707);
        let one_shot = read_f32(
            &svf_one_shot
                .process(&make_f32_mono(&input), f32_mono(fs))
                .unwrap()[0],
        );

        let mut svf_split = SvfFilter::low_pass(2_000.0, 0.707);
        let head = read_f32(
            &svf_split
                .process(&make_f32_mono(&input[..n / 2]), f32_mono(fs))
                .unwrap()[0],
        );
        let tail = read_f32(
            &svf_split
                .process(&make_f32_mono(&input[n / 2..]), f32_mono(fs))
                .unwrap()[0],
        );

        let mut combined = head;
        combined.extend(tail);
        assert_eq!(combined.len(), one_shot.len());
        for (a, b) in combined.iter().zip(one_shot.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "streaming continuity broken: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn sample_rate_change_recomputes_coefs() {
        // First call at 48 kHz, second at 96 kHz — the cached f must
        // shrink (since f = 2·sin(π·f_c/f_s) and f_s doubles).
        let mut svf = SvfFilter::low_pass(1_000.0, 0.707);
        let _ = svf
            .process(&make_f32_mono(&[0.5; 32]), f32_mono(48_000))
            .unwrap();
        let f_at_48k = svf.f_coef;
        let _ = svf
            .process(&make_f32_mono(&[0.5; 32]), f32_mono(96_000))
            .unwrap();
        let f_at_96k = svf.f_coef;
        assert!(
            f_at_96k < f_at_48k,
            "f must shrink with higher fs (got {} → {})",
            f_at_48k,
            f_at_96k
        );
        // Approximately half (small-angle limit: sin(π·f_c/f_s) ≈ π·f_c/f_s).
        let ratio = f_at_96k / f_at_48k;
        assert!(
            (ratio - 0.5).abs() < 0.02,
            "f ratio at 48k→96k expected ≈ 0.5, got {}",
            ratio
        );
    }
}
