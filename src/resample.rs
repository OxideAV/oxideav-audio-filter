//! Polyphase windowed-sinc rate conversion.
//!
//! Given input/output sample rates `src_rate` / `dst_rate`, the resampler
//! builds an analytic polyphase filter bank at construction time:
//!
//! ```text
//! L = lcm(src_rate, dst_rate)
//! up   = L / src_rate           // rational upsample factor
//! down = L / dst_rate           // rational downsample factor
//! ```
//!
//! The bank has `up` phases and `taps_per_phase` taps per phase. The
//! prototype filter is a sinc with cutoff `1 / (2 * max(up, down))`,
//! windowed by a Kaiser window with `beta = 8.0` (~80 dB stopband). Per
//! output sample we identify the integer source-sample index plus a phase,
//! then convolve the corresponding `taps_per_phase` history samples with
//! that phase's tap row.
//!
//! The prototype's *length* scales with the resampling ratio:
//! `max(up, down)` controls the cutoff, so the filter is sized to span a
//! fixed number of sinc zero-crossings (`HALF_ZERO_CROSSINGS` per side)
//! *of that cutoff*. For interpolation (`up ≥ down`) this is the same as
//! a fixed history depth; for decimation (`down > up`) the input-rate
//! history depth grows by the decimation factor so the narrower
//! anti-aliasing low-pass is resolved with the same transition sharpness
//! and stop-band rejection (otherwise a short fixed-tap decimator leaks
//! aliasing — measured ≈ −13 dB rejection at 48 k→16 k with a 32-tap
//! fixed window vs ≥ 80 dB with the ratio-scaled length).
//!
//! State (the per-channel sample history) is preserved across `process`
//! calls so streaming yields the same output as a one-shot call.
//!
//! # Parameters
//! * `src_rate` — input sample rate in Hz.
//! * `dst_rate` — output sample rate in Hz.
//!
//! # Limitations
//! * Sample format is preserved (input format == output format).
//! * Channel count is preserved.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Error, Result};

/// Number of sinc zero-crossings (of the design cutoff) the windowed
/// prototype spans on each side of its centre. The total prototype length
/// is `2·HALF_ZERO_CROSSINGS·max(up, down) + 1` L-rate taps; for the
/// `up ≥ down` (interpolation / equal-rate) case this is `≈ 2·HZC` input
/// samples of history, matching the historical 32-tap window at
/// `HZC = 16`.
const HALF_ZERO_CROSSINGS: usize = 16;
const KAISER_BETA: f32 = 8.0;

/// Polyphase windowed-sinc resampler.
pub struct Resample {
    src_rate: u32,
    /// Target output sample rate (exposed via [`Resample::dst_rate`]); the
    /// resampler operates on the precomputed `up`/`down` ratio derived from
    /// it, not on the raw rate.
    dst_rate: u32,
    /// `up` (== L / src_rate) — number of phases.
    up: u32,
    /// `down` (== L / dst_rate) — phase increment per output sample.
    down: u32,
    /// Input-rate history depth: how many recent input samples each output
    /// convolves. Equal to `ceil(prototype_len / up)`, so it grows with the
    /// decimation factor when `down > up`.
    taps_per_phase: usize,
    /// Filter bank: `up * taps_per_phase` floats. Row `p` contains the taps
    /// for phase `p`.
    taps: Vec<f32>,
    state: Option<ResampleState>,
}

/// Input-rate history depth required for the given rational ratio: enough
/// input samples to cover the full ratio-scaled prototype.
fn taps_per_phase_for(up: u32, down: u32) -> usize {
    let proto_len = 2 * HALF_ZERO_CROSSINGS * up.max(down) as usize + 1;
    proto_len.div_ceil(up as usize)
}

struct ResampleState {
    channels: usize,
    /// Per-channel ring of recent input samples (length `taps_per_phase`).
    history: Vec<Vec<f32>>,
    /// Per-channel write cursor (next slot to write).
    cursor: Vec<usize>,
    /// Number of input samples consumed so far.
    samples_in: u64,
    /// Number of output samples produced so far.
    samples_out: u64,
}

struct ProduceCfg<'a> {
    taps: &'a [f32],
    up: u32,
    down: u32,
    taps_per_phase: usize,
    samples_in_before: u64,
    samples_out_before: u64,
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn lcm(a: u32, b: u32) -> u64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a as u64 / gcd(a, b) as u64) * b as u64
}

/// Modified Bessel function of the first kind, order 0. Series expansion
/// converges quickly for the small arguments we use (Kaiser window taps).
fn bessel_i0(x: f32) -> f32 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half_x_sq = (x as f64 * x as f64) / 4.0;
    for k in 1..50 {
        term *= half_x_sq / (k as f64 * k as f64);
        sum += term;
        if term < 1.0e-12 * sum {
            break;
        }
    }
    sum as f32
}

/// Build the windowed-sinc prototype low-pass filter (before polyphase
/// decomposition). The prototype runs at the `L = lcm(src, dst)` rate;
/// `proto[n]` is the `n`-th tap of a length-`up·taps_per_phase`
/// linear-phase FIR. The continuous-time impulse this approximates is
/// the ideal bandlimited-interpolation kernel
/// `h_s(t) = 2·f_c·sinc(2·f_c·t)` of the resampling theory doc
/// (`docs/audio/filter/jos-theory-of-sample-rate-conversion.html`),
/// sampled at the L-rate and Kaiser-windowed to finite length.
fn build_prototype(up: u32, down: u32, taps_per_phase: usize, beta: f32) -> Vec<f32> {
    let total = (up as usize) * taps_per_phase;
    let center = (total as f32 - 1.0) / 2.0;
    // Cutoff in cycles per L-rate sample.
    let cutoff = 1.0 / (2.0 * up.max(down) as f32);
    let i0_beta = bessel_i0(beta);
    // Window half-width in L-rate samples (so the Kaiser taper spans the
    // whole prototype symmetrically about `center`).
    let half = total as f32 / 2.0;

    let mut proto = vec![0.0f32; total];
    for (n, slot) in proto.iter_mut().enumerate().take(total) {
        let m = n as f32 - center;
        let s = if m == 0.0 {
            2.0 * cutoff
        } else {
            let arg = 2.0 * std::f32::consts::PI * cutoff * m;
            arg.sin() / (std::f32::consts::PI * m)
        };
        let r = (n as f32 - center) / half;
        let w = if r.abs() >= 1.0 {
            0.0
        } else {
            bessel_i0(beta * (1.0 - r * r).sqrt()) / i0_beta
        };
        *slot = s * w;
    }
    proto
}

fn build_polyphase(up: u32, down: u32, taps_per_phase: usize, beta: f32) -> Vec<f32> {
    let total = (up as usize) * taps_per_phase;
    let proto = build_prototype(up, down, taps_per_phase, beta);

    // Polyphase decomposition: row `p`, column `k` holds proto[p + k*up].
    // Apply gain of `up` to compensate for the upsample-by-zeros operation.
    // The prototype may be shorter than `up * taps_per_phase` (its true
    // length is `2·HZC·max(up,down)+1`); indices past the end read 0.
    let mut bank = vec![0.0f32; total];
    for p in 0..(up as usize) {
        for k in 0..taps_per_phase {
            let proto_idx = p + k * (up as usize);
            if proto_idx < proto.len() {
                bank[p * taps_per_phase + k] = proto[proto_idx] * up as f32;
            }
        }
    }
    bank
}

impl Resample {
    /// Build a new resampler. Returns `Error::Unsupported` if either rate is
    /// zero or the rate ratio leads to an unreasonable LCM.
    pub fn new(src_rate: u32, dst_rate: u32) -> Result<Self> {
        if src_rate == 0 || dst_rate == 0 {
            return Err(Error::unsupported("resample rate must be non-zero"));
        }
        let l = lcm(src_rate, dst_rate);
        if l > 100_000_000 {
            return Err(Error::unsupported(
                "resample sample-rate ratio too extreme for LCM-polyphase design",
            ));
        }
        let up = (l / src_rate as u64) as u32;
        let down = (l / dst_rate as u64) as u32;
        let taps_per_phase = taps_per_phase_for(up, down);
        let taps = build_polyphase(up, down, taps_per_phase, KAISER_BETA);
        Ok(Self {
            src_rate,
            dst_rate,
            up,
            down,
            taps_per_phase,
            taps,
            state: None,
        })
    }

    /// Input sample rate this resampler was built for.
    pub fn src_rate(&self) -> u32 {
        self.src_rate
    }

    /// Target output sample rate.
    pub fn dst_rate(&self) -> u32 {
        self.dst_rate
    }

    /// Rational up-sampling factor `up = L / src_rate` (number of polyphase
    /// branches, where `L = lcm(src_rate, dst_rate)`).
    pub fn up_factor(&self) -> u32 {
        self.up
    }

    /// Rational down-sampling factor `down = L / dst_rate` (the phase
    /// increment per output sample).
    pub fn down_factor(&self) -> u32 {
        self.down
    }

    /// Passband edge of the anti-aliasing / anti-imaging prototype filter,
    /// in Hz. Following the resampling theory (`docs/audio/filter/
    /// jos-theory-of-sample-rate-conversion.html`): the low-pass cutoff is
    /// placed at half the *lower* of the two sample rates so that, on
    /// decimation, the reconstructed continuous signal is bandlimited
    /// below half the new (lower) rate — no aliasing — and, on
    /// interpolation, the spectral images introduced by zero-stuffing are
    /// rejected. With the design cutoff `1/(2·max(up,down))` cycles per
    /// L-rate sample and the L-rate equal to `up·src_rate`, the passband
    /// edge in Hz is `min(src_rate, dst_rate) / 2`.
    pub fn passband_edge_hz(&self) -> f64 {
        self.src_rate.min(self.dst_rate) as f64 / 2.0
    }

    /// Closed-form magnitude response (in dB) of the anti-aliasing
    /// prototype low-pass filter at `freq_hz`, referenced to the *input*
    /// sample rate. This is the DTFT of the windowed-sinc prototype that
    /// the polyphase bank realises, evaluated at
    /// `Ω = 2π·freq_hz / L` (L = `up·src_rate`, the prototype's own rate):
    ///
    /// ```text
    /// |H(e^{jΩ})| = |Σ_n proto[n]·e^{-jΩn}|
    /// ```
    ///
    /// referenced so that the in-band (DC) response is exactly 0 dB. Pure
    /// function of the rate ratio — touches no streaming state. Useful for
    /// asserting the band-limiting design (passband flatness, stop-band
    /// rejection / anti-aliasing) without running samples through the
    /// convolution. `freq_hz` must lie in `[0, L/2]`; frequencies above the
    /// input Nyquist (`src_rate/2`) still evaluate (the prototype must
    /// reject the images that live there during interpolation).
    pub fn prototype_response_db(&self, freq_hz: f64) -> f64 {
        let proto = build_prototype(self.up, self.down, self.taps_per_phase, KAISER_BETA);
        let l_rate = self.up as f64 * self.src_rate as f64;
        let omega = 2.0 * std::f64::consts::PI * freq_hz / l_rate;
        // DTFT at Ω, and at DC (Ω = 0) for the 0 dB reference.
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        let mut dc = 0.0f64;
        for (n, &h) in proto.iter().enumerate() {
            let h = h as f64;
            let phase = omega * n as f64;
            re += h * phase.cos();
            im -= h * phase.sin();
            dc += h;
        }
        let mag = (re * re + im * im).sqrt();
        20.0 * (mag / dc.abs().max(1.0e-30)).log10()
    }

    fn ensure_state(&mut self, channels: usize) {
        let needs_rebuild = match &self.state {
            Some(s) => s.channels != channels,
            None => true,
        };
        if needs_rebuild {
            let tpp = self.taps_per_phase;
            self.state = Some(ResampleState {
                channels,
                history: (0..channels).map(|_| vec![0.0; tpp]).collect(),
                cursor: vec![0; channels],
                samples_in: 0,
                samples_out: 0,
            });
        }
    }

    #[inline]
    fn read_back(history: &[f32], cursor: usize, back: usize) -> f32 {
        let n = history.len();
        let idx = (cursor + n - 1 - back) % n;
        history[idx]
    }

    #[inline]
    fn push_sample(history: &mut [f32], cursor: &mut usize, sample: f32) {
        history[*cursor] = sample;
        *cursor = (*cursor + 1) % history.len();
    }

    /// Inner kernel parameters so we can avoid borrowing `&self` while we
    /// hold `&mut self.state` and stay under the clippy argument limit.
    fn produce_for_channel(
        cfg: ProduceCfg<'_>,
        ch_in: &[f32],
        history: &mut [f32],
        cursor: &mut usize,
        out_buf: &mut Vec<f32>,
    ) {
        let up_u64 = cfg.up as u64;
        let down_u64 = cfg.down as u64;
        let tpp = cfg.taps_per_phase;

        for (i, x) in ch_in.iter().enumerate() {
            Self::push_sample(history, cursor, *x);
            let new_in_pos = cfg.samples_in_before + i as u64 + 1;
            loop {
                let next_out_idx = cfg.samples_out_before + out_buf.len() as u64;
                let phase_acc = next_out_idx * down_u64;
                let src_pos = phase_acc / up_u64;
                if src_pos + 1 > new_in_pos {
                    break;
                }
                let phase = (phase_acc % up_u64) as usize;
                let back0 = (new_in_pos - 1 - src_pos) as usize;
                if back0 >= tpp {
                    out_buf.push(0.0);
                    continue;
                }
                let row = &cfg.taps[phase * tpp..(phase + 1) * tpp];
                let mut acc = 0.0f32;
                for (k, tap) in row.iter().enumerate().take(tpp) {
                    let back = back0 + k;
                    if back >= tpp {
                        break;
                    }
                    acc += *tap * Self::read_back(history, *cursor, back);
                }
                out_buf.push(acc);
            }
        }
    }
}

impl AudioFilter for Resample {
    /// The symmetric windowed-sinc prototype has
    /// `2·HALF_ZERO_CROSSINGS·max(up, down) + 1` taps at the
    /// L-rate (`fs·up`), so its group delay is
    /// `HALF_ZERO_CROSSINGS·max(up, down)` L-rate samples =
    /// `HALF_ZERO_CROSSINGS·max(up, down) / up` input samples,
    /// reported rounded to the nearest input sample.
    fn latency_samples(&self, _params: AudioStreamParams) -> usize {
        let l_rate_delay = HALF_ZERO_CROSSINGS as f64 * self.up.max(self.down) as f64;
        (l_rate_delay / self.up as f64).round() as usize
    }

    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        if params.sample_rate != self.src_rate {
            return Err(Error::invalid(
                "Resample: input stream sample_rate does not match constructor",
            ));
        }
        let channels = decode_to_f32(input, params.format, params.channels)?;
        let n_chan = channels.len();
        self.ensure_state(n_chan);

        let taps = &self.taps;
        let up = self.up;
        let down = self.down;

        let taps_per_phase = self.taps_per_phase;
        let state = self.state.as_mut().expect("state ensured above");
        let samples_in_before = state.samples_in;
        let samples_in_after = samples_in_before + channels[0].len() as u64;
        let samples_out_before = state.samples_out;

        let mut out_per_channel: Vec<Vec<f32>> = (0..n_chan).map(|_| Vec::new()).collect();
        for (ch, out_ch) in out_per_channel.iter_mut().enumerate().take(n_chan) {
            let history = &mut state.history[ch];
            let cursor_ref = &mut state.cursor[ch];
            Self::produce_for_channel(
                ProduceCfg {
                    taps,
                    up,
                    down,
                    taps_per_phase,
                    samples_in_before,
                    samples_out_before,
                },
                &channels[ch],
                history,
                cursor_ref,
                out_ch,
            );
        }
        let produced = out_per_channel[0].len() as u64;
        state.samples_in = samples_in_after;
        state.samples_out += produced;

        if produced == 0 {
            return Ok(Vec::new());
        }

        // Output frame inherits format + channels from the input stream
        // (resample preserves both); only the sample rate changes, and
        // that lives on the downstream port spec, not the frame.
        let frame = encode_from_f32(params.format, params.channels, input, &out_per_channel)?;
        Ok(vec![frame])
    }

    fn flush(&mut self, params: AudioStreamParams) -> Result<Vec<AudioFrame>> {
        let n_chan = match &self.state {
            Some(s) => s.channels,
            None => return Ok(Vec::new()),
        };
        // Push half a tap window of zeros to flush the tail.
        let pad = self.taps_per_phase / 2;
        let template = AudioFrame {
            samples: 0,
            pts: None,
            data: vec![Vec::new()],
        };

        let taps = &self.taps;
        let up = self.up;
        let down = self.down;
        let taps_per_phase = self.taps_per_phase;

        let state = self.state.as_mut().expect("state checked");
        let samples_in_before = state.samples_in;
        let samples_out_before = state.samples_out;
        let mut out_per_channel: Vec<Vec<f32>> = (0..n_chan).map(|_| Vec::new()).collect();

        for (ch, out_ch) in out_per_channel.iter_mut().enumerate().take(n_chan) {
            let history = &mut state.history[ch];
            let cursor_ref = &mut state.cursor[ch];
            let zero_pad = vec![0.0f32; pad];
            Self::produce_for_channel(
                ProduceCfg {
                    taps,
                    up,
                    down,
                    taps_per_phase,
                    samples_in_before,
                    samples_out_before,
                },
                &zero_pad,
                history,
                cursor_ref,
                out_ch,
            );
        }
        let produced = out_per_channel[0].len() as u64;
        state.samples_in += pad as u64;
        state.samples_out += produced;

        if produced == 0 {
            return Ok(Vec::new());
        }

        let frame = encode_from_f32(params.format, params.channels, &template, &out_per_channel)?;
        Ok(vec![frame])
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

    fn sine_f32(freq: f32, rate: u32, n: usize) -> AudioFrame {
        let mut bytes = Vec::with_capacity(n * 4);
        for i in 0..n {
            let t = i as f32 / rate as f32;
            let s = (2.0 * std::f32::consts::PI * freq * t).sin();
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
    fn round_trip_48k_44k_48k_within_40db() {
        // Round-trip a 1 kHz sine through 48000 → 44100 → 48000 and verify
        // the reconstruction error is below -40 dB. Because the resampler
        // applies a non-integer group delay, integer-sample RMS comparison
        // would be dominated by sub-sample misalignment. Instead we project
        // the round-tripped signal onto sin/cos at the same frequency and
        // measure the residual.
        let n = 48_000;
        let freq = 1000.0_f64;
        let rate = 48_000.0_f64;
        let original_frame = sine_f32(freq as f32, 48_000, n);

        let mut down = Resample::new(48_000, 44_100).unwrap();
        let mut mid_frames = down.process(&original_frame, f32_mono(48_000)).unwrap();
        mid_frames.extend(down.flush(f32_mono(44_100)).unwrap());

        let mut up = Resample::new(44_100, 48_000).unwrap();
        let mut out_frames: Vec<AudioFrame> = Vec::new();
        for f in &mid_frames {
            let outs = up.process(f, f32_mono(44_100)).unwrap();
            out_frames.extend(outs);
        }
        out_frames.extend(up.flush(f32_mono(48_000)).unwrap());

        let mut result: Vec<f32> = Vec::new();
        for f in &out_frames {
            result.extend(read_f32(f));
        }

        let mid_start = 5_000;
        let mid_end = (n - 5_000).min(result.len());
        assert!(
            mid_end > mid_start + 1_000,
            "round-trip output too short: {} samples",
            result.len()
        );

        // Least-squares fit of (a*sin + b*cos) to the round-tripped signal.
        let mut sum_s2 = 0.0f64;
        let mut sum_c2 = 0.0f64;
        let mut sum_xs = 0.0f64;
        let mut sum_xc = 0.0f64;
        for (i, x) in result.iter().enumerate().take(mid_end).skip(mid_start) {
            let t = i as f64 / rate;
            let s = (2.0 * std::f64::consts::PI * freq * t).sin();
            let c = (2.0 * std::f64::consts::PI * freq * t).cos();
            let x = *x as f64;
            sum_s2 += s * s;
            sum_c2 += c * c;
            sum_xs += x * s;
            sum_xc += x * c;
        }
        let a = sum_xs / sum_s2;
        let b = sum_xc / sum_c2;
        let mag = (a * a + b * b).sqrt();

        let mut sum_r2 = 0.0f64;
        let mut sum_x2 = 0.0f64;
        for (i, x) in result.iter().enumerate().take(mid_end).skip(mid_start) {
            let t = i as f64 / rate;
            let s = (2.0 * std::f64::consts::PI * freq * t).sin();
            let c = (2.0 * std::f64::consts::PI * freq * t).cos();
            let model = a * s + b * c;
            let x = *x as f64;
            let resid = x - model;
            sum_r2 += resid * resid;
            sum_x2 += x * x;
        }
        let count = (mid_end - mid_start) as f64;
        let rms_resid = (sum_r2 / count).sqrt();
        let rms_signal = (sum_x2 / count).sqrt();
        let snr_db = 20.0 * (rms_resid / rms_signal).log10();
        eprintln!(
            "round-trip: amplitude={:.6}, residual SNR={:.2} dB",
            mag, snr_db
        );
        assert!(
            (mag - 1.0).abs() < 0.01,
            "round-trip amplitude {} is far from unity",
            mag
        );
        assert!(
            snr_db < -40.0,
            "round-trip residual SNR {} dB exceeded -40 dB threshold",
            snr_db
        );
    }

    #[test]
    fn output_rate_matches_target() {
        let n = 9_600;
        let frame = sine_f32(440.0, 48_000, n);
        let mut r = Resample::new(48_000, 24_000).unwrap();
        let outs = r.process(&frame, f32_mono(48_000)).unwrap();
        let total: usize = outs.iter().map(|f| f.samples as usize).sum();
        assert!(
            (total as i32 - 4_800).abs() < 64,
            "expected ~4800 samples, got {}",
            total
        );
        // Output sample rate is the resampler's `dst_rate` and lives on
        // the downstream port spec (the registry builds an output
        // PortSpec at `dst_rate`); the AudioFrame itself no longer
        // carries it.
    }

    // ---- Band-limited-interpolation design verification (round 352).
    // Grounded in `docs/audio/filter/jos-theory-of-sample-rate-conversion.html`
    // / `jos-bandlimited-interpolation.html` / `jos-resample.html`. These
    // assert the anti-aliasing / anti-imaging *prototype* low-pass filter
    // directly via the closed-form `prototype_response_db` evaluator (no
    // sample-domain estimation error), plus one end-to-end aliasing check
    // through the convolution.

    /// Test rate pairs spanning integer-decimate, integer-interpolate,
    /// and the awkward 48k↔44.1k coprime ratio (large polyphase bank).
    const RATE_PAIRS: [(u32, u32); 5] = [
        (48_000, 24_000), // ×½ decimate (up=1, down=2)
        (8_000, 48_000),  // ×6 interpolate (up=6, down=1)
        (48_000, 44_100), // coprime down (up=147, down=160)
        (44_100, 48_000), // coprime up   (up=160, down=147)
        (48_000, 16_000), // ×⅓ decimate (up=1, down=3)
    ];

    #[test]
    fn rational_factors_satisfy_lcm_identity() {
        // The design invariant: up·src_rate = down·dst_rate = L, and the
        // bank has exactly `up` phases. This pins the polyphase geometry
        // the theory doc derives (`L = lcm`, evaluate the reconstruction
        // at integer multiples of the new period).
        for &(s, d) in &RATE_PAIRS {
            let r = Resample::new(s, d).unwrap();
            let l_from_up = r.up_factor() as u64 * s as u64;
            let l_from_down = r.down_factor() as u64 * d as u64;
            assert_eq!(
                l_from_up, l_from_down,
                "{}->{}: up·src ({}) != down·dst ({})",
                s, d, l_from_up, l_from_down
            );
            assert_eq!(r.src_rate(), s);
            assert_eq!(r.dst_rate(), d);
            // up and down must be coprime (L is the *least* common multiple).
            assert_eq!(
                gcd(r.up_factor(), r.down_factor()),
                1,
                "{}->{}: up/down not coprime ({}/{})",
                s,
                d,
                r.up_factor(),
                r.down_factor()
            );
        }
    }

    #[test]
    fn prototype_passband_is_flat_to_unity() {
        // The resampling theory places the cutoff at half the *lower* rate
        // with "the scale factor [maintaining] unity gain in the
        // passband". Up to 0.8·edge the windowed-sinc prototype must stay
        // within ±0.05 dB of 0 dB — no passband droop or ripple that would
        // colour the audible band.
        for &(s, d) in &RATE_PAIRS {
            let r = Resample::new(s, d).unwrap();
            let edge = r.passband_edge_hz();
            for frac in [0.0, 0.1, 0.25, 0.5, 0.7, 0.8] {
                let g = r.prototype_response_db(edge * frac);
                assert!(
                    g.abs() < 0.05,
                    "{}->{}: passband gain at {:.0} Hz ({:.2}·edge) = {} dB (expected ≈ 0)",
                    s,
                    d,
                    edge * frac,
                    frac,
                    g
                );
            }
        }
    }

    #[test]
    fn prototype_is_minus6db_at_the_band_edge() {
        // With the cutoff at the half-band frequency (= edge = min-rate/2),
        // the windowed-sinc transition is symmetric about the cutoff, so
        // |H| there is exactly half the passband amplitude → −6.02 dB. This
        // is the canonical −6 dB crossover point of a half-band linear-
        // phase low-pass; assert it within ±0.1 dB for every rate pair.
        for &(s, d) in &RATE_PAIRS {
            let r = Resample::new(s, d).unwrap();
            let g = r.prototype_response_db(r.passband_edge_hz());
            assert!(
                (g + 6.0206).abs() < 0.1,
                "{}->{}: band-edge gain = {} dB (expected ≈ -6.02)",
                s,
                d,
                g
            );
        }
    }

    #[test]
    fn prototype_stopband_rejects_aliasing_images() {
        // The Kaiser β = 8 design targets ≈ 80 dB stop-band rejection. By
        // 1.2·edge the prototype must already be ≥ 70 dB down — this is the
        // rejection that suppresses the spectral images (interpolation) and
        // the would-be alias band (decimation) the theory doc requires the
        // low-pass to remove. We probe a sweep of stop-band points up to
        // the input Nyquist and demand monotone-deep attenuation.
        for &(s, d) in &RATE_PAIRS {
            let r = Resample::new(s, d).unwrap();
            let edge = r.passband_edge_hz();
            let l_nyq = r.up_factor() as f64 * s as f64 / 2.0;
            for frac in [1.2, 1.5, 2.0, 3.0] {
                let f = edge * frac;
                if f >= l_nyq {
                    continue;
                }
                let g = r.prototype_response_db(f);
                assert!(
                    g < -70.0,
                    "{}->{}: stop-band gain at {:.0} Hz ({:.1}·edge) = {} dB (expected < -70)",
                    s,
                    d,
                    f,
                    frac,
                    g
                );
            }
        }
    }

    #[test]
    fn prototype_response_is_monotone_through_transition() {
        // A windowed-sinc low-pass falls monotonically from DC through the
        // transition band down to its first stop-band null; beyond that
        // null the stop-band ripple lobes rise and fall (a sharp design
        // *should* exhibit that structure), so we assert monotonicity only
        // over `[0, 1.05·edge]` — the passband + transition — where a
        // clean roll-off with no resonant passband bump is the design
        // requirement. Tolerance covers f32 tap quantisation.
        let r = Resample::new(48_000, 24_000).unwrap();
        let edge = r.passband_edge_hz();
        let mut prev = r.prototype_response_db(0.0);
        for i in 1..=120 {
            let f = edge * 1.05 * (i as f64 / 120.0);
            let g = r.prototype_response_db(f);
            assert!(
                g <= prev + 1.0e-3,
                "response rose in the transition band: {} dB → {} dB at {:.0} Hz",
                prev,
                g,
                f
            );
            prev = g;
        }
    }

    #[test]
    fn downsample_rejects_tone_above_new_nyquist() {
        // End-to-end anti-aliasing through the convolution: a 9 kHz sine
        // sits *above* the new Nyquist (8 kHz) when 48 kHz is decimated to
        // 16 kHz. Without band-limiting it would fold back to 16 − 9 =
        // 7 kHz and pollute the output; the prototype low-pass must reject
        // it. We measure the residual energy in the decimated stream and
        // require it ≥ 40 dB below the energy of an in-band 1 kHz tone put
        // through the same path.
        let n = 48_000;
        let fs_in = 48_000u32;
        let fs_out = 16_000u32;

        let measure = |freq: f32| -> f64 {
            let frame = sine_f32(freq, fs_in, n);
            let mut r = Resample::new(fs_in, fs_out).unwrap();
            let mut outs = r.process(&frame, f32_mono(fs_in)).unwrap();
            outs.extend(r.flush(f32_mono(fs_out)).unwrap());
            let mut out: Vec<f32> = Vec::new();
            for f in &outs {
                out.extend(read_f32(f));
            }
            // RMS over the settled middle, skipping filter start-up.
            let lo = 2_000.min(out.len());
            let hi = out.len().saturating_sub(2_000);
            if hi <= lo {
                return 0.0;
            }
            let sum: f64 = out[lo..hi].iter().map(|x| (*x as f64).powi(2)).sum();
            (sum / (hi - lo) as f64).sqrt()
        };

        let in_band = measure(1_000.0); // passes
        let alias = measure(9_000.0); // must be rejected (would fold to 7 kHz)
        let rej_db = 20.0 * (alias / in_band.max(1.0e-12)).log10();
        assert!(
            rej_db < -40.0,
            "above-Nyquist tone leaked into decimated output: {} dB relative to in-band (expected < -40)",
            rej_db
        );
    }

    #[test]
    fn integer_upsample_preserves_dc() {
        // Bandlimited interpolation passes exactly through the existing
        // samples (sinc zero-crossings at every nonzero integer); a
        // constant (DC) input is the simplest witness — every output
        // sample of a 1→3 interpolation of a constant must equal that
        // constant once the line is primed, since the prototype has unity
        // DC gain and the polyphase rows sum to 1 per phase.
        let n = 4_000;
        let level = 0.5f32;
        let mut bytes = Vec::with_capacity(n * 4);
        for _ in 0..n {
            bytes.extend_from_slice(&level.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut r = Resample::new(16_000, 48_000).unwrap();
        let outs = r.process(&frame, f32_mono(16_000)).unwrap();
        let mut out: Vec<f32> = Vec::new();
        for f in &outs {
            out.extend(read_f32(f));
        }
        // Skip the start-up transient; the settled output must sit at the
        // DC level.
        let tail = &out[out.len().saturating_sub(2_000)..];
        let max_dev = tail
            .iter()
            .map(|x| (*x - level).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_dev < 1.0e-3,
            "interpolated DC drifted: max deviation {} from {}",
            max_dev,
            level
        );
    }

    /// End-to-end spectral-purity contract: a pure tone in is a pure
    /// tone out — same frequency in Hz, same amplitude, and nothing
    /// else. The residual after subtracting the best least-squares
    /// quadrature fit `a·sin(ωt) + b·cos(ωt)` at the target frequency
    /// captures everything that is NOT that tone (aliasing images,
    /// interpolation distortion, phase jitter across polyphase
    /// branches) in one number.
    ///
    /// The LS fit is evaluated at the exact target frequency, so —
    /// unlike a DFT bin — it has no spectral-leakage penalty for a
    /// non-integer number of cycles per window.
    #[test]
    fn pure_tone_survives_conversion_spectrally_clean() {
        for &(src, dst) in &[
            (48_000u32, 44_100u32), // non-trivial rational (147/160)
            (48_000, 96_000),       // integer up
            (44_100, 48_000),       // non-trivial up
            (48_000, 32_000),       // rational down
        ] {
            let freq = 1_000.0f32;
            let n_in = src as usize; // 1 s
            let mut rs = Resample::new(src, dst).unwrap();
            let mut out: Vec<f32> = Vec::new();
            let produced = rs
                .process(&sine_f32(freq, src, n_in), f32_mono(src))
                .unwrap();
            for fr in &produced {
                for c in fr.data[0].chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            // Skip both edges: kernel warm-up at the head, un-flushed
            // ramp at the tail.
            let skip = (dst as usize) / 10;
            let y = &out[skip..out.len() - skip];
            let w = 2.0 * std::f64::consts::PI * freq as f64 / dst as f64;

            // Least-squares quadrature fit at the mapped frequency.
            let (mut ss, mut sc, mut scc, mut sss, mut ssc) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for (i, &v) in y.iter().enumerate() {
                let (s, c) = ((i as f64 * w).sin(), (i as f64 * w).cos());
                ss += v as f64 * s;
                sc += v as f64 * c;
                sss += s * s;
                scc += c * c;
                ssc += s * c;
            }
            // Solve the 2×2 normal equations.
            let det = sss * scc - ssc * ssc;
            let a = (ss * scc - sc * ssc) / det;
            let b = (sc * sss - ss * ssc) / det;
            let amplitude = (a * a + b * b).sqrt();

            let mut resid_sq = 0.0f64;
            let mut tot_sq = 0.0f64;
            for (i, &v) in y.iter().enumerate() {
                let fit = a * (i as f64 * w).sin() + b * (i as f64 * w).cos();
                resid_sq += (v as f64 - fit).powi(2);
                tot_sq += (v as f64).powi(2);
            }
            let purity_db = 10.0 * (resid_sq / tot_sq).max(1.0e-30).log10();

            assert!(
                (amplitude - 1.0).abs() < 0.005,
                "{src}->{dst}: tone amplitude {amplitude:.5} drifted from 1.0"
            );
            assert!(
                purity_db < -55.0,
                "{src}->{dst}: non-tone residual only {purity_db:.1} dB down"
            );
        }
    }
}
