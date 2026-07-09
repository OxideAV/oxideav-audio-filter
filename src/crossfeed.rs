//! Crossfeed — headphone spatialisation via opposite-channel bleed.
//!
//! On loudspeakers each ear hears BOTH channels: the far speaker's
//! signal arrives at the opposite ear slightly later (the interaural
//! time difference, ~250–350 µs for a typical ±30° stereo triangle),
//! quieter, and with its high frequencies shadowed by the head.
//! Headphones deliver each channel to exactly one ear, which renders
//! hard-panned material unnaturally wide and fatiguing. A crossfeed
//! filter restores the acoustic summing:
//!
//! ```text
//! out_L[n] = d · L[n] + g · LP( R[n − D] )
//! out_R[n] = d · R[n] + g · LP( L[n − D] )
//! ```
//!
//! where
//!
//! * `g = 10^(level_db / 20)` — the crossfeed level (how loud the
//!   opposite channel bleeds through), default −6 dB;
//! * `d = 1 − g/(1 + g)` and the effective cross gain is `g/(1 + g)`,
//!   so `d + cross = 1`: **mono content passes at unity** at low
//!   frequencies instead of gaining `+level_db` of build-up;
//! * `LP` — one-pole low-pass at `cutoff_hz` (default 700 Hz)
//!   modelling the head shadow (high frequencies diffract around the
//!   head far less than they pass through/around it);
//! * `D = delay_us · fs / 10⁶` — the interaural delay (default
//!   300 µs), realised with the crate's fractional delay line so
//!   non-integer sample offsets are exact at any rate.
//!
//! Channels 0/1 are treated as L/R; mono or >2-channel input passes
//! through unchanged (a crossfeed between surround channels is not
//! meaningful).
//!
//! Distinct from [`StereoWidener`](crate::StereoWidener) /
//! [`StereoImager`](crate::StereoImager) (which *increase* or shape
//! width via M/S scaling) — crossfeed *narrows* the image the way a
//! pair of loudspeakers in a room does, direction-preserving and
//! delay-aware.

use crate::frac_delay::{FracDelayLine, Interp};
use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Streaming headphone crossfeed.
#[derive(Debug)]
pub struct Crossfeed {
    level_db: f32,
    cutoff_hz: f32,
    delay_us: f32,
    state: Option<State>,
}

#[derive(Debug)]
struct State {
    sample_rate: u32,
    /// Two-channel fractional delay line (L, R histories).
    line: FracDelayLine,
    /// Delay in samples derived from `delay_us` at `sample_rate`.
    delay_samples: f32,
    /// One-pole low-pass coefficient `a = 1 − e^(−2π·fc/fs)`.
    lp_a: f32,
    /// Head-shadow LPF state for the L→R and R→L bleed paths.
    lp_state: [f32; 2],
}

impl Crossfeed {
    /// Natural preset: −6 dB bleed, 700 Hz head-shadow cutoff, 300 µs
    /// interaural delay.
    pub fn new() -> Self {
        Self::with(-6.0, 700.0, 300.0)
    }

    /// Custom-parameter constructor.
    ///
    /// * `level_db` — crossfeed level, clamped to `[-30, 0]` dB
    ///   (NaN → −6). `0` is full bleed (mono-fold at low frequency),
    ///   `-30` is barely audible.
    /// * `cutoff_hz` — head-shadow low-pass cutoff, clamped to
    ///   `[100, 2000]` Hz (NaN → 700).
    /// * `delay_us` — interaural delay, clamped to `[0, 1000]` µs
    ///   (NaN → 300).
    pub fn with(level_db: f32, cutoff_hz: f32, delay_us: f32) -> Self {
        Self {
            level_db: crate::clamp_param(level_db, -6.0, -30.0, 0.0),
            cutoff_hz: crate::clamp_param(cutoff_hz, 700.0, 100.0, 2_000.0),
            delay_us: crate::clamp_param(delay_us, 300.0, 0.0, 1_000.0),
            state: None,
        }
    }

    /// Crossfeed level in dB.
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// Head-shadow low-pass cutoff in Hz.
    pub fn cutoff_hz(&self) -> f32 {
        self.cutoff_hz
    }

    /// Interaural delay in microseconds.
    pub fn delay_us(&self) -> f32 {
        self.delay_us
    }

    /// Reset delay-line and low-pass state.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// `(direct, cross)` gain pair. `direct + cross = 1` so mono
    /// content passes at unity at low frequencies.
    fn gains(&self) -> (f32, f32) {
        let g = 10.0f32.powf(self.level_db / 20.0);
        let cross = g / (1.0 + g);
        (1.0 - cross, cross)
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs = match &self.state {
            Some(s) => s.sample_rate != sample_rate,
            None => true,
        };
        if needs {
            let fs = sample_rate.max(1) as f32;
            let delay_samples = self.delay_us * 1.0e-6 * fs;
            // Linear interpolation reaches 1 sample past the integer
            // delay; +4 gives comfortable headroom.
            let capacity = delay_samples.ceil() as usize + 4;
            let lp_a = 1.0 - (-2.0 * std::f32::consts::PI * self.cutoff_hz / fs).exp();
            self.state = Some(State {
                sample_rate,
                line: FracDelayLine::new(2, capacity, Interp::Linear),
                delay_samples,
                lp_a,
                lp_state: [0.0; 2],
            });
        }
    }
}

impl Default for Crossfeed {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioFilter for Crossfeed {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        // Crossfeed is only meaningful between an L/R pair.
        if channels.len() != 2 {
            let out = encode_from_f32(params.format, params.channels, input, &channels)?;
            return Ok(vec![out]);
        }
        self.ensure_state(params.sample_rate);
        let (direct, cross) = self.gains();
        let st = self.state.as_mut().expect("state ensured above");
        let n = channels[0].len().min(channels[1].len());

        let (left, right) = channels.split_at_mut(1);
        for (ls, rs) in left[0].iter_mut().zip(right[0].iter_mut()).take(n) {
            let (l, r) = (*ls, *rs);
            st.line.push(&[l, r]);
            let dl = st.line.read(0, st.delay_samples);
            let dr = st.line.read(1, st.delay_samples);
            // Head-shadow one-pole LP on each bleed path. Flushed to
            // zero so a decaying tail terminates instead of dwelling
            // subnormal (crate::ftz).
            st.lp_state[0] = crate::ftz(st.lp_state[0] + st.lp_a * (dl - st.lp_state[0]));
            st.lp_state[1] = crate::ftz(st.lp_state[1] + st.lp_a * (dr - st.lp_state[1]));
            *ls = direct * l + cross * st.lp_state[1];
            *rs = direct * r + cross * st.lp_state[0];
        }

        let out = encode_from_f32(params.format, params.channels, input, &channels)?;
        Ok(vec![out])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::SampleFormat;

    const FS: u32 = 48_000;

    fn stereo_params() -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32,
            channels: 2,
            sample_rate: FS,
        }
    }

    fn stereo_frame(l: &[f32], r: &[f32]) -> AudioFrame {
        assert_eq!(l.len(), r.len());
        let mut bytes = Vec::with_capacity(l.len() * 8);
        for i in 0..l.len() {
            bytes.extend_from_slice(&l[i].to_le_bytes());
            bytes.extend_from_slice(&r[i].to_le_bytes());
        }
        AudioFrame {
            samples: l.len() as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    fn split(frame: &AudioFrame) -> (Vec<f32>, Vec<f32>) {
        let all: Vec<f32> = frame.data[0]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let l = all.iter().step_by(2).copied().collect();
        let r = all.iter().skip(1).step_by(2).copied().collect();
        (l, r)
    }

    fn rms(x: &[f32]) -> f32 {
        let s: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
        (s / x.len().max(1) as f64).sqrt() as f32
    }

    /// A hard-left impulse must appear in the right channel delayed by
    /// the interaural delay, attenuated by the cross gain, and
    /// low-pass smeared (peak below the raw cross gain).
    #[test]
    fn hard_left_impulse_bleeds_delayed_and_shadowed() {
        let n = 512;
        let mut l = vec![0.0f32; n];
        l[0] = 1.0;
        let r = vec![0.0f32; n];
        let mut cf = Crossfeed::new();
        let out = cf.process(&stereo_frame(&l, &r), stereo_params()).unwrap();
        let (ol, or) = split(&out[0]);

        // Direct path: L keeps its impulse at n=0 scaled by `direct`.
        let g = 10.0f32.powf(-6.0 / 20.0);
        let cross = g / (1.0 + g);
        let direct = 1.0 - cross;
        assert!(
            (ol[0] - direct).abs() < 1.0e-6,
            "direct gain wrong: {} vs {}",
            ol[0],
            direct
        );

        // Bleed: nothing in R before the interaural delay…
        let d = (300.0e-6 * FS as f32) as usize; // ≈ 14 samples
        for (i, v) in or.iter().enumerate().take(d - 2) {
            assert!(
                v.abs() < 1.0e-4,
                "right channel rings before the ITD at {i}: {v}"
            );
        }
        // …then energy appears, capped below the raw cross gain
        // because the head-shadow LPF smears the impulse.
        let peak = or.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.01, "no crossfeed bleed at all");
        assert!(
            peak < cross,
            "bleed peak {peak} not smeared below cross gain {cross}"
        );
        // Total bleed energy integrates to ~cross at DC; check the sum
        // is in the right ballpark (LP passes DC exactly).
        let sum: f32 = or.iter().sum();
        assert!(
            (sum - cross).abs() < 0.05 * cross,
            "bleed DC sum {sum} vs cross gain {cross}"
        );
    }

    /// Mono material (L == R) must pass at unity level at low
    /// frequencies — the `direct + cross = 1` compensation.
    #[test]
    fn mono_low_frequency_is_unity() {
        let n = FS as usize; // 1 s
        let tone: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 100.0 * i as f32 / FS as f32).sin())
            .collect();
        let mut cf = Crossfeed::new();
        let out = cf
            .process(&stereo_frame(&tone, &tone), stereo_params())
            .unwrap();
        let (ol, or) = split(&out[0]);
        // Skip the first 50 ms (delay-line fill + LP settle).
        let warm = FS as usize / 20;
        let in_rms = rms(&tone[warm..]);
        let l_rms = rms(&ol[warm..]);
        let r_rms = rms(&or[warm..]);
        let l_db = 20.0 * (l_rms / in_rms).log10();
        let r_db = 20.0 * (r_rms / in_rms).log10();
        assert!(
            l_db.abs() < 0.35 && r_db.abs() < 0.35,
            "mono 100 Hz level shifted: L {l_db:+.2} dB, R {r_db:+.2} dB"
        );
        // And both channels remain identical (symmetry).
        for i in warm..n {
            assert!(
                (ol[i] - or[i]).abs() < 1.0e-6,
                "mono symmetry broken at {i}"
            );
        }
    }

    /// Crossfeed narrows the image: the L/R correlation of a
    /// hard-panned (fully uncorrelated) programme must increase.
    #[test]
    fn correlation_increases_on_hard_panned_material() {
        let n = FS as usize / 2;
        // L = 440 Hz, R = 3.1 kHz: essentially orthogonal signals.
        let l: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / FS as f32).sin())
            .collect();
        let r: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 3_100.0 * i as f32 / FS as f32).sin())
            .collect();
        let corr = |a: &[f32], b: &[f32]| -> f64 {
            let n = a.len() as f64;
            let (mut sa, mut sb, mut saa, mut sbb, mut sab) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
            for i in 0..a.len() {
                let (x, y) = (a[i] as f64, b[i] as f64);
                sa += x;
                sb += y;
                saa += x * x;
                sbb += y * y;
                sab += x * y;
            }
            let cov = sab - sa * sb / n;
            let var_a = saa - sa * sa / n;
            let var_b = sbb - sb * sb / n;
            cov / (var_a * var_b).sqrt().max(1.0e-30)
        };
        let before = corr(&l, &r).abs();
        // Strong crossfeed for a clear measurement.
        let mut cf = Crossfeed::with(0.0, 2_000.0, 300.0);
        let out = cf.process(&stereo_frame(&l, &r), stereo_params()).unwrap();
        let (ol, or) = split(&out[0]);
        let warm = FS as usize / 20;
        let after = corr(&ol[warm..], &or[warm..]).abs();
        assert!(
            after > before + 0.1,
            "crossfeed did not narrow the image: |corr| {before:.3} -> {after:.3}"
        );
    }

    /// Mono (1-channel) input passes through bit-exact.
    #[test]
    fn mono_input_passes_through() {
        let n = 256;
        let tone: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() * 0.4).collect();
        let mut bytes = Vec::with_capacity(n * 4);
        for s in &tone {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let p = AudioStreamParams {
            format: SampleFormat::F32,
            channels: 1,
            sample_rate: FS,
        };
        let mut cf = Crossfeed::new();
        let out = cf.process(&frame, p).unwrap();
        assert_eq!(out[0].data[0], frame.data[0]);
    }
}
