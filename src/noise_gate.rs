//! Per-sample noise gate.
//!
//! A noise gate attenuates audio that falls below a threshold. The internal
//! envelope follower watches the absolute value of the signal and either
//! ramps the output gain toward `1.0` (when above threshold) or toward `0.0`
//! (when continuously below threshold for `hold` samples).
//!
//! For multi-channel input the channels share a single linked envelope: the
//! per-sample drive value is `max(|x_ch_0|, |x_ch_1|, ...)`. This keeps the
//! channels gated together so a stereo image does not collapse when only one
//! side dips below the threshold.
//!
//! # Hysteresis and soft-knee (r181)
//!
//! Two upgrades on the original binary single-threshold gate:
//!
//! 1. **Hysteresis** — separate `open_db` and `close_db` thresholds with
//!    `close_db ≤ open_db`. The gate opens only when the drive rises above
//!    `open_db`; once open it stays open until the drive falls below
//!    `close_db` (and the hold timer has elapsed). The default
//!    `hysteresis_db` is 6 dB. Setting `hysteresis_db = 0` reduces to the
//!    classic single-threshold behaviour.
//!
//! 2. **Soft-knee** — `knee_db` widens the on/off transition into a smooth
//!    region centred on the relevant threshold. Within the knee, the
//!    instantaneous target gain interpolates smoothly between fully open
//!    (`0 dB`) and fully closed (`gain_floor_db`, default `-∞`). This
//!    keeps attack/release ramping behaviour while removing the harsh
//!    on/off edge near the threshold. `knee_db = 0` is a hard knee.
//!
//! Both upgrades are opt-in via [`NoiseGate::with`]; the original
//! [`NoiseGate::new`] constructor stays binary single-threshold for
//! back-compat (callers that rely on the legacy behaviour see no change).
//!
//! # Parameters
//!
//! * `threshold_db` — gate opens when |signal| > 10^(threshold_db / 20). A
//!   typical value is `-40.0` dBFS.
//! * `attack_ms` — time over which the gain ramps from current to `1.0`
//!   when the signal exceeds the threshold.
//! * `release_ms` — time over which the gain ramps from current to `0.0`
//!   after the hold period elapses.
//! * `hold_ms` — how long the signal must remain below threshold before
//!   the release ramp begins.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Lowest non-zero linear amplitude we consider when converting drive to
/// dB. Anything below this clamps to `-∞ dB` for the knee computation.
const DRIVE_FLOOR_LIN: f32 = 1.0e-12;

#[derive(Debug, Clone)]
pub struct NoiseGate {
    open_db: f32,
    close_db: f32,
    knee_db: f32,
    attack_ms: f32,
    release_ms: f32,
    hold_ms: f32,
    // Cached state, updated lazily when the sample rate changes
    state: Option<GateState>,
}

#[derive(Debug, Clone)]
struct GateState {
    sample_rate: u32,
    open_db: f32,
    close_db: f32,
    knee_db: f32,
    attack_step: f32,
    release_step: f32,
    hold_samples: u32,
    /// Current ramped output gain (linear, `[0, 1]`).
    gain: f32,
    /// Hysteresis latch: once the drive has crossed `open_db + knee/2`,
    /// stay open until it falls below `close_db - knee/2` for
    /// `hold_samples`.
    is_open: bool,
    /// Number of samples since the drive last rose above `close_db`.
    /// While the gate is open, this counts time spent below the close
    /// threshold; once it exceeds `hold_samples` the gate latches
    /// closed.
    below_count: u32,
}

impl NoiseGate {
    /// Build a classic single-threshold noise gate (no hysteresis, hard
    /// knee). Kept for back-compat — existing call-sites are unchanged.
    pub fn new(threshold_db: f32, attack_ms: f32, release_ms: f32, hold_ms: f32) -> Self {
        Self {
            open_db: threshold_db,
            close_db: threshold_db,
            knee_db: 0.0,
            attack_ms,
            release_ms,
            hold_ms,
            state: None,
        }
    }

    /// Build a gate with explicit open / close thresholds (hysteresis) and
    /// optional soft-knee width.
    ///
    /// * `open_db` — drive level (dBFS) that opens the gate. Default in
    ///   `new` is `threshold_db`.
    /// * `close_db` — drive level (dBFS) that closes the gate after the
    ///   hold timer expires. Must satisfy `close_db ≤ open_db`; the
    ///   difference is the **hysteresis margin** (typically 3–12 dB).
    /// * `knee_db` — soft-knee width in dB, centred on the active
    ///   threshold. `0.0` is a hard knee identical to [`Self::new`].
    /// * `attack_ms`, `release_ms`, `hold_ms` — same as [`Self::new`].
    ///
    /// If `close_db > open_db` it is clamped down to `open_db` (zero
    /// hysteresis) rather than rejected — a wider open-than-close
    /// threshold has no meaningful interpretation.
    pub fn with(
        open_db: f32,
        close_db: f32,
        knee_db: f32,
        attack_ms: f32,
        release_ms: f32,
        hold_ms: f32,
    ) -> Self {
        let close_db = close_db.min(open_db);
        Self {
            open_db,
            close_db,
            knee_db: knee_db.max(0.0),
            attack_ms,
            release_ms,
            hold_ms,
            state: None,
        }
    }

    /// Reset the runtime gain envelope so the next `process` call starts
    /// from a closed gate. Sample-rate-derived coefficients are kept.
    pub fn reset(&mut self) {
        if let Some(st) = self.state.as_mut() {
            st.gain = 0.0;
            st.is_open = false;
            st.below_count = 0;
        }
    }

    /// Open / close thresholds in dBFS, in evaluation order
    /// `(open_db, close_db)`. Mostly useful for tests / inspection.
    pub fn thresholds_db(&self) -> (f32, f32) {
        (self.open_db, self.close_db)
    }

    /// Soft-knee width in dB.
    pub fn knee_db(&self) -> f32 {
        self.knee_db
    }

    fn ensure_state(&mut self, sample_rate: u32) {
        let needs_rebuild = match &self.state {
            Some(s) => {
                s.sample_rate != sample_rate
                    || s.open_db != self.open_db
                    || s.close_db != self.close_db
                    || s.knee_db != self.knee_db
            }
            None => true,
        };
        if needs_rebuild {
            // Preserve the running gain across param changes so live
            // knob-twisting doesn't click; only the very first build
            // starts from `gain = 0`.
            let prev_gain = self.state.as_ref().map(|s| s.gain).unwrap_or(0.0);
            let prev_open = self.state.as_ref().map(|s| s.is_open).unwrap_or(false);
            let prev_below = self.state.as_ref().map(|s| s.below_count).unwrap_or(0);
            let attack_samples = ((self.attack_ms / 1000.0) * sample_rate as f32).max(1.0);
            let release_samples = ((self.release_ms / 1000.0) * sample_rate as f32).max(1.0);
            let hold_samples = ((self.hold_ms / 1000.0) * sample_rate as f32).max(0.0) as u32;
            self.state = Some(GateState {
                sample_rate,
                open_db: self.open_db,
                close_db: self.close_db,
                knee_db: self.knee_db,
                attack_step: 1.0 / attack_samples,
                release_step: 1.0 / release_samples,
                hold_samples,
                gain: prev_gain,
                is_open: prev_open,
                below_count: prev_below,
            });
        }
    }
}

/// Convert a linear drive value to dBFS with a floor to keep `log10` finite.
#[inline]
fn drive_to_db(drive: f32) -> f32 {
    20.0 * drive.max(DRIVE_FLOOR_LIN).log10()
}

/// Soft-knee target gain (`0..=1`) centred on `threshold_db` of total
/// width `knee_db`. Returns `1.0` (fully open) above the knee, `0.0`
/// (fully closed) below it, and a smooth interpolation inside.
///
/// The interpolation is a Hermite smoothstep on the knee position, which
/// gives a C¹-continuous transition (no audible step in the gain
/// derivative). At `knee_db = 0` the function degenerates to a step
/// (hard knee).
#[inline]
fn knee_target_gain(drive_db: f32, threshold_db: f32, knee_db: f32) -> f32 {
    if knee_db <= 0.0 {
        return if drive_db >= threshold_db { 1.0 } else { 0.0 };
    }
    let half = knee_db * 0.5;
    let lo = threshold_db - half;
    let hi = threshold_db + half;
    if drive_db <= lo {
        0.0
    } else if drive_db >= hi {
        1.0
    } else {
        // Smoothstep: t ∈ [0,1] ↦ t² (3 - 2t).
        let t = (drive_db - lo) / knee_db;
        t * t * (3.0 - 2.0 * t)
    }
}

impl AudioFilter for NoiseGate {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        self.ensure_state(params.sample_rate);
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        let n_samples = channels.first().map(|c| c.len()).unwrap_or(0);
        let n_chan = channels.len();
        let state = self.state.as_mut().expect("ensure_state succeeded");

        // Convert thresholds to dB once per call.
        let open_db = state.open_db;
        let close_db = state.close_db;
        let knee_db = state.knee_db;
        let half_knee = knee_db * 0.5;
        let open_upper = open_db + half_knee;
        let close_lower = close_db - half_knee;

        for s in 0..n_samples {
            let mut drive = 0.0f32;
            for ch in channels.iter().take(n_chan) {
                let abs = ch[s].abs();
                if abs > drive {
                    drive = abs;
                }
            }
            let drive_db = drive_to_db(drive);

            // Hysteresis latch update. The latch flips on definitive
            // crossings of the *outer* edges of the knee; inside the knee
            // the previous latch state is sticky, so soft-knee interp
            // continues to use the active threshold.
            if !state.is_open {
                if drive_db >= open_upper {
                    state.is_open = true;
                    state.below_count = 0;
                }
            } else {
                // Already open. Track how long we've been below the
                // close threshold; only flip to closed once we've spent
                // `hold_samples` continuously below it.
                if drive_db <= close_lower {
                    state.below_count = state.below_count.saturating_add(1);
                    if state.below_count > state.hold_samples {
                        state.is_open = false;
                    }
                } else {
                    state.below_count = 0;
                }
            }

            // Target gain depends on which side of the hysteresis loop
            // we're on. While the gate is open the *close* threshold is
            // active (signal needs to drop into the close knee to start
            // attenuating); while closed the *open* threshold is active
            // (signal needs to climb into the open knee to start
            // un-attenuating). Hard knee (`knee_db = 0`) reduces this
            // to the original `target ∈ {0, 1}` step.
            let target = if state.is_open {
                knee_target_gain(drive_db, close_db, knee_db)
            } else {
                knee_target_gain(drive_db, open_db, knee_db)
            };

            // Per-sample linear ramp toward `target` at the
            // attack/release rate. Matches the legacy behaviour
            // (`attack_step` / `release_step` are 1/N-sample slopes), but
            // the destination is now `target` rather than `{0, 1}`.
            if target > state.gain {
                state.gain = (state.gain + state.attack_step).min(target);
            } else if target < state.gain {
                state.gain = (state.gain - state.release_step).max(target);
            }

            for ch in channels.iter_mut().take(n_chan) {
                ch[s] *= state.gain;
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

    const F32_MONO: AudioStreamParams = AudioStreamParams {
        format: SampleFormat::F32,
        channels: 1,
        sample_rate: 48_000,
    };

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

    #[test]
    fn quiet_signal_is_attenuated() {
        // -80 dBFS noise at -40 dBFS threshold → gate must close.
        let samples = vec![0.0001f32; 48_000];
        let frame = make_f32_mono(&samples);
        let mut g = NoiseGate::new(-40.0, 1.0, 1.0, 0.0);
        let out = g.process(&frame, F32_MONO).unwrap();
        let got = read_f32(&out[0]);
        let last = *got.last().unwrap();
        assert!(last.abs() < 1.0e-6, "expected gate closed, got {}", last);
    }

    #[test]
    fn loud_signal_passes_through() {
        // Half-scale tone, well above -40 dBFS
        let mut samples = Vec::with_capacity(2_000);
        for i in 0..2_000 {
            samples.push(0.5 * ((i as f32) * 0.1).sin());
        }
        let frame = make_f32_mono(&samples);
        let mut g = NoiseGate::new(-40.0, 1.0, 50.0, 5.0);
        let out = g.process(&frame, F32_MONO).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[got.len() - 100..];
        let peak = tail.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(peak > 0.4, "expected open gate, peak={}", peak);
    }

    // ---------- r181 additions: hysteresis + soft-knee ----------

    #[test]
    fn knee_target_step_at_zero_knee() {
        // Hard-knee → exact step function.
        assert_eq!(knee_target_gain(-39.0, -40.0, 0.0), 1.0);
        assert_eq!(knee_target_gain(-40.0, -40.0, 0.0), 1.0);
        assert_eq!(knee_target_gain(-40.01, -40.0, 0.0), 0.0);
        assert_eq!(knee_target_gain(-60.0, -40.0, 0.0), 0.0);
    }

    #[test]
    fn knee_target_smoothstep_in_band() {
        // 12 dB knee centred on -40 → fully open above -34, fully closed
        // below -46, smoothstep monotonic in between.
        let thr = -40.0;
        let kn = 12.0;
        assert_eq!(knee_target_gain(-34.0, thr, kn), 1.0);
        assert_eq!(knee_target_gain(-46.0, thr, kn), 0.0);
        // Midpoint of smoothstep at t=0.5 is exactly 0.5.
        let mid = knee_target_gain(thr, thr, kn);
        assert!(
            (mid - 0.5).abs() < 1.0e-6,
            "smoothstep midpoint = {} (expected 0.5)",
            mid
        );
        // Monotonic.
        let mut prev = 0.0f32;
        for i in 0..=60 {
            let d_db = -46.0 + (i as f32) * (12.0 / 60.0);
            let g = knee_target_gain(d_db, thr, kn);
            assert!(g + 1.0e-6 >= prev, "non-monotonic at {} dB: {}", d_db, g);
            prev = g;
        }
    }

    #[test]
    fn hysteresis_prevents_chatter() {
        // Signal dances right at the open threshold. Without hysteresis
        // a single-threshold gate chatters; with 6 dB hysteresis the
        // gate either stays closed (signal never reaches open_db) or
        // stays open (signal never falls below close_db). Drive level
        // here is -42 dBFS — between the legacy threshold (-40) and the
        // hysteresis open threshold (-36). Gate must stay CLOSED.
        let level = 10.0f32.powf(-42.0 / 20.0);
        let mut samples = Vec::with_capacity(48_000);
        for i in 0..48_000 {
            samples.push(level * ((i as f32) * 0.1).sin());
        }
        let frame = make_f32_mono(&samples);
        // Hysteresis: open at -36, close at -44, 0 knee. The -42 dBFS
        // tone sits between them, so a never-opened gate must remain
        // closed for the full duration.
        let mut g = NoiseGate::with(-36.0, -44.0, 0.0, 1.0, 1.0, 0.0);
        let out = g.process(&frame, F32_MONO).unwrap();
        let got = read_f32(&out[0]);
        let peak = got.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            peak < 1.0e-6,
            "hysteresis open threshold breached: peak={}",
            peak
        );
    }

    #[test]
    fn hysteresis_latch_holds_open_state_between_thresholds() {
        // Verify the LATCH itself stays open while the drive sits in
        // the hysteresis band — the gain envelope's exact value at
        // sine zero-crossings is a separate concern (it's released by
        // the envelope follower regardless of the latch). What
        // hysteresis guarantees is that the *latch* doesn't snap back
        // to closed and force the gate to re-attack from zero on the
        // next sine peak.
        //
        // First half: -20 dBFS tone opens the latch. Second half: -42
        // dBFS tone (above close, below open). With a long hold timer
        // the latch must remain open through the entire dip.
        let loud = 10.0f32.powf(-20.0 / 20.0);
        let quiet = 10.0f32.powf(-42.0 / 20.0);
        let mut samples = Vec::with_capacity(96_000);
        for i in 0..48_000 {
            samples.push(loud * ((i as f32) * 0.1).sin());
        }
        for i in 0..48_000 {
            samples.push(quiet * ((i as f32) * 0.1).sin());
        }
        let frame = make_f32_mono(&samples);
        // Hold 100 ms so a single sine zero-crossing run doesn't
        // exceed it. close_db − knee/2 = −44 dBFS so the quiet tone's
        // peak (−42 dBFS) never triggers below_count to increment from
        // its peak; below_count only counts up at the sine's
        // zero-crossing valleys, which is < 100 ms per cycle.
        let mut g = NoiseGate::with(-36.0, -44.0, 0.0, 1.0, 1.0, 100.0);
        let _ = g.process(&frame, F32_MONO).unwrap();
        // Inspect the latch after processing — it must STILL be open.
        let st = g.state.as_ref().unwrap();
        assert!(
            st.is_open,
            "hysteresis latch closed despite drive remaining above close_db"
        );
    }

    #[test]
    fn soft_knee_smooths_attenuation_at_threshold() {
        // 12 dB knee centred on -40. Sustained drive at -40 dBFS (the
        // exact knee centre) should settle to a gain of ~0.5 (smoothstep
        // midpoint), not 0 or 1. We use instant attack/release so the
        // envelope tracks the static target exactly.
        let level = 10.0f32.powf(-40.0 / 20.0);
        let mut samples = Vec::with_capacity(48_000);
        for i in 0..48_000 {
            samples.push(level * ((i as f32) * 0.1).sin());
        }
        let frame = make_f32_mono(&samples);
        // Open / close at -40 (no hysteresis), 12 dB knee.
        // attack=release=0 ms means slopes default to 1/sample (instant).
        let mut g = NoiseGate::with(-40.0, -40.0, 12.0, 0.0, 0.0, 0.0);
        let out = g.process(&frame, F32_MONO).unwrap();
        let got = read_f32(&out[0]);
        let tail = &got[got.len() - 4_000..];
        let peak_out = tail.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        // Roughly half the input peak (smoothstep at midpoint = 0.5).
        let ratio = peak_out / level;
        assert!(
            ratio > 0.3 && ratio < 0.7,
            "expected mid-knee gain ≈ 0.5, ratio = {}",
            ratio
        );
    }

    #[test]
    fn close_db_clamped_when_above_open_db() {
        // Invalid (close > open) collapses to zero hysteresis (the more
        // conservative interpretation).
        let g = NoiseGate::with(-40.0, -30.0, 0.0, 1.0, 1.0, 0.0);
        let (open_db, close_db) = g.thresholds_db();
        assert_eq!(open_db, -40.0);
        assert_eq!(close_db, -40.0);
    }

    #[test]
    fn new_preserves_legacy_step_behaviour() {
        // The legacy `new` constructor must remain a hard-knee single
        // threshold gate (knee_db = 0, open_db = close_db).
        let g = NoiseGate::new(-30.0, 1.0, 1.0, 0.0);
        let (open_db, close_db) = g.thresholds_db();
        assert_eq!(open_db, -30.0);
        assert_eq!(close_db, -30.0);
        assert_eq!(g.knee_db(), 0.0);
    }

    #[test]
    fn reset_returns_gate_to_closed() {
        // Run loud input to open the gate, then `reset` and verify the
        // next process starts from a closed state.
        let loud = 0.5f32;
        let mut samples = Vec::with_capacity(2_000);
        for i in 0..2_000 {
            samples.push(loud * ((i as f32) * 0.1).sin());
        }
        let frame = make_f32_mono(&samples);
        let mut g = NoiseGate::new(-40.0, 1.0, 50.0, 5.0);
        let _ = g.process(&frame, F32_MONO).unwrap();
        let st_before = g.state.as_ref().unwrap().gain;
        assert!(st_before > 0.0);
        g.reset();
        let st_after = g.state.as_ref().unwrap().gain;
        assert_eq!(st_after, 0.0);
        assert!(!g.state.as_ref().unwrap().is_open);
    }
}
