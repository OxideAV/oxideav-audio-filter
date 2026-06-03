//! Sliding-window median filter — non-linear impulse-noise restoration.
//!
//! Each output sample is the **median** of the most-recent `window`
//! input samples on the same channel. The filter is non-linear: unlike
//! every IIR / FIR filter in this crate, it does not satisfy
//! superposition. Its defining property is that it suppresses isolated
//! impulse-noise samples (clicks, single-sample glitches, "salt and
//! pepper" transients) while preserving step edges that span more
//! than `window / 2` samples — a behaviour no linear low-pass filter
//! achieves. Within the restoration family it complements
//! [`HumFilter`](crate::HumFilter) (cyclic-mains denoising) and
//! [`DcBlocker`](crate::DcBlocker) (DC drift removal): the median
//! filter targets *transient* impulse noise instead.
//!
//! # Algorithm
//!
//! Per channel:
//!
//! ```text
//! ring[k mod window] := x[n]                    // store newest input
//! sorted := sort(ring)                          // ascending copy
//! y[n] := sorted[window / 2]                    // odd-window mid sample
//!         (sorted[w/2 - 1] + sorted[w/2]) / 2   // even-window mean of two centres
//! ```
//!
//! The sort is an insertion sort over the window's contents. For the
//! window sizes typical of click removal (`window ∈ [3, 31]`) this is
//! cache-friendly and beats heap-/quickselect-based selection on real
//! per-sample dispatch. The cost is `O(window²)` per sample in the
//! worst case but `O(window)` in the common case where the ring is
//! already nearly sorted (steady-state signal).
//!
//! # Warm-up
//!
//! Before `window` samples have been seen the ring is partially
//! populated. The default policy is **zero-fill**: the unfilled slots
//! contribute `0.0` to the sort. Callers that need a non-zero
//! warm-up convention should pre-feed a quiet leader frame and call
//! [`MedianFilter::reset`] before the real input.
//!
//! # Window size
//!
//! - `window = 1` is the identity (output equals input). Allowed for
//!   parameter-sweep convenience; the filter still pays its ring
//!   storage cost.
//! - Odd window sizes give a clean single-sample median. The mid index
//!   is `window / 2` (integer division, so e.g. for `window = 5` the
//!   third sorted sample).
//! - Even window sizes return the mean of the two middle sorted
//!   samples (e.g. `window = 4` averages indices `1` and `2`). This
//!   adds a tiny amount of low-pass smoothing on top of the median
//!   selection.
//!
//! `window` is clamped to `[1, MAX_WINDOW]` (`MAX_WINDOW = 257`).
//! Beyond ~31 samples a median filter starts visibly attenuating
//! signal transients; the explicit upper bound just defends against
//! pathological per-sample allocations.
//!
//! # Why a separate filter (vs. setting `BiquadKind::LowPass`?)
//!
//! Linear LPFs cannot remove isolated impulses without also softening
//! the surrounding signal; the spectral content of a single-sample
//! click is broadband, so any linear filter that kills the click also
//! kills equivalent broadband signal energy. The median filter is the
//! textbook non-linear answer to this: it discards the outlier sample
//! without touching the rest of the window's content. This is why the
//! filter lives outside the [`Biquad`](crate::Biquad) family even
//! though it occupies the same "click cleanup" niche callers reach for
//! a steep low-pass for.
//!
//! # Per-channel state
//!
//! Each channel keeps its own ring buffer and write index, so
//! channels do not cross-talk. [`MedianFilter::reset`] zeros every
//! channel's ring without changing the configured window size.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Result};

/// Upper bound on the configurable window length.
pub const MAX_WINDOW: usize = 257;

/// Per-channel ring buffer + write index.
#[derive(Debug, Clone)]
struct ChState {
    ring: Vec<f32>,
    /// Next slot to overwrite. After `n` samples,
    /// `write_idx = n % window` and `ring` holds the most-recent
    /// `min(n, window)` samples (older slots are still zero-filled
    /// during warm-up).
    write_idx: usize,
}

impl ChState {
    fn new(window: usize) -> Self {
        Self {
            ring: vec![0.0; window],
            write_idx: 0,
        }
    }
    fn reset(&mut self) {
        for v in self.ring.iter_mut() {
            *v = 0.0;
        }
        self.write_idx = 0;
    }
}

/// Sliding-window median filter.
///
/// See the [module docs](self) for the algorithm and the rationale
/// for picking a non-linear filter over a steep low-pass for
/// impulse-noise removal.
#[derive(Debug, Clone)]
pub struct MedianFilter {
    window: usize,
    state: Vec<ChState>,
    /// Scratch buffer reused across every sample of every `process`
    /// call to avoid per-sample allocation. Sized to `window` at
    /// construction.
    scratch: Vec<f32>,
}

impl MedianFilter {
    /// Build a median filter with a sliding window of `window`
    /// samples. The window is clamped to `[1, MAX_WINDOW]`.
    pub fn new(window: usize) -> Self {
        let w = window.clamp(1, MAX_WINDOW);
        Self {
            window: w,
            state: Vec::new(),
            scratch: vec![0.0; w],
        }
    }

    /// Currently-configured window length (after clamping).
    pub fn window(&self) -> usize {
        self.window
    }

    /// Reset every channel's ring buffer to zero. The configured
    /// `window` length is preserved.
    pub fn reset(&mut self) {
        for st in self.state.iter_mut() {
            st.reset();
        }
    }

    fn ensure_state(&mut self, channels: usize) {
        if self.state.len() != channels {
            self.state = (0..channels).map(|_| ChState::new(self.window)).collect();
        }
    }
}

impl Default for MedianFilter {
    /// 5-sample window — the canonical click-removal default. Wide
    /// enough to mask a couple of adjacent impulse samples, narrow
    /// enough to preserve transients of musical interest.
    fn default() -> Self {
        Self::new(5)
    }
}

/// Sort `scratch[..w]` in place by insertion-sort and return the
/// median value. For odd `w` this is `scratch[w / 2]` after sorting.
/// For even `w` it is the mean of the two middle entries.
fn median_of(scratch: &mut [f32], w: usize) -> f32 {
    // Insertion sort: in the steady-state case the ring is *almost*
    // already sorted (one element changed at the write index), which
    // is the best case for insertion sort — `O(window)` rather than
    // the worst-case `O(window²)`. For the small window sizes this
    // filter is configured with this beats quickselect-style
    // partial-sorting on real per-sample dispatch.
    for i in 1..w {
        let key = scratch[i];
        let mut j = i;
        while j > 0 && scratch[j - 1] > key {
            scratch[j] = scratch[j - 1];
            j -= 1;
        }
        scratch[j] = key;
    }
    if w == 0 {
        return 0.0;
    }
    if w % 2 == 1 {
        scratch[w / 2]
    } else {
        // Mean of the two middle entries. `f64` intermediate to
        // avoid the small ULP error a `(a + b) * 0.5` `f32` chain
        // accumulates on near-equal `a`, `b`.
        let mid = w / 2;
        ((scratch[mid - 1] as f64 + scratch[mid] as f64) * 0.5) as f32
    }
}

impl AudioFilter for MedianFilter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        let mut channels = decode_to_f32(input, params.format, params.channels)?;
        self.ensure_state(channels.len());

        let w = self.window;
        if self.scratch.len() != w {
            self.scratch.resize(w, 0.0);
        }
        for (ch_idx, buf) in channels.iter_mut().enumerate() {
            let st = &mut self.state[ch_idx];
            for s in buf.iter_mut() {
                // Overwrite the oldest ring slot with the newest input.
                st.ring[st.write_idx] = *s;
                st.write_idx = (st.write_idx + 1) % w;
                // Copy ring into scratch, sort, pick median.
                self.scratch.copy_from_slice(&st.ring);
                *s = median_of(&mut self.scratch, w);
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

    #[test]
    fn window_of_one_is_identity() {
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let frame = make_f32_mono(&input);
        let mut m = MedianFilter::new(1);
        let out = m.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        for (a, b) in input.iter().zip(got.iter()) {
            assert!(
                (a - b).abs() < 1.0e-7,
                "window=1 should be identity: in={} out={}",
                a,
                b
            );
        }
    }

    #[test]
    fn window_clamps_to_max() {
        let m = MedianFilter::new(10_000);
        assert_eq!(m.window(), MAX_WINDOW);
    }

    #[test]
    fn window_clamps_to_one_floor() {
        let m = MedianFilter::new(0);
        assert_eq!(m.window(), 1);
    }

    #[test]
    fn default_window_is_five() {
        assert_eq!(MedianFilter::default().window(), 5);
    }

    #[test]
    fn isolated_impulse_is_suppressed() {
        // Constant 0.2 signal with one isolated +0.9 impulse at index
        // 10. A 5-sample median must reject the impulse entirely.
        let mut input = vec![0.2f32; 64];
        input[10] = 0.9;
        let frame = make_f32_mono(&input);
        let mut m = MedianFilter::new(5);
        let out = m.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);

        // After warm-up (≥ 4 samples), output should match the
        // baseline 0.2 even at and around the impulse location.
        let warm = 5;
        for (i, &v) in got.iter().enumerate().skip(warm).take(input.len() - warm) {
            assert!(
                (v - 0.2).abs() < 1.0e-5,
                "median did not suppress impulse at i={}: got={}",
                i,
                v
            );
        }
    }

    #[test]
    fn two_adjacent_impulses_pass_through_3_tap_but_not_5_tap() {
        // Two adjacent +0.9 impulses on a 0.2 baseline. A 3-sample
        // median window CAN be fooled (median of {0.2, 0.9, 0.9} =
        // 0.9), but a 5-sample window cannot (median of any 5-sample
        // window containing two impulses on a 0.2 baseline = 0.2).
        let mut input = vec![0.2f32; 64];
        input[20] = 0.9;
        input[21] = 0.9;

        // 3-tap median: at i = 21 the window is {0.9, 0.9, 0.2 or
        // 0.2 — depending on warm-up} which yields 0.9.
        let mut m3 = MedianFilter::new(3);
        let out3 = m3
            .process(&make_f32_mono(&input), f32_mono(48_000))
            .unwrap();
        let got3 = read_f32(&out3[0]);
        let peaked3 = got3.iter().skip(15).take(15).any(|&v| v > 0.5);
        assert!(
            peaked3,
            "3-tap median should let two adjacent impulses through"
        );

        // 5-tap median: the window is dominated by 0.2 (3 of 5).
        let mut m5 = MedianFilter::new(5);
        let out5 = m5
            .process(&make_f32_mono(&input), f32_mono(48_000))
            .unwrap();
        let got5 = read_f32(&out5[0]);
        for (i, &v) in got5.iter().enumerate().skip(15).take(15) {
            assert!(
                (v - 0.2).abs() < 1.0e-5,
                "5-tap median did not suppress 2-impulse burst at i={}: got={}",
                i,
                v
            );
        }
    }

    #[test]
    fn step_edge_is_preserved() {
        // Step from -0.3 to +0.4 at i = 30. After at least
        // (window / 2 + 1) samples on the new side, output must
        // match +0.4 exactly. Median preserves step edges that span
        // more than half the window — unlike a linear LPF, which
        // ringing-smooths them.
        let mut input = vec![-0.3f32; 64];
        for v in &mut input[30..] {
            *v = 0.4;
        }
        let frame = make_f32_mono(&input);
        let mut m = MedianFilter::new(5);
        let out = m.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);

        // From i = 32 (step + 2 = past the median centre) onward the
        // output must equal +0.4.
        for (i, &v) in got.iter().enumerate().skip(32) {
            assert!(
                (v - 0.4).abs() < 1.0e-5,
                "step edge not preserved at i={}: got={}",
                i,
                v
            );
        }
    }

    #[test]
    fn even_window_averages_two_centres() {
        // 4-sample window. After 4 samples of {0.1, 0.2, 0.3, 0.4}
        // the sorted window is the same and median = (0.2+0.3)/2 = 0.25.
        let input = vec![0.1f32, 0.2, 0.3, 0.4];
        let frame = make_f32_mono(&input);
        let mut m = MedianFilter::new(4);
        let out = m.process(&frame, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        // At i = 3 the window is exactly {0.1, 0.2, 0.3, 0.4}.
        assert!(
            (got[3] - 0.25).abs() < 1.0e-6,
            "even-window mean wrong: got {}",
            got[3]
        );
    }

    #[test]
    fn channels_do_not_cross_talk() {
        // Stereo: L = constant 0.5, R = silence with one impulse at
        // index 10. The L channel must remain 0.5 throughout, and
        // the R channel's impulse must be suppressed independently —
        // neither channel's state may leak into the other.
        let n = 32usize;
        let mut bytes = Vec::with_capacity(n * 2 * 4);
        for i in 0..n {
            let l = 0.5f32;
            let r = if i == 10 { 0.9f32 } else { 0.0f32 };
            bytes.extend_from_slice(&l.to_le_bytes());
            bytes.extend_from_slice(&r.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        };
        let mut m = MedianFilter::new(5);
        let out = m
            .process(
                &frame,
                AudioStreamParams {
                    format: SampleFormat::F32,
                    channels: 2,
                    sample_rate: 48_000,
                },
            )
            .unwrap();
        let bytes = &out[0].data[0];
        // Skip warm-up: after 5 samples both channels should be at
        // their respective steady-state values.
        for i in 5..n {
            let lo = i * 2 * 4;
            let l = f32::from_le_bytes([bytes[lo], bytes[lo + 1], bytes[lo + 2], bytes[lo + 3]]);
            let ro = lo + 4;
            let r = f32::from_le_bytes([bytes[ro], bytes[ro + 1], bytes[ro + 2], bytes[ro + 3]]);
            assert!(
                (l - 0.5).abs() < 1.0e-5,
                "L channel polluted at i={}: got {}",
                i,
                l
            );
            assert!(
                r.abs() < 1.0e-5,
                "R channel impulse leaked at i={}: got {}",
                i,
                r
            );
        }
    }

    #[test]
    fn reset_clears_ring_buffer() {
        // Run a non-trivial signal through, then reset, then verify
        // the next sample sees the warm-up (zero-filled) ring rather
        // than the previous run's history.
        let mut m = MedianFilter::new(5);
        let frame1 = make_f32_mono(&[0.7f32; 16]);
        let _ = m.process(&frame1, f32_mono(48_000)).unwrap();
        m.reset();
        // After reset, a single non-zero sample with a 5-wide window
        // is in a ring of {x, 0, 0, 0, 0}; median = 0.
        let frame2 = make_f32_mono(&[0.9f32]);
        let out = m.process(&frame2, f32_mono(48_000)).unwrap();
        let got = read_f32(&out[0]);
        assert!(
            got[0].abs() < 1.0e-7,
            "reset failed: got {} (expected 0 from {{0.9, 0, 0, 0, 0}})",
            got[0]
        );
    }

    #[test]
    fn streaming_continuity_matches_single_shot() {
        // Same input processed (a) in one frame and (b) split across
        // two frames must give identical output: the ring state
        // persists across `process` calls.
        let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.21).sin() * 0.4).collect();
        let mut m_single = MedianFilter::new(5);
        let single = m_single
            .process(&make_f32_mono(&input), f32_mono(48_000))
            .unwrap();
        let single_out = read_f32(&single[0]);

        let mut m_split = MedianFilter::new(5);
        let part1 = m_split
            .process(&make_f32_mono(&input[..23]), f32_mono(48_000))
            .unwrap();
        let part2 = m_split
            .process(&make_f32_mono(&input[23..]), f32_mono(48_000))
            .unwrap();
        let mut split_out = read_f32(&part1[0]);
        split_out.extend(read_f32(&part2[0]));

        assert_eq!(single_out.len(), split_out.len());
        for (i, (a, b)) in single_out.iter().zip(split_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0e-6,
                "streaming mismatch at i={}: single={} split={}",
                i,
                a,
                b
            );
        }
    }

    #[test]
    fn median_of_helper_odd_window() {
        let mut s = [0.5f32, 0.1, 0.9, 0.3, 0.7];
        assert!((median_of(&mut s, 5) - 0.5).abs() < 1.0e-7);
    }

    #[test]
    fn median_of_helper_even_window() {
        let mut s = [0.1f32, 0.2, 0.3, 0.4];
        // Mean of indices 1 and 2 = (0.2 + 0.3) / 2 = 0.25.
        assert!((median_of(&mut s, 4) - 0.25).abs() < 1.0e-7);
    }

    #[test]
    fn median_of_already_sorted_is_cheap() {
        // Sorted input is the insertion-sort best case; the helper
        // must still produce the right median. Defensive coverage
        // (the perf claim is in the module docs).
        let mut s = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        assert!((median_of(&mut s, 5) - 0.0).abs() < 1.0e-7);
    }

    #[test]
    fn s16_format_roundtrips_through_filter() {
        // The filter's promise is sample-format-agnostic: an i16
        // input frame must come out as an i16 output frame with the
        // median computed in f32 internally and re-quantised on output.
        let input_i16: Vec<i16> = (0..16).map(|i| (i * 1000 - 8000) as i16).collect();
        let mut bytes = Vec::with_capacity(input_i16.len() * 2);
        for s in &input_i16 {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: input_i16.len() as u32,
            pts: None,
            data: vec![bytes],
        };
        let params = AudioStreamParams {
            format: SampleFormat::S16,
            channels: 1,
            sample_rate: 48_000,
        };
        let mut m = MedianFilter::new(3);
        let out = m.process(&frame, params).unwrap();
        // Output bytes should be a clean i16 stream of the same length.
        assert_eq!(out[0].data[0].len(), input_i16.len() * 2);
        let got_i16: Vec<i16> = out[0].data[0]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got_i16.len(), input_i16.len());
        // Smoke: monotone ramp through a 3-tap median is still
        // monotone (median tracks the middle of the local window).
        for w in got_i16.windows(2).skip(2) {
            assert!(w[1] >= w[0], "monotone ramp violated: {} -> {}", w[0], w[1]);
        }
    }
}
