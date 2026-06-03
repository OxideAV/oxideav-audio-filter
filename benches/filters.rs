// Parallel-array index loops are idiomatic in DSP / bench code where
// each iteration touches the same index across several channel
// buffers — readability beats the iterator rewrite. Mirrors the
// sibling-codec benches in `oxideav-flac` and `oxideav-tta`.
#![allow(clippy::needless_range_loop)]

//! Criterion benchmarks for the audio-filter hot paths.
//!
//! Round 215 (depth-mode benchmarks, no behavioural change). This
//! crate is feature-complete (~50 filters as of round 209); per the
//! workspace "saturated → bench/fuzz/profile" doctrine, the round
//! adds an observational measurement harness so future rounds can
//! A/B their per-filter algorithm tweaks against a stable baseline.
//!
//! Every scenario:
//!
//!   * Synthesises its PCM input deterministically from an xorshift32
//!     seed — no committed fixture bytes, no audio file format
//!     parsing inside the bench.
//!   * Runs the filter through its **public API** (`AudioFilter::process`
//!     for the streaming filters; `process_in_place` for the
//!     `Biquad` / `Equalizer` direct paths). No private-field pokes.
//!   * Builds the input frame once **outside** the timed region so
//!     the measurement reflects the per-sample DSP cost, not
//!     `AudioFrame` construction or sample-format conversion churn
//!     specific to the bench.
//!   * Uses `Criterion::throughput(Throughput::Bytes(...))` so the
//!     report quotes per-byte timings, matching the convention used
//!     by the sibling-crate decoder benches (`oxideav-flac`,
//!     `oxideav-tta`).
//!
//! Filter coverage spans one representative per architectural family
//! the crate ships:
//!
//!   * **biquad_lpf** — single second-order IIR (DF-II-T, `f64`
//!     state). The building block under [`Equalizer`], the
//!     `Crossover` slopes, `HumFilter`, `DeEsser`, `Wah`,
//!     `StereoImager`, `MultibandCompressor`, …
//!   * **equalizer_3band** — three cascaded `Biquad` sections in
//!     series (low-shelf + peaking + high-shelf). Measures the
//!     cost of section-chaining without the `AudioFilter` decode
//!     /encode round-trip.
//!   * **loudness_itu** — BS.1770-4 K-weighting pipeline
//!     (high-shelf pre-filter + RLB high-pass + sum-of-squares
//!     accumulator + channel-weighted mean). Stereo path.
//!   * **compressor** — peak-detector compressor with soft-knee
//!     gain curve and one-pole attack/release follower. Exercises
//!     the dB-domain log/exp branch the linear filters never reach.
//!   * **reverb** — Schroeder algorithmic reverb (4 parallel combs
//!     ║ 2 serial all-passes). Delay-line heavy, dominates the
//!     stereo wet/dry mixer.
//!   * **resample_44k1_48k** — polyphase windowed-sinc rate
//!     conversion at the canonical 44.1 → 48 kHz CD-to-TV ratio
//!     (`L = 160`, `M = 147`). Single-channel; the per-sample cost
//!     scales linearly in channel count.
//!   * **true_peak_4x** — 4× polyphase Kaiser-FIR oversampling
//!     for inter-sample peak detection (dBTP). Stereo, single
//!     pass; the FIR work dominates over the `max |y|` tracking.
//!
//! Run with:
//!     CARGO_TARGET_DIR=/tmp/oxideav-audio-filter-bench \
//!         cargo bench -p oxideav-audio-filter --bench filters
//!
//! Add `-- --baseline <name>` to compare against a saved baseline.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use oxideav_audio_filter::{
    biquad::{Biquad, BiquadKind},
    AudioFilter, AudioStreamParams, Compressor, Equalizer, LoudnessITU, Resample, Reverb,
    TruePeakDetector,
};
use oxideav_core::{AudioFrame, SampleFormat};

/// Deterministic 32-bit xorshift PRNG. Same recurrence the sibling
/// codec benches (`oxideav-flac`, `oxideav-tta`) already use; small
/// enough to be obvious, "random enough" for a DSP exercise.
fn xorshift32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Build per-channel `f32` PCM in `[-0.5, +0.5)` containing a slow
/// triangular ramp plus broadband xorshift noise. The triangular
/// component keeps low-Q filters from short-circuiting to DC; the
/// noise component keeps the dynamics-section filters' detectors
/// from settling onto a single value.
fn build_f32_pcm(n_samples: usize, channels: u16) -> Vec<Vec<f32>> {
    let nch = channels as usize;
    let mut out: Vec<Vec<f32>> = (0..nch).map(|_| Vec::with_capacity(n_samples)).collect();
    let mut state: u32 = 0xCAFE_F00D;
    for s in 0..n_samples {
        // Slow triangle: period 1024 samples, amplitude 0.25.
        let phase = ((s as i32) % 1024) - 512;
        let env = (phase as f32) * (0.25 / 512.0);
        for ch in out.iter_mut().take(nch) {
            // Map the upper 24 bits of the PRNG into [-0.25, +0.25).
            let raw = (xorshift32(&mut state) >> 8) as i32 - (1 << 23);
            let noise = (raw as f32) * (0.25 / (1u32 << 23) as f32);
            ch.push(env + noise);
        }
    }
    out
}

/// Pack per-channel `f32` PCM into an `AudioFrame` with interleaved
/// little-endian `f32` samples. Matches `SampleFormat::F32`.
fn build_audio_frame_f32(pcm_per_channel: &[Vec<f32>]) -> AudioFrame {
    let nch = pcm_per_channel.len();
    let n = pcm_per_channel[0].len();
    let mut interleaved: Vec<u8> = Vec::with_capacity(n * nch * 4);
    for i in 0..n {
        for c in 0..nch {
            interleaved.extend_from_slice(&pcm_per_channel[c][i].to_le_bytes());
        }
    }
    AudioFrame {
        samples: n as u32,
        pts: Some(0),
        data: vec![interleaved],
    }
}

/// Interleave per-channel `f32` PCM into a flat `Vec<f32>` for the
/// `process_in_place` direct paths on `Biquad` / `Equalizer`. The
/// in-place call is given the channel count, so the layout is
/// `[ch0[0], ch1[0], ch0[1], ch1[1], …]`.
fn interleave_f32(pcm_per_channel: &[Vec<f32>]) -> Vec<f32> {
    let nch = pcm_per_channel.len();
    let n = pcm_per_channel[0].len();
    let mut out = Vec::with_capacity(n * nch);
    for i in 0..n {
        for c in 0..nch {
            out.push(pcm_per_channel[c][i]);
        }
    }
    out
}

// --- Per-filter scenarios --------------------------------------------

fn bench_biquad_lpf(c: &mut Criterion) {
    // 1 s of stereo F32 PCM @ 48 kHz, second-order Butterworth-style
    // LPF at 4 kHz. Measures the DF-II-T per-sample recurrence with
    // both channels (independent state).
    let fs = 48_000u32;
    let n = fs as usize;
    let pcm = build_f32_pcm(n, 2);
    let interleaved = interleave_f32(&pcm);

    let mut g = c.benchmark_group("biquad_lpf");
    g.throughput(Throughput::Bytes((n * 2 * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("stereo/f32/48k/1s"), |b| {
        b.iter_batched(
            || interleaved.clone(),
            |mut buf| {
                let mut bq = Biquad::new(BiquadKind::LowPass {
                    cutoff_hz: 4_000.0,
                    q: 0.707,
                });
                bq.process_in_place(&mut buf, 2, fs);
                criterion::black_box(buf);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_equalizer_3band(c: &mut Criterion) {
    // 1 s of stereo F32 PCM @ 48 kHz through a 3-band parametric EQ:
    // low-shelf (boost at 120 Hz), peaking (cut at 1 kHz), high-shelf
    // (boost at 8 kHz). Three cascaded biquads + the section-chain
    // dispatch cost.
    let fs = 48_000u32;
    let n = fs as usize;
    let pcm = build_f32_pcm(n, 2);
    let interleaved = interleave_f32(&pcm);

    let mut g = c.benchmark_group("equalizer_3band");
    g.throughput(Throughput::Bytes((n * 2 * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("stereo/f32/48k/1s"), |b| {
        b.iter_batched(
            || interleaved.clone(),
            |mut buf| {
                let mut eq = Equalizer::new(fs)
                    .with_low_shelf(120.0, 0.707, 3.0)
                    .with_peaking(1_000.0, 1.0, -4.0)
                    .with_high_shelf(8_000.0, 0.707, 2.0);
                eq.process_in_place(&mut buf, 2, fs);
                criterion::black_box(buf);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    g.finish();
}

fn bench_loudness_itu(c: &mut Criterion) {
    // BS.1770-4 K-weighting + sum-of-squares accumulator over 1 s of
    // stereo F32 PCM @ 48 kHz. Includes the pre-filter (high-shelf),
    // RLB (HPF), per-channel mean-square sum, and the final
    // `integrated_lufs` read on the running accumulator.
    let fs = 48_000u32;
    let n = fs as usize;
    let pcm = build_f32_pcm(n, 2);
    let frame = build_audio_frame_f32(&pcm);
    let params = AudioStreamParams {
        format: SampleFormat::F32,
        channels: 2,
        sample_rate: fs,
    };

    let mut g = c.benchmark_group("loudness_itu");
    g.throughput(Throughput::Bytes((n * 2 * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("stereo/f32/48k/1s"), |b| {
        b.iter(|| {
            let mut meter = LoudnessITU::new();
            let _ = meter.process(criterion::black_box(&frame), params);
            criterion::black_box(meter.integrated_lufs());
        });
    });
    g.finish();
}

fn bench_compressor(c: &mut Criterion) {
    // Stereo F32 compressor over 1 s @ 48 kHz: -18 dBFS threshold,
    // 4:1 ratio, 5 ms attack / 80 ms release, 6 dB soft knee, +3 dB
    // make-up. Exercises the dB-domain knee + one-pole follower
    // hot path.
    let fs = 48_000u32;
    let n = fs as usize;
    let pcm = build_f32_pcm(n, 2);
    let frame = build_audio_frame_f32(&pcm);
    let params = AudioStreamParams {
        format: SampleFormat::F32,
        channels: 2,
        sample_rate: fs,
    };

    let mut g = c.benchmark_group("compressor");
    g.throughput(Throughput::Bytes((n * 2 * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("stereo/f32/48k/1s"), |b| {
        b.iter(|| {
            let mut comp = Compressor::new(-18.0, 4.0, 5.0, 80.0, 6.0, 3.0);
            let out = comp.process(criterion::black_box(&frame), params).unwrap();
            criterion::black_box(out);
        });
    });
    g.finish();
}

fn bench_reverb(c: &mut Criterion) {
    // Schroeder reverb over 1 s of stereo F32 PCM @ 48 kHz. The
    // four parallel combs + two serial all-passes dominate the
    // per-sample cost; the dry/wet sum is a couple of `f32` muls.
    let fs = 48_000u32;
    let n = fs as usize;
    let pcm = build_f32_pcm(n, 2);
    let frame = build_audio_frame_f32(&pcm);
    let params = AudioStreamParams {
        format: SampleFormat::F32,
        channels: 2,
        sample_rate: fs,
    };

    let mut g = c.benchmark_group("reverb");
    g.throughput(Throughput::Bytes((n * 2 * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("stereo/f32/48k/1s"), |b| {
        b.iter(|| {
            let mut rv = Reverb::new(0.7, 0.4, 0.3, 0.7);
            let out = rv.process(criterion::black_box(&frame), params).unwrap();
            criterion::black_box(out);
        });
    });
    g.finish();
}

fn bench_resample_44k1_48k(c: &mut Criterion) {
    // Polyphase windowed-sinc resampling, 44.1 -> 48 kHz, mono F32,
    // 1 s of input. The LCM at this ratio is 7_056_000, giving
    // `L = 160`, `M = 147`; per-output sample cost is one polyphase
    // sub-filter convolution.
    let src = 44_100u32;
    let dst = 48_000u32;
    let n = src as usize;
    let pcm = build_f32_pcm(n, 1);
    let frame = build_audio_frame_f32(&pcm);
    let params = AudioStreamParams {
        format: SampleFormat::F32,
        channels: 1,
        sample_rate: src,
    };

    let mut g = c.benchmark_group("resample_44k1_48k");
    g.throughput(Throughput::Bytes((n * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("mono/f32/44k1->48k/1s"), |b| {
        b.iter(|| {
            let mut rs = Resample::new(src, dst).expect("resample build");
            let out = rs.process(criterion::black_box(&frame), params).unwrap();
            criterion::black_box(out);
        });
    });
    g.finish();
}

fn bench_true_peak_4x(c: &mut Criterion) {
    // 4x polyphase Kaiser-FIR inter-sample peak detection over 1 s
    // of stereo F32 PCM @ 48 kHz. Per sample: four polyphase
    // sub-filters of 12 taps each (48-tap default FIR), `f64`
    // accumulation, `max |y|` tracking. Pass-through audio path
    // makes the timing reflect just the FIR overhead.
    let fs = 48_000u32;
    let n = fs as usize;
    let pcm = build_f32_pcm(n, 2);
    let frame = build_audio_frame_f32(&pcm);
    let params = AudioStreamParams {
        format: SampleFormat::F32,
        channels: 2,
        sample_rate: fs,
    };

    let mut g = c.benchmark_group("true_peak_4x");
    g.throughput(Throughput::Bytes((n * 2 * 4) as u64));
    g.bench_function(BenchmarkId::from_parameter("stereo/f32/48k/1s"), |b| {
        b.iter(|| {
            let mut tp = TruePeakDetector::new();
            let _ = tp.process(criterion::black_box(&frame), params);
            criterion::black_box(tp.max_dbtp());
        });
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_biquad_lpf,
    bench_equalizer_3band,
    bench_loudness_itu,
    bench_compressor,
    bench_reverb,
    bench_resample_44k1_48k,
    bench_true_peak_4x,
);
criterion_main!(benches);
