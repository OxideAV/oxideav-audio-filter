//! Latency-reporting contract: `AudioFilter::latency_samples` must
//! match the group delay actually measured from the filter's impulse
//! (or step) response.
//!
//! A declared latency that drifts from the real one silently breaks a
//! host's delay compensation, so every non-zero reporter is pinned
//! against a measurement here, and the zero-latency default is spot-
//! checked on filters whose direct path really is instantaneous.

use oxideav_audio_filter::*;
use oxideav_core::{AudioFrame, SampleFormat};

const FS: u32 = 48_000;

fn params() -> AudioStreamParams {
    AudioStreamParams {
        format: SampleFormat::F32,
        channels: 1,
        sample_rate: FS,
    }
}

fn frame(s: &[f32]) -> AudioFrame {
    let mut b = Vec::with_capacity(s.len() * 4);
    for v in s {
        b.extend_from_slice(&v.to_le_bytes());
    }
    AudioFrame {
        samples: s.len() as u32,
        pts: None,
        data: vec![b],
    }
}

fn run(f: &mut dyn AudioFilter, x: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut frames = f.process(&frame(x), params()).expect("process");
    frames.extend(f.flush(params()).expect("flush"));
    for fr in &frames {
        for c in fr.data[0].chunks_exact(4) {
            out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
    }
    out
}

/// Declared latency via the trait (avoids inherent-method shadowing —
/// `Limiter` also has an inherent `latency_samples()`).
fn declared(f: &dyn AudioFilter) -> usize {
    f.latency_samples(params())
}

/// Index of the sample with the largest magnitude.
fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Look-ahead limiter: a below-ceiling impulse passes unattenuated,
/// delayed by exactly the configured look-ahead.
#[test]
fn limiter_latency_equals_lookahead() {
    for look_ahead in [0usize, 1, 32, 480] {
        let mut lim = Limiter::new(0.0, 50.0, look_ahead);
        let declared = declared(&lim);
        assert_eq!(declared, look_ahead);
        let mut x = vec![0.0f32; 4096];
        x[0] = 0.5; // −6 dBFS, well under the 0 dBFS ceiling
        let out = run(&mut lim, &x);
        assert_eq!(
            argmax(&out),
            declared,
            "limiter(look_ahead={look_ahead}): measured delay != declared"
        );
        assert!((out[declared] - 0.5).abs() < 1.0e-6, "impulse attenuated");
    }
}

/// Frequency shifter at Δf = 0: the in-phase path is a pure delay to
/// the Hilbert FIR's centre tap.
#[test]
fn freq_shifter_latency_equals_half_taps() {
    for half_taps in [15usize, 63, 255] {
        let mut fsh = FreqShifter::with(0.0, half_taps);
        let declared = declared(&fsh);
        assert_eq!(declared, half_taps);
        let mut x = vec![0.0f32; 2048];
        x[0] = 0.5;
        let out = run(&mut fsh, &x);
        assert_eq!(
            argmax(&out),
            declared,
            "freq_shifter(half_taps={half_taps}): measured delay != declared"
        );
    }
}

/// Median filter: a step edge crosses the output `window / 2` samples
/// after the input edge (an isolated impulse is REJECTED by a median,
/// so latency is measured with a step).
#[test]
fn median_filter_latency_equals_half_window() {
    for window in [3usize, 9, 31, 257] {
        let mut mf = MedianFilter::new(window);
        let declared = declared(&mf);
        assert_eq!(declared, window / 2);
        let edge = 300usize;
        let n = edge + window + 64;
        let step: Vec<f32> = (0..n).map(|i| if i >= edge { 1.0 } else { 0.0 }).collect();
        let out = run(&mut mf, &step);
        let onset = out
            .iter()
            .position(|v| *v > 0.5)
            .expect("step never crossed");
        assert_eq!(
            onset - edge,
            declared,
            "median(window={window}): measured step delay != declared"
        );
    }
}

/// Resampler: the symmetric polyphase kernel delays the signal by its
/// group delay. Measured at the OUTPUT rate, converted back to input
/// samples; ±1 input sample of tolerance covers the fractional part
/// (the true delay is generally not an integer number of input
/// samples) plus output-grid rounding.
#[test]
fn resample_latency_matches_kernel_group_delay() {
    for (dst, name) in [
        (96_000u32, "48k->96k"),
        (44_100, "48k->44.1k"),
        (32_000, "48k->32k"),
    ] {
        let mut rs = Resample::new(FS, dst).expect("resample");
        let declared = declared(&rs) as f64;
        let mut x = vec![0.0f32; 4096];
        x[0] = 0.5;
        let out = run(&mut rs, &x);
        let measured_in = argmax(&out) as f64 * FS as f64 / dst as f64;
        assert!(
            (measured_in - declared).abs() <= 1.0,
            "{name}: measured {measured_in:.2} input-samples vs declared {declared}"
        );
    }
}

/// Spot-check the zero-latency default on filters whose direct path is
/// genuinely instantaneous.
#[test]
fn zero_latency_filters_measure_zero() {
    let mut x = vec![0.0f32; 1024];
    x[0] = 0.5;

    let checks: Vec<(&str, Box<dyn AudioFilter>)> = vec![
        ("volume", Box::new(Volume::new(0.8))),
        ("hard_clipper", Box::new(HardClipper::new(1.0, 1.0))),
        ("true_peak_detector", Box::new(TruePeakDetector::new())),
        ("tape_saturation", Box::new(TapeSaturation::new(1.0, 0.0))),
    ];
    for (name, mut f) in checks {
        assert_eq!(f.latency_samples(params()), 0, "{name} declares non-zero");
        let out = run(&mut *f, &x);
        assert_eq!(argmax(&out), 0, "{name}: direct path is not instantaneous");
    }
}
