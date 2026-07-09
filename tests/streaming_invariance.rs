//! Chunk-size independence contract for every stateful filter.
//!
//! A streaming filter's output must depend only on the sample sequence it
//! has consumed, never on how that sequence was sliced into frames.
//! Feeding the same programme in 1-sample frames, odd-sized frames, or one
//! giant frame must produce bit-identical output (after concatenating the
//! emitted frames and draining `flush`).
//!
//! Violations are real bugs: an LFO that advances per *frame* instead of
//! per *sample*, an envelope that resets at frame boundaries, a block
//! accumulator keyed to the input frame length, a delay line primed from
//! the first frame's size, …
//!
//! The harness runs every public filter (each with parameters chosen to
//! exercise its state: feedback on, LFO on, look-ahead on, …) over a
//! deterministic composite programme — tone mix + impulse + loud burst
//! into near-silence — and compares the f32 output streams bit-for-bit.

use oxideav_audio_filter::*;
use oxideav_core::{AudioFrame, ChannelLayout, SampleFormat};

const FS: u32 = 48_000;
const N: usize = 4_800; // 100 ms

fn params(channels: u16) -> AudioStreamParams {
    AudioStreamParams {
        format: SampleFormat::F32,
        channels,
        sample_rate: FS,
    }
}

/// Deterministic composite programme, `channels`-interleaved f32.
///
/// - two-tone mix (440 Hz + 2.5 kHz) so filters with frequency-dependent
///   state see spectral content,
/// - an isolated impulse at n = 100 (transient detectors, medians, peaks),
/// - loud first half, near-silent second half (gates / compressors /
///   normalisers traverse both sides of their thresholds mid-stream),
/// - small DC bias (DC blockers / offset meters have work to do),
/// - channel 1 (if present) phase-shifted and scaled so stereo filters
///   see genuinely different channels.
fn programme(channels: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(N * channels);
    for n in 0..N {
        let t = n as f32 / FS as f32;
        let level = if n < N / 2 { 0.9 } else { 0.02 };
        for ch in 0..channels {
            let phase = ch as f32 * 0.7;
            let mut s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t + phase).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 2_500.0 * t + 0.3 + phase).sin();
            s *= level * (1.0 - 0.15 * ch as f32);
            if n == 100 {
                s = 0.95; // isolated impulse on every channel
            }
            s += 0.01; // DC bias
            out.push(s.clamp(-1.0, 1.0));
        }
    }
    out
}

fn frame_from(interleaved: &[f32], channels: usize) -> AudioFrame {
    let samples = interleaved.len() / channels;
    let mut bytes = Vec::with_capacity(interleaved.len() * 4);
    for s in interleaved {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    AudioFrame {
        samples: samples as u32,
        pts: None,
        data: vec![bytes],
    }
}

/// Run `filter` over the programme in frames of `chunk` samples
/// (`chunk == 0` means one single frame), returning every emitted output
/// byte (process outputs then flush outputs, concatenated in order).
fn run(
    filter: &mut dyn AudioFilter,
    interleaved: &[f32],
    channels: usize,
    chunk: usize,
) -> Vec<u8> {
    let p = params(channels as u16);
    let mut out = Vec::new();
    let step = if chunk == 0 { N } else { chunk };
    for slice in interleaved.chunks(step * channels) {
        let frame = frame_from(slice, channels);
        for f in filter.process(&frame, p).expect("process") {
            for plane in &f.data {
                out.extend_from_slice(plane);
            }
        }
    }
    for f in filter.flush(p).expect("flush") {
        for plane in &f.data {
            out.extend_from_slice(plane);
        }
    }
    out
}

/// Assert chunk-size independence for one filter configuration.
fn assert_invariant(name: &str, channels: usize, mut make: impl FnMut() -> Box<dyn AudioFilter>) {
    let programme = programme(channels);
    let reference = run(&mut *make(), &programme, channels, 0);
    for &chunk in &[1usize, 17, 480] {
        let got = run(&mut *make(), &programme, channels, chunk);
        assert_eq!(
            got.len(),
            reference.len(),
            "{name}: chunk={chunk} emitted {} bytes vs one-shot {}",
            got.len(),
            reference.len()
        );
        if got != reference {
            // Find the first differing f32 for a readable failure.
            let (mut idx, mut a, mut b) = (0usize, 0.0f32, 0.0f32);
            for (i, (ga, gb)) in got
                .chunks_exact(4)
                .zip(reference.chunks_exact(4))
                .enumerate()
            {
                if ga != gb {
                    idx = i;
                    a = f32::from_le_bytes([ga[0], ga[1], ga[2], ga[3]]);
                    b = f32::from_le_bytes([gb[0], gb[1], gb[2], gb[3]]);
                    break;
                }
            }
            panic!("{name}: chunk={chunk} diverges at f32 #{idx}: chunked={a} one-shot={b}");
        }
    }
}

macro_rules! invariant {
    ($test:ident, $name:literal, $channels:expr, $make:expr) => {
        #[test]
        fn $test() {
            assert_invariant($name, $channels, || Box::new($make));
        }
    };
}

// --- gain / distortion / waveshaping ---------------------------------
invariant!(volume, "volume", 2, Volume::new(0.8));
invariant!(hard_clipper, "hard_clipper", 1, HardClipper::new(2.0, 0.8));
invariant!(
    tape_saturation,
    "tape_saturation",
    1,
    TapeSaturation::new(2.0, 0.2)
);
invariant!(exciter, "exciter", 1, Exciter::with(3_000.0, 2.0, 0.5));
invariant!(
    octave_doubler,
    "octave_doubler",
    1,
    OctaveDoubler::with(0.7, 0.7, true)
);
invariant!(bitcrusher, "bitcrusher", 1, Bitcrusher::new(8, 4));
invariant!(
    dither16,
    "dither",
    1,
    Dither::with_seed(16, DitherMode::Tpdf, NoiseShaping::SecondOrder, 7)
);
invariant!(slew_limiter, "slew_limiter", 1, SlewLimiter::new(2_000.0));

// --- dynamics ----------------------------------------------------------
invariant!(
    compressor_peak,
    "compressor(peak,feedforward)",
    2,
    Compressor::new(-20.0, 4.0, 5.0, 50.0, 6.0, 3.0)
);
invariant!(
    compressor_rms_feedback,
    "compressor(rms,feedback)",
    2,
    Compressor::with_topology(
        -20.0,
        4.0,
        5.0,
        50.0,
        6.0,
        3.0,
        EnvelopeMode::Rms,
        DetectorTopology::Feedback,
    )
);
invariant!(
    parallel_compressor,
    "parallel_compressor",
    2,
    ParallelCompressor::new(-30.0, 8.0, 1.0, 100.0, 6.0)
);
invariant!(
    multiband_compressor,
    "multiband_compressor",
    2,
    MultibandCompressor::new()
);
invariant!(
    limiter,
    "limiter(look-ahead)",
    2,
    Limiter::new(-3.0, 50.0, 32)
);
invariant!(
    expander,
    "expander",
    1,
    Expander::new(-30.0, 2.0, 5.0, 50.0, 6.0, 0.0)
);
invariant!(
    upward_compressor,
    "upward_compressor",
    1,
    UpwardCompressor::new(-30.0, 2.0, 5.0, 50.0, 6.0, 24.0)
);
invariant!(
    upward_expander,
    "upward_expander",
    1,
    UpwardExpander::new(-20.0, 2.0, 5.0, 50.0, 6.0, 12.0)
);
invariant!(
    noise_gate,
    "noise_gate",
    1,
    NoiseGate::new(-30.0, 5.0, 50.0, 10.0)
);
invariant!(
    noise_gate_hysteresis,
    "noise_gate(hysteresis+knee)",
    1,
    NoiseGate::with(-28.0, -34.0, 5.0, 50.0, 10.0, 6.0)
);
invariant!(
    adaptive_noise_gate,
    "adaptive_noise_gate",
    1,
    AdaptiveNoiseGate::new()
);
invariant!(ducker, "ducker", 2, Ducker::new());
invariant!(gain_normalizer, "gain_normalizer", 1, GainNormalizer::new());
invariant!(
    transient_designer,
    "transient_designer",
    1,
    TransientDesigner::new()
);
invariant!(de_esser, "de_esser", 1, DeEsser::new());

// --- EQ / spectral -----------------------------------------------------
invariant!(
    biquad_lpf,
    "biquad(lpf)",
    2,
    Biquad::new(BiquadKind::LowPass {
        cutoff_hz: 1_000.0,
        q: 0.707,
    })
);
invariant!(
    biquad_peaking,
    "biquad(peaking)",
    1,
    Biquad::new(BiquadKind::Peaking {
        center_hz: 1_000.0,
        q: 2.0,
        gain_db: 6.0,
    })
);
invariant!(
    svf,
    "svf(bandpass)",
    1,
    SvfFilter::new(SvfMode::BandPass, 1_000.0, 2.0)
);
invariant!(
    equalizer,
    "equalizer",
    2,
    Equalizer::new(FS)
        .with_low_pass(8_000.0, 0.707)
        .with_band_pass(1_000.0, 1.0)
);
invariant!(dc_blocker, "dc_blocker", 1, DcBlocker::new());
invariant!(hum_filter, "hum_filter", 1, HumFilter::new(50.0, 30.0, 4));
invariant!(
    crossover_lr4,
    "crossover(lr4)",
    2,
    Crossover::with_slope(1_000.0, 0.707, CrossoverSlope::LinkwitzRiley4)
);
invariant!(
    pre_emphasis,
    "pre_emphasis(fm50)",
    1,
    PreEmphasis::new(EmphasisCurve::Fm50us)
);
invariant!(
    de_emphasis,
    "de_emphasis(fm75)",
    1,
    DeEmphasis::new(EmphasisCurve::Fm75us)
);
invariant!(
    comb_feedback,
    "comb_filter(feedback,damped)",
    1,
    CombFilter::with_delay_ms(
        CombMode::Feedback {
            gain: 0.8,
            damping: 0.3,
        },
        5.0,
    )
);
invariant!(median_filter, "median_filter", 1, MedianFilter::new(5));
invariant!(
    freq_shifter,
    "freq_shifter",
    1,
    FreqShifter::with(100.0, 32)
);

// --- delay / modulation -------------------------------------------------
invariant!(echo, "echo", 2, Echo::new(25.0, 0.4, 0.5));
invariant!(reverb, "reverb", 2, Reverb::new(0.7, 0.4, 0.3, 0.7));
invariant!(chorus, "chorus", 2, Chorus::new(3, 20.0, 5.0, 0.8, 0.5));
invariant!(flanger, "flanger", 2, Flanger::new(0.5, 3.0, 0.5, 0.5));
invariant!(phaser, "phaser", 2, Phaser::new(4, 0.5, 800.0, 0.5, 0.5));
invariant!(vibrato, "vibrato", 1, Vibrato::new(5.0, 2.0));
invariant!(tremolo, "tremolo", 1, Tremolo::new(5.0, 0.8));
invariant!(auto_pan, "auto_pan", 2, AutoPan::new(1.0, 1.0));
invariant!(wah, "wah", 1, Wah::new());
invariant!(
    ring_modulator,
    "ring_modulator",
    1,
    RingModulator::new(440.0, 1.0)
);
invariant!(talkbox, "talkbox", 1, Talkbox::new());
invariant!(pitch_shift, "pitch_shift", 1, PitchShift::new(3.0));

// --- rate / format ------------------------------------------------------
invariant!(
    resample_down,
    "resample(48k->44.1k)",
    2,
    Resample::new(48_000, 44_100).expect("resample")
);
invariant!(
    resample_up,
    "resample(48k->96k)",
    1,
    Resample::new(48_000, 96_000).expect("resample")
);
invariant!(
    downmix,
    "downmix(stereo->mono)",
    2,
    DownmixFilter::new(
        ChannelLayout::Stereo,
        ChannelLayout::Mono,
        DownmixMode::Average
    )
    .expect("downmix")
);

// --- stereo field -------------------------------------------------------
invariant!(stereo_widener, "stereo_widener", 2, StereoWidener::new(1.5));
invariant!(stereo_imager, "stereo_imager", 2, StereoImager::new());
invariant!(mid_side_encode, "mid_side(encode)", 2, MidSide::encoder());

// --- generators ---------------------------------------------------------
invariant!(
    white_noise,
    "white_noise",
    1,
    WhiteNoise::with_seed(0.5, 42)
);
invariant!(pink_noise, "pink_noise", 1, PinkNoise::with_seed(0.5, 42));
invariant!(
    brown_noise,
    "brown_noise",
    1,
    BrownNoise::with_seed(0.5, 42)
);

// --- meters / observers (pass-through must still be pass-through) -------
invariant!(
    envelope_follower,
    "envelope_follower",
    1,
    EnvelopeFollower::new(5.0, 50.0)
);
invariant!(loudness, "loudness_itu", 2, LoudnessITU::new());
invariant!(
    true_peak_detector,
    "true_peak_detector",
    1,
    TruePeakDetector::new()
);
invariant!(
    silence_detector,
    "silence_detector",
    1,
    SilenceDetector::new(-50.0, 20.0)
);
invariant!(
    crest_factor_meter,
    "crest_factor_meter",
    1,
    CrestFactorMeter::new()
);
invariant!(dc_offset_meter, "dc_offset_meter", 1, DcOffsetMeter::new());
invariant!(
    stereo_correlation_meter,
    "stereo_correlation_meter",
    2,
    StereoCorrelationMeter::new()
);
invariant!(
    stereo_balance_meter,
    "stereo_balance_meter",
    2,
    StereoBalanceMeter::new()
);
invariant!(
    zero_crossing_rate,
    "zero_crossing_rate",
    1,
    ZeroCrossingRateMeter::new()
);

// --- image output -------------------------------------------------------
// Spectrogram has its own feed/finalize API (it is not an `AudioFilter`),
// so it gets a bespoke invariance check over the rendered RGB raster.
// Its default stream shape is F32 stereo, so feed 2-channel frames.
#[test]
fn spectrogram() {
    let opts = SpectrogramOptions {
        fft_size: 256,
        hop_size: 64,
        width: 64,
        height: 64,
        ..SpectrogramOptions::default()
    };
    let programme = programme(2);
    let render = |chunk: usize| {
        let mut sg = Spectrogram::new(opts.clone()).expect("spectrogram");
        let step = if chunk == 0 { N } else { chunk };
        for slice in programme.chunks(step * 2) {
            sg.feed(&frame_from(slice, 2)).expect("feed");
        }
        sg.finalize_rgb()
    };
    let reference = render(0);
    for &chunk in &[1usize, 17, 480] {
        assert_eq!(
            render(chunk),
            reference,
            "spectrogram raster differs at chunk={chunk}"
        );
    }
}
