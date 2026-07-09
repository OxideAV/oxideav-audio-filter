//! Parameter-edge hardening contract.
//!
//! Every filter constructor must accept arbitrary garbage — NaN,
//! ±infinity, huge magnitudes, zero, negative values — without
//! panicking, without attempting absurd allocations, and without
//! poisoning the audio path: processing a finite input signal through a
//! filter built from hostile parameters must yield entirely finite
//! output (constructors returning `Result` may instead reject the
//! parameters with a typed error).
//!
//! `f32::clamp` PROPAGATES NaN, so "the constructor clamps" is not
//! enough on its own — every parameter needs an explicit non-finite
//! scrub before (or instead of) its range clamp. This suite pins that
//! contract for the whole filter family.

use oxideav_audio_filter::spectrogram::SpectrogramOptions;
use oxideav_audio_filter::*;
use oxideav_core::{AudioFrame, ChannelLayout, SampleFormat};

const FS: u32 = 48_000;
const N: usize = 512;

/// The hostile parameter values every f32 knob is exercised with.
const HOSTILE: [f32; 7] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    1.0e30,
    -1.0e30,
    0.0,
    -1.0,
];

fn params(channels: u16) -> AudioStreamParams {
    AudioStreamParams {
        format: SampleFormat::F32,
        channels,
        sample_rate: FS,
    }
}

fn tone(channels: usize) -> AudioFrame {
    let mut bytes = Vec::with_capacity(N * channels * 4);
    for n in 0..N {
        for ch in 0..channels {
            let w = 2.0 * std::f32::consts::PI * 440.0 / FS as f32;
            let s = 0.5 * (n as f32 * w + ch as f32).sin();
            bytes.extend_from_slice(&s.to_le_bytes());
        }
    }
    AudioFrame {
        samples: N as u32,
        pts: None,
        data: vec![bytes],
    }
}

/// Drive `filter` over a finite tone and assert every output sample is
/// finite. `label` identifies the construction in failures.
fn assert_finite_output(label: &str, channels: usize, filter: &mut dyn AudioFilter) {
    let p = params(channels as u16);
    let input = tone(channels);
    let mut frames = filter.process(&input, p).expect("process must not error");
    frames.extend(filter.flush(p).expect("flush must not error"));
    for f in &frames {
        for plane in &f.data {
            for c in plane.chunks_exact(4) {
                let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                assert!(
                    v.is_finite(),
                    "{label}: non-finite output sample {v} from finite input"
                );
            }
        }
    }
}

/// For each hostile value `h`, build the filter with `make(h)` (every
/// f32 knob set to `h`) and check the finite-output contract.
fn check(name: &str, channels: usize, mut make: impl FnMut(f32) -> Box<dyn AudioFilter>) {
    for &h in &HOSTILE {
        let mut f = make(h);
        assert_finite_output(&format!("{name}(h={h})"), channels, &mut *f);
    }
}

macro_rules! hostile {
    ($test:ident, $name:literal, $channels:expr, |$h:ident| $make:expr) => {
        #[test]
        fn $test() {
            check($name, $channels, |$h| Box::new($make));
        }
    };
}

// --- gain / distortion / waveshaping ------------------------------------
hostile!(volume, "volume", 1, |h| Volume::new(h));
hostile!(volume_db, "volume_db", 1, |h| Volume::from_db(h));
hostile!(hard_clipper, "hard_clipper", 1, |h| HardClipper::new(h, h));
hostile!(tape_saturation, "tape_saturation", 1, |h| {
    TapeSaturation::new(h, h)
});
hostile!(exciter, "exciter", 1, |h| Exciter::with(h, h, h));
hostile!(
    octave_doubler,
    "octave_doubler",
    1,
    |h| OctaveDoubler::with(h, h, true)
);
hostile!(slew_limiter, "slew_limiter", 1, |h| SlewLimiter::new(h));
hostile!(slew_limiter_asym, "slew_limiter_asym", 1, |h| {
    SlewLimiter::with_asymmetric(h, h)
});

// --- dynamics ------------------------------------------------------------
hostile!(compressor, "compressor", 1, |h| Compressor::new(
    h, h, h, h, h, h
));
hostile!(parallel_compressor, "parallel_compressor", 1, |h| {
    ParallelCompressor::new(h, h, h, h, h)
});
hostile!(limiter, "limiter", 1, |h| Limiter::new(h, h, 32));
hostile!(expander, "expander", 1, |h| Expander::new(h, h, h, h, h, h));
hostile!(upward_compressor, "upward_compressor", 1, |h| {
    UpwardCompressor::new(h, h, h, h, h, h)
});
hostile!(upward_expander, "upward_expander", 1, |h| {
    UpwardExpander::new(h, h, h, h, h, h)
});
hostile!(noise_gate, "noise_gate", 1, |h| NoiseGate::new(h, h, h, h));
hostile!(noise_gate_with, "noise_gate_with", 1, |h| NoiseGate::with(
    h, h, h, h, h, h
));
hostile!(adaptive_noise_gate, "adaptive_noise_gate", 1, |h| {
    AdaptiveNoiseGate::with(h, h, h, h)
});
hostile!(ducker, "ducker", 2, |h| Ducker::with(h, h, h, h)
    .with_max_reduction_db(h));
hostile!(gain_normalizer, "gain_normalizer", 1, |h| {
    GainNormalizer::with(h, h, h)
        .with_max_gain_db(h)
        .with_max_atten_db(h)
});
hostile!(transient_designer, "transient_designer", 1, |h| {
    TransientDesigner::with(h, h, h, h)
});
hostile!(de_esser, "de_esser", 1, |h| DeEsser::with(h, h, h, h, h));
hostile!(multiband_compressor, "multiband_compressor", 1, |h| {
    MultibandCompressor::with(
        h,
        h,
        BandSettings {
            threshold_db: h,
            ratio: h,
            attack_ms: h,
            release_ms: h,
            knee_db: h,
            makeup_gain_db: h,
        },
        BandSettings {
            threshold_db: h,
            ratio: h,
            attack_ms: h,
            release_ms: h,
            knee_db: h,
            makeup_gain_db: h,
        },
        BandSettings {
            threshold_db: h,
            ratio: h,
            attack_ms: h,
            release_ms: h,
            knee_db: h,
            makeup_gain_db: h,
        },
    )
});

// --- EQ / spectral ---------------------------------------------------------
hostile!(biquad_lpf, "biquad_lpf", 1, |h| Biquad::new(
    BiquadKind::LowPass { cutoff_hz: h, q: h }
));
hostile!(biquad_peaking, "biquad_peaking", 1, |h| {
    Biquad::new(BiquadKind::Peaking {
        center_hz: h,
        q: h,
        gain_db: h,
    })
});
hostile!(biquad_shelf, "biquad_shelf", 1, |h| {
    Biquad::new(BiquadKind::LowShelf {
        cutoff_hz: h,
        q: h,
        gain_db: h,
    })
});
hostile!(svf, "svf", 1, |h| SvfFilter::new(SvfMode::LowPass, h, h));
hostile!(equalizer, "equalizer", 1, |h| Equalizer::new(FS)
    .with_low_pass(h, h)
    .with_band_pass(h, h));
hostile!(dc_blocker, "dc_blocker", 1, |h| DcBlocker::with_pole(h));
hostile!(hum_filter, "hum_filter", 1, |h| HumFilter::new(h, h, 4));
hostile!(crossover, "crossover", 1, |h| Crossover::new(h, h));
hostile!(pre_emphasis, "pre_emphasis", 1, |h| {
    PreEmphasis::with_gain(EmphasisCurve::Custom { tau_s: h }, h)
});
hostile!(de_emphasis, "de_emphasis", 1, |h| {
    DeEmphasis::with_gain(EmphasisCurve::Custom { tau_s: h }, h)
});
hostile!(comb_ff, "comb_ff", 1, |h| CombFilter::with_delay_ms(
    CombMode::Feedforward { gain: h },
    h
));
hostile!(comb_fb, "comb_fb", 1, |h| CombFilter::with_delay_ms(
    CombMode::Feedback {
        gain: h,
        damping: h
    },
    h
));
hostile!(freq_shifter, "freq_shifter", 1, |h| FreqShifter::with(
    h,
    1 << 20
));

// --- delay / modulation ------------------------------------------------
hostile!(echo, "echo", 1, |h| Echo::new(h, h, h));
hostile!(reverb, "reverb", 1, |h| Reverb::new(h, h, h, h));
hostile!(chorus, "chorus", 1, |h| Chorus::new(u8::MAX, h, h, h, h));
hostile!(flanger, "flanger", 1, |h| Flanger::new(h, h, h, h));
hostile!(phaser, "phaser", 1, |h| Phaser::new(u8::MAX, h, h, h, h));
hostile!(vibrato, "vibrato", 1, |h| Vibrato::new(h, h));
hostile!(tremolo, "tremolo", 1, |h| Tremolo::new(h, h));
hostile!(auto_pan, "auto_pan", 2, |h| AutoPan::new(h, h));
hostile!(wah, "wah", 1, |h| Wah::with(h, h, h, h, h));
hostile!(ring_modulator, "ring_modulator", 1, |h| RingModulator::new(
    h, h
));
hostile!(talkbox, "talkbox", 1, |h| Talkbox::with(
    Vowel::Ah,
    Vowel::Ee,
    h,
    h,
    h
));
hostile!(pitch_shift, "pitch_shift", 1, |h| PitchShift::new(h));

// --- stereo field --------------------------------------------------------
hostile!(stereo_widener, "stereo_widener", 2, |h| StereoWidener::new(
    h
));
hostile!(stereo_imager, "stereo_imager", 2, |h| StereoImager::with(
    h, h, h
));

// --- generators ------------------------------------------------------------
hostile!(white_noise, "white_noise", 1, |h| WhiteNoise::new(h));
hostile!(pink_noise, "pink_noise", 1, |h| PinkNoise::new(h));
hostile!(brown_noise, "brown_noise", 1, |h| BrownNoise::new(h));

// --- meters / observers -------------------------------------------------
hostile!(envelope_follower, "envelope_follower", 1, |h| {
    EnvelopeFollower::new(h, h)
});
hostile!(silence_detector, "silence_detector", 1, |h| {
    SilenceDetector::with_env(h, h, h, h)
});
hostile!(crest_factor_meter, "crest_factor_meter", 1, |h| {
    CrestFactorMeter::with_window_ms(h)
});
hostile!(dc_offset_meter, "dc_offset_meter", 1, |h| {
    DcOffsetMeter::with_window_ms(h)
});
hostile!(
    stereo_correlation_meter,
    "stereo_correlation_meter",
    2,
    |h| StereoCorrelationMeter::with_window_ms(h)
);
hostile!(stereo_balance_meter, "stereo_balance_meter", 2, |h| {
    StereoBalanceMeter::with_window_ms(h)
});
hostile!(zero_crossing_rate, "zero_crossing_rate", 1, |h| {
    ZeroCrossingRateMeter::with_window_ms(h)
});
hostile!(loudness, "loudness", 2, |h| {
    let _ = h;
    LoudnessITU::new()
});
hostile!(true_peak, "true_peak", 1, |h| {
    let _ = h;
    TruePeakDetector::new()
});

// --- integer-parameter extremes -----------------------------------------
hostile!(bitcrusher_extremes, "bitcrusher", 1, |h| {
    let _ = h;
    Bitcrusher::new(0, 0)
});
hostile!(bitcrusher_max, "bitcrusher_max", 1, |h| {
    let _ = h;
    Bitcrusher::new(u8::MAX, u32::MAX)
});
hostile!(median_filter_extremes, "median_filter", 1, |h| {
    let _ = h;
    MedianFilter::new(usize::MAX)
});
hostile!(median_filter_zero, "median_filter_zero", 1, |h| {
    let _ = h;
    MedianFilter::new(0)
});
hostile!(limiter_lookahead_max, "limiter_lookahead_max", 1, |h| {
    let _ = h;
    Limiter::new(-1.0, 50.0, usize::MAX)
});
hostile!(dither_extremes, "dither", 1, |h| {
    let _ = h;
    Dither::with(0, DitherMode::Tpdf, NoiseShaping::SecondOrder)
});
hostile!(dither_max_bits, "dither_max_bits", 1, |h| {
    let _ = h;
    Dither::with(u8::MAX, DitherMode::Tpdf, NoiseShaping::SecondOrder)
});

// --- Result-returning constructors must reject, not panic ----------------
#[test]
fn resample_rejects_bad_rates() {
    assert!(Resample::new(0, 48_000).is_err());
    assert!(Resample::new(48_000, 0).is_err());
    assert!(Resample::new(1, u32::MAX).is_err()); // extreme LCM ratio
}

#[test]
fn downmix_rejects_discrete_layouts() {
    assert!(DownmixFilter::new(
        ChannelLayout::DiscreteN(8),
        ChannelLayout::Mono,
        DownmixMode::Average
    )
    .is_err());
}

#[test]
fn spectrogram_rejects_hostile_options() {
    // fft_size: not a power of two / too small / absurdly large.
    for fft in [0usize, 7, 1 << 30, usize::MAX] {
        assert!(
            Spectrogram::new(SpectrogramOptions {
                fft_size: fft,
                ..SpectrogramOptions::default()
            })
            .is_err(),
            "fft_size={fft} must be rejected"
        );
    }
    // hop_size: zero / larger than fft.
    for hop in [0usize, 2048] {
        assert!(
            Spectrogram::new(SpectrogramOptions {
                fft_size: 1024,
                hop_size: hop,
                ..SpectrogramOptions::default()
            })
            .is_err(),
            "hop_size={hop} must be rejected"
        );
    }
    // width/height: zero / absurdly large.
    for dim in [0u32, u32::MAX] {
        assert!(
            Spectrogram::new(SpectrogramOptions {
                width: dim,
                ..SpectrogramOptions::default()
            })
            .is_err(),
            "width={dim} must be rejected"
        );
        assert!(
            Spectrogram::new(SpectrogramOptions {
                height: dim,
                ..SpectrogramOptions::default()
            })
            .is_err(),
            "height={dim} must be rejected"
        );
    }
}

// --- FracDelayLine primitive ------------------------------------------
#[test]
fn frac_delay_capacity_is_bounded() {
    // usize::MAX capacity must be capped, not allocated.
    let line = FracDelayLine::new(1, usize::MAX, Interp::Linear);
    assert!(line.capacity() <= frac_delay::MAX_CAPACITY);
    let line = FracDelayLine::new(0, 0, Interp::Lagrange(usize::MAX));
    assert!(line.capacity() >= 1);
}
