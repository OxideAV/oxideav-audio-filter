//! Denormal-flush contract for the feedback / recursive filters.
//!
//! A recursive filter excited by an impulse and then fed silence decays
//! exponentially. Left alone, the tail glides into the f32 subnormal
//! range (|x| < 2⁻¹²⁶ ≈ 1.2e-38) and dwells there for hundreds of
//! milliseconds to seconds before underflowing to true zero — inaudible
//! by ~250 dB, but on hardware without fast subnormal arithmetic each
//! of those samples costs a micro-trap, so a "silent" reverb tail can
//! consume more CPU than programme audio.
//!
//! The contract pinned here:
//!  1. **No subnormal output, ever** — the filter must flush its
//!     feedback state to exact zero once it decays below audibility
//!     instead of letting it creep through the subnormal range.
//!  2. **The tail terminates** — after an impulse and 10 s of silence,
//!     the final second of output is bit-exact zero (no self-
//!     oscillation, no state stuck above the flush threshold).

use oxideav_audio_filter::*;
use oxideav_core::{AudioFrame, SampleFormat};

const FS: u32 = 48_000;
/// Frames of 100 ms.
const FRAME: usize = 4_800;
/// 10 s of post-impulse silence.
const SILENCE_FRAMES: usize = 100;

fn params() -> AudioStreamParams {
    AudioStreamParams {
        format: SampleFormat::F32,
        channels: 1,
        sample_rate: FS,
    }
}

fn frame_from(samples: &[f32]) -> AudioFrame {
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

/// Drive `filter` with a unit impulse then 10 s of silence; enforce the
/// no-subnormal + tail-terminates contract.
fn assert_decays(name: &str, filter: &mut dyn AudioFilter) {
    let p = params();

    let mut impulse = vec![0.0f32; FRAME];
    impulse[0] = 1.0;
    let silence = vec![0.0f32; FRAME];

    let check = |frame_idx: usize, frames: &[AudioFrame]| {
        for f in frames {
            for plane in &f.data {
                for c in plane.chunks_exact(4) {
                    let v = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    assert!(
                        v == 0.0 || v.abs() >= f32::MIN_POSITIVE,
                        "{name}: subnormal output {v:e} in frame {frame_idx} — \
                         feedback state is dwelling below the flush threshold"
                    );
                    if frame_idx > SILENCE_FRAMES - 10 {
                        assert!(
                            v == 0.0,
                            "{name}: tail still ringing at {v:e} in frame {frame_idx} \
                             (~{} s after the impulse)",
                            frame_idx / 10
                        );
                    }
                }
            }
        }
    };

    let out = filter.process(&frame_from(&impulse), p).expect("process");
    check(0, &out);
    for i in 1..=SILENCE_FRAMES {
        let out = filter.process(&frame_from(&silence), p).expect("process");
        check(i, &out);
    }
}

macro_rules! decays {
    ($test:ident, $name:literal, $make:expr) => {
        #[test]
        fn $test() {
            let mut f = $make;
            assert_decays($name, &mut f);
        }
    };
}

decays!(
    biquad_lpf,
    "biquad(lpf)",
    Biquad::new(BiquadKind::LowPass {
        cutoff_hz: 1_000.0,
        q: 0.707,
    })
);
decays!(
    biquad_peaking_high_q,
    "biquad(peaking q=20)",
    Biquad::new(BiquadKind::Peaking {
        center_hz: 1_000.0,
        q: 20.0,
        gain_db: 12.0,
    })
);
decays!(
    svf,
    "svf(lowpass)",
    SvfFilter::new(SvfMode::LowPass, 1_000.0, 0.707)
);
decays!(dc_blocker, "dc_blocker", DcBlocker::new());
decays!(echo, "echo(fb 0.5)", Echo::new(10.0, 0.5, 1.0));
decays!(
    comb_feedback,
    "comb(fb 0.5)",
    CombFilter::with_delay_ms(
        CombMode::Feedback {
            gain: 0.5,
            damping: 0.2,
        },
        5.0,
    )
);
decays!(flanger, "flanger", Flanger::new(0.5, 3.0, 0.5, 1.0));
decays!(phaser, "phaser", Phaser::new(4, 0.5, 800.0, 0.5, 1.0));
decays!(reverb, "reverb(room 0)", Reverb::new(0.0, 0.5, 1.0, 0.0));
decays!(
    pre_emphasis,
    "pre_emphasis",
    PreEmphasis::new(EmphasisCurve::Fm50us)
);
decays!(
    de_emphasis,
    "de_emphasis",
    DeEmphasis::new(EmphasisCurve::Fm75us)
);
// Q = 10 here, not the sharper default 30: a Q = 30 notch at 50 Hz has
// a pole radius of ~0.99989, whose impulse tail legitimately takes ~11 s
// to decay below the 1e-25 flush threshold — longer than this harness's
// silence window. Q = 10 decays in ~4 s while exercising the same
// cascaded-notch feedback path.
decays!(hum_filter, "hum_filter", HumFilter::new(50.0, 10.0, 4));
decays!(
    crossover_lr4,
    "crossover(lr4)",
    Crossover::with_slope(1_000.0, 0.707, CrossoverSlope::LinkwitzRiley4)
);
decays!(
    equalizer,
    "equalizer",
    Equalizer::new(FS)
        .with_low_pass(8_000.0, 0.707)
        .with_band_pass(1_000.0, 1.0)
);
decays!(wah, "wah", Wah::new());
decays!(de_esser, "de_esser", DeEsser::new());
decays!(
    de_emphasis_riaa,
    "de_emphasis(riaa 2nd-order)",
    DeEmphasis::new(EmphasisCurve::Riaa3180_318_75)
);

/// Regression: a low-frequency, moderate-Q notch must decay to EXACT
/// zero, not hang in a flush-induced limit cycle.
///
/// The biquad's `(s1, s2)` state pair nearly cancels for low-cutoff,
/// high-pole-radius configurations. An early flush-to-zero
/// implementation truncated each component independently; the
/// asymmetric truncation re-injected ~1e-25 of error per sample, which
/// the resonant pole amplified into a SUSTAINED ~3e-23 oscillation that
/// never decayed (and slowly grew). The joint flush
/// (`biquad::State::flush_denormals`) zeroes the pair atomically and
/// the tail dies for good.
#[test]
fn notch_50hz_no_limit_cycle() {
    let mut bq = Biquad::new(BiquadKind::Notch {
        center_hz: 50.0,
        q: 10.0,
    });
    let mut buf = vec![0.0f32; 48_000 * 8];
    buf[0] = 1.0;
    bq.process_in_place(&mut buf, 1, FS);
    // By 6 s the tail must be bit-exact zero and STAY zero.
    for (i, v) in buf[48_000 * 6..].iter().enumerate() {
        assert!(
            *v == 0.0,
            "50 Hz notch still ringing at {v:e}, sample {} after 6 s — \
             flush-induced limit cycle?",
            i
        );
    }
}
