//! Pure-Rust audio filters for the oxideav framework.
//!
//! Each filter implements the [`AudioFilter`] trait and operates on
//! [`AudioFrame`](oxideav_core::AudioFrame) values. All filters convert input
//! samples to `f32` internally via [`sample_convert`] and convert back to the
//! input format on output. The exception is [`Resample`](resample::Resample),
//! whose output frame's `sample_rate` differs from its input but whose sample
//! format is preserved.
//!
//! # Streaming model
//!
//! Filters maintain internal state between calls to [`AudioFilter::process`].
//! A single input frame may produce zero, one, or many output frames depending
//! on the filter's buffering behaviour. After the last input frame, callers
//! should invoke [`AudioFilter::flush`] to drain any retained samples.
//!
//! Filters declare themselves `Send` so they can be moved between threads, but
//! they are not required to be `Sync`.
//!
//! # Available filters
//!
//! - [`Volume`](volume::Volume) — gain (linear or dB) with hard clipping.
//! - [`NoiseGate`](noise_gate::NoiseGate) — threshold-based gate with attack,
//!   release, and hold.
//! - [`Echo`](echo::Echo) — single-tap circular delay line with feedback and
//!   wet/dry mix.
//! - [`Resample`](resample::Resample) — polyphase windowed-sinc rate
//!   conversion.
//! - [`Spectrogram`](spectrogram::Spectrogram) — STFT-based image renderer
//!   with PNG output.
//! - [`DownmixFilter`](downmix::DownmixFilter) — channel-layout fold-down
//!   (LoRo / LtRt / Average / Binaural).
//! - [`Biquad`](biquad::Biquad) — second-order IIR EQ family (LPF/HPF
//!   /BPF/notch/peaking/low-shelf/high-shelf) with bilinear-transform
//!   coefficient derivation.
//! - [`Compressor`](compressor::Compressor) — peak-detector compressor
//!   with soft-knee + attack/release follower + make-up gain.
//! - [`Limiter`](limiter::Limiter) — brickwall peak limiter with
//!   optional look-ahead.
//! - [`DcBlocker`](dc_blocker::DcBlocker) — first-order IIR HPF at
//!   sub-audible cutoff to remove DC offsets.
//! - [`StereoWidener`](stereo_widener::StereoWidener) — M/S width
//!   control with `width ∈ [0, 2]`.
//! - [`Reverb`](reverb::Reverb) — Schroeder-style algorithmic reverb
//!   (4 parallel combs ║ 2 serial all-passes).
//! - [`Tremolo`](tremolo::Tremolo) — sine-LFO amplitude modulation.
//! - [`LoudnessITU`](loudness::LoudnessITU) — ITU-R BS.1770-4 / EBU
//!   R128 integrated-loudness measurement (LUFS).
//! - [`PitchShift`](pitch_shift::PitchShift) — time-domain SOLA-style
//!   granular pitch shifter (`-12..=+12` semitones, no FFT).
//! - [`Chorus`](chorus::Chorus) — 1..=4 LFO-modulated short delay taps.
//! - [`Flanger`](flanger::Flanger) — short-delay comb with positive
//!   feedback (sweeping resonance).
//! - [`Phaser`](phaser::Phaser) — N cascaded all-pass sections with
//!   LFO-modulated cutoffs.
//! - [`Equalizer`](equalizer::Equalizer) — builder over N [`Biquad`]
//!   sections in series.
//! - [`WhiteNoise`](white_noise::WhiteNoise) /
//!   [`PinkNoise`](pink_noise::PinkNoise) /
//!   [`BrownNoise`](brown_noise::BrownNoise) — splitmix64-seeded noise
//!   generators with flat / 1/f / 1/f² spectra.
//! - [`SilenceDetector`](silence_detector::SilenceDetector) — RMS-based
//!   silence flag with attack/release envelope + hold.
//! - [`Vibrato`](vibrato::Vibrato) — LFO-modulated delay-line pitch
//!   shift (complement to [`Tremolo`](tremolo::Tremolo) which modulates
//!   amplitude).
//! - [`AutoPan`](auto_pan::AutoPan) — LFO-modulated L/R stereo
//!   placement.
//! - [`Bitcrusher`](bitcrusher::Bitcrusher) — bit-depth quantisation
//!   plus sample-and-hold rate reduction.
//! - [`TapeSaturation`](tape_saturation::TapeSaturation) — `tanh`
//!   soft-clip with asymmetric drive.
//! - [`HumFilter`](hum_filter::HumFilter) — cascaded narrow notches at
//!   line-mains fundamental + harmonics.
//! - [`Crossover`](crossover::Crossover) — two-way LPF/HPF band split
//!   at a configurable cutoff (output frame carries 2× input channels).

pub mod auto_pan;
pub mod biquad;
pub mod bitcrusher;
pub mod brown_noise;
pub mod chorus;
pub mod compressor;
pub mod crossover;
pub mod dc_blocker;
pub mod downmix;
pub mod echo;
pub mod equalizer;
pub mod fft;
pub mod flanger;
pub mod hum_filter;
pub mod limiter;
pub mod loudness;
pub mod noise_gate;
pub mod phaser;
pub mod pink_noise;
pub mod pitch_shift;
pub mod registry;
pub mod resample;
pub mod reverb;
pub mod sample_convert;
pub mod silence_detector;
pub mod spectrogram;
pub mod stereo_widener;
pub mod tape_saturation;
pub mod tremolo;
pub mod vibrato;
pub mod volume;
pub mod white_noise;

pub use auto_pan::AutoPan;
pub use biquad::{Biquad, BiquadKind};
pub use bitcrusher::Bitcrusher;
pub use brown_noise::BrownNoise;
pub use chorus::Chorus;
pub use compressor::Compressor;
pub use crossover::Crossover;
pub use dc_blocker::DcBlocker;
pub use downmix::{auto_downmix, DownmixFilter, DownmixMode};
pub use echo::Echo;
pub use equalizer::Equalizer;
pub use flanger::Flanger;
pub use hum_filter::HumFilter;
pub use limiter::Limiter;
pub use loudness::LoudnessITU;
pub use noise_gate::NoiseGate;
pub use phaser::Phaser;
pub use pink_noise::PinkNoise;
pub use pitch_shift::PitchShift;
pub use registry::{__oxideav_entry, register};
pub use resample::Resample;
pub use reverb::Reverb;
pub use silence_detector::SilenceDetector;
pub use spectrogram::{Colormap, Spectrogram, SpectrogramOptions, Window};
pub use stereo_widener::StereoWidener;
pub use tape_saturation::TapeSaturation;
pub use tremolo::Tremolo;
pub use vibrato::Vibrato;
pub use volume::Volume;
pub use white_noise::WhiteNoise;

use oxideav_core::{AudioFrame, Result, SampleFormat};

/// Stream-level parameters threaded into every [`AudioFilter`] call.
///
/// Used to live on every `AudioFrame` (`format` / `channels` /
/// `sample_rate`); the slim moved them to the stream's
/// [`CodecParameters`](oxideav_core::CodecParameters). The
/// [`AudioFilterAdapter`](crate::registry) shim reads them once from
/// the input port spec at construction and re-supplies them per call so
/// concrete filters don't have to negotiate per-frame.
#[derive(Clone, Copy, Debug)]
pub struct AudioStreamParams {
    pub format: SampleFormat,
    pub channels: u16,
    pub sample_rate: u32,
}

/// Streaming audio filter.
///
/// Implementors process one input frame at a time and may emit zero or more
/// output frames. Internal state (delay lines, envelopes, sample histories,
/// resampler phase, FFT accumulators, …) lives in `self` and is preserved
/// across calls.
///
/// At end-of-stream callers invoke [`AudioFilter::flush`] to obtain any
/// frames still buffered inside the filter. The default implementation
/// returns an empty `Vec` for filters that do not buffer.
pub trait AudioFilter: Send {
    /// Process one input frame, returning zero or more output frames.
    fn process(&mut self, input: &AudioFrame, params: AudioStreamParams)
        -> Result<Vec<AudioFrame>>;

    /// Drain any internally buffered samples at end-of-stream. `params`
    /// describes the same stream-level shape passed to
    /// [`AudioFilter::process`].
    fn flush(&mut self, _params: AudioStreamParams) -> Result<Vec<AudioFrame>> {
        Ok(Vec::new())
    }
}
