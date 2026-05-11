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

pub mod biquad;
pub mod compressor;
pub mod dc_blocker;
pub mod downmix;
pub mod echo;
pub mod fft;
pub mod limiter;
pub mod loudness;
pub mod noise_gate;
pub mod pitch_shift;
pub mod registry;
pub mod resample;
pub mod reverb;
pub mod sample_convert;
pub mod spectrogram;
pub mod stereo_widener;
pub mod tremolo;
pub mod volume;

pub use biquad::{Biquad, BiquadKind};
pub use compressor::Compressor;
pub use dc_blocker::DcBlocker;
pub use downmix::{auto_downmix, DownmixFilter, DownmixMode};
pub use echo::Echo;
pub use limiter::Limiter;
pub use loudness::LoudnessITU;
pub use noise_gate::NoiseGate;
pub use pitch_shift::PitchShift;
pub use registry::{__oxideav_entry, register};
pub use resample::Resample;
pub use reverb::Reverb;
pub use spectrogram::{Colormap, Spectrogram, SpectrogramOptions, Window};
pub use stereo_widener::StereoWidener;
pub use tremolo::Tremolo;
pub use volume::Volume;

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
