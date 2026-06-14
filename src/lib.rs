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
//! - [`Biquad`](biquad::Biquad) — second-order IIR EQ family (LPF /
//!   HPF / BPF in constant-skirt and constant-0dB-peak variants /
//!   notch / peaking / low + high shelves in Q- and slope-(S)-
//!   parameterised forms / all-pass) with bilinear-transform
//!   coefficient derivation and a closed-form
//!   `magnitude_response_db` evaluator.
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
//!   at a configurable cutoff (output frame carries 2× input channels);
//!   Butterworth-2 (12 dB/oct) or magnitude-flat Linkwitz-Riley-4
//!   (24 dB/oct) slope via [`CrossoverSlope`](crossover::CrossoverSlope).
//! - [`MidSide`](mid_side::MidSide) — explicit L/R ↔ M/S transcoder
//!   (stateless, exact roundtrip).
//! - [`EnvelopeFollower`](envelope_follower::EnvelopeFollower) — peak
//!   / RMS amplitude envelope detector (pass-through, observe via
//!   `current()` / `current_db()`).
//! - [`DeEsser`](de_esser::DeEsser) — split-band downward compressor
//!   targeting sibilance (LPF/HPF + per-band compressor on the high
//!   half).
//! - [`Wah`](wah::Wah) — LFO-swept resonant band-pass (Cry-Baby-style
//!   sweep, logarithmic centre interpolation).
//! - [`OctaveDoubler`](octave_doubler::OctaveDoubler) — full-wave-
//!   rectifier-based octave-up layer (Octavia-style), DC-blocked.
//! - [`AdaptiveNoiseGate`](adaptive_noise_gate::AdaptiveNoiseGate) —
//!   gate with a self-learned noise floor (slow one-pole learner +
//!   margin-based open/close decision).
//! - [`Exciter`](exciter::Exciter) — high-band saturation
//!   psycho-acoustic enhancer (HPF + `tanh` waveshaper, dry-mixed).
//! - [`MultibandCompressor`](multiband_compressor::MultibandCompressor) —
//!   three-band (low / mid / high) independent compression with
//!   Butterworth-2 crossovers.
//! - [`StereoImager`](stereo_imager::StereoImager) — frequency-
//!   dependent stereo width (M/S side split into two bands with
//!   independent width factors).
//! - [`Talkbox`](talkbox::Talkbox) — LFO-morphed parallel formant
//!   band-pass bank (vowel filter, no carrier required).
//! - [`TransientDesigner`](transient_designer::TransientDesigner) —
//!   two-envelope (fast / slow) attack + sustain shaper.
//! - [`Ducker`](ducker::Ducker) — internally-keyed sidechain
//!   compressor with safety-floored gain reduction.
//! - [`GainNormalizer`](gain_normalizer::GainNormalizer) — slow-AGC
//!   programme-level normaliser with silence-freeze integrator.
//! - [`FreqShifter`](freq_shifter::FreqShifter) — Hilbert-FIR SSB
//!   frequency shifter (additive `Δf` in Hz, harmonic-destroying).
//! - [`RingModulator`](ring_modulator::RingModulator) — sine-carrier
//!   amplitude multiplication (double-sideband suppressed-carrier AM:
//!   `y = x · cos(2π·fc·n/fs)`, audible-carrier Dalek / bell effect).
//! - [`HardClipper`](hard_clipper::HardClipper) — memoryless symmetric
//!   clipping distortion (`y = clamp(drive·x, -ceiling, +ceiling)`;
//!   odd-harmonic fuzz / overdrive, distinct from the `tanh` soft-clip
//!   in [`TapeSaturation`](tape_saturation::TapeSaturation)).
//! - [`SlewLimiter`](slew_limiter::SlewLimiter) — slope-limited smoother
//!   that bounds the per-sample amplitude change to `max_slew_per_sec
//!   / fs`. Linear-ramp response (vs the LPF's exponential response);
//!   classic anti-zipper / portamento / anti-pop primitive. Supports
//!   asymmetric rise / fall caps.
//! - [`Expander`](expander::Expander) — proportional downward expander
//!   (slope `-(R - 1)` per dB under threshold). Distinct from the
//!   binary [`NoiseGate`](noise_gate::NoiseGate): a `2:1` setting
//!   gives `2 dB` attenuation for each `1 dB` of under-shoot, fading
//!   gracefully into silence instead of slamming closed. `ratio = ∞`
//!   collapses to a hard downward gate; soft-knee width smooths the
//!   threshold transition.
//! - [`UpwardCompressor`](upward_compressor::UpwardCompressor) — the
//!   complementary dynamics quadrant: *boosts* quiet signal **below**
//!   the threshold by `(1 - 1/R)` of each dB of under-shoot (capped at
//!   `range_db`), leaving peaks above the threshold untouched. Narrows
//!   the dynamic range from the bottom while preserving transients —
//!   distinct from [`Compressor`](compressor::Compressor) (reduces
//!   *above* threshold) and [`Expander`](expander::Expander)
//!   (attenuates *below*; this boosts). `ratio = ∞` lifts quiet signal
//!   up to the threshold; soft-knee + range-cap.
//! - [`TruePeakDetector`](true_peak_detector::TruePeakDetector) — 4×
//!   polyphase oversampled inter-sample peak observer (dBTP).
//!   Pass-through; reports `current_dbtp` / `max_dbtp` / `overs`
//!   count. Distinct from [`Limiter`](limiter::Limiter) (sample-peak
//!   reduction), [`EnvelopeFollower`](envelope_follower::EnvelopeFollower)
//!   (smoothed envelope), and [`LoudnessITU`](loudness::LoudnessITU)
//!   (K-weighted integrated loudness). 48-tap Kaiser-windowed FIR
//!   (β chosen for ~100 dB stop-band) split into 4 polyphase
//!   sub-filters of 12 taps each; `f64` accumulation; recovers the
//!   ≈ 0 dBTP inter-sample peak of an `fs/4 + π/4`-phase full-scale
//!   sine whose sample-peak is only `-3.01 dBFS`.
//! - [`SvfFilter`](state_variable::SvfFilter) — Chamberlin
//!   two-integrator-loop State Variable Filter. Simultaneous
//!   LP / BP / HP / Notch taps from one recurrence, `f`-modulatable
//!   per sample (single `sin` on `set_cutoff`). State-space topology
//!   distinct from [`Biquad`](biquad::Biquad)'s Direct-Form-II-
//!   Transposed transfer-function realisation. Stable while
//!   `f_c < f_s / 6` and `Q ∈ [0.5, 50]`; clamped at the boundary.
//! - [`MedianFilter`](median_filter::MedianFilter) — non-linear
//!   sliding-window median filter for impulse-noise (click / pop)
//!   removal. Per channel: keeps an `N`-sample ring buffer and emits
//!   the median of the latest `N` samples as the output for every
//!   input. Step edges that span more than `N / 2` samples pass
//!   through unaltered, but isolated impulse outliers are entirely
//!   discarded — a behaviour no linear LPF achieves. Complements the
//!   restoration family ([`HumFilter`](hum_filter::HumFilter) on
//!   cyclic-mains, [`DcBlocker`](dc_blocker::DcBlocker) on DC drift)
//!   by targeting transient impulse noise instead.
//! - [`CrestFactorMeter`] —
//!   pass-through observer reporting the peak-to-RMS ratio in dB over a
//!   sliding rectangular window (default 400 ms, EBU R128 short-term
//!   window). Distinct from
//!   [`TruePeakDetector`](true_peak_detector::TruePeakDetector)
//!   (absolute oversampled peak only), [`LoudnessITU`](loudness::LoudnessITU)
//!   (K-weighted integrated loudness only), and
//!   [`EnvelopeFollower`](envelope_follower::EnvelopeFollower) (single
//!   one-pole envelope, not a ratio). Reports `0 dB` on a square wave,
//!   `3.01 dB` on a sine, `~5–8 dB` on heavy broadcast pop, `> 20 dB`
//!   on sparse drum transients. Maintains running sum-of-squares for
//!   RMS plus a monotonic deque for sliding `max |x|` — both `O(1)`
//!   amortised per sample; periodic per-window sum-of-squares rebuild
//!   bounds `f64` drift on long streams.
//! - [`CombFilter`](comb_filter::CombFilter) — tunable comb filter
//!   (feedforward FIR `y[n] = x[n] + g·x[n−D]` or feedback IIR
//!   `y[n] = x[n] + g·damped(y[n−D])`) with optional one-pole LP
//!   damping on the feedback path. Building block for resonators,
//!   short-delay flange/chorus voices, plucked-string
//!   ([`CombFilter::karplus_strong`]) synthesis, and comb-EQ
//!   effects. Distinct from [`Echo`](echo::Echo) (single-tap audible
//!   repetition, no resonator framing), from
//!   [`Flanger`](flanger::Flanger) (LFO-modulated delay length on
//!   top of a feedback comb), and from [`Reverb`](reverb::Reverb)
//!   (four parallel combs + two serial all-passes presented as a
//!   room simulator).
//! - [`StereoCorrelationMeter`] — pass-through observer reporting the
//!   Pearson correlation coefficient between L and R channels over a
//!   sliding rectangular window (default 400 ms). Maps to the
//!   goniometer's `0° / 90° / 180°` angular axis via
//!   [`current_degrees`](stereo_correlation_meter::StereoCorrelationMeter::current_degrees).
//!   Reports `+1` for identical channels (mono content panned centre),
//!   `0` for orthogonal channels (uncorrelated stereo bed), and `-1`
//!   for phase-inverted channels (canonical mono-fold-down
//!   cancellation hazard). Five-sum incremental closed form
//!   (Σx, Σy, Σx², Σy², Σxy) at `O(1)` per sample, with a periodic
//!   per-window rebuild bounding `f64` drift on long streams. Distinct
//!   from
//!   [`StereoWidener`](stereo_widener::StereoWidener) and
//!   [`StereoImager`](stereo_imager::StereoImager) (both of which
//!   *process* the stereo image), and from the single-channel meters
//!   [`CrestFactorMeter`], [`TruePeakDetector`](true_peak_detector::TruePeakDetector),
//!   and [`LoudnessITU`](loudness::LoudnessITU) (all of which collapse
//!   stereo input to a per-channel-then-max summary, losing the
//!   inter-channel phase term the correlation meter exposes).
//! - [`PreEmphasis`](pre_emphasis::PreEmphasis) /
//!   [`DeEmphasis`](de_emphasis::DeEmphasis) — paired analog-broadcast
//!   / tape / FM record EQ shelving filters with a shared
//!   [`Curve`](pre_emphasis::Curve) family covering FM 50 µs / FM 75 µs
//!   / ITU-R J.17 / RIAA 3180 + 318 + 75 µs / user-specified
//!   single-time-constant. Coefficients are derived in-source via
//!   bilinear transform of `H_pre(s) = (1 + s·τ) / (1 + s·τ/G)` and
//!   its inverse; `pre · de` cascades to the identity by
//!   construction (within `f64` round-off).
//! - [`DcOffsetMeter`] — pass-through observer reporting the
//!   per-channel running mean (DC component) over a sliding
//!   rectangular window (default 400 ms). Where
//!   [`DcBlocker`](dc_blocker::DcBlocker) *removes* a DC bias with a
//!   single-pole HPF, this meter *reports* it without altering the
//!   signal. Distinct from the energy / phase observers —
//!   [`CrestFactorMeter`] is insensitive to a constant offset (peak
//!   and RMS grow together), [`LoudnessITU`](loudness::LoudnessITU)
//!   has its own ITU-BS.1770 RLB HPF filtering DC away before the
//!   K-weighting sum, [`StereoCorrelationMeter`] is mean-centred by
//!   construction, [`ZeroCrossingRateMeter`] is only *indirectly*
//!   sensitive (DC shifts the sign-change axis). Algorithm: per
//!   channel an `f32` ring of `N` samples + an `f64` running sum
//!   `S = Σ x` updated incrementally (`S ← S + x_new − x_old`) at
//!   `O(1)` per sample; periodic per-window rebuild bounds `f64`
//!   drift on long streams. Channel-link picks the per-channel mean
//!   with largest `|·|`, *sign preserved*, so equal-and-opposite
//!   biases on a split stereo bed do not cancel in the readout.
//! - [`Dither`](dither::Dither) — word-length-reduction requantizer
//!   with TPDF / RPDF dither and first- / second-order error-feedback
//!   noise shaping. The transparency-grade end-of-chain primitive for
//!   float → fixed-point output: rounds onto the exact `bits`-wide
//!   code grid (`Δ = 2^(1-bits)`) while decorrelating the rounding
//!   error from the programme. TPDF (triangular, peak-to-peak `2Δ`)
//!   renders mean *and* variance of the total error
//!   signal-independent (constant `Δ²/4`); error feedback through
//!   `NTF = 1 - z⁻¹` (+3 dB total, tilted +6 dB/oct) or
//!   `NTF = (1 - z⁻¹)²` (+7.8 dB total, double zero at DC) moves the
//!   noise out of the sensitive low/mid band. Distinct from
//!   [`Bitcrusher`](bitcrusher::Bitcrusher) (creative degradation:
//!   bare quantisation + aliasing sample-and-hold, no dither, no
//!   shaping).
//! - [`ZeroCrossingRateMeter`] — pass-through observer reporting the
//!   number of sign changes in the signal per unit time over a
//!   sliding rectangular window (default 25 ms, the canonical speech-
//!   analysis short-time frame). Per-sample running count is `O(1)`
//!   via a ring buffer of `N + 1` samples (one extra slot so we can
//!   reach the trailing sample's predecessor to subtract its
//!   contribution as it rotates out). Distinct from the energy /
//!   amplitude observers — [`CrestFactorMeter`] reports peak-to-RMS,
//!   [`TruePeakDetector`](true_peak_detector::TruePeakDetector)
//!   reports oversampled inter-sample peak, [`LoudnessITU`](loudness::LoudnessITU)
//!   reports K-weighted integrated loudness, [`EnvelopeFollower`](envelope_follower::EnvelopeFollower)
//!   reports a smoothed envelope; the zero-crossing meter alone
//!   exposes the per-sample sign sequence, the proxy for spectral
//!   centroid that classifies voiced vs unvoiced speech and tone vs
//!   percussion. Rate in Hz scales to `≈ 2·f₀` on a pure sine and
//!   saturates at `fs` on an alternating-sign signal.

pub mod adaptive_noise_gate;
pub mod auto_pan;
pub mod biquad;
pub mod bitcrusher;
pub mod brown_noise;
pub mod chorus;
pub mod comb_filter;
pub mod compressor;
pub mod crest_factor_meter;
pub mod crossover;
pub mod dc_blocker;
pub mod dc_offset_meter;
pub mod de_emphasis;
pub mod de_esser;
pub mod dither;
pub mod downmix;
pub mod ducker;
pub mod echo;
pub mod envelope_follower;
pub mod equalizer;
pub mod exciter;
pub mod expander;
pub mod fft;
pub mod flanger;
pub mod frac_delay;
pub mod freq_shifter;
pub mod gain_normalizer;
pub mod hard_clipper;
pub mod hum_filter;
pub mod limiter;
pub mod loudness;
pub mod median_filter;
pub mod mid_side;
pub mod multiband_compressor;
pub mod noise_gate;
pub mod octave_doubler;
pub mod phaser;
pub mod pink_noise;
pub mod pitch_shift;
pub mod pre_emphasis;
pub mod registry;
pub mod resample;
pub mod reverb;
pub mod ring_modulator;
pub mod sample_convert;
pub mod silence_detector;
pub mod slew_limiter;
pub mod spectrogram;
pub mod state_variable;
pub mod stereo_balance_meter;
pub mod stereo_correlation_meter;
pub mod stereo_imager;
pub mod stereo_widener;
pub mod talkbox;
pub mod tape_saturation;
pub mod transient_designer;
pub mod tremolo;
pub mod true_peak_detector;
pub mod upward_compressor;
pub mod vibrato;
pub mod volume;
pub mod wah;
pub mod white_noise;
pub mod window;
pub mod zero_crossing_rate;

pub use adaptive_noise_gate::AdaptiveNoiseGate;
pub use auto_pan::AutoPan;
pub use biquad::{Biquad, BiquadKind};
pub use bitcrusher::Bitcrusher;
pub use brown_noise::BrownNoise;
pub use chorus::Chorus;
pub use comb_filter::{CombFilter, CombMode};
pub use compressor::Compressor;
pub use crest_factor_meter::CrestFactorMeter;
pub use crossover::{Crossover, CrossoverSlope};
pub use dc_blocker::DcBlocker;
pub use dc_offset_meter::DcOffsetMeter;
pub use de_emphasis::DeEmphasis;
pub use de_esser::DeEsser;
pub use dither::{Dither, DitherMode, NoiseShaping};
pub use downmix::{auto_downmix, DownmixFilter, DownmixMode};
pub use ducker::Ducker;
pub use echo::Echo;
pub use envelope_follower::{EnvelopeFollower, EnvelopeMode};
pub use equalizer::Equalizer;
pub use exciter::Exciter;
pub use expander::Expander;
pub use flanger::Flanger;
pub use frac_delay::{FracDelayLine, Interp};
pub use freq_shifter::FreqShifter;
pub use gain_normalizer::GainNormalizer;
pub use hard_clipper::HardClipper;
pub use hum_filter::HumFilter;
pub use limiter::Limiter;
pub use loudness::LoudnessITU;
pub use median_filter::MedianFilter;
pub use mid_side::{MidSide, MidSideMode};
pub use multiband_compressor::{BandSettings, MultibandCompressor};
pub use noise_gate::NoiseGate;
pub use octave_doubler::OctaveDoubler;
pub use phaser::Phaser;
pub use pink_noise::PinkNoise;
pub use pitch_shift::PitchShift;
pub use pre_emphasis::{Curve as EmphasisCurve, PreEmphasis};
pub use registry::{__oxideav_entry, register};
pub use resample::Resample;
pub use reverb::Reverb;
pub use ring_modulator::RingModulator;
pub use silence_detector::SilenceDetector;
pub use slew_limiter::SlewLimiter;
pub use spectrogram::{Colormap, Spectrogram, SpectrogramOptions, Window};
pub use state_variable::{SvfFilter, SvfMode};
pub use stereo_balance_meter::StereoBalanceMeter;
pub use stereo_correlation_meter::StereoCorrelationMeter;
pub use stereo_imager::StereoImager;
pub use stereo_widener::StereoWidener;
pub use talkbox::{Talkbox, Vowel};
pub use tape_saturation::TapeSaturation;
pub use transient_designer::TransientDesigner;
pub use tremolo::Tremolo;
pub use true_peak_detector::TruePeakDetector;
pub use upward_compressor::UpwardCompressor;
pub use vibrato::Vibrato;
pub use volume::Volume;
pub use wah::Wah;
pub use white_noise::WhiteNoise;
pub use window::Window as WindowFunction;
pub use zero_crossing_rate::ZeroCrossingRateMeter;

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
