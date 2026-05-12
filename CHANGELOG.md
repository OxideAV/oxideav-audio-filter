# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- round 4: eight new filter families — `chorus` (1..=4 LFO-modulated
  short delay taps with per-voice phase offsets), `flanger` (short-delay
  feedback comb with swept resonance, 1..=15 ms), `phaser` (N=2..=12
  cascaded first-order all-pass sections with LFO-swept cutoffs +
  optional feedback), `equalizer` (builder over N `Biquad` sections in
  series), `white_noise` / `pink_noise` / `brown_noise` (splitmix64-seeded
  generators with flat / 1/f / 1/f² spectra — Paul Kellet's pink-noise
  filter, leaky-integrator brown), and `silence_detector` (pass-through
  RMS envelope observer with attack/release + hold-threshold flag). All
  eight registered in `registry::register` and wired through the standard
  `AudioFilter` contract.
- round 3: six new filter families — `dc_blocker` (single-pole IIR
  HPF at sub-audible cutoff for DC-offset removal), `stereo_widener`
  (Mid/Side width control with `width ∈ [0, 2]`), `reverb` (Schroeder
  algorithmic reverb: 4 parallel combs ║ 2 serial all-passes with
  room_size / damping / wet / dry knobs), `tremolo` (sine-LFO
  amplitude modulation with rate / depth), `loudness_itu` (ITU-R
  BS.1770-4 / EBU R128 integrated-loudness meter with K-weighting
  via bilinear-transformed analog prototypes + channel weights), and
  `pitch_shift` (time-domain SOLA-style granular pitch shifter,
  `-12..=+12` semitones, no FFT). All six registered in
  `registry::register` and wired through the standard `AudioFilter`
  contract.
- round 2: three new filter families — `biquad` (seven-config IIR EQ with
  bilinear-transform coefficient derivation: LPF/HPF/BPF/notch/peaking
  /low-shelf/high-shelf, DF-II-T core with `f64` state), `compressor`
  (peak detector with soft-knee + attack/release one-pole follower +
  make-up gain), and `limiter` (brickwall peak limiter with optional
  look-ahead, 0..=2048 samples). All three registered in
  `registry::register` and wired through the standard `AudioFilter`
  contract.

## [0.1.1](https://github.com/OxideAV/oxideav-audio-filter/compare/v0.1.0...v0.1.1) - 2026-05-06

### Other

- reframe FFI claim — HW-engine crates use OS FFI by necessity
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- replace never-match regex with semver_check = false
- release v0.0.7 ([#4](https://github.com/OxideAV/oxideav-audio-filter/pull/4))

## [0.1.0](https://github.com/OxideAV/oxideav-audio-filter/compare/v0.0.6...v0.1.0) - 2026-05-02

### Other

- promote to 0.1.0
- migrate to centralized OxideAV/.github reusable workflows

## [0.0.6](https://github.com/OxideAV/oxideav-audio-filter/compare/v0.0.5...v0.0.6) - 2026-05-02

### Added

- *(downmix)* add DownmixFilter with LoRo / LtRt / Average / Binaural modes

### Other

- stay on 0.1.x during heavy dev (semver_check=false)
- round 17: SampleFormat non_exhaustive arms + dead_code allow + fmt
- adopt slim VideoFrame/AudioFrame shape
- pin release-plz to patch-only bumps

## [0.0.5](https://github.com/OxideAV/oxideav-audio-filter/compare/v0.0.4...v0.0.5) - 2026-04-25

### Other

- release v0.0.4

## [0.0.4](https://github.com/OxideAV/oxideav-audio-filter/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- add `register(&mut RuntimeContext)` + adopt audio-filter factories
- with_audio_input() pre-seeds ports before first push
- implement StreamFilter::reset() for seek barriers
- ride the audio time_base for A/V sync
- proper scrolling-waterfall render before buffer fills
- implement StreamFilter with rolling-scroll 30 fps
- emit VideoFrame, drop png dep
- bump png 0.17 → 0.18
- drop Cargo.lock — this crate is a library
- bump to oxideav-core 0.1.1 + codec 0.1.1
- bump oxideav-core + oxideav-codec deps to "0.1"
