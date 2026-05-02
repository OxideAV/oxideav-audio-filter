# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
