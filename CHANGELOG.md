# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
