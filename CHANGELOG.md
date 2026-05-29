# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- round 181: hysteresis + soft-knee upgrades for the existing `noise_gate`
  filter. New constructor `NoiseGate::with(open_db, close_db, knee_db,
  attack_ms, release_ms, hold_ms)` exposes the two-threshold + smooth-knee
  parameters. **Hysteresis**: the gate opens only when drive rises above
  `open_db` and re-closes only when drive falls below `close_db` (must
  satisfy `close_db ≤ open_db`; an inverted spec is clamped to
  `close_db = open_db` = no hysteresis); the latch is sticky inside the
  band, eliminating the chatter the original single-threshold gate emits
  when the signal dances around the threshold. **Soft-knee**: `knee_db`
  widens the transition into a Hermite-smoothstep region (C¹-continuous,
  midpoint = 0.5 at the threshold centre); `knee_db = 0.0` reproduces the
  original hard step exactly. Both upgrades are opt-in — the legacy
  `NoiseGate::new(threshold_db, attack_ms, release_ms, hold_ms)`
  constructor still builds a hard-knee single-threshold gate with
  `open_db = close_db = threshold_db` and `knee_db = 0`, so existing
  call-sites are byte-for-byte unaffected. The registry `noise_gate`
  factory gains optional `hysteresis_db` (default 6 dB when present),
  explicit `close_db` override, and `knee_db` keys; specs that omit all
  three route through the legacy constructor unchanged. Also adds
  `NoiseGate::reset()` (clear latch + envelope without rebuilding
  sample-rate coefficients) and `thresholds_db()` / `knee_db()`
  accessors. 8 new unit tests: hard-knee step verification, smoothstep
  monotonicity + midpoint, hysteresis prevents chatter (drive between
  thresholds → gate stays closed), hysteresis latch holds open through
  a dip into the hysteresis band, soft-knee gain ≈ 0.5 at knee centre,
  invalid `close_db > open_db` clamps to zero hysteresis, legacy `new`
  constructor preserves binary-step behaviour, `reset` returns the
  gate to closed state. Existing `quiet_signal_is_attenuated` and
  `loud_signal_passes_through` tests stay green unchanged.

- round 174: one new filter family — `expander` (proportional downward
  expander, the under-threshold mirror of the existing `compressor`).
  Static curve `gr_db = -(R - 1) · max(0, threshold_db - env_db)` for
  finite `ratio`, with an optional soft-knee width `W` blending the
  unity-gain and slope segments via the quadratic
  `gr_db = -(R - 1) · (W/2) · ((W/2 - over)/W)²` for `over ∈ (-W/2,
  +W/2)`. `ratio = 1.0` collapses to identity (bypass);
  `ratio = f32::INFINITY` is the brick-wall downward-gate limit
  case (signal under the lower-knee edge multiplied by zero, with a
  quadratic in-knee fade preserving continuity). Same one-pole
  envelope follower as `compressor` (`α = exp(-1 / (τ · fs))`, peak-
  linked across channels so the stereo image is preserved). Distinct
  from `noise_gate`: the latter is a *binary* device that ramps the
  output gain between `0.0` and `1.0` via attack/hold/release timing,
  whereas an expander is *proportional* — a `-50 dBFS` signal with
  `-40 dBFS` threshold and `2:1` ratio gets exactly `10 dB`
  attenuation, while a `-60 dBFS` signal gets `20 dB`, fading
  gracefully into silence as the input falls further. Constructors:
  `Expander::new(threshold_db, ratio, attack_ms, release_ms, knee_db,
  makeup_gain_db)`, `Expander::downward(t, r, atk, rel)` (hard knee +
  unity make-up), `Expander::gate(t, atk, rel)` (`ratio = ∞`).
  Registered in `registry::register` as `"expander"` accepting JSON
  `threshold_db` / `ratio` / `attack_ms` / `release_ms` / `knee_db` /
  `makeup_gain_db` knobs; default `{-40, 2:1, 5 ms, 50 ms, 0 dB
  knee, 0 dB makeup}` is a sensible general noise-floor management
  preset. 10 hand-derived unit tests: above-threshold pass-through
  (≤ 0.5 dB residual); steady-state `-10 dB` reduction at `2:1` with
  `10 dB` under-shoot; steady-state `-40 dB` reduction at `3:1` with
  `20 dB` under-shoot; `ratio = 1` identity; `ratio = ∞` hard-gate
  silence; soft-knee continuity (closed-form middle-of-knee + lower-
  knee-edge + well-below-knee values match the analytic formula,
  monotonic non-increasing as drive falls); peak-linked detector
  preserves stereo image when only one channel is loud; make-up gain
  applies post-curve; parameter clamping (`ratio ≥ 1`, times ≥ 0,
  knee ≥ 0); rate invariance (44.1 kHz vs 96 kHz steady-state
  attenuation within 1.5 dB); streaming continuity (single-call vs
  split-call sample-by-sample identity within 1 µ FS). Algorithm
  derived from first principles by mirroring `compressor.rs`'s
  static curve across the threshold; no external library source
  consulted.
- round 132: one new biquad configuration — `BiquadKind::AllPass`
  (second-order all-pass / phase rotator). Analog prototype
  `H(s) = (s² − s/Q + 1) / (s² + s/Q + 1)` — numerator and
  denominator are mirror images so `|H(jω)| ≡ 1` for every analog
  frequency. Bilinear transform gives `b = (1 − α, −2cosω, 1 + α)`,
  `a = (1 + α, −2cosω, 1 − α)`; the digital numerator is the bit-
  reversal of the denominator, which preserves the flat-magnitude
  property in `z`. Phase response rotates from `0` at DC through
  `−π` at the centre frequency to `−2π` at Nyquist; `Q` sets the
  width of the phase-rotation skirt (higher `Q` → sharper sweep).
  Cookbook formula transcribed from the documented analog `H(s)`
  in our own variable names — no reference C source consulted.
  Used as a phase-alignment / decorrelation primitive in reverb
  tanks, phaser stages, and crossover phase-correction networks;
  algorithmically distinct from the 7 existing biquad kinds (LPF
  / HPF / BPF / notch / peaking / low-shelf / high-shelf), each of
  which has a frequency-dependent magnitude response. Convenience
  constructor `Biquad::all_pass(sample_rate_hz, center_hz, q)`
  matches the existing `low_pass` / `high_pass` / ... ergonomics.
  Registered in `registry::make_biquad` under JSON kind aliases
  `"all_pass"` / `"allpass"` / `"apf"` so callers can spec it from
  pipeline config. 3 hand-derived unit tests: flat magnitude at
  three probe frequencies (200 Hz passband, 1 kHz centre /
  transition, 8 kHz stopband — all within ±0.1 dB of unity gain
  with `Q=2`); phase-inversion (correlation of input vs settled
  output at `f_c` ≈ −1, confirming the `−π` phase shift at the
  centre); high-Q (`Q=50`) impulse-response numerical stability
  (every sample finite, last-quarter L² < first-quarter L²,
  L1 norm finite and bounded < 1000).
- round 106: one new filter family — `slew_limiter` (slope-limited
  smoother that bounds the per-sample output change). Per channel the
  recurrence is `Δ = x[n] − y[n−1]; y[n] = y[n−1] + clamp(Δ, −s, +s)`
  where `s = max_slew_per_sec / fs` so the same instance is stream-
  rate-agnostic (re-derived per `process` call against
  `AudioStreamParams::sample_rate`). Linear-ramp response — input
  jumps larger than `s` ramp at exactly the cap rate until the
  output catches up, then snap to bit-exact pass-through — vs the
  exponential decay of a one-pole / biquad LPF. Asymmetric variant
  (`SlewLimiter::with_asymmetric(up, dn)`) lets rise and fall caps
  differ independently (e.g. fast attack / slow release, or one-sided
  rate-limit). `with_initial_value(v)` seeds the held value so a
  spliced segment doesn't ramp up from zero. Classic anti-zipper /
  portamento glide / anti-pop primitive used in analog-style synth
  modulation smoothers and click-prone parameter changes (volume,
  pan, EQ Q-factor). Registered in `registry::register` as
  `"slew_limiter"`, with the JSON spec accepting either the symmetric
  `max_slew_per_sec` knob or the explicit `slew_up_per_sec` /
  `slew_dn_per_sec` pair (plus optional `initial`). 10 hand-derived
  unit tests (closed-form ramp at cap rate, within-budget bit-exact
  pass-through, downward step with seed, asymmetric attack/release
  independence, streaming continuity across split calls, stereo
  channel independence, zero-slew freeze, very-high-slew pass-through,
  rate-invariance of per-second slope across fs ∈ {10, 100, 1000} Hz,
  parameter clamping). Distinct from any existing filter — the LPF
  family (`biquad`, `dc_blocker`) all have exponential-decay
  smoothing; only the slew limiter gives a strictly linear ramp
  response.
- round 101: one new filter family — `hard_clipper` (memoryless
  symmetric clipping distortion: `y = clamp(drive·x, -ceiling,
  +ceiling)`). The transfer curve is piecewise linear with slope
  `drive` inside `|drive·x| ≤ ceiling` and flat `±ceiling` rails
  beyond, so it generates strong odd harmonics — the classic
  fuzz / overdrive timbre, and the limiting case approximates a
  square wave. Symmetric clipping is an odd function `f(-x) =
  -f(x)`, so a full-period sine yields zero-mean output (no even
  harmonics, no DC). Distinct from `volume` (gain then a *fixed*
  `±1.0` clip) because the ceiling is a separate configurable knob
  and drive is applied *before* the clamp; distinct from
  `tape_saturation` (smooth `tanh` knee → soft saturation). Stateless
  / memoryless: no state across samples or channels. `drive` clamped
  to `[0, 64]`, `ceiling` clamped to `[1e-6, 1.0]` (kept strictly
  positive so the curve never collapses to constant zero). Registered
  in `registry::register` as `"hard_clipper"` and wired through the
  standard `AudioFilter` contract. 7 hand-derived unit tests
  (closed-form transfer-curve match for drive=2 / custom-ceiling
  clamp / unity pass-through / output-bounded-by-ceiling with
  rail-reached check / symmetric-clip zero DC / stereo channel
  independence / parameter clamping).
- round 92: one new filter family — `ring_modulator` (sine-carrier
  double-sideband suppressed-carrier amplitude multiplier; the
  product-to-sum identity `sin(2π f t) · cos(2π fc t) =
  ½·[sin(2π(f+fc)t) + sin(2π(f−fc)t)]` shows a single input tone is
  mapped to mirror sidebands at `|f − fc|` and `f + fc` of half
  amplitude each, with the carrier itself suppressed because a full
  period of `cos(2π fc t)` has zero mean. `f64` phase accumulator
  preserves precision across long streams; phase wraps to `[0, 2π)`
  per sample. `mix = 0` is a bit-exact bypass; `carrier_hz = 0`
  reduces the carrier to `cos(0) = 1` so output equals input.
  Registered in `registry::register` as `"ring_modulator"` and
  wired through the standard `AudioFilter` contract). Distinct
  from `tremolo` (sub-audible LFO; same DSP but audibly different
  because `fc < 20 Hz`) and `freq_shifter` (Hilbert-FIR SSB shifter
  that cancels the lower sideband). 9 hand-derived unit tests
  (carrier-shape match against `cos(n·π/4)` at fs=8 kHz fc=1 kHz,
  half-mix dry/wet superposition, streaming continuity across
  successive `process` calls, stereo phase coherence, zero-mean
  over one carrier period, parameter clamping).
- round 81: four new filter families — `transient_designer`
  (two-envelope fast/slow detector — `α = 1 − exp(−1/(τ·fₛ))` —
  with attack-factor `max(0, env_fast − env_slow)/env_slow` and
  sustain-factor `max(0, env_slow − env_fast)/env_slow` driving the
  per-sample gain; SPL-Transient-Designer-style time constants:
  1 ms fast / 35 ms slow with 10× release; gain clamped to `[0, 8]`),
  `ducker` (internally-keyed sidechain compressor: fast 1 ms detector
  + dB-domain static curve with slope `(ratio − 1)/ratio` + separate
  attack/release gain trajectory + safety-floored `max_reduction_db`
  — broadcast voice-over duck; optional `key_channel` to key off a
  single channel), `gain_normalizer` (slow programme-level AGC: long-
  window RMS² detector → dB-domain error → smoothed gain trajectory
  with `max_gain_db` / `max_atten_db` clamps and silence-freeze
  integrator to prevent wind-up during pauses), and `freq_shifter`
  (true single-sideband Hilbert-FIR frequency shifter: windowed-sinc
  Hilbert kernel `h[n] = 2/(π·n) · blackman[n]` for odd `n`, zero
  taps for even `n`; per-channel ring buffer + SSB combine
  `y = r·cos(ωΔ·t) − q·sin(ωΔ·t)` — adds a constant `Δf` in Hz to
  every spectral component, harmonic-destroying ring-mod-style
  effect). All four registered in `registry::register` and wired
  through the standard `AudioFilter` contract.
- round 72: four new filter families — `exciter` (HPF +
  `tanh(k·x)/tanh(k)` waveshaper added on top of the dry signal;
  the saturator generates harmonics in the high band only and is
  drive-normalised so peak gain stays bounded), `multiband_compressor`
  (three-band Low/Mid/High parallel Butterworth-2 split at 250 Hz +
  2500 Hz with an independent `Compressor` per band — each band's
  `threshold / ratio / attack / release / knee / makeup` is
  configurable via `BandSettings`), `stereo_imager` (frequency-
  dependent stereo widener: M/S encode, split the side channel into
  low + high bands with two Butterworth-2 biquads, apply different
  width multipliers per band, recombine — classic "mono bass + wide
  treble" mastering preset), and `talkbox` (LFO-morphed parallel
  formant band-pass bank: 6 vowel presets `{Ah, Eh, Ee, Oh, Oo, Uh}`
  with two formants each from the Hillenbrand dataset, log-frequency
  interpolation between `from` and `to` vowels driven by a sine LFO,
  Q≈8 BPF realisation — no carrier required, no FFT). All four
  registered in `registry::register` and wired through the standard
  `AudioFilter` contract.
- round 6: six new filter families — `mid_side` (explicit L/R ↔ M/S
  transcoder, stateless `M = (L+R)/2 / S = (L-R)/2` with bit-exact
  roundtrip), `envelope_follower` (one-pole peak / RMS amplitude
  detector, pass-through with `current()` / `current_db()` query
  hooks), `de_esser` (split-band downward compressor: LPF/HPF
  Butterworth pair at the configurable split point + per-band peak
  detector + hard-knee compression curve on the high band only),
  `wah` (Cry-Baby-style LFO-swept resonant band-pass with
  logarithmic centre interpolation between `f_min` and `f_max`),
  `octave_doubler` (Tycobrahe-Octavia-style full-wave-rectifier
  octave-up layer with one-pole DC block on the rectified path),
  and `adaptive_noise_gate` (self-learning noise gate with
  asymmetric one-pole floor tracker — 64× faster downward than
  upward — driving a margin-thresholded open/close decision). All
  six registered in `registry::register` and wired through the
  standard `AudioFilter` contract.
- round 5: six new filter families — `vibrato` (LFO-modulated
  fractional-delay pitch shift, pitch counterpart to `tremolo`),
  `auto_pan` (LFO-modulated L/R placement with conservative
  mono-sum pan law), `bitcrusher` (bit-depth quantisation +
  sample-and-hold rate reduction), `tape_saturation` (`tanh`
  soft-clip waveshaper with asymmetric drive — odd + even
  harmonics), `hum_filter` (cascaded narrow notches at line-mains
  fundamental + harmonics; `eu_50()` / `us_60()` presets), and
  `crossover` (two-way LPF/HPF Butterworth-2 band split; output
  frame carries `2× input channels` with low band first, high
  band second). All six registered in `registry::register` and
  wired through the standard `AudioFilter` contract.
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
