# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- round 368: `compressor` **feedback detector topology**, grounded in the
  "Design" section of
  `docs/audio/filter/wikipedia-dynamic-range-compression.html`
  (feed-forward vs feedback layout — "earlier designs … measured the
  signal level after the amplifier"). New `DetectorTopology { FeedForward,
  Feedback }` enum, orthogonal to the existing peak/RMS `EnvelopeMode`
  sensing. Feed-forward (default) senses the input; feedback senses the
  previous output `y[n-1]` (one-sample loop delay), a self-stabilising
  loop that settles on a softer, program-dependent gain-reduction curve
  (vintage opto / vari-mu character). `Compressor::with_topology`
  constructor + `topology()` accessor; `new()` / `with_detector()` keep
  the feed-forward default byte-for-byte; `prev_out` carry keeps `y[n-1]`
  continuous across block boundaries. Registry `topology` JSON key
  (`"feedforward"` default / `"feedback"` / `"fb"`). Seven new tests:
  defaults, param preservation, below-threshold identity, feedback
  reduces less than feed-forward, self-consistent loop fixed point,
  block-boundary continuity, and registry key parsing.

- round 352: `resample` band-limited-interpolation milestone, grounded in
  `docs/audio/filter/jos-theory-of-sample-rate-conversion.html` /
  `jos-bandlimited-interpolation.html` / `jos-resample.html`. Added a
  closed-form `prototype_response_db(freq_hz)` DTFT evaluator of the
  windowed-sinc anti-aliasing prototype (0 dB DC reference) plus
  `passband_edge_hz`, `up_factor`, `down_factor`, `src_rate`, `dst_rate`
  accessors (retiring the `#[allow(dead_code)]` `dst_rate` field). Seven
  new property tests assert the design directly: `up·src = down·dst = L`
  with coprime `up`/`down`, passband flat to ±0.05 dB out to 0.8·edge,
  exactly −6.02 dB at the band edge, ≥ 70 dB stop-band rejection by
  1.2·edge, monotone transition band, end-to-end anti-aliasing
  (above-new-Nyquist tone rejected ≥ 40 dB on a 48 k→16 k downsample),
  and DC preservation through integer interpolation.

### Changed

- round 352: `resample` anti-aliasing prototype length now **scales with
  `max(up, down)`** (spanning a fixed number of sinc zero-crossings of the
  design cutoff) instead of a fixed 32-tap window. A short fixed window
  produced a wide transition under decimation that leaked aliasing —
  measured ≈ −13 dB rejection at 48 k→16 k (an above-Nyquist tone folding
  back into band); the ratio-scaled length restores ≥ 80 dB prototype
  stop-band rejection and ≥ 40 dB end-to-end alias rejection while
  preserving the existing round-trip and output-rate behaviour. The
  per-output history depth (`taps_per_phase`) became a runtime field.

- round 336: `parallel_compressor` — parallel ("New York" / "Motown")
  compression, grounded in the "Parallel compression" section of
  `docs/audio/filter/wikipedia-dynamic-range-compression.html`
  ("Inserting the compressor in a parallel signal path is known as
  parallel compression. It is a form of upward compression … Combining
  a linear signal with a compressor and then reducing the output gain
  of the compression chain results in low-level detail enhancement
  without any peak reduction; the compressor significantly adds to the
  combined gain at low levels only."). The dry input is split: one copy
  passes through untouched, the other through a soft-knee compressor,
  and the two are summed (`y = x·dry_lin + comp(x)·wet_lin`). Reuses the
  exact static soft-knee curve, one-pole attack/release follower, and
  selectable peak/RMS detector of `Compressor`, peak-linked across
  channels so both paths' stereo image is preserved.
  `ParallelCompressor::new` gives unity dry/wet trims + peak detector;
  `with_mix` exposes `dry_db` / `wet_db` trims (the inner make-up gain
  is folded into `wet_db`) + detector mode. Registry entry
  `"parallel_compressor"` accepts JSON `threshold_db` / `ratio` /
  `attack_ms` / `release_ms` / `knee_db` / `dry_db` / `wet_db` /
  `detector` keys. 9 new unit tests: dry-only passthrough (wet muted →
  bit-identical), quiet-passage +6 dB lift from the un-reduced wet path,
  loud peak gains markedly less than a quiet passage (peaks preserved),
  `wet_db = -6` halves the lift to +3.52 dB, `dry_db = +6` doubles the
  dry path, peak-linked detector across channels, unity-trim/peak
  defaults, empty-block no-op, sample-rate-change coefficient rebuild.
- round 329: `compressor` peak / RMS detector — added a selectable
  sidechain sensing mode to `Compressor`, grounded in the "Peak vs RMS
  sensing" section of
  `docs/audio/filter/wikipedia-dynamic-range-compression.html` ("Some
  compressors apply a power measurement function (commonly root mean
  square or RMS) on the input signal before comparing its level to the
  threshold. This produces a more relaxed compression that more closely
  relates to human perception of loudness."). The new
  `Compressor::with_detector(…, EnvelopeMode::Rms)` runs the existing
  one-pole attack/release follower on the squared drive `max(x_0², …)`
  and reports `√env`, so compression is driven off the power
  (root-mean-square) level rather than the rectified peak; reusing the
  crate's existing `EnvelopeMode { Peak, Rms }` enum (shared with
  `EnvelopeFollower`). `Compressor::new` is unchanged and defaults to
  `EnvelopeMode::Peak` — legacy peak-sensing behaviour is byte-for-byte
  identical. Registry `make_compressor` gained a `"detector"` JSON key
  (`"peak"` default / `"rms"`), mirroring the envelope-follower `"mode"`
  convention. 4 new unit tests: detector defaults to `Peak`; RMS mode
  reproduces the peak detector exactly on a DC drive (no peak-vs-RMS gap
  there); RMS mode settles onto the sine's true RMS level (`−3.01 dB`
  below the peak) so a sine whose RMS sits 12 dB over the threshold gets
  the expected `−9 dB` 4:1 reduction; plus a `detector()` getter.

- round 324: `upward_expander` — the fourth and final quadrant of the
  dynamic-range-processor taxonomy described in
  `docs/audio/filter/wikipedia-dynamic-range-compression.html`
  ("upward expansion makes the louder sounds above the threshold even
  louder"). Boosts signal **above** `threshold_db` by `(ratio − 1)` of
  each dB of over-shoot (`boost_db = min(range_db, (R − 1)·over)`),
  leaving signal below the threshold untouched — widening the dynamic
  range from the top (re-opens flattened transients, accentuates
  crescendos). The above-threshold mirror of `upward_compressor`'s
  below-threshold boost: same one-pole peak-linked detector (shared
  attack/release, channel-linked by peak), same C¹ soft-knee quadratic
  blend reflected across the threshold, same `range_db` cap (default
  `12 dB` via `UpwardExpander::upward`) so an unbounded slope cannot
  drive the loudest peaks past full-scale. `ratio = 1.0` or
  `range_db = 0` → identity. Completes the four quadrants:
  `compressor` (reduce above), `upward_compressor` (boost below),
  `expander` (attenuate below), `upward_expander` (boost above). 13
  unit tests (static-curve slope at 1.5:1 / 2:1, range cap, identity
  cases, soft-knee continuity + C¹ slope match at the upper edge,
  monotonicity, linked-detector stereo behaviour, rate invariance,
  split-call streaming continuity, and the headline range-widening
  programme test). Registry entry `"upward_expander"` accepts JSON
  `threshold_db` / `ratio` / `attack_ms` / `release_ms` / `knee_db` /
  `range_db` keys.

- round 313: `window` — two new polynomial / B-spline analysis windows
  added to the `Window` catalogue, transcribed from
  `docs/audio/filter/wikipedia-window-function.html` (§ "Other
  polynomial windows" / § "B-spline windows"): `Welch` (a single
  parabolic section `w[n] = 1 − ((n − N/2)/(N/2))²` on the `0 ≤ n ≤ N`
  convention; nulls both endpoints exactly, peaks at unity at the
  centre — the canonical window of Welch's periodogram-averaging PSD
  estimate) and `Parzen` (the 4th-order B-spline / de la Vallée Poussin
  window, the smoothest piecewise-cubic polynomial taper, defined
  zero-phase on `|m| ≤ L/2` with `m = n − N/2` by the staged two-segment
  cubic; endpoints `2·(1/L)³`). Both reuse the existing `value` /
  `generate` / `coherent_gain` / `equivalent_noise_bandwidth` API.
  Completes the polynomial B-spline family (Triangular = 1st order,
  Welch ≈ 2nd, Parzen = 4th), whose ENBW widens monotonically with
  order: rectangular 1.0 < Welch ≈ 1.20 < Triangular ≈ 1.34 <
  Parzen ≈ 1.92. 5 new unit tests (Welch parabola values + concavity
  + non-negativity, Parzen centre/endpoint formula + smoothness, and
  the polynomial-B-spline ENBW ordering). Pure utility — not an
  `AudioFilter`, no registry entry.
- round 305: `frac_delay` — new `FracDelayLine` DSP primitive providing
  selectable-kernel reads of a per-channel ring buffer at arbitrary
  fractional (between-sample) delays. This is the bandlimited-
  interpolation problem from `docs/audio/filter/`
  (`jos-bandlimited-interpolation.html`,
  `jos-theory-of-sample-rate-conversion.html`, `jos-resample.html`):
  reconstruct `x(p)` for non-integer `p` from samples bandlimited to
  `Fs/2`. Four `Interp` kernels: `Linear` (two-tap blend), `Hermite`
  (four-tap Catmull–Rom cubic), `Lagrange(n)` (order-`n` polynomial via
  the product Lagrange basis `Lⱼ(x)=Πₘ≠ⱼ(x−xₘ)/(xⱼ−xₘ)`, `n=1`≡linear,
  clamped to `[1, 8]`), and `Sinc { half_taps, beta }` (windowed-sinc
  reconstruction `Σ x[k]·sinc(p−k)`, Kaiser-tapered via
  `window::Window::Kaiser`, normalised for unity DC gain, `half_taps`
  clamped to `[1, 32]`). Edge-extension clamp on the newest boundary;
  `max_delay()` reports the kernel-reach-aware safe delay. Pure utility
  (not an `AudioFilter`, no registry entry), mirroring the `window`
  catalogue. Consolidates the two-tap linear reads currently hand-rolled
  in `chorus` / `flanger` / `vibrato`. 14 unit tests: integer-delay
  exactness (all kernels), linear midpoint average, DC reconstruction
  (all kernels), order-3 Lagrange exactness on a quadratic,
  `Lagrange(1)`≡`Linear`, Hermite endpoint pass-through, sinc beating
  linear ≥4× on an 8 kHz sine at worst-case `f=0.5`, channel
  independence, under-filled / negative / non-finite handling, reset,
  parameter clamping, kernel-reach `max_delay` ordering.
- round 299: `upward_compressor` — new `UpwardCompressor` filter
  completing the four-quadrant dynamic-range-processor taxonomy
  (`docs/audio/filter/` dynamic-range reference). Boosts signal
  *below* the threshold by `(1 − 1/R)` of each dB of under-shoot,
  capped at `range_db`, leaving peaks above the threshold untouched
  (narrows dynamic range from the bottom while preserving transients).
  `ratio = 1` / `range_db = 0` → identity; `ratio = ∞` lifts quiet
  signal up to the threshold; soft-knee C¹ quadratic blend and one-pole
  peak-linked detector mirror `compressor` / `expander`. Registry entry
  `"upward_compressor"` (`threshold_db` / `ratio` / `attack_ms` /
  `release_ms` / `knee_db` / `range_db` JSON keys). 13 unit tests:
  above-threshold pass-through, 2:1 / 4:1 / ∞:1 boost depth, range cap,
  ratio-1 + range-0 identity, soft-knee continuity + monotonicity,
  linked-detector stereo image preservation, parameter clamping,
  rate-invariance, split-call streaming continuity, and a
  two-level-programme dynamic-range-narrowing test.
- round 292: `window` — new FIR analysis-window catalogue module
  (`WindowFunction` re-export) gathering the full closed-form window
  family from `docs/audio/filter/wikipedia-window-function.html` in
  one reusable place: `Rectangular`, `Triangular` (Bartlett), `Hann`,
  `Hamming` (optimal `0.53836 / 0.46164`), `Blackman` (classic
  `α = 0.16`), `BlackmanExact` (rational `7938/18608 …`), `Nuttall`,
  `BlackmanNuttall`, `BlackmanHarris`, `FlatTop` (five-term cosine
  sum, partially negative), `Sine` / cosine, `Lanczos` (sinc), and the
  three parameterised adjustable windows `Gaussian(σ)`, `Tukey(α)`,
  `Kaiser(β)` (sharing a local `f64` `I₀` Bessel series). General
  alternating cosine-sum evaluation `w[n] = a0 − a1·cos(2πn/N) +
  a2·cos(4πn/N) − …` on the symmetric `N = L − 1` convention (matching
  the spectrogram's existing `denom = n − 1`). Per-window
  `value(n, len)` / `generate(len)` / `generate_f32(len)` plus
  `coherent_gain` and `equivalent_noise_bandwidth` (ENBW in DFT bins)
  metrics. 51 unit tests against worked values: exact endpoint /
  centre coefficients for every cosine-sum window, Tukey `α = 0 ≡`
  rectangular and `α = 1 ≡` Hann, Kaiser `β = 0 ≡` rectangular,
  `I₀(1) ≈ 1.26607` / `I₀(2) ≈ 2.27959` reference Bessel values,
  flat-top negativity, full-catalogue symmetry, and the ENBW ordering
  `rect (1.0) < Hann (→1.5) < Blackman < flat-top`.
- round 284: `biquad` — completed the staged EQ-cookbook catalogue
  (`docs/audio/filter/audio-eq-cookbook.html`), growing the family
  from eight to eleven configurations. New `BiquadKind` variants:
  `BandPassConstantPeak` (numerator `(α, 0, −α)` — exactly 0 dB at
  the centre frequency for every `Q`, so `Q` is a pure bandwidth
  knob; the existing constant-skirt `BandPass` peaks at
  `20·log10(Q)` dB), plus `LowShelfSlope` / `HighShelfSlope`
  parameterised by the cookbook shelf slope `S` via
  `α = (sinω/2)·√((A + 1/A)(1/S − 1) + 2)` instead of `Q`
  (`S = 1` → the steepest monotonic transition, identical to
  `Q = 1/√2` for any gain; `S > 1` steepens further with response
  overshoot; out-of-range `S` for the chosen gain is clamped to keep
  `α` real). New convenience constructors
  (`Biquad::band_pass_constant_peak` / `low_shelf_slope` /
  `high_shelf_slope`) and a closed-form
  `Biquad::magnitude_response_db(freq_hz, sample_rate_hz)` evaluator
  (`|H(e^{jω})|` straight from the compiled coefficients — pure
  function, no state touched) for response plotting and analytic
  design assertions. Registry kinds `band_pass_0db` (aliases
  `band_pass_constant_peak` / `bpf0`), `low_shelf_slope` /
  `high_shelf_slope` (JSON `slope` key, default 1.0) on both the
  `biquad` and `equalizer` factories; `equalizer` bands also accept
  `all_pass` now. Nine new frequency-response assertion tests pin
  the design algebra at ≤ 1e-9 dB: constant-peak BPF unity at fc for
  `Q ∈ {0.3, 1/√2, 2, 8, 32}`; skirt-vs-peak separation exactly
  `20·log10(Q)` dB at every probe frequency; `S = 1` shelf ≡
  `Q = 1/√2` shelf across gains ±6 / ±15 dB; shelf midpoint gain
  exactly `gain_db/2`; DC/Nyquist plateaus within 0.01 dB; `S = 1`
  monotonic over a 1/12-octave 20 Hz–20 kHz sweep while `S = 2`
  overshoots (measured +1.38 dB over the +12 dB shelf top and
  −1.38 dB under unity); peaking analytic-vs-recurrence cross-check
  (exactly +6 dB at fc analytically, sine-measured within 0.6 dB).

- round 280: `dither` — word-length-reduction requantizer with TPDF /
  RPDF dither and first- / second-order error-feedback noise shaping.
  The transparency-grade end-of-chain primitive for a float pipeline
  feeding a fixed-point encode: rounds every sample onto the exact
  `bits`-wide signed mid-tread code grid (`Δ = 2^(1-bits)`, `bits`
  clamped to `[2, 24]`, default 16 → every output is an exact 16-bit
  PCM code) while decorrelating the rounding error from the
  programme. Bare rounding error is a deterministic function of the
  input — harmonic distortion on low-level periodic material and a
  deadband that erases any sine under `Δ/2` outright. Non-subtractive
  dither fixes this per the classical moment analysis (Schuchman
  condition on the dither's characteristic function): RPDF (uniform,
  peak-to-peak `Δ`) zeroes the error *mean* for every input but
  leaves the variance signal-dependent (noise modulation); TPDF
  (triangular, peak-to-peak `2Δ`, sum of two RPDF draws — the
  default) renders mean *and* variance signal-independent at a
  constant total `Δ²/4` (`Δ²/12` quantisation + `2·Δ²/12` dither,
  i.e. +4.77 dB over bare rounding). Optional error feedback
  `v[n] = x[n] - c₁e[n-1] - c₂e[n-2]` shapes the noise through
  `NTF = 1 - C(z)` while the signal passes untouched: first order
  (`c = [1]`, `NTF = 1 - z⁻¹`, `|NTF|² = 4sin²(ω/2)`) tilts the
  noise +6 dB/oct with a DC zero at total power ×2; second order
  (`c = [2, -1]`, `NTF = (1 - z⁻¹)²`, `|NTF|² = 16sin⁴(ω/2)`)
  doubles the DC zero at total power ×6, pushing noise out of the
  ear's sensitive low/mid band into the top octave. Dither PRNG is
  the same splitmix64 as the noise generators, seedable for
  bit-reproducible output; per-channel feedback state, channels draw
  mutually independent dither. Distinct from `bitcrusher` (creative
  degradation: bare quantisation + aliasing sample-and-hold, no
  dither, no shaping). Registry entry `"dither"` accepts JSON `bits`
  / `mode` (`"none"` / `"rpdf"` / `"tpdf"`) / `shaping` (`"off"` /
  `"first"` / `"second"`) / `seed` keys. 14 hand-derived unit tests:
  exact code-grid + code-range membership under dither + 2nd-order
  shaping, bare-rounding `Δ/2` / RPDF `Δ` / TPDF `3Δ/2` error
  bounds, TPDF zero-mean error, the deadband test pair (0.4Δ sine →
  exact silence undithered, fundamental FFT bin survives ≈ 25× above
  the noise floor with TPDF), Parseval check of measured error power
  against the closed-form NTF gains (×1 / ×2 / ×6), 2nd-order
  spectral tilt (low band < 0.35×, top band > 2× vs flat),
  full-scale code clamps (`-1` exact, `+1 → 1 - Δ`), bits clamp,
  seed determinism, per-channel dither independence, and bit-exact
  streaming continuity across a frame split.
- round 272: `stereo_balance_meter` — pass-through observer reporting
  the left / right *energy* balance of a stereo signal over a sliding
  rectangular window. The textbook normalised level-difference scalar
  `B = (R_rms - L_rms) / (R_rms + L_rms) ∈ [-1, +1]` says where the
  stereo energy sits: `B = 0` for a centred image (equal-energy
  channels — mono panned dead-centre or a symmetric bed), `B = -1`
  when all energy is on the left (right silent), `B = +1` when all
  energy is on the right, `+1/3` when the right channel is twice as
  loud as the left, `-1/3` for the mirror case. This is the *level*
  complement to `stereo_correlation_meter`: correlation reports the
  inter-channel *phase* relationship and is mean-centred and
  scale-invariant, so it is blind to a level imbalance — two
  perfectly correlated channels at `+12 dB` / `-12 dB` still read
  `ρ = +1`. Balance reports exactly the dimension correlation throws
  away, and the two meters together pin down both axes of a stereo
  image (phase + level) on a shared time axis. A persistent non-zero
  balance flags an accidental pan offset, a channel-trim mismatch in
  the capture chain, one dead or intermittent channel, or a mono
  source mis-routed to a single leg of a stereo bus. Algorithm:
  per-channel ring of `N` samples plus an `f64` running
  sum-of-squares `Q = Σ x²` updated incrementally (`Q ← Q + x_new² -
  x_old²`) at `O(1)` per sample; windowed RMS is `sqrt(Q / N)` and the
  balance follows in closed form. Periodic per-window rebuild of both
  sums bounds `f64` round-off drift on long streams (same cadence as
  `crest_factor_meter` / `stereo_correlation_meter`). Stereo input
  only — mono and multichannel (channel count not equal to two)
  layouts pass through unchanged with the meter state untouched.
  Window default `400 ms` (matching the EBU R128 short-term loudness
  window the other R128-aligned meters default to), clamped to
  `[0.1, 10_000] ms` and additionally to `[1, 192_000]` samples.
  Until the window first fills the readout returns the neutral `0.0`
  (centred); when both channels are bit-exact silent over the window
  the balance is undefined and likewise reads `0.0`. Consumers poll
  `current()` (balance), `rms_left()` / `rms_right()` (per-channel
  windowed RMS), `max_abs()` / `reset_max()` (running `|balance|`
  high-water mark), and `samples_seen()`; `reset()` wipes all state.
  Registry entry `"stereo_balance_meter"` accepts JSON `window_ms`
  key. 19 hand-derived unit tests cover pass-through byte
  preservation, warm-up neutral-zero, the four canonical readings
  (centred / hard-left `-1` / hard-right `+1` / `±1/3`), per-channel
  RMS-equals-constant-level cross-check with derived balance,
  bit-exact-silence neutral reading, the correlated-but-unequal-level
  case that distinguishes balance from correlation, channel-swap sign
  flip, mono pass-through with untouched state, reset / reset_max
  separation, streaming continuity (one call = two halves),
  long-stream round-off drift bounded by the periodic rebuild,
  construction-time window clamp, sample-rate-resolved window at
  16 / 48 / 96 kHz, and sample-rate-change re-derivation.
- round 263: `dc_offset_meter` — pass-through observer reporting the
  per-channel running mean (DC component) of the signal over a
  sliding rectangular window. The textbook scalar `mean = (1/N) · Σ
  x[n]` quantifies any bias the signal sits on top of, exposed both
  in linear amplitude (`current()` / `per_channel()`) and as
  `20·log10(|mean|)` dB (`current_db()`). Where `dc_blocker` *removes*
  the DC component with a single-pole HPF, this meter *reports* it
  without altering the signal — useful for diagnosing preamp bias
  trim drift, ADC quantiser-midpoint offsets, sagging field-recorder
  rails, accidental unipolar oscillators pushed through to the
  output bus. A non-zero mean leaves the speaker cone parked off
  centre, wastes a chunk of available headroom on a constant —
  inaudible — push, and starves transient peaks of the linear range
  they would otherwise have used. Algorithm: per-channel `f32` ring
  of `N` samples plus an `f64` running sum `S = Σ x` updated
  incrementally on every sample (`S ← S + x_new − x_old`); cost is
  `O(1)` per sample, no sort, no deque, just the ring rotation.
  Periodic per-window rebuild of `S` from the ring contents bounds
  `f64` round-off drift on long streams (same cadence as
  `crest_factor_meter` and `stereo_correlation_meter`).
  Channel-link picks the per-channel mean with largest `|·|`, *sign
  preserved* — so a `+0.05` channel and a `-0.20` channel report
  `-0.20`, and equal-and-opposite biases (`+0.1` / `-0.1`) do not
  cancel in the readout. Window default `400 ms` (matching the EBU
  R128 short-term loudness window that `crest_factor_meter` and
  `stereo_correlation_meter` also default to, so the three meters
  share a time axis on a display); clamped to `[0.1, 10_000] ms`
  and additionally to `[1, 192_000]` samples. Until the window
  first fills the readout returns `0.0` (linear) /
  `f32::NEG_INFINITY` (dB) so callers can branch on "not yet
  ready"; `samples_seen()` exposes the count. `reset()` wipes all
  per-channel state; `reset_max()` clears only the running-`|mean|`
  high water mark. Registry entry `"dc_offset_meter"` accepts JSON
  `window_ms` key. 16 hand-derived unit tests cover pass-through
  byte preservation, warm-up zero/`NEG_INFINITY` semantics, constant
  DC fidelity (positive and negative), bit-exact silence, zero-mean
  integer-period sine, biased-sine DC isolation, steady-state
  invariance, reset / reset_max separation, stereo channel-link by
  largest-`|·|` with sign preserved, equal-and-opposite-bias
  non-cancellation, streaming continuity (one call = two halves),
  long-stream round-off drift bounded by the periodic rebuild,
  construction-time window clamp, first-process sample-rate
  resolution at 16 / 48 / 96 kHz, sample-rate-change re-derivation,
  and the dB readout cross-check (`20·log10(0.5) = -6.02 dB`).
- round 258: `zero_crossing_rate` — pass-through observer reporting
  the number of sign changes in the signal per unit time over a
  sliding rectangular window. The zero-crossing rate (`ZCR`) is the
  textbook scalar that counts `sign(x[n]) != sign(x[n-1])` events
  over a window of `N` adjacent-sample pairs, exposed either as the
  per-sample fraction `count / N ∈ [0, 1]` or — more usefully on an
  audio meter — in crossings per second (Hz) as `count · fs / N`.
  It is a cheap proxy for the spectral centroid: a pure sine at
  `f_0` produces `≈ 2·f_0` crossings per second (every period has
  exactly one positive-going and one negative-going crossing); an
  alternating-sign signal saturates at `fs`; a constant DC or
  bit-exact-silence signal sits at `0`. ZCR is the canonical front-end
  feature for voiced/unvoiced speech classification — voiced phonemes
  (vowels, nasals) sit at low ZCR (≤ 1500 Hz typical), unvoiced
  fricatives ('s', 'f', 'sh') push the ZCR into the multiple-kHz
  range — and is also widely used for tone-pitch proxying,
  percussion vs harmonic separation, and silence / noise-floor
  gating. Algorithm: per channel keep a ring of `N` boolean
  "crossing flags" (`true` if `sign(x[t-1]) != sign(x[t])`) plus a
  one-sample latch holding the most recent previously-seen sample.
  For every incoming sample `x[t]`: form the pair `(prev, x[t])`,
  compute `crossed = sign(prev) != sign(x[t])`, subtract the flag
  about to be overwritten from the running `count` if the flag-ring
  is already full, write the new flag in and add it to `count`,
  update the latch to `x[t]`. That's `O(1)` per sample with no `f64`
  drift bookkeeping (every increment / decrement is on a `u32`
  counter rather than a floating-point sum-of-squares). The latch
  survives across `process()` calls so streaming a long input as
  many small frames is bit-identical to a single large call. Sign
  is reduced to `{-1, +1}` with the convention `sign(0.0) = +1` so
  a run of bit-exact zeros doesn't manufacture phantom crossings
  (`f32::signum` returns `-0.0` for `-0.0` which would be exactly
  that bug). Channel-link is by `max`: a transiently noisy channel
  of a split stereo bed isn't masked by a quieter average on the
  other — the crate's convention across all observation filters.
  Window default `25 ms` (the canonical short-time speech-analysis
  frame), clamped to `[0.1, 10_000] ms` at construction and to
  `[1, 192_000]` samples (4 s at 48 kHz) after sample-rate
  resolution. Until the flag-ring has filled at least once
  (`pairs_seen < window_samples`), the readouts return `f32::NAN`
  (rate Hz), `0.0` (fraction), and `0` (count) so callers can
  branch on "not yet ready". `reset()` wipes all per-channel state
  (rings, latches, counters); `reset_max()` clears only the running
  max linear fraction. Registered in `registry::register` as
  `"zero_crossing_rate"` accepting JSON `window_ms` key. Within the
  observation family, the zero-crossing meter sits orthogonal to
  [`CrestFactorMeter`] (peak-to-RMS ratio, no crossing-rate info),
  [`TruePeakDetector`] (oversampled inter-sample peak, no
  crossing-rate info), [`LoudnessITU`] (K-weighted integrated
  loudness, insensitive to the per-sample sign sequence),
  [`EnvelopeFollower`] (smoothed peak / RMS envelope, not a
  ratio), and [`SilenceDetector`] (single binary above/below RMS
  threshold flag); the zero-crossing meter alone exposes the
  per-sample sign sequence, the spectral-centroid proxy widely used
  in speech and music classification front-ends. 17 hand-derived
  unit tests: pass-through preserves audio bytes byte-for-byte
  (observation-only contract); before the window first fills the
  readouts return `NAN` / `0.0` / `0`; constant DC yields a count
  of exactly zero (no sign changes); bit-exact silence yields zero
  (sign convention defends against the `-0.0` phantom-crossing
  bug); alternating `±0.5` saturates the count to exactly `N` and
  the rate to `fs`; a pure 1 kHz sine at 48 kHz with a window
  holding an integer number of periods reads `≈ 2 · f_0` crossings
  per second (allowing a small tolerance for the boundary-pair
  count); doubling the sine frequency ~doubles the reported rate
  across 500/1000/2000/4000 Hz; the readout is amplitude-invariant
  (only the sign sequence matters); `reset()` wipes state, `reset_max()`
  clears only the running max; stereo channel-link by max picks the
  louder-ZCR channel; one-call equivalent to two-halves call (latch
  survives the frame boundary); `window_ms` clamp at construction;
  `window_samples` resolves correctly to 25 ms at multiple sample
  rates (16/48/96 kHz); rate scales linearly with the sample rate
  on a fresh window; `current_fraction()` always lies in `[0, 1]`
  on deterministic xorshift32 noise; long-running alternating-sign
  input keeps the count locked at exactly `N` (no drift, no
  overflow, no double-counting). Test total rises from 377 to 394
  (+17).

- round 248: `comb_filter` — tunable single-tap comb filter exposing
  both the feedforward FIR `y[n] = x[n] + g · x[n − D]` and the
  feedback IIR `y[n] = x[n] + g · damped(y[n − D])` forms behind a
  single [`CombMode`] selector, with an optional one-pole low-pass
  in the feedback path (`damped(·)`). The comb filter is one of the
  fundamental DSP primitives — its transfer function is a set of
  evenly-spaced peaks / notches (the "teeth of a comb" in the
  frequency domain) — and the existing crate uses it as a building
  block of `reverb` (four parallel combs + two serial all-passes)
  and `flanger` (feedback comb with an LFO modulating the delay
  length), but never exposes the bare primitive. This round adds the
  bare primitive as `CombFilter`. The feedforward form has transfer
  function `H(z) = 1 + g · z^{-D}` with magnitude
  `√(1 + g² + 2g·cos(ωD))` — `D + 1` evenly-spaced extrema in
  `[0, π]` (`[0, fs/2]`), with `g > 0` giving peaks of `1 + g` at
  `ω = 2πk/D` and troughs of `|1 − g|` at `ω = (2k+1)π/D`, and
  `g < 0` swapping the two roles. Used in stereo widener side-paths,
  decorrelation networks, and frequency-domain dereverberation
  prefilters. The feedback form has transfer function
  `H(z) = 1 / (1 − g · z^{-D})` with magnitude
  `1 / √(1 + g² − 2g·cos(ωD))` — resonant peaks of `1 / (1 − g)` at
  `ω = 2πk/D` (resonance frequencies `f_k = k · fs / D`) with a
  `−3 dB` bandwidth `≈ (1 − g) · fs / (π · D)`, and `D` poles on a
  circle of radius `|g|^{1/D}` in the `z` plane. Stable iff
  `|g| < 1`; the constructors clamp `|g|` to `0.999` for a safety
  margin (`g = 1` is marginally stable, self-oscillates on a
  denormal). The optional damping factor `a ∈ [0, 0.999]` inserts a
  one-pole low-pass `s[n] = (1 − a)·y[n − D] + a·s[n − 1]` in the
  feedback path so high-frequency overtones decay faster than the
  fundamental — the natural plucked-string behaviour. `a = 0` is the
  bare feedback comb. [`CombFilter::karplus_strong(freq_hz, decay)`]
  picks `D = round(fs / freq_hz)` and `damping = 0.5` for the
  classic Karplus-Strong plucked-string tone; feed a short noise
  burst (e.g. from [`WhiteNoise`]) into the filter and the loop
  circulates and decays into the audible string tone. Delay can be
  specified either in exact samples (`with_delay_samples`,
  sample-rate-dependent) or in milliseconds (`with_delay_ms`,
  rate-portable — `D = round(delay_ms · fs / 1000)` resolved on the
  first `process()` call against the input stream's `sample_rate`);
  both clamp the resolved `D` to `[1, MAX_DELAY_SAMPLES = 192_000]`
  (4 s at 48 kHz). Per-channel ring buffers + per-channel one-pole
  LP state (feedback-with-damping path only); channels do not
  cross-talk. `reset()` zeros every channel's ring buffer + LP
  state without changing the configured mode or delay. `set_mode`
  swaps the recurrence while preserving the delay-line contents
  (useful for live morphing between flange and resonator
  presentations). Registered in `registry::register` as
  `"comb_filter"` accepting JSON `mode`
  (`"feedforward"` / `"fir"` / `"ff"` /
  `"feedback"` / `"iir"` / `"fb"` /
  `"karplus_strong"` / `"ks"` / `"plucked_string"`), `delay_ms` (or
  `delay_samples`), `gain`, `damping`, and — for the karplus_strong
  shortcut — `freq_hz` + `decay`. Distinct from existing delay-line
  filters: [`Echo`] (single-tap delay with wet/dry mix, used as an
  audible repetition effect with delays in the tens-to-hundreds of
  ms range), [`Flanger`] (feedback comb with an LFO modulating the
  delay length on top), and [`Reverb`] (four parallel combs + two
  serial all-passes presented collectively as a room simulator).
  15 hand-derived unit tests: feedforward impulse response matches
  the closed-form two-tap FIR (`y = 1, 0, 0, 0.5, 0, …` for
  `D = 3`, `g = 0.5`); feedback impulse response is the exact
  geometric decay `y[k·D] = g^k` for the bare (un-damped) loop;
  feedforward at `g = +1` notches the trough frequency
  `fs / (2D)` to within `0.01` after a `2·D`-sample warm-up;
  feedforward at `g = +1` doubles DC after warm-up; feedback at
  `g = 0` is bit-exact identity; `with_delay_ms` resolves to the
  right sample count at 48 kHz then re-resolves at 96 kHz on a
  rate change (`D` ratio = 2.0); Karplus-Strong tuning resolves
  `D = round(fs / freq_hz)` exactly and the loop's tail energy
  after a noise burst is non-trivial; stereo per-channel state
  isolation (impulse on left, silence on right → right stays
  bit-exact silent, left rings); streaming continuity (single
  call on a 128-sample frame ≡ two calls on two 64-sample halves,
  bit-identical within `1e-7`); feedback gain clamp at both ends
  (`±2.0` / `±3.0` requested → `±0.999` retained); delay clamp at
  both ends (zero bumped to 1, `MAX_DELAY_SAMPLES × 10` capped at
  `MAX_DELAY_SAMPLES`); `reset()` zeros the ring (zero input
  post-reset → zero output, no residual decay); `set_mode`
  preserves the delay buffer (feedforward → feedback transition
  rings on the next impulse with the expected `g^k` decay
  pattern); feedback damping reduces resonance peak energy vs a
  bare feedback comb at the same `g` (LP-in-loop drains energy
  faster); mode accessor returns the clamped value, not the raw
  user argument. Test total rises from 362 to 377 (+15).

- round 231: `stereo_correlation_meter` — pass-through observer
  reporting the windowed Pearson correlation coefficient
  `ρ ∈ [-1, +1]` between the L and R channels. The Pearson coefficient
  is a unit-free scalar that classifies the stereo image at a glance:
  `+1` for identical channels (mono content panned centre); `≈ 0` for
  orthogonal channels (uncorrelated stereo bed, ambient reverb tails,
  decorrelated chorus voices); `-1` for phase-inverted channels
  (the canonical broadcast hazard — most TV / radio chains downstream
  from the mastering bus still emit a mono sum and a programme that
  correlates strongly negative dies on the way out). Algorithm: five
  per-window incremental running sums (`Σx, Σy, Σx², Σy², Σxy`)
  updated in `O(1)` per sample pair via the standard add-new /
  subtract-old form; the windowed correlation falls out of the closed
  form `ρ = (N·Σxy − Σx·Σy) / √((N·Σx² − Σx²)·(N·Σy² − Σy²))` by
  algebraic identity, with no statistical assumption. To bound `f64`
  round-off drift on long streams the meter rebuilds all five sums
  from the active ring contents once per full window — `O(N)` every
  `N` samples, i.e. `O(1)` amortised. Polar reading
  `current_degrees() = acos(ρ)·180/π` exposes the classical
  goniometer's `0° / 90° / 180°` angular axis for direct UI display
  (perfect-mono / decorrelated / phase-inverted). Running `min()`
  latches the worst-case correlation seen since construction or last
  `reset_min`, so a transiently-inverted frame survives a quieter
  tail. Window default `400 ms` (matching `crest_factor_meter`, so
  the two readouts can share a time axis on a meter display);
  clamped to `[0.1, 10_000] ms` and additionally to `[1, 192_000]`
  samples (4 s at 48 kHz). Stereo input only — mono and multichannel
  (channel count not equal to two) layouts pass through unchanged
  with the meter state untouched (`current()` keeps its previous
  value, or stays at `0.0` if no stereo input has been seen). Pearson
  is mean-centred (DC offsets don't bias the metric) and
  scale-invariant (per-channel volume changes don't bias it). Until
  the window first fills, `current()` returns `0.0` and
  `current_degrees()` returns `90.0` (the neutral reading); the
  `samples_seen()` accessor exposes the warm-up count explicitly.
  Silent windows (either channel has zero variance) also map to the
  neutral reading rather than NaN, matching the convention of every
  other observation filter in the crate. API surface: `current` /
  `current_degrees` (snapshot at frame close), `min` / `reset_min`
  (running min over history), `samples_seen` / `window_samples` /
  `window_ms` (introspection), `reset` (wipe all state). Registered
  in `registry::register` as `"stereo_correlation_meter"` accepting
  JSON `window_ms` key (default `400.0`). Distinct from
  `stereo_widener` / `stereo_imager` (both *processors* of the stereo
  image), from the single-channel meters `crest_factor_meter` /
  `true_peak_detector` / `loudness_itu` (none of which carries
  inter-channel phase information), and from `silence_detector`
  (single-channel binary threshold flag). 19 hand-derived unit tests:
  pass-through preserves audio bytes byte-for-byte; before-window-full
  returns the neutral `0.0` reading; identical channels correlate to
  exactly `+1` (and `0°` on the goniometer); phase-inverted channels
  correlate to exactly `-1` (and `180°`); quadrature `(sin, cos)`
  channels correlate to ≈ `0` (and ≈ `90°`); silent windows return
  the neutral reading rather than NaN; asymmetric zero variance
  (one silent channel) also returns the neutral reading; DC offsets
  don't bias the metric; per-channel scaling doesn't bias the metric;
  `min` latches on the worst correlation across frames; `reset_min`
  clears only the min; `reset` wipes everything; split-call vs
  single-call streaming continuity identical within `1e-3`; mono
  input passes through without updating; `window_ms` clamp at both
  ends; sample-count clamp to `SCM_MAX_WINDOW_SAMPLES`; sample-rate
  change rebuilds `window_samples` proportionally (48 → 96 kHz
  ratio ≈ 2.0); long-stream sums stay bit-stable (the periodic
  rebuild safeguard); the per-sample clamp keeps `ρ` strictly in
  `[-1, +1]` and `current_degrees()` strictly in `[0, 180]`.
  Test total rises from 343 to 362 (+19).

- round 226: `crest_factor_meter` — pass-through observer reporting
  peak-to-RMS ratio (crest factor) in dB over a sliding rectangular
  window. The crest factor `CF = peak / rms` (or
  `20·log10(peak/rms)` in dB) is the textbook scalar that quantifies
  how "spiky" or transient-rich a signal is: a sine wave reads
  `3.01 dB`, a symmetric square wave reads `0 dB`, broadband noise
  lands near `11 dB`, heavily-compressed broadcast pop sits in the
  `5..8 dB` range, sparse drum transients can push beyond `20 dB`.
  Window defaults to `400 ms` (matching the EBU R128 short-term
  loudness window), clamped to `[0.1, 10_000]` ms; the sample-count
  form is additionally clamped to `[1, 192_000]` samples (4 s at
  48 kHz) — the cap defends against pathological per-sample
  allocations without rejecting any realistic broadcast / mastering
  window. Two running statistics drive the meter: (a) per-channel
  running sum-of-squares `S = Σ x²` updated incrementally on each
  in-and-out sample pair (`S ← S + x_new² − x_old²`), with a
  per-window rebuild from ring contents that bounds `f64` round-off
  drift on long streams; (b) per-channel monotonic-decreasing deque
  of `(|x|, sample_index)` — the classical sliding-maximum
  primitive — where the deque front always holds the active-window
  maximum, the rear pops while the incoming `|x|` dominates, and
  the front pops when its sample index falls outside
  `[n − window_samples + 1, n]`. Both run in `O(1)` amortised per
  sample (each sample is pushed and popped at most once). Channels
  are linked by `max` of the per-channel peak and `max` of the
  per-channel RMS so a loud transient on one half of a split stereo
  bed isn't masked by a quieter average on the other. Until the
  window first fills, `current_db` returns `f32::NEG_INFINITY` and
  `current_linear` returns `0.0` so callers can branch on "not yet
  ready"; the `samples_seen()` accessor exposes the warm-up count
  explicitly. Silent windows (rms = 0) also map to NEG_INFINITY,
  matching the convention of every other observation filter in the
  crate (`true_peak_detector`, `loudness_itu`). Distinct from
  `true_peak_detector` (absolute oversampled inter-sample peak only;
  says nothing about average power), `loudness_itu` (K-weighted
  integrated LUFS; says nothing about transient peaks),
  `envelope_follower` (single one-pole peak or RMS envelope; not a
  ratio), and `silence_detector` (binary above/below RMS-threshold
  flag). API surface: `current_db` / `current_linear` /
  `current_peak` / `current_rms` (snapshot at frame close);
  `max_db` / `reset_max` (running max over history);
  `samples_seen` / `window_samples` / `window_ms` (introspection);
  `reset` (wipe all state). Registered in `registry::register` as
  `"crest_factor_meter"` accepting JSON `window_ms` key (default
  `400.0`). 18 hand-derived unit tests: pass-through preserves audio
  bytes byte-for-byte; before-window-full returns NEG_INFINITY;
  DC input yields exactly `0 dB` (peak == rms); full-scale sine
  yields exactly `3.0103 dB` (`20·log10(√2)`) when the window holds
  an integer number of periods; symmetric square wave yields exactly
  `0 dB` (`|x|` constant); single transient spike on a quiet
  baseline pushes the meter above `15 dB`; silent window reports
  NEG_INFINITY; stereo peak-link picks the louder channel; running
  max latches on the loudest frame and survives a quieter trailing
  frame; `reset_max` clears only the max, leaving `current_db`
  intact; `reset` wipes everything (samples_seen → 0,
  window_samples → 0); split-call vs single-call streaming continuity
  identical within `1e-3`; window_ms clamp at both ends; sample-count
  clamp to `CFM_MAX_WINDOW_SAMPLES`; sample-rate change rebuilds
  `window_samples` proportionally (48 → 96 kHz ratio ≈ 2.0);
  `linear_to_db` helper maps `≤ 0` ratios to NEG_INFINITY (no NaN);
  long-stream sum-of-squares stays bit-stable (the per-window rebuild
  safeguard); sliding-max correctly drops a loud sample once its
  sample index expires (the monotonic-deque pop_front branch fires).
  Test total rises from 325 to 343 (+18).

- round 220: `median_filter` — non-linear sliding-window median filter
  for impulse-noise (click / pop) restoration. Per channel the filter
  maintains an `N`-sample ring buffer; every output sample is the
  **median** of the latest `N` ring contents (insertion-sorted into a
  per-`process()` scratch buffer to avoid per-sample allocation). The
  filter is non-linear — unlike every IIR / FIR filter the crate
  already ships, it does not satisfy superposition — which is exactly
  what gives it the property linear LPFs cannot achieve: it kills
  isolated impulse outliers without softening the surrounding signal.
  Step edges that span more than `window / 2` samples pass through
  unaltered; a single outlier sample on an otherwise quiet baseline
  is entirely discarded. Within the crate's restoration family it
  complements `hum_filter` (cyclic-mains denoising) and `dc_blocker`
  (DC drift removal) by targeting *transient* impulse noise instead
  of cyclic or DC content. `MedianFilter::new(window)` clamps the
  window into `[1, MedianFilter::MAX_WINDOW]` (`= [1, 257]`);
  `window = 1` is the identity (allowed for parameter-sweep
  convenience); odd windows return the central sorted sample; even
  windows return the `f64` mean of the two central sorted samples
  (a tiny low-pass smoothing on top of the median pick).
  `MedianFilter::default()` picks `window = 5`, the canonical
  click-removal default — wide enough to mask a couple of adjacent
  impulses, narrow enough to preserve transients of musical interest.
  Per-channel state (each channel keeps its own ring buffer and write
  index, so stereo input does not cross-talk through the filter);
  `reset()` zeros every ring without changing the configured window.
  Registered in `registry::register` as `"median_filter"` accepting a
  JSON `window` key. 13 hand-derived unit tests: window=1 identity;
  window clamps at zero and at `MAX_WINDOW`; `Default` is `5`;
  isolated impulse fully suppressed on a constant baseline; two
  adjacent impulses fool a 3-tap window but are killed by a 5-tap
  one (the textbook "two-impulse robustness" boundary); ramp / step
  edge preserved exactly; even-window centre-pair mean; per-channel
  no-cross-talk; `reset()` zeros the ring; streaming continuity (one
  frame ≡ two split frames with the same state path); `median_of`
  helper standalone (odd window, even window, already-sorted ring);
  S16 sample format roundtrips through the f32-internal filter.
  Insertion-sort hot path documented as the steady-state best case
  (`O(window)` on near-sorted ring contents); cap `MAX_WINDOW = 257`
  defends against pathological per-sample allocations without
  rejecting any realistic configuration.

- round 215 (depth-mode benchmarks): Criterion harness `benches/filters.rs`
  covering seven representative filters across the architectural families the
  crate ships — `biquad_lpf` (single second-order DF-II-T IIR),
  `equalizer_3band` (cascaded biquads), `loudness_itu` (BS.1770-4 K-weighting +
  RLB HPF + per-channel mean-square), `compressor` (peak-detector soft-knee
  follower), `reverb` (Schroeder 4 combs ║ 2 all-passes), `resample_44k1_48k`
  (polyphase windowed-sinc), and `true_peak_4x` (4× polyphase Kaiser-FIR
  inter-sample peak detector). Every PCM input is synthesised in-bench from a
  deterministic xorshift32 seed and fed through the public `AudioFilter::process`
  / `process_in_place` surface, so per-filter algorithm tweaks in future rounds
  (compressor follower constants, resampler polyphase taps, true-peak FIR width,
  …) have a stable per-byte throughput baseline. Criterion pinned to the `0.5`
  line the sibling audio crates (`oxideav-flac`, `oxideav-tta`) already track.
  No behavioural change to any filter; the harness is observational only.

- round 209: `pre_emphasis` + `de_emphasis` — paired analog-broadcast
  / tape / FM record EQ shelving filters. Shared `EmphasisCurve`
  family covers `Fm50us` (European FM, 50 µs), `Fm75us` (North-American
  FM, 75 µs), `J17` (ITU-R J.17 voice-band, 50 µs), `Custom { tau_s }`
  (user-specified single-time-constant), and `Riaa3180_318_75`
  (phonograph / vinyl second-order three-time-constant 3180 + 318 +
  75 µs curve). Coefficients derived in-source by bilinear transform
  of `H_pre(s) = (1 + s·τ) / (1 + s·τ/G)` and its inverse `H_de(s) =
  (1 + s·τ/G) / (1 + s·τ)`. For RIAA the second-order analog transfers
  `H_rec(s) = (1 + s·τ₁)·(1 + s·τ₃) / (1 + s·τ₂)` and `H_play(s) =
  (1 + s·τ₂) / ((1 + s·τ₁)·(1 + s·τ₃))` are bilinear-mapped to
  direct-form-I `(b₀, b₁, b₂, a₁, a₂)` coefficients, with the matching
  `(z + 1)` factor between analog numerator and denominator cancelled
  symbolically before extraction. Full derivation (DC-gain = 1 sanity
  check, Nyquist-gain = G / 1/G asymptote check, pole-location
  stability proof) is in the module headers as the central audit aid
  for clean-room provenance — no table-lifted coefficients from any
  external source. The asymptotic shelf-top gain `G` (default 10,
  clamped to [1, 1000]) bounds the discrete shelf's HF response; for
  FM emphasis the analog standard takes `G → ∞`, but a finite `G`
  whose equivalent pole `G/τ` sits above Nyquist is acoustically
  equivalent and guarantees a strictly-stable discrete filter. The
  cascade `pre_emphasis(curve, G) · de_emphasis(curve, G)` is the
  exact algebraic inverse `H_pre(z) · H_de(z) = 1`; in `f64`
  floating-point the cascade error on a splitmix64-driven broadband
  pseudo-noise probe is ≤ −60 dB RMS for FM curves and ≤ −50 dB RMS
  for RIAA. Per-channel state (one `(x_prev, y_prev)` pair for
  first-order curves, `(x_prev, x_prev2, y_prev, y_prev2)` for RIAA);
  `reset()`; `set_sample_rate` rederives coefficients. Registered in
  `registry::register` as `"pre_emphasis"` / `"de_emphasis"` accepting
  JSON `curve` (`"fm_50us"` / `"fm_75us"` / `"j17"` / `"riaa"` /
  `"custom"` + `"tau_us"` for custom) and `g` (asymptotic shelf gain)
  keys. 15 hand-derived unit tests per filter: DC gain unity
  (first- and second-order); Nyquist asymptote equals `G` (pre) /
  `1/G` (de); derived corner frequency at `f_c = 1/(2π·τ) ≈ 3183 Hz`
  for FM-50 / 2122 Hz for FM-75; +20 dB/decade slope verified between
  two decade-spaced frequencies; J.17 ≡ FM-50 first-order-identity
  check; `Custom { tau_s }` ≡ named first-order-identity check;
  per-channel state isolation; streaming continuity (split-vs-whole
  call bit-identical); sample-rate change rederives `(c, c/G)`; reset
  clears state but preserves coefficients; gain clamped at `G = 1`
  reduces to identity; RIAA qualitative bass/treble asymmetry matches
  the curve's record / playback bias. Three cascade-identity tests
  (FM-50 / FM-75 / RIAA) verify the inverse property. Test total
  rises from 280 to 310 (+30).

- round 205: `svf` — Chamberlin two-integrator-loop State Variable
  Filter (a state-space topology distinct from the existing `biquad`
  family's bilinear-transform Direct-Form-II-Transposed realisation).
  Single recurrence
  `hp[n] = x[n] - q·bp[n-1] - lp[n-1]; bp[n] = bp[n-1] + f·hp[n];
  lp[n] = lp[n-1] + f·bp[n]; notch[n] = hp[n] + lp[n]` produces all
  four canonical taps in one update, so [`SvfMode`] selects which is
  emitted without touching state. Coefficients are
  `f = 2·sin(π·f_c/f_s)` (frequency parameter) and `q = 1/Q` (damping);
  cutoff modulation is a single `sin` on `set_cutoff` with no
  pre-warping or coefficient resolve required, making this the
  canonical synth filter for envelope-/LFO-swept cutoff sweeps where
  the bilinear biquad would need to rebuild all six tap coefficients
  per sample. The discrete two-integrator loop is conditionally
  stable: clamps enforce `f_c ≤ f_s / 6.5` and `Q ∈ [0.5, 50.0]` at
  construction and on `set_cutoff` / `set_q`. The Notch tap is
  `hp + lp`, which in the discrete form degrades as Q rises (deep at
  low Q, can lift above unity gain at Q ≥ 10); the docs flag this and
  point callers wanting sharp narrow-band reject at high Q to the
  bilinear-biquad Notch instead. Constructors: `SvfFilter::new(mode,
  cutoff, q)`, `SvfFilter::low_pass / band_pass / high_pass / notch`
  (mode-shorthand factories), plus mode-/cutoff-/Q-mutating
  `set_mode` / `set_cutoff` / `set_q` and `reset()`. Registered in
  `registry::register` as `"svf"` accepting JSON `mode` (`"low_pass"`
  / `"band_pass"` / `"high_pass"` / `"notch"` with `lp` / `bp` / `hp`
  / `bs` aliases), `cutoff_hz` (or `center_hz`), and `q` keys;
  defaults `{low_pass, 1 kHz, Q = 0.707}` give a Butterworth-equivalent
  LPF. 12 hand-derived unit tests: LP pass-band ≤ 0.5 dB at one decade
  below cutoff + stop-band ≤ −20 dB three octaves above; HP mirror
  (one octave below cutoff stop ≤ −20 dB, pass-band within 1 dB at
  three octaves above); BP centre gain peaks dominant over both
  skirts at Q = 4; Notch centre cut ≥ 15 dB at Q = 0.5 with DC-band
  flatness ≤ 1 dB; mode-switch preserves integrator state; stereo
  channels do not cross-talk through the resonant loop; `reset()`
  clears state but keeps cached coefficients; `set_cutoff` defers
  recompute until first sample rate is observed; Q clamp at both ends
  of the documented range; cutoff above stability bound clamps
  internally (impulse-train probe stays bounded); streaming
  continuity across split calls is bit-identical; sample-rate change
  recomputes `f` proportionally (48 → 96 kHz ratio ≈ 0.5 to within
  2 %). Algorithm derived from first principles by matching the
  analog `H_lp(s) = 1 / (s² + s/Q + 1)` against the discrete
  two-integrator loop; classical reference *Hal Chamberlin, "Musical
  Applications of Microprocessors" (2nd ed., Hayden Books, 1985,
  ch. 19)* cited in module docs.

- round 198: `true_peak_detector` — 4× polyphase Kaiser-windowed FIR
  oversampling inter-sample peak observer (dBTP). Pass-through audio;
  exposes `current_dbtp` / `max_dbtp` / `overs` count. Distinct from
  `limiter` (sample-peak reduction), `envelope_follower` (smoothed
  envelope), and `loudness_itu` (K-weighted integrated loudness). 48-tap
  base FIR (12 taps per polyphase sub-filter) designed at construction
  via the Kaiser–Schafer empirical β formula; `f64` accumulation for
  the convolution. Recovers `≈ 0 dBTP` on the canonical `fs/4 + π/4`
  full-scale sine whose sample-peak is only `-3.01 dBFS`. New filter
  registry entry `"true_peak_detector"`.

## [0.1.2](https://github.com/OxideAV/oxideav-audio-filter/compare/v0.1.1...v0.1.2) - 2026-05-29

### Other

- fix stale +3 dB-lump claim on Butterworth2 doc variant
- add Linkwitz-Riley 4th-order (LR4) slope (round 188)
- add hysteresis + soft-knee upgrades (round 181)
- add expander — proportional downward expander (round 174)
- add second-order all-pass (APF) phase-rotator kind (round 132)
- add slew_limiter slope-bounded smoother (round 106)
- add hard_clipper memoryless distortion (round 101)
- round 92: ring_modulator — DSBSC audible-carrier amplitude multiplier
- round 81: add 4 filter families (transient_designer / ducker / gain_normalizer / freq_shifter)
- round 72: add 4 filter families (exciter / multiband_compressor / stereo_imager / talkbox)
- round 6: add 6 filter families (mid_side / envelope_follower / de_esser / wah / octave_doubler / adaptive_noise_gate)
- round 5: add 6 filter families (vibrato / auto_pan / bitcrusher / tape_saturation / hum_filter / crossover)
- round 4: add 8 filter families (chorus / flanger / phaser / equalizer / white_noise / pink_noise / brown_noise / silence_detector)
- round 3: add 6 filter families (dc_blocker / stereo_widener / reverb / tremolo / loudness_itu / pitch_shift)
- round 2: add biquad EQ + compressor + limiter filter families

### Added

- round 188: 4th-order Linkwitz-Riley (LR4) slope for the existing
  `crossover` filter. New [`CrossoverSlope`] enum
  (`Butterworth2` / `LinkwitzRiley4`) plus constructors
  `Crossover::with_slope(cutoff_hz, q, slope)` and
  `Crossover::linkwitz_riley(cutoff_hz)`, and a `slope()` accessor. LR4 is
  built as **two cascaded Butterworth-2 sections** (`q = 1/√2`) per band,
  giving a 24 dB/oct split that is −6 dB at the crossover and *in phase*
  between bands, so the summed output is a 2nd-order all-pass:
  magnitude-flat reconstruction (`|low + high| = 1` at every frequency).
  This is the standard fix for the reconstruction defect of the simple
  parallel Butterworth-2 form, whose LPF and HPF are 180° apart at the
  crossover and therefore *null* on direct summation. The legacy
  `Crossover::new` / `Crossover::butterworth` constructors keep the
  Butterworth-2 (12 dB/oct) behaviour byte-for-byte; for LR4 the
  per-section Q is forced to `1/√2` regardless of the `q` argument (two
  cascaded Butterworth-2 is what defines LR4). The registry `crossover`
  factory gains an optional `"slope"` key (`"butterworth2"` default, or
  `"lr4"` / `"linkwitz_riley"`). 5 new unit tests: slope/Q reporting,
  −6 dB-per-band at fc, LR4 high-band rejecting >10 dB steeper than
  Butterworth-2 two octaves below fc, magnitude-flat L+H reconstruction
  across five probe frequencies, and the contrasting Butterworth-2
  summation null at fc.

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
  static curve across the threshold.
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
  in our own variable names.
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
