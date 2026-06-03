# oxideav-audio-filter

Pure-Rust audio filters (volume, gate, echo, resample, spectrogram) for oxideav

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a pure-Rust media transcoding and streaming stack. Codec, container, and filter crates are implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates). Optional hardware-engine crates (`oxideav-videotoolbox` / `-audiotoolbox` / `-vaapi` / `-vdpau` / `-nvidia` / `-vulkan-video`) bridge to OS APIs via runtime `libloading`; pass `--no-hwaccel` (or omit the `hwaccel` feature) to opt out.

## Filters

- **volume** — linear / dB gain with hard clipping
- **noise_gate** — threshold + attack/release/hold envelope gate; optional two-threshold hysteresis (`open_db` / `close_db`, sticky latch eliminates chatter when the drive dances around the threshold) and Hermite-smoothstep soft-knee (`knee_db`, C¹-continuous transition with 0.5 gain at the knee centre). `NoiseGate::new` keeps the legacy hard-knee single-threshold behaviour byte-for-byte; `NoiseGate::with` exposes the upgraded parameters.
- **echo** — single-tap delay line with feedback and wet/dry mix
- **resample** — polyphase windowed-sinc rate conversion
- **spectrogram** — STFT renderer with PNG / video output
- **downmix** — channel-layout fold-down (LoRo / LtRt / Average / Binaural)
- **biquad** — eight-config IIR EQ family (LPF / HPF / BPF / notch / peaking / low-shelf / high-shelf / all-pass) with DF-II-T `f64` state
- **compressor** — peak compressor with soft-knee, attack/release follower, and make-up gain
- **limiter** — brickwall peak limiter with optional look-ahead (0..=2048 samples)
- **dc_blocker** — single-pole IIR HPF (`y[n] = x[n] - x[n-1] + R·y[n-1]`) for DC-offset removal
- **stereo_widener** — Mid/Side width control with `width ∈ [0, 2]` (0 = mono, 1 = bypass, 2 = wide)
- **reverb** — Schroeder algorithmic reverb (4 parallel combs ║ 2 serial all-passes) with room_size / damping / wet / dry knobs
- **tremolo** — sine-LFO amplitude modulation with rate / depth knobs
- **loudness_itu** — ITU-R BS.1770-4 / EBU R128 integrated loudness meter (LUFS) with K-weighting + channel weights
- **pitch_shift** — time-domain SOLA-style granular pitch shifter (`-12..=+12` semitones, no FFT)
- **chorus** — 1..=4 LFO-modulated short delay taps with phase-offset voices
- **flanger** — short-delay feedback comb (1..=15 ms, swept resonance)
- **phaser** — N (2..=12) cascaded first-order all-pass sections with LFO-swept cutoffs
- **equalizer** — builder over N `Biquad` sections in series (low/high-pass, BPF, notch, peaking, shelves)
- **white_noise** / **pink_noise** / **brown_noise** — splitmix64-seeded generators with flat / 1/f / 1/f² spectra
- **silence_detector** — pass-through observer with attack/release RMS envelope + hold-threshold flag
- **vibrato** — LFO-modulated fractional-delay pitch shift (pitch counterpart to `tremolo`)
- **auto_pan** — LFO-modulated L/R stereo placement (mono-sum conservative pan law)
- **bitcrusher** — bit-depth quantisation + sample-and-hold rate reduction
- **tape_saturation** — `tanh` soft-clip waveshaper with asymmetric drive (odd + even harmonics)
- **hum_filter** — cascaded narrow notches at 50/60 Hz mains fundamental + harmonics
- **crossover** — two-way LPF/HPF band split (`output channels = 2× input`); Butterworth-2 (12 dB/oct, default, byte-for-byte legacy via `Crossover::new`/`butterworth`) or `CrossoverSlope::LinkwitzRiley4` (24 dB/oct, two cascaded Butterworth-2 per band → −6 dB at fc, in-phase, magnitude-flat `|low + high| = 1` reconstruction). Registry `slope` key (`"lr4"`)
- **mid_side** — explicit L/R ↔ M/S transcoder (stateless, exact roundtrip)
- **envelope_follower** — peak / RMS amplitude envelope detector (pass-through, observe via API)
- **de_esser** — split-band downward compressor targeting sibilance
- **wah** — LFO-swept resonant band-pass (Cry-Baby-style log sweep)
- **octave_doubler** — Octavia-style full-wave-rectifier octave-up layer with DC block
- **adaptive_noise_gate** — gate with self-learned asymmetric noise floor
- **exciter** — high-band `tanh` saturation enhancer (HPF + dry mix; adds "air" without raising broadband level)
- **multiband_compressor** — three-band (low / mid / high) independent compression with Butterworth-2 crossovers (default 250 Hz / 2500 Hz)
- **stereo_imager** — frequency-dependent stereo width via M/S side split (mono bass + wide treble preset)
- **talkbox** — LFO-morphed parallel formant band-pass bank (vowel filter, 6 vowels, no FFT)
- **transient_designer** — two-envelope (fast vs slow) attack + sustain shaper (`attack`/`sustain ∈ [-1, +1]`)
- **ducker** — internally-keyed sidechain compressor with safety-floored gain reduction (broadcast voice-over duck)
- **gain_normalizer** — slow programme-level AGC with silence-freeze integrator (`target_db`, default ~1 s response)
- **freq_shifter** — Hilbert-FIR SSB frequency shifter (additive `Δf` in Hz, harmonic-destroying)
- **ring_modulator** — sine-carrier double-sideband suppressed-carrier AM (`y = x · cos(2π·fc·n/fs)`, audible carrier; produces mirror sidebands at `f ± fc`, Dalek / bell timbre)
- **hard_clipper** — memoryless symmetric clipping distortion (`y = clamp(drive·x, ±ceiling)`; odd-harmonic fuzz / overdrive, distinct from the `tanh` soft-clip of `tape_saturation`)
- **slew_limiter** — slope-limited smoother (`Δ = x − y_prev; y = y_prev + clamp(Δ, ±s)` with `s = max_slew_per_sec / fs`); linear-ramp response (vs the LPF's exponential), with optional asymmetric rise / fall caps. Anti-zipper / portamento / anti-pop primitive.
- **expander** — proportional downward expander: `gr_db = -(R − 1) · max(0, threshold_db − env_db)`. Distinct from `noise_gate` (binary open/close) and from `compressor` (slope on the *above*-threshold side). `ratio = 1.0` → identity, `ratio = ∞` → hard downward gate, finite ratios in between trade off between gentle noise-floor management (`1.5:1` / `2:1`) and aggressive expansion (`4:1`+). Soft-knee width smooths the threshold transition; attack/release one-pole follower; peak-linked detector across channels.
- **true_peak_detector** — pass-through inter-sample peak observer reporting dBTP via 4× polyphase Kaiser-windowed FIR oversampling. Sample-domain `max |x[n]|` understates the band-limited reconstructed peak (canonical example: an `fs/4 + π/4`-phase full-scale sine reads `-3.01 dBFS` on the grid but reconstructs to `≈ 0 dBTP`); this filter reveals the true peak by upsampling by `L = 4` with a unit-DC-gain FIR (default 48 taps → 12 taps per polyphase sub-filter, Kaiser β chosen for ~100 dB stop-band) and tracking `max |y|` in `f64`. Pass-through audio (no modification); consumers read `current_dbtp` (last frame), `max_dbtp` (running), and `overs` (count of oversampled samples above the configurable linear threshold, default `1.0` = 0 dBTP). Distinct from `limiter` (sample-peak gain reduction), `envelope_follower` (smoothed envelope), and `loudness_itu` (K-weighted integrated loudness).
- **svf** — Chamberlin two-integrator-loop State Variable Filter. Simultaneous LP / BP / HP / Notch taps from a single recurrence (`hp = x - q·bp_prev - lp_prev; bp += f·hp; lp += f·bp`), so the output mode can be reconfigured without touching state. `f = 2·sin(π·f_c/f_s)`, `q = 1/Q`; per-sample modulation of cutoff is cheap (one `sin` on `set_cutoff`) and avoids the bilinear-transform coefficient resolve that a regular biquad needs — this is the canonical synth filter for envelope-/LFO-swept cutoff. State-space topology distinct from the `biquad` family (transfer-function DF-II-T realisation). Stable while `f_c < f_s / 6` and `Q ∈ [0.5, 50]`; values outside that range are clamped at construction. Notch tap is `hp + lp`, deep at low Q (~−24 dB at Q = 0.5) and degrading as Q rises — use `biquad` notch for high-Q narrow-band reject.
- **median_filter** — non-linear sliding-window median filter for impulse-noise (click / pop) removal. Per channel keeps an `N`-sample ring; output is the median of the latest `N` samples. Step edges spanning more than `N / 2` samples pass through unaltered, but isolated impulse outliers are entirely discarded — a behaviour no linear LPF achieves. Window `∈ [1, MedianFilter::MAX_WINDOW]` (= `[1, 257]`); default `5` (canonical click-removal value). Even windows return the mean of the two centre sorted samples. Insertion-sort over the ring (best-case `O(window)` on near-sorted data, steady-state common case). Joins the restoration family next to `hum_filter` (cyclic-mains) and `dc_blocker` (DC drift) — its niche is *transient* impulse noise instead of cyclic / DC. Registry entry `"median_filter"` accepts JSON `window` key.
- **pre_emphasis** / **de_emphasis** — paired analog-broadcast / tape / FM record EQ shelving filters. The cascade `pre · de` is the identity by construction (within `f64` round-off; ≤ −60 dB RMS error per sample on white-noise round-trip). Shared `EmphasisCurve` enum covers `Fm50us` (European FM), `Fm75us` (North-American FM), `J17` (ITU-R J.17 voice-band), `Custom { tau_s }` (user-specified single-time-constant), and `Riaa3180_318_75` (phonograph / vinyl second-order three-time-constant curve). Coefficients are derived in-source by bilinear transform of `H_pre(s) = (1 + s·τ) / (1 + s·τ/G)` and its inverse, with the asymptotic shelf-top gain `G` (default 10, capped at 1000) bounding HF boost so the discrete transfer stays well-behaved at Nyquist. First-order `(b₀, b₁, a₁)` direct-form-I IIR per channel for FM / J.17 / Custom; second-order `(b₀, b₁, b₂, a₁, a₂)` for RIAA. Per-channel state, `reset()`, `set_sample_rate` re-derives coefficients. Registry entries `"pre_emphasis"` / `"de_emphasis"` accept JSON `curve` key (`"fm_50us"` / `"fm_75us"` / `"j17"` / `"riaa"` / `"custom"` + `"tau_us"` for the custom case) and `g` for the asymptotic gain. 15 hand-derived unit tests per filter including DC-gain-unity (both first- and second-order), Nyquist-asymptote-equals-G, derived corner frequencies (3183 Hz @ 50 µs, 2122 Hz @ 75 µs), +20 dB/decade slope-between-corners, channel-independence, streaming continuity, sample-rate-change re-derivation, alias-curve identity (J.17 ≡ FM-50, Custom ≡ named); plus three cascade-identity tests (FM-50 / FM-75 / RIAA) proving the inverse property.

## Usage

```toml
[dependencies]
oxideav-audio-filter = "0.0"
```

## Benchmarks

Criterion harness `benches/filters.rs` measures the per-sample DSP cost of seven
representative filter families on deterministic xorshift32-synthesised PCM:

- `biquad_lpf` — single second-order IIR (DF-II-T, `f64` state), stereo F32 @ 48 kHz
- `equalizer_3band` — low-shelf + peaking + high-shelf cascade, stereo F32 @ 48 kHz
- `loudness_itu` — BS.1770-4 K-weighting + RLB HPF + per-channel mean-square, stereo F32 @ 48 kHz
- `compressor` — peak-detector soft-knee compressor, stereo F32 @ 48 kHz
- `reverb` — Schroeder 4 combs ║ 2 all-passes, stereo F32 @ 48 kHz
- `resample_44k1_48k` — polyphase windowed-sinc rate conversion, mono F32
- `true_peak_4x` — 4× polyphase Kaiser-FIR inter-sample peak detection, stereo F32 @ 48 kHz

Every PCM input is built once **outside** the timed region; the bench body
exercises only the filter's public `AudioFilter::process` / `process_in_place`
path. Run with:

```sh
cargo bench -p oxideav-audio-filter --bench filters
```

## License

MIT — see [LICENSE](LICENSE).
