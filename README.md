# oxideav-audio-filter

Pure-Rust audio filters (volume, gate, echo, resample, spectrogram) for oxideav

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace) framework — a pure-Rust media transcoding and streaming stack. Codec, container, and filter crates are implemented from the spec (no C codec libraries linked or wrapped, no `*-sys` crates). Optional hardware-engine crates (`oxideav-videotoolbox` / `-audiotoolbox` / `-vaapi` / `-vdpau` / `-nvidia` / `-vulkan-video`) bridge to OS APIs via runtime `libloading`; pass `--no-hwaccel` (or omit the `hwaccel` feature) to opt out.

## Filters

- **volume** — linear / dB gain with hard clipping
- **noise_gate** — threshold + attack/release/hold envelope gate
- **echo** — single-tap delay line with feedback and wet/dry mix
- **resample** — polyphase windowed-sinc rate conversion
- **spectrogram** — STFT renderer with PNG / video output
- **downmix** — channel-layout fold-down (LoRo / LtRt / Average / Binaural)
- **biquad** — seven-config IIR EQ family (LPF / HPF / BPF / notch / peaking / low-shelf / high-shelf) with DF-II-T `f64` state
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
- **crossover** — two-way LPF/HPF band split (`output channels = 2× input`)
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

## Usage

```toml
[dependencies]
oxideav-audio-filter = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
