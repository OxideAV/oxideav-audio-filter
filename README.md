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

## Usage

```toml
[dependencies]
oxideav-audio-filter = "0.0"
```

## License

MIT — see [LICENSE](LICENSE).
