//! Factory glue + `register` entry point.
//!
//! Lifts the audio-filter factories (volume / noise_gate / echo /
//! resample / spectrogram) and the legacy `AudioFilterAdapter` shim
//! out of the old `oxideav-pipeline::filter_registry` module — they
//! live with the concrete filters now so that `oxideav-pipeline`
//! itself doesn't need to depend on this crate.

use oxideav_core::{
    filter::FilterContext, ChannelLayout, Error, Frame, MediaType, PortParams, PortSpec, Result,
    RuntimeContext, SampleFormat, StreamFilter,
};
use serde_json::Value;
use std::str::FromStr;

use crate::{AudioFilter, AudioStreamParams};

/// Install Volume, NoiseGate, Echo, Resample, and Spectrogram into the
/// runtime context's filter registry. Idempotent — last write wins
/// per filter name.
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    ctx.filters.register("volume", Box::new(make_volume));
    ctx.filters
        .register("noise_gate", Box::new(make_noise_gate));
    ctx.filters.register("echo", Box::new(make_echo));
    ctx.filters.register("resample", Box::new(make_resample));
    ctx.filters
        .register("spectrogram", Box::new(make_spectrogram));
    ctx.filters.register("downmix", Box::new(make_downmix));
    ctx.filters.register("biquad", Box::new(make_biquad));
    ctx.filters
        .register("compressor", Box::new(make_compressor));
    ctx.filters.register("limiter", Box::new(make_limiter));
    ctx.filters
        .register("dc_blocker", Box::new(make_dc_blocker));
    ctx.filters
        .register("stereo_widener", Box::new(make_stereo_widener));
    ctx.filters.register("reverb", Box::new(make_reverb));
    ctx.filters.register("tremolo", Box::new(make_tremolo));
    ctx.filters
        .register("loudness_itu", Box::new(make_loudness_itu));
    ctx.filters
        .register("pitch_shift", Box::new(make_pitch_shift));
    ctx.filters.register("chorus", Box::new(make_chorus));
    ctx.filters.register("flanger", Box::new(make_flanger));
    ctx.filters.register("phaser", Box::new(make_phaser));
    ctx.filters.register("equalizer", Box::new(make_equalizer));
    ctx.filters
        .register("white_noise", Box::new(make_white_noise));
    ctx.filters
        .register("pink_noise", Box::new(make_pink_noise));
    ctx.filters
        .register("brown_noise", Box::new(make_brown_noise));
    ctx.filters
        .register("silence_detector", Box::new(make_silence_detector));
    ctx.filters.register("vibrato", Box::new(make_vibrato));
    ctx.filters.register("auto_pan", Box::new(make_auto_pan));
    ctx.filters
        .register("bitcrusher", Box::new(make_bitcrusher));
    ctx.filters
        .register("tape_saturation", Box::new(make_tape_saturation));
    ctx.filters
        .register("hum_filter", Box::new(make_hum_filter));
    ctx.filters.register("crossover", Box::new(make_crossover));
    ctx.filters.register("mid_side", Box::new(make_mid_side));
    ctx.filters
        .register("envelope_follower", Box::new(make_envelope_follower));
    ctx.filters.register("de_esser", Box::new(make_de_esser));
    ctx.filters.register("wah", Box::new(make_wah));
    ctx.filters
        .register("octave_doubler", Box::new(make_octave_doubler));
    ctx.filters
        .register("adaptive_noise_gate", Box::new(make_adaptive_noise_gate));
    ctx.filters.register("exciter", Box::new(make_exciter));
    ctx.filters
        .register("multiband_compressor", Box::new(make_multiband_compressor));
    ctx.filters
        .register("stereo_imager", Box::new(make_stereo_imager));
    ctx.filters.register("talkbox", Box::new(make_talkbox));
    ctx.filters
        .register("transient_designer", Box::new(make_transient_designer));
    ctx.filters.register("ducker", Box::new(make_ducker));
    ctx.filters
        .register("gain_normalizer", Box::new(make_gain_normalizer));
    ctx.filters
        .register("freq_shifter", Box::new(make_freq_shifter));
    ctx.filters
        .register("ring_modulator", Box::new(make_ring_modulator));
    ctx.filters
        .register("hard_clipper", Box::new(make_hard_clipper));
    ctx.filters
        .register("slew_limiter", Box::new(make_slew_limiter));
    ctx.filters.register("expander", Box::new(make_expander));
    ctx.filters
        .register("true_peak_detector", Box::new(make_true_peak_detector));
    ctx.filters.register("svf", Box::new(make_svf));
    ctx.filters
        .register("pre_emphasis", Box::new(make_pre_emphasis));
    ctx.filters
        .register("de_emphasis", Box::new(make_de_emphasis));
    ctx.filters
        .register("median_filter", Box::new(make_median_filter));
    ctx.filters
        .register("crest_factor_meter", Box::new(make_crest_factor_meter));
    ctx.filters.register(
        "stereo_correlation_meter",
        Box::new(make_stereo_correlation_meter),
    );
    ctx.filters
        .register("comb_filter", Box::new(make_comb_filter));
    ctx.filters
        .register("zero_crossing_rate", Box::new(make_zero_crossing_rate));
    ctx.filters
        .register("dc_offset_meter", Box::new(make_dc_offset_meter));
    ctx.filters
        .register("stereo_balance_meter", Box::new(make_stereo_balance_meter));
    ctx.filters.register("dither", Box::new(make_dither));
}

oxideav_core::register!("audio_filter", register);

/// Wraps a legacy [`AudioFilter`] in the [`StreamFilter`] contract.
/// Single audio port in, single audio port out; both inherit params
/// from the upstream input port. The stream-level audio shape
/// ([`AudioStreamParams`]) is cached once at construction off the
/// input port and threaded into every `process()` / `flush()` call —
/// the trait used to read these off the frame, but they live on the
/// stream's `CodecParameters` now.
struct AudioFilterAdapter {
    inner: Box<dyn AudioFilter>,
    inp: [PortSpec; 1],
    outp: [PortSpec; 1],
    params: AudioStreamParams,
}

impl AudioFilterAdapter {
    fn new(inner: Box<dyn AudioFilter>, in_port: PortSpec, out_port: PortSpec) -> Self {
        let params = match &in_port.params {
            PortParams::Audio {
                format,
                channels,
                sample_rate,
            } => AudioStreamParams {
                format: *format,
                channels: *channels,
                sample_rate: *sample_rate,
            },
            // Non-audio input ports shouldn't reach this adapter, but pick
            // a defensible default so we don't panic on misuse.
            _ => AudioStreamParams {
                format: SampleFormat::F32,
                channels: 2,
                sample_rate: 48_000,
            },
        };
        Self {
            inner,
            inp: [in_port],
            outp: [out_port],
            params,
        }
    }
}

impl StreamFilter for AudioFilterAdapter {
    fn input_ports(&self) -> &[PortSpec] {
        &self.inp
    }
    fn output_ports(&self) -> &[PortSpec] {
        &self.outp
    }
    fn push(&mut self, ctx: &mut dyn FilterContext, port: usize, frame: &Frame) -> Result<()> {
        if port != 0 {
            return Err(Error::invalid(format!(
                "audio-filter adapter: unknown input port {port}"
            )));
        }
        let Frame::Audio(a) = frame else {
            return Err(Error::invalid(
                "audio-filter adapter: input port 0 only accepts audio frames",
            ));
        };
        let outs = self.inner.process(a, self.params)?;
        for o in outs {
            ctx.emit(0, Frame::Audio(o))?;
        }
        Ok(())
    }
    fn flush(&mut self, ctx: &mut dyn FilterContext) -> Result<()> {
        let outs = self.inner.flush(self.params)?;
        for o in outs {
            ctx.emit(0, Frame::Audio(o))?;
        }
        Ok(())
    }
}

/// Pull the single audio port spec from `inputs`, or fall back to a
/// sane default if none is provided.
fn audio_in_port(inputs: &[PortSpec]) -> PortSpec {
    inputs
        .iter()
        .find(|p| matches!(p.params, PortParams::Audio { .. }))
        .cloned()
        .unwrap_or_else(|| PortSpec::audio("in", 48_000, 2, SampleFormat::F32))
}

fn make_volume(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Volume;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let volume = if let Some(db) = get_f64("gain_db") {
        let linear = 10f32.powf((db as f32) / 20.0);
        Volume::new(linear)
    } else if let Some(g) = get_f64("gain") {
        Volume::new(g as f32)
    } else {
        return Err(Error::invalid(
            "job: filter 'volume' needs `gain` or `gain_db`",
        ));
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(volume),
        in_port,
        out_port,
    )))
}

fn make_noise_gate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::NoiseGate;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let has = |k: &str| p.and_then(|m| m.get(k)).is_some();
    let threshold_db = get_f64("threshold_db", -40.0) as f32;
    let attack_ms = get_f64("attack_ms", 10.0) as f32;
    let release_ms = get_f64("release_ms", 100.0) as f32;
    let hold_ms = get_f64("hold_ms", 50.0) as f32;
    // Optional hysteresis + soft-knee upgrades (r181). If the job
    // spec omits both `hysteresis_db`/`close_db` AND `knee_db`, fall
    // through to the legacy hard-knee single-threshold constructor so
    // existing job specs are byte-for-byte unaffected.
    let gate = if has("hysteresis_db") || has("close_db") || has("knee_db") {
        // `close_db` overrides; otherwise derive it from `hysteresis_db`
        // (default 6 dB, a common broadcast value).
        let close_db = if has("close_db") {
            get_f64("close_db", (threshold_db - 6.0) as f64) as f32
        } else {
            let hyst = get_f64("hysteresis_db", 6.0) as f32;
            threshold_db - hyst.max(0.0)
        };
        let knee_db = get_f64("knee_db", 0.0) as f32;
        NoiseGate::with(
            threshold_db,
            close_db,
            knee_db,
            attack_ms,
            release_ms,
            hold_ms,
        )
    } else {
        NoiseGate::new(threshold_db, attack_ms, release_ms, hold_ms)
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(gate),
        in_port,
        out_port,
    )))
}

fn make_echo(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Echo;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let e = Echo::new(
        get_f64("delay_ms", 250.0) as f32,
        get_f64("feedback", 0.35) as f32,
        get_f64("mix", 0.5) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(e),
        in_port,
        out_port,
    )))
}

fn make_resample(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Resample;
    let p = params.as_object();
    let dst_rate = p
        .and_then(|m| m.get("rate"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::invalid("job: filter 'resample' needs `rate` (output sample rate)"))?
        as u32;
    let in_port = audio_in_port(inputs);
    let (src_rate, channels, format) = match &in_port.params {
        PortParams::Audio {
            sample_rate,
            channels,
            format,
        } => (*sample_rate, *channels, *format),
        _ => (48_000, 2, SampleFormat::F32),
    };
    let filter = Resample::new(src_rate, dst_rate)?;
    let out_port = PortSpec::audio("audio", dst_rate, channels, format);
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(filter),
        in_port,
        out_port,
    )))
}

fn make_spectrogram(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::spectrogram::{Colormap, Spectrogram, SpectrogramOptions, Window};
    let p = params.as_object();
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let mut opts = SpectrogramOptions::default();
    if let Some(v) = get_u64("fft_size") {
        opts.fft_size = v as usize;
    }
    if let Some(v) = get_u64("hop_size") {
        opts.hop_size = v as usize;
    }
    if let Some(v) = get_u64("width") {
        opts.width = v as u32;
    }
    if let Some(v) = get_u64("height") {
        opts.height = v as u32;
    }
    opts.window = match get_str("window") {
        Some("hamming") => Window::Hamming,
        Some("blackman") => Window::Blackman,
        _ => Window::Hann,
    };
    opts.colormap = match get_str("colormap") {
        Some("grayscale") | Some("gray") => Colormap::Grayscale,
        Some("magma") => Colormap::Magma,
        _ => Colormap::Viridis,
    };
    if let Some(lo) = get_f64("db_low") {
        opts.db_range.0 = lo as f32;
    }
    if let Some(hi) = get_f64("db_high") {
        opts.db_range.1 = hi as f32;
    }
    let fps = get_u64("fps").unwrap_or(30) as u32;
    let mut s = Spectrogram::new(opts)?.with_video_fps(fps);
    // Pre-seed the audio-input params so the output port's time_base +
    // sample_rate are correct BEFORE the first push.
    if let Some(audio) = inputs.iter().find(|p| p.kind == MediaType::Audio) {
        s = s.with_audio_input(audio);
    }
    Ok(Box::new(s) as Box<dyn StreamFilter>)
}

/// `{"filter": "downmix", "to": "stereo", "mode": "loro"}` — fold a
/// surround source into a smaller layout.
///
/// Required params:
///   - `to`: destination layout name (`"stereo"`, `"mono"`, `"5.1"`, …),
///     parsed via [`ChannelLayout::from_str`].
///
/// Optional params:
///   - `mode`: `"loro"` (default for surround→stereo), `"ltrt"`,
///     `"average"` / `"avg"`, or `"binaural"` / `"hrtf"`. Omitted →
///     [`crate::auto_downmix`] picks one.
///   - `from`: source layout name. When absent the source is inferred
///     from the upstream port's channel count.
fn make_downmix(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{auto_downmix, DownmixFilter, DownmixMode};

    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let in_port = audio_in_port(inputs);
    let (src_rate, src_channels, src_format) = match &in_port.params {
        PortParams::Audio {
            sample_rate,
            channels,
            format,
        } => (*sample_rate, *channels, *format),
        _ => (48_000, 2, SampleFormat::F32),
    };

    let src_layout = if let Some(name) = get_str("from") {
        ChannelLayout::from_str(name)
            .map_err(|e| Error::invalid(format!("downmix: invalid `from` layout {name:?}: {e}")))?
    } else {
        ChannelLayout::from_count(src_channels)
    };

    let dst_name = get_str("to").ok_or_else(|| {
        Error::invalid("job: filter 'downmix' needs `to` (destination channel layout)")
    })?;
    let dst_layout = ChannelLayout::from_str(dst_name)
        .map_err(|e| Error::invalid(format!("downmix: invalid `to` layout {dst_name:?}: {e}")))?;

    let filter = if let Some(mode_name) = get_str("mode") {
        let mode = DownmixMode::from_name(mode_name)?;
        DownmixFilter::new(src_layout, dst_layout, mode)?
    } else {
        auto_downmix(src_layout, dst_layout)?
    };

    let out_port = PortSpec::audio("audio", src_rate, dst_layout.channel_count(), src_format);
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(filter),
        in_port,
        out_port,
    )))
}

/// `{"filter": "biquad", "kind": "low_pass", "cutoff_hz": 1000.0, "q": 0.707}`.
///
/// Required: `kind` — one of `"low_pass"`, `"high_pass"`, `"band_pass"`,
/// `"notch"`, `"peaking"`, `"low_shelf"`, `"high_shelf"`.
///
/// `cutoff_hz` / `center_hz` and `q` are required for every kind;
/// `gain_db` is required for `peaking` / `low_shelf` / `high_shelf`.
fn make_biquad(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::biquad::{Biquad, BiquadKind};

    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());

    let kind_name =
        get_str("kind").ok_or_else(|| Error::invalid("job: filter 'biquad' needs `kind`"))?;
    let freq = get_f64("cutoff_hz")
        .or_else(|| get_f64("center_hz"))
        .ok_or_else(|| Error::invalid("job: filter 'biquad' needs `cutoff_hz` or `center_hz`"))?
        as f32;
    let q = get_f64("q").unwrap_or(std::f64::consts::FRAC_1_SQRT_2) as f32;
    let gain_db = get_f64("gain_db").unwrap_or(0.0) as f32;

    let kind = match kind_name {
        "low_pass" | "lpf" => BiquadKind::LowPass { cutoff_hz: freq, q },
        "high_pass" | "hpf" => BiquadKind::HighPass { cutoff_hz: freq, q },
        "band_pass" | "bpf" => BiquadKind::BandPass { center_hz: freq, q },
        "notch" => BiquadKind::Notch { center_hz: freq, q },
        "peaking" | "peak" => BiquadKind::Peaking {
            center_hz: freq,
            q,
            gain_db,
        },
        "low_shelf" | "lowshelf" => BiquadKind::LowShelf {
            cutoff_hz: freq,
            q,
            gain_db,
        },
        "high_shelf" | "highshelf" => BiquadKind::HighShelf {
            cutoff_hz: freq,
            q,
            gain_db,
        },
        "all_pass" | "allpass" | "apf" => BiquadKind::AllPass { center_hz: freq, q },
        other => return Err(Error::invalid(format!("biquad: unknown kind {other:?}"))),
    };

    let bq = Biquad::new(kind);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(bq),
        in_port,
        out_port,
    )))
}

/// `{"filter": "compressor", "threshold_db": -18.0, "ratio": 4.0,
/// "attack_ms": 10.0, "release_ms": 100.0, "knee_db": 6.0,
/// "makeup_gain_db": 0.0}`.
fn make_compressor(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Compressor;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let comp = Compressor::new(
        get_f64("threshold_db", -18.0) as f32,
        get_f64("ratio", 4.0) as f32,
        get_f64("attack_ms", 10.0) as f32,
        get_f64("release_ms", 100.0) as f32,
        get_f64("knee_db", 0.0) as f32,
        get_f64("makeup_gain_db", 0.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(comp),
        in_port,
        out_port,
    )))
}

/// `{"filter": "limiter", "ceiling_db": -0.3, "release_ms": 50.0,
/// "look_ahead_samples": 64}`.
fn make_limiter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Limiter;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let lim = Limiter::new(
        get_f64("ceiling_db", -0.3) as f32,
        get_f64("release_ms", 50.0) as f32,
        get_u64("look_ahead_samples", 0) as usize,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(lim),
        in_port,
        out_port,
    )))
}

/// `{"filter": "dc_blocker", "pole": 0.995}` — single-pole DC remover.
fn make_dc_blocker(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DcBlocker;
    let p = params.as_object();
    let get_f64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_f64());
    let bl = match get_f64("pole") {
        Some(v) => DcBlocker::with_pole(v as f32),
        None => DcBlocker::new(),
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(bl),
        in_port,
        out_port,
    )))
}

/// `{"filter": "stereo_widener", "width": 1.5}` — M/S width control.
fn make_stereo_widener(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::StereoWidener;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let w = StereoWidener::new(get_f64("width", 1.0) as f32);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(w),
        in_port,
        out_port,
    )))
}

/// `{"filter": "reverb", "room_size": 0.5, "damping": 0.5,
/// "wet": 0.33, "dry": 0.67}` — Schroeder algorithmic reverb.
fn make_reverb(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Reverb;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let r = Reverb::new(
        get_f64("room_size", 0.5) as f32,
        get_f64("damping", 0.5) as f32,
        get_f64("wet", 0.33) as f32,
        get_f64("dry", 0.67) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(r),
        in_port,
        out_port,
    )))
}

/// `{"filter": "tremolo", "rate_hz": 5.0, "depth": 0.5}` — sine LFO AM.
fn make_tremolo(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Tremolo;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let t = Tremolo::new(get_f64("rate_hz", 5.0) as f32, get_f64("depth", 0.5) as f32);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(t),
        in_port,
        out_port,
    )))
}

/// `{"filter": "loudness_itu"}` — ITU-R BS.1770-4 loudness meter.
/// Has no parameters; reads back via API. Output port is preserved
/// (the meter passes nothing through downstream).
fn make_loudness_itu(_params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::LoudnessITU;
    let m = LoudnessITU::new();
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(m),
        in_port,
        out_port,
    )))
}

/// `{"filter": "pitch_shift", "semitones": 7.0}` — granular pitch shift.
fn make_pitch_shift(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::PitchShift;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let ps = PitchShift::new(get_f64("semitones", 0.0) as f32);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(ps),
        in_port,
        out_port,
    )))
}

/// `{"filter": "chorus", "n_voices": 2, "base_delay_ms": 25.0,
/// "depth_ms": 5.0, "rate_hz": 1.0, "mix": 0.5}` — multi-voice chorus.
fn make_chorus(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Chorus;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let c = Chorus::new(
        get_u64("n_voices", 2) as u8,
        get_f64("base_delay_ms", 25.0) as f32,
        get_f64("depth_ms", 5.0) as f32,
        get_f64("rate_hz", 1.0) as f32,
        get_f64("mix", 0.5) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(c),
        in_port,
        out_port,
    )))
}

/// `{"filter": "flanger", "rate_hz": 0.5, "depth_ms": 5.0,
/// "feedback": 0.5, "mix": 0.5}` — feedback comb flanger.
fn make_flanger(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Flanger;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let f = Flanger::new(
        get_f64("rate_hz", 0.5) as f32,
        get_f64("depth_ms", 5.0) as f32,
        get_f64("feedback", 0.5) as f32,
        get_f64("mix", 0.5) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// `{"filter": "phaser", "n_stages": 4, "rate_hz": 0.5,
/// "depth_hz": 1000.0, "feedback": 0.3, "mix": 0.5}` — N-stage AP phaser.
fn make_phaser(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Phaser;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let f = Phaser::new(
        get_u64("n_stages", 4) as u8,
        get_f64("rate_hz", 0.5) as f32,
        get_f64("depth_hz", 1_000.0) as f32,
        get_f64("feedback", 0.0) as f32,
        get_f64("mix", 0.5) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(f),
        in_port,
        out_port,
    )))
}

/// `{"filter": "equalizer", "bands": [
///   {"kind": "low_shelf", "freq_hz": 100, "q": 0.707, "gain_db": 3},
///   {"kind": "peaking",   "freq_hz": 1000, "q": 1.0,   "gain_db": -2},
///   {"kind": "high_shelf","freq_hz": 10000,"q": 0.707, "gain_db": 4}
/// ]}` — N-band parametric EQ.
fn make_equalizer(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::biquad::BiquadKind;
    use crate::Equalizer;
    let in_port = audio_in_port(inputs);
    let (sample_rate, _channels, _format) = match &in_port.params {
        PortParams::Audio {
            sample_rate,
            channels,
            format,
        } => (*sample_rate, *channels, *format),
        _ => (48_000, 2, SampleFormat::F32),
    };
    let mut eq = Equalizer::new(sample_rate);
    if let Some(bands) = params.get("bands").and_then(|v| v.as_array()) {
        for band in bands {
            let kind = band.get("kind").and_then(|v| v.as_str()).unwrap_or("");
            let freq = band
                .get("freq_hz")
                .and_then(|v| v.as_f64())
                .or_else(|| band.get("cutoff_hz").and_then(|v| v.as_f64()))
                .or_else(|| band.get("center_hz").and_then(|v| v.as_f64()))
                .unwrap_or(1_000.0) as f32;
            let q = band
                .get("q")
                .and_then(|v| v.as_f64())
                .unwrap_or(std::f64::consts::FRAC_1_SQRT_2) as f32;
            let gain_db = band.get("gain_db").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let bk = match kind {
                "low_pass" | "lpf" => BiquadKind::LowPass { cutoff_hz: freq, q },
                "high_pass" | "hpf" => BiquadKind::HighPass { cutoff_hz: freq, q },
                "band_pass" | "bpf" => BiquadKind::BandPass { center_hz: freq, q },
                "notch" => BiquadKind::Notch { center_hz: freq, q },
                "peaking" | "peak" => BiquadKind::Peaking {
                    center_hz: freq,
                    q,
                    gain_db,
                },
                "low_shelf" | "lowshelf" => BiquadKind::LowShelf {
                    cutoff_hz: freq,
                    q,
                    gain_db,
                },
                "high_shelf" | "highshelf" => BiquadKind::HighShelf {
                    cutoff_hz: freq,
                    q,
                    gain_db,
                },
                other => {
                    return Err(Error::invalid(format!(
                        "equalizer: unknown band kind {other:?}"
                    )))
                }
            };
            eq = eq.add_band(bk);
        }
    }
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(eq),
        in_port,
        out_port,
    )))
}

/// `{"filter": "white_noise", "amplitude": 0.5, "seed": 42}` — uniform PRNG.
fn make_white_noise(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::WhiteNoise;
    let p = params.as_object();
    let amplitude = p
        .and_then(|m| m.get("amplitude"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let seed = p.and_then(|m| m.get("seed")).and_then(|v| v.as_u64());
    let g = match seed {
        Some(s) => WhiteNoise::with_seed(amplitude, s),
        None => WhiteNoise::new(amplitude),
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(g),
        in_port,
        out_port,
    )))
}

/// `{"filter": "pink_noise", "amplitude": 0.5, "seed": 42}` — 1/f Kellet.
fn make_pink_noise(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::PinkNoise;
    let p = params.as_object();
    let amplitude = p
        .and_then(|m| m.get("amplitude"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let seed = p.and_then(|m| m.get("seed")).and_then(|v| v.as_u64());
    let g = match seed {
        Some(s) => PinkNoise::with_seed(amplitude, s),
        None => PinkNoise::new(amplitude),
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(g),
        in_port,
        out_port,
    )))
}

/// `{"filter": "brown_noise", "amplitude": 0.5, "seed": 42}` — 1/f² leaky.
fn make_brown_noise(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::BrownNoise;
    let p = params.as_object();
    let amplitude = p
        .and_then(|m| m.get("amplitude"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5) as f32;
    let seed = p.and_then(|m| m.get("seed")).and_then(|v| v.as_u64());
    let g = match seed {
        Some(s) => BrownNoise::with_seed(amplitude, s),
        None => BrownNoise::new(amplitude),
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(g),
        in_port,
        out_port,
    )))
}

/// `{"filter": "silence_detector", "threshold_dbfs": -60.0, "hold_ms": 100.0}`.
fn make_silence_detector(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::SilenceDetector;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let det = SilenceDetector::new(
        get_f64("threshold_dbfs", -60.0) as f32,
        get_f64("hold_ms", 100.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(det),
        in_port,
        out_port,
    )))
}

/// `{"filter": "vibrato", "rate_hz": 5.0, "depth_ms": 2.0}` — LFO-modulated
/// delay-line pitch shift.
fn make_vibrato(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Vibrato;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let v = Vibrato::new(
        get_f64("rate_hz", 5.0) as f32,
        get_f64("depth_ms", 2.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(v),
        in_port,
        out_port,
    )))
}

/// `{"filter": "auto_pan", "rate_hz": 1.5, "depth": 1.0}` — LFO L/R pan.
fn make_auto_pan(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AutoPan;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let ap = AutoPan::new(get_f64("rate_hz", 1.5) as f32, get_f64("depth", 1.0) as f32);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(ap),
        in_port,
        out_port,
    )))
}

/// `{"filter": "bitcrusher", "bits": 6, "decimation": 4}` — bit-depth +
/// sample-and-hold rate reduction.
fn make_bitcrusher(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Bitcrusher;
    let p = params.as_object();
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let bc = Bitcrusher::new(get_u64("bits", 8) as u8, get_u64("decimation", 1) as u32);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(bc),
        in_port,
        out_port,
    )))
}

/// `{"filter": "tape_saturation", "drive": 2.0, "asymmetry": 0.3}` —
/// tanh soft-clip with optional asymmetric drive.
fn make_tape_saturation(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::TapeSaturation;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let ts = TapeSaturation::new(
        get_f64("drive", 2.0) as f32,
        get_f64("asymmetry", 0.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(ts),
        in_port,
        out_port,
    )))
}

/// `{"filter": "hum_filter", "fundamental_hz": 60.0, "q": 30.0,
/// "n_harmonics": 5}` — line-mains hum suppression.
fn make_hum_filter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::HumFilter;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let hf = HumFilter::new(
        get_f64("fundamental_hz", 60.0) as f32,
        get_f64("q", 30.0) as f32,
        get_u64("n_harmonics", 5) as u8,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(hf),
        in_port,
        out_port,
    )))
}

/// `{"filter": "crossover", "cutoff_hz": 1000.0, "q": 0.707}` — two-way
/// LPF/HPF split; output port carries `2× input channels`.
///
/// Optional `"slope"` selects the topology: `"butterworth2"` (default,
/// 12 dB/oct) or `"lr4"` / `"linkwitz_riley"` (24 dB/oct, magnitude-flat
/// summation). For LR4 the per-section Q is forced to `1/√2`.
fn make_crossover(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Crossover, CrossoverSlope};
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let cutoff_hz = get_f64("cutoff_hz", 1_000.0) as f32;
    let q = get_f64("q", std::f64::consts::FRAC_1_SQRT_2) as f32;
    let slope = match get_str("slope") {
        Some("lr4") | Some("linkwitz_riley") | Some("linkwitz-riley") => {
            CrossoverSlope::LinkwitzRiley4
        }
        _ => CrossoverSlope::Butterworth2,
    };
    let xo = Crossover::with_slope(cutoff_hz, q, slope);
    let in_port = audio_in_port(inputs);
    let (sample_rate, channels, format) = match &in_port.params {
        PortParams::Audio {
            sample_rate,
            channels,
            format,
        } => (*sample_rate, *channels, *format),
        _ => (48_000, 1, SampleFormat::F32),
    };
    let out_port = PortSpec::audio("audio", sample_rate, channels * 2, format);
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(xo),
        in_port,
        out_port,
    )))
}

/// `{"filter": "mid_side", "mode": "encode"}` — L/R ↔ M/S transcoder.
/// `mode` is `"encode"` (L/R → M/S, default) or `"decode"` (M/S → L/R).
fn make_mid_side(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::MidSide;
    let p = params.as_object();
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let ms = match get_str("mode") {
        Some("decode") | Some("ms_to_lr") => MidSide::decoder(),
        _ => MidSide::encoder(),
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(ms),
        in_port,
        out_port,
    )))
}

/// `{"filter": "envelope_follower", "attack_ms": 5.0, "release_ms": 50.0,
/// "mode": "peak"}` — amplitude-envelope detector (pass-through).
fn make_envelope_follower(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{EnvelopeFollower, EnvelopeMode};
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let mode = match get_str("mode") {
        Some("rms") => EnvelopeMode::Rms,
        _ => EnvelopeMode::Peak,
    };
    let ef = EnvelopeFollower::with_mode(
        get_f64("attack_ms", 5.0) as f32,
        get_f64("release_ms", 50.0) as f32,
        mode,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(ef),
        in_port,
        out_port,
    )))
}

/// `{"filter": "de_esser", "cutoff_hz": 6000.0, "threshold_db": -20.0,
/// "ratio": 4.0, "attack_ms": 1.0, "release_ms": 30.0}` — split-band
/// downward compressor targeting sibilance.
fn make_de_esser(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DeEsser;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let de = DeEsser::with(
        get_f64("cutoff_hz", 6_000.0) as f32,
        get_f64("threshold_db", -20.0) as f32,
        get_f64("ratio", 4.0) as f32,
        get_f64("attack_ms", 1.0) as f32,
        get_f64("release_ms", 30.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(de),
        in_port,
        out_port,
    )))
}

/// `{"filter": "wah", "rate_hz": 0.8, "f_min": 400.0, "f_max": 2200.0,
/// "q": 2.5, "mix": 1.0}` — LFO-swept resonant band-pass.
fn make_wah(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Wah;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let w = Wah::with(
        get_f64("rate_hz", 0.8) as f32,
        get_f64("f_min", 400.0) as f32,
        get_f64("f_max", 2_200.0) as f32,
        get_f64("q", 2.5) as f32,
        get_f64("mix", 1.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(w),
        in_port,
        out_port,
    )))
}

/// `{"filter": "octave_doubler", "dry": 1.0, "wet": 0.5,
/// "dc_block": true}` — full-wave rectifier + DC block.
fn make_octave_doubler(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::OctaveDoubler;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_bool = |k: &str, dflt: bool| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_bool())
            .unwrap_or(dflt)
    };
    let od = OctaveDoubler::with(
        get_f64("dry", 1.0) as f32,
        get_f64("wet", 0.5) as f32,
        get_bool("dc_block", true),
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(od),
        in_port,
        out_port,
    )))
}

/// `{"filter": "adaptive_noise_gate", "margin_db": 12.0, "learn_ms": 2000.0,
/// "attack_ms": 5.0, "release_ms": 100.0}` — adaptive gate with learned
/// noise floor.
fn make_adaptive_noise_gate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::AdaptiveNoiseGate;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let g = AdaptiveNoiseGate::with(
        get_f64("margin_db", 12.0) as f32,
        get_f64("learn_ms", 2_000.0) as f32,
        get_f64("attack_ms", 5.0) as f32,
        get_f64("release_ms", 100.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(g),
        in_port,
        out_port,
    )))
}

/// `{"filter": "exciter", "cutoff_hz": 4000.0, "drive": 3.0,
/// "mix": 0.4}` — high-band saturation enhancer.
fn make_exciter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Exciter;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let ex = Exciter::with(
        get_f64("cutoff_hz", 4_000.0) as f32,
        get_f64("drive", 3.0) as f32,
        get_f64("mix", 0.4) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(ex),
        in_port,
        out_port,
    )))
}

/// `{"filter": "multiband_compressor", "low_cutoff_hz": 250.0,
/// "high_cutoff_hz": 2500.0, "low": {…}, "mid": {…}, "high": {…}}`.
/// Each band table accepts the standard
/// `threshold_db / ratio / attack_ms / release_ms / knee_db /
/// makeup_gain_db` keys; omitted fields fall back to
/// [`BandSettings::default_low`] / `default_mid` / `default_high`.
fn make_multiband_compressor(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::multiband_compressor::{BandSettings, MultibandCompressor};

    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let band = |key: &str, base: BandSettings| -> BandSettings {
        let obj = params.get(key).and_then(|v| v.as_object());
        let g = |k: &str, dflt: f32| -> f32 {
            obj.and_then(|m| m.get(k))
                .and_then(|v| v.as_f64())
                .map(|x| x as f32)
                .unwrap_or(dflt)
        };
        BandSettings {
            threshold_db: g("threshold_db", base.threshold_db),
            ratio: g("ratio", base.ratio),
            attack_ms: g("attack_ms", base.attack_ms),
            release_ms: g("release_ms", base.release_ms),
            knee_db: g("knee_db", base.knee_db),
            makeup_gain_db: g("makeup_gain_db", base.makeup_gain_db),
        }
    };
    let mbc = MultibandCompressor::with(
        get_f64("low_cutoff_hz", 250.0) as f32,
        get_f64("high_cutoff_hz", 2_500.0) as f32,
        band("low", BandSettings::default_low()),
        band("mid", BandSettings::default_mid()),
        band("high", BandSettings::default_high()),
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(mbc),
        in_port,
        out_port,
    )))
}

/// `{"filter": "stereo_imager", "cutoff_hz": 250.0, "low_width": 0.0,
/// "high_width": 1.5}` — band-split M/S widener.
fn make_stereo_imager(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::StereoImager;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let im = StereoImager::with(
        get_f64("cutoff_hz", 250.0) as f32,
        get_f64("low_width", 0.0) as f32,
        get_f64("high_width", 1.5) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(im),
        in_port,
        out_port,
    )))
}

/// `{"filter": "talkbox", "from": "ah", "to": "ee", "rate_hz": 0.5,
/// "q": 8.0, "mix": 1.0}` — LFO-morphed vowel formant filter.
fn make_talkbox(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::talkbox::{Talkbox, Vowel};
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_str = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_str());
    let parse_vowel = |name: Option<&str>, dflt: Vowel| -> Vowel {
        match name.map(|s| s.to_ascii_lowercase()) {
            Some(ref s) if s == "ah" || s == "a" => Vowel::Ah,
            Some(ref s) if s == "eh" || s == "e" => Vowel::Eh,
            Some(ref s) if s == "ee" || s == "i" => Vowel::Ee,
            Some(ref s) if s == "oh" || s == "o" => Vowel::Oh,
            Some(ref s) if s == "oo" || s == "u" => Vowel::Oo,
            Some(ref s) if s == "uh" => Vowel::Uh,
            _ => dflt,
        }
    };
    let from = parse_vowel(get_str("from"), Vowel::Ah);
    let to = parse_vowel(get_str("to"), Vowel::Ee);
    let tb = Talkbox::with(
        from,
        to,
        get_f64("rate_hz", 0.5) as f32,
        get_f64("q", 8.0) as f32,
        get_f64("mix", 1.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(tb),
        in_port,
        out_port,
    )))
}

/// `{"filter": "transient_designer", "attack": 0.5, "sustain": 0.0,
/// "attack_ms_fast": 1.0, "attack_ms_slow": 35.0}` — two-envelope
/// attack/sustain shaper.
fn make_transient_designer(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::TransientDesigner;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let td = TransientDesigner::with(
        get_f64("attack", 0.0) as f32,
        get_f64("sustain", 0.0) as f32,
        get_f64("attack_ms_fast", 1.0) as f32,
        get_f64("attack_ms_slow", 35.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(td),
        in_port,
        out_port,
    )))
}

/// `{"filter": "ducker", "threshold_db": -20.0, "ratio": 8.0,
/// "attack_ms": 5.0, "release_ms": 250.0, "max_reduction_db": -30.0,
/// "key_channel": null}` — internally-keyed sidechain compressor.
fn make_ducker(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Ducker;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str| p.and_then(|m| m.get(k)).and_then(|v| v.as_u64());
    let mut d = Ducker::with(
        get_f64("threshold_db", -20.0) as f32,
        get_f64("ratio", 8.0) as f32,
        get_f64("attack_ms", 5.0) as f32,
        get_f64("release_ms", 250.0) as f32,
    );
    if let Some(mr) = p
        .and_then(|m| m.get("max_reduction_db"))
        .and_then(|v| v.as_f64())
    {
        d = d.with_max_reduction_db(mr as f32);
    }
    if let Some(kc) = get_u64("key_channel") {
        d = d.with_key_channel(Some(kc as usize));
    }
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(d),
        in_port,
        out_port,
    )))
}

/// `{"filter": "gain_normalizer", "target_db": -16.0, "detector_ms": 500.0,
/// "gain_ms": 200.0, "max_gain_db": 24.0, "max_atten_db": -24.0,
/// "silence_threshold_db": -60.0}` — slow AGC programme-level normaliser.
fn make_gain_normalizer(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::GainNormalizer;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let mut a = GainNormalizer::with(
        get_f64("target_db", -16.0) as f32,
        get_f64("detector_ms", 500.0) as f32,
        get_f64("gain_ms", 200.0) as f32,
    );
    if let Some(v) = p
        .and_then(|m| m.get("max_gain_db"))
        .and_then(|v| v.as_f64())
    {
        a = a.with_max_gain_db(v as f32);
    }
    if let Some(v) = p
        .and_then(|m| m.get("max_atten_db"))
        .and_then(|v| v.as_f64())
    {
        a = a.with_max_atten_db(v as f32);
    }
    if let Some(v) = p
        .and_then(|m| m.get("silence_threshold_db"))
        .and_then(|v| v.as_f64())
    {
        a = a.with_silence_threshold_db(v as f32);
    }
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(a),
        in_port,
        out_port,
    )))
}

/// `{"filter": "freq_shifter", "delta_hz": 100.0, "half_taps": 63}` —
/// Hilbert-FIR SSB frequency shifter.
fn make_freq_shifter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::FreqShifter;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let fs = FreqShifter::with(
        get_f64("delta_hz", 100.0) as f32,
        get_u64("half_taps", 63) as usize,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(fs),
        in_port,
        out_port,
    )))
}

/// `{"filter": "ring_modulator", "carrier_hz": 440.0, "mix": 1.0}` —
/// sine-carrier double-sideband suppressed-carrier amplitude modulator
/// (Dalek / bell effect for audible carriers).
fn make_ring_modulator(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::RingModulator;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let rm = RingModulator::new(
        get_f64("carrier_hz", 440.0) as f32,
        get_f64("mix", 1.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(rm),
        in_port,
        out_port,
    )))
}

/// `{"filter": "hard_clipper", "drive": 2.0, "ceiling": 1.0}` —
/// memoryless symmetric clipping distortion
/// (`y = clamp(drive·x, -ceiling, +ceiling)`; odd-harmonic fuzz).
fn make_hard_clipper(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::HardClipper;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let hc = HardClipper::new(get_f64("drive", 1.0) as f32, get_f64("ceiling", 1.0) as f32);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(hc),
        in_port,
        out_port,
    )))
}

/// `{"filter": "slew_limiter", "max_slew_per_sec": 2.0}` for the
/// symmetric form, or `{"filter": "slew_limiter", "slew_up_per_sec":
/// 10.0, "slew_dn_per_sec": 1.0, "initial": 0.0}` for the asymmetric
/// form — bounds the per-sample output change to `max_slew_per_sec /
/// fs`; linear-ramp anti-zipper / portamento smoother.
fn make_slew_limiter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::SlewLimiter;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    // Prefer asymmetric spec when *either* axis-specific key is given;
    // otherwise fall back to the symmetric `max_slew_per_sec`.
    let has_up = p.and_then(|m| m.get("slew_up_per_sec")).is_some();
    let has_dn = p.and_then(|m| m.get("slew_dn_per_sec")).is_some();
    let initial = get_f64("initial", 0.0) as f32;
    let sl = if has_up || has_dn {
        let up = get_f64("slew_up_per_sec", 2.0) as f32;
        let dn = get_f64("slew_dn_per_sec", 2.0) as f32;
        SlewLimiter::with_asymmetric(up, dn).with_initial_value(initial)
    } else {
        let s = get_f64("max_slew_per_sec", 2.0) as f32;
        SlewLimiter::new(s).with_initial_value(initial)
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(sl),
        in_port,
        out_port,
    )))
}

/// `{"filter": "expander", "threshold_db": -40.0, "ratio": 2.0,
/// "attack_ms": 5.0, "release_ms": 50.0, "knee_db": 0.0,
/// "makeup_gain_db": 0.0}` — proportional downward expander
/// (`-(R - 1) dB` per dB below threshold). Use a large `ratio` (e.g.
/// `1.0e6`) or JSON `null` substituted for `ratio` to approach the
/// hard-gate limit; in code [`crate::Expander::gate`] gives the
/// `ratio = ∞` form directly. Distinct from `noise_gate` (binary
/// open/close) — fades gracefully into silence instead of slamming
/// shut.
fn make_expander(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::Expander;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let exp = Expander::new(
        get_f64("threshold_db", -40.0) as f32,
        get_f64("ratio", 2.0) as f32,
        get_f64("attack_ms", 5.0) as f32,
        get_f64("release_ms", 50.0) as f32,
        get_f64("knee_db", 0.0) as f32,
        get_f64("makeup_gain_db", 0.0) as f32,
    );
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(exp),
        in_port,
        out_port,
    )))
}

/// `{"filter": "true_peak_detector", "oversample": 4, "taps": 48,
/// "kaiser_db": 100.0, "overs_threshold": 1.0}` — pass-through
/// observer that reports inter-sample peak level (dBTP) via 4×
/// polyphase Kaiser-windowed FIR oversampling. All keys optional;
/// defaults give the conventional broadcast-loudness 4× / 12-tap
/// per-phase / ~100 dB stop-band / 0 dBTP overs threshold setup. The
/// detector emits audio frames unchanged; downstream stages observe
/// `current_dbtp` / `max_dbtp` / `overs` by holding a direct handle
/// to the [`TruePeakDetector`](crate::TruePeakDetector). Within the
/// `StreamFilter` graph the detector simply forwards the input
/// without modification — wire it inline with the audio it should
/// observe.
fn make_true_peak_detector(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::TruePeakDetector;
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };
    let oversample = get_u64("oversample", 4) as usize;
    let taps = get_u64("taps", 48) as usize;
    let kaiser_db = get_f64("kaiser_db", 100.0);
    let overs_threshold = get_f64("overs_threshold", 1.0) as f32;
    let det = TruePeakDetector::with_params(oversample, taps, kaiser_db, overs_threshold);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(det),
        in_port,
        out_port,
    )))
}

/// `{"filter": "svf", "mode": "low_pass", "cutoff_hz": 1000.0,
/// "q": 0.707}` — Chamberlin State Variable Filter. All keys optional;
/// defaults give a 1 kHz Butterworth-equivalent LPF. The `mode` key
/// accepts `"low_pass"` / `"lp"`, `"band_pass"` / `"bp"`, `"high_pass"`
/// / `"hp"`, or `"notch"`. Distinct topology from the bilinear-transform
/// [`Biquad`](crate::biquad::Biquad): a state-space two-integrator
/// loop where cutoff and `Q` can be modulated per-sample without
/// recomputing transfer-function coefficients. Stable while
/// `f_c < f_s / 6` and `Q ∈ [0.5, 50]`; out-of-range arguments are
/// clamped at construction.
fn make_svf(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{SvfFilter, SvfMode};
    let p = params.as_object();
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let mode_str = p
        .and_then(|m| m.get("mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("low_pass");
    let mode = match mode_str {
        "low_pass" | "lp" | "lowpass" => SvfMode::LowPass,
        "band_pass" | "bp" | "bandpass" => SvfMode::BandPass,
        "high_pass" | "hp" | "highpass" => SvfMode::HighPass,
        "notch" | "band_stop" | "bandstop" | "bs" => SvfMode::Notch,
        other => {
            return Err(Error::invalid(format!(
                "job: filter 'svf' unknown mode '{other}' (expected low_pass/band_pass/high_pass/notch)"
            )));
        }
    };
    let cutoff_hz = get_f64("cutoff_hz", get_f64("center_hz", 1_000.0)) as f32;
    let q = get_f64("q", 0.707) as f32;
    let svf = SvfFilter::new(mode, cutoff_hz, q);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(svf),
        in_port,
        out_port,
    )))
}

/// Resolve the JSON `curve` key into a [`crate::pre_emphasis::Curve`].
/// Accepts `"fm_50us"` / `"fm_75us"` / `"j17"` / `"riaa"` /
/// `"custom"` (with `"tau_us"` companion key for the custom case).
fn resolve_emphasis_curve(
    p: Option<&serde_json::Map<String, Value>>,
) -> Result<crate::pre_emphasis::Curve> {
    use crate::pre_emphasis::Curve;
    let name = p
        .and_then(|m| m.get("curve"))
        .and_then(|v| v.as_str())
        .unwrap_or("fm_50us");
    match name {
        "fm_50us" | "fm50" | "50us" => Ok(Curve::Fm50us),
        "fm_75us" | "fm75" | "75us" => Ok(Curve::Fm75us),
        "j17" | "j_17" | "j.17" => Ok(Curve::J17),
        "riaa" | "riaa_3180_318_75" => Ok(Curve::Riaa3180_318_75),
        "custom" => {
            let tau_us = p
                .and_then(|m| m.get("tau_us"))
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    Error::invalid(
                        "job: filter '*emphasis' curve = 'custom' requires \
                         'tau_us' key (time constant in microseconds)",
                    )
                })?;
            Ok(Curve::Custom {
                tau_s: (tau_us as f32) * 1.0e-6,
            })
        }
        other => Err(Error::invalid(format!(
            "job: filter '*emphasis' unknown curve '{other}' \
             (expected fm_50us / fm_75us / j17 / riaa / custom)"
        ))),
    }
}

/// `{"filter": "pre_emphasis", "curve": "fm_50us", "g": 10.0}` —
/// analog-broadcast / tape / FM record EQ pre-emphasis. Default curve
/// is FM 50 µs; default asymptotic shelf gain is 10× (20 dB HF boost).
fn make_pre_emphasis(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::PreEmphasis;
    let p = params.as_object();
    let curve = resolve_emphasis_curve(p)?;
    let g = p
        .and_then(|m| m.get("g"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10.0) as f32;
    let flt = PreEmphasis::with_gain(curve, g);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(flt),
        in_port,
        out_port,
    )))
}

/// `{"filter": "de_emphasis", "curve": "fm_50us", "g": 10.0}` —
/// analog-broadcast / tape / FM playback EQ de-emphasis (inverse of
/// `pre_emphasis` with matching `curve` + `g`).
fn make_de_emphasis(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DeEmphasis;
    let p = params.as_object();
    let curve = resolve_emphasis_curve(p)?;
    let g = p
        .and_then(|m| m.get("g"))
        .and_then(|v| v.as_f64())
        .unwrap_or(10.0) as f32;
    let flt = DeEmphasis::with_gain(curve, g);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(flt),
        in_port,
        out_port,
    )))
}

/// `{"filter": "median_filter", "window": 5}` — sliding-window median
/// filter (non-linear impulse-noise restoration). `window` is the
/// number of samples in the per-channel ring; defaults to `5`
/// (canonical click-removal value). The runtime clamps `window` into
/// `[1, MedianFilter::MAX_WINDOW]` (= `[1, 257]`); an out-of-range
/// request is accepted but silently saturated rather than rejected,
/// matching the other "parametric knob" filters in this registry.
fn make_median_filter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::MedianFilter;
    let p = params.as_object();
    let window = p
        .and_then(|m| m.get("window"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(5);
    let flt = MedianFilter::new(window);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(flt),
        in_port,
        out_port,
    )))
}

/// `{"filter": "crest_factor_meter", "window_ms": 400.0}` — pass-through
/// observer reporting the peak-to-RMS ratio (crest factor) in dB over a
/// sliding rectangular window. Window defaults to 400 ms (EBU R128
/// short-term); the JSON `window_ms` key overrides it. Observation-only;
/// consumers poll the meter's accessors directly.
fn make_crest_factor_meter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::CrestFactorMeter;
    let p = params.as_object();
    let window_ms = p
        .and_then(|m| m.get("window_ms"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(crate::crest_factor_meter::CFM_DEFAULT_WINDOW_MS);
    let m = CrestFactorMeter::with_window_ms(window_ms);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(m),
        in_port,
        out_port,
    )))
}

/// `{"filter": "stereo_correlation_meter", "window_ms": 400.0}` —
/// pass-through observer reporting the Pearson correlation coefficient
/// between the L and R channels over a sliding rectangular window.
/// Window defaults to 400 ms (matching `crest_factor_meter` so the two
/// readouts can share a time axis on a meter display); the JSON
/// `window_ms` key overrides it. Observation-only; consumers poll
/// `current()` / `current_degrees()` / `min()` on the meter handle
/// directly. Stereo input only — mono and >2-channel layouts pass
/// through unchanged with the meter state untouched.
fn make_stereo_correlation_meter(
    params: &Value,
    inputs: &[PortSpec],
) -> Result<Box<dyn StreamFilter>> {
    use crate::StereoCorrelationMeter;
    let p = params.as_object();
    let window_ms = p
        .and_then(|m| m.get("window_ms"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(crate::stereo_correlation_meter::SCM_DEFAULT_WINDOW_MS);
    let m = StereoCorrelationMeter::with_window_ms(window_ms);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(m),
        in_port,
        out_port,
    )))
}

/// `{"filter": "comb_filter", "mode": "feedforward" | "feedback",
///   "delay_ms": 10.0, "delay_samples": 480, "gain": 0.5,
///   "damping": 0.0}` — single-tap tunable comb (FIR or IIR form).
///
/// `mode` selects the topology (defaults to `"feedforward"`).  Delay
/// may be set either in milliseconds (`delay_ms`, rate-portable) or
/// in exact samples (`delay_samples`, sample-rate-dependent).  If
/// both are present, `delay_samples` wins; if neither is present
/// the default is `delay_ms = 10.0`.  `gain` defaults to `0.5`;
/// for feedback mode it is clamped into `[-0.999, +0.999]` so the
/// recurrence is strictly stable.  `damping` is only consulted in
/// feedback mode (clamped to `[0.0, 0.999]`, default `0.0`).
///
/// A `"karplus_strong"` shortcut also exists — pass
/// `"mode": "karplus_strong"` plus `"freq_hz"` (default `220.0`)
/// and `"decay"` (default `0.99`) and the filter is configured as
/// a feedback comb tuned to that fundamental with the canonical
/// half-damping plucked-string tail.
fn make_comb_filter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{CombFilter, CombMode};
    let p = params.as_object();
    let mode_str = p
        .and_then(|m| m.get("mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("feedforward");
    let get_f64 = |k: &str, dflt: f64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_f64())
            .unwrap_or(dflt)
    };
    let get_u64 = |k: &str, dflt: u64| {
        p.and_then(|m| m.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(dflt)
    };

    let gain = get_f64("gain", 0.5) as f32;
    let damping = get_f64("damping", 0.0) as f32;

    let flt = match mode_str {
        "feedforward" | "fir" | "ff" => {
            let mode = CombMode::Feedforward { gain };
            if let Some(s) = p.and_then(|m| m.get("delay_samples")) {
                CombFilter::with_delay_samples(mode, s.as_u64().unwrap_or(0) as usize)
            } else {
                CombFilter::with_delay_ms(mode, get_f64("delay_ms", 10.0) as f32)
            }
        }
        "feedback" | "iir" | "fb" => {
            let mode = CombMode::Feedback { gain, damping };
            if let Some(s) = p.and_then(|m| m.get("delay_samples")) {
                CombFilter::with_delay_samples(mode, s.as_u64().unwrap_or(0) as usize)
            } else {
                CombFilter::with_delay_ms(mode, get_f64("delay_ms", 10.0) as f32)
            }
        }
        "karplus_strong" | "ks" | "plucked_string" => {
            let freq = get_f64("freq_hz", 220.0) as f32;
            let decay = get_f64("decay", 0.99) as f32;
            CombFilter::karplus_strong(freq, decay)
        }
        other => {
            return Err(Error::invalid(format!(
                "job: filter 'comb_filter' unknown mode '{other}' (expected feedforward/feedback/karplus_strong)"
            )));
        }
    };
    // Touch unused locals to make the borrow checker happy in the
    // karplus_strong branch where `gain`/`damping`/etc. aren't read.
    let _ = get_u64;

    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(flt),
        in_port,
        out_port,
    )))
}

/// `{"filter": "zero_crossing_rate", "window_ms": 25.0}` — pass-through
/// observer reporting the number of sign changes in the signal per
/// unit time (crossings per second) over a sliding rectangular
/// window. Window defaults to 25 ms (the canonical short-time
/// speech-analysis frame); the JSON `window_ms` key overrides it.
/// Observation-only; consumers poll
/// [`current_rate_hz`](crate::ZeroCrossingRateMeter::current_rate_hz)
/// /
/// [`current_fraction`](crate::ZeroCrossingRateMeter::current_fraction)
/// /
/// [`current_count`](crate::ZeroCrossingRateMeter::current_count)
/// on the meter handle directly. Per-sample work is `O(1)` via a
/// ring buffer of `N + 1` samples and an incrementally-updated
/// running count.
fn make_zero_crossing_rate(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::ZeroCrossingRateMeter;
    let p = params.as_object();
    let window_ms = p
        .and_then(|m| m.get("window_ms"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(crate::zero_crossing_rate::ZCR_DEFAULT_WINDOW_MS);
    let m = ZeroCrossingRateMeter::with_window_ms(window_ms);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(m),
        in_port,
        out_port,
    )))
}

/// `{"filter": "dc_offset_meter", "window_ms": 400.0}` — pass-through
/// observer reporting the per-channel running mean (DC component)
/// over a sliding rectangular window. Window defaults to 400 ms
/// (matching `crest_factor_meter` / `stereo_correlation_meter` so the
/// three meters share a time axis on a display); the JSON
/// `window_ms` key overrides it. Observation-only; consumers poll
/// [`current`](crate::DcOffsetMeter::current) /
/// [`current_db`](crate::DcOffsetMeter::current_db) /
/// [`per_channel`](crate::DcOffsetMeter::per_channel) /
/// [`max_abs`](crate::DcOffsetMeter::max_abs) on the meter handle
/// directly. Per-sample work is `O(1)` via a ring buffer of `N`
/// samples and an incrementally-updated running sum; periodic
/// per-window rebuild bounds `f64` round-off drift on long streams.
fn make_dc_offset_meter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::DcOffsetMeter;
    let p = params.as_object();
    let window_ms = p
        .and_then(|m| m.get("window_ms"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(crate::dc_offset_meter::DCM_DEFAULT_WINDOW_MS);
    let m = DcOffsetMeter::with_window_ms(window_ms);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(m),
        in_port,
        out_port,
    )))
}

/// `{"filter": "stereo_balance_meter", "window_ms": 400.0}` —
/// pass-through observer reporting the left / right *energy* balance
/// `(R_rms - L_rms) / (R_rms + L_rms) ∈ [-1, +1]` of a stereo signal
/// over a sliding rectangular window. `-1` = all energy on the left,
/// `0` = centred, `+1` = all energy on the right. Window defaults to
/// 400 ms (matching the other R128-aligned meters so they share a
/// time axis on a display); the JSON `window_ms` key overrides it.
/// Observation-only; consumers poll
/// [`current`](crate::StereoBalanceMeter::current) /
/// [`rms_left`](crate::StereoBalanceMeter::rms_left) /
/// [`rms_right`](crate::StereoBalanceMeter::rms_right) /
/// [`max_abs`](crate::StereoBalanceMeter::max_abs) on the meter handle
/// directly. The level complement to `stereo_correlation_meter`
/// (phase): correlation is scale-invariant and blind to a level
/// imbalance, balance reports exactly that. Per-sample work is `O(1)`
/// via per-channel ring buffers and incrementally-updated
/// sums-of-squares; periodic per-window rebuild bounds `f64`
/// round-off drift on long streams. Mono / multichannel input passes
/// through with the meter state untouched.
/// `{"filter": "dither", "bits": 16, "mode": "tpdf", "shaping": "off",
/// "seed": 305419896}` — word-length-reduction requantizer with
/// dither + error-feedback noise shaping. Rounds every sample onto
/// the exact `bits`-wide signed code grid (`Δ = 2^(1-bits)`, `bits`
/// clamped to `[2, 24]`, default 16) so a downstream fixed-point
/// encode is lossless. `mode` selects the dither density: `"tpdf"`
/// (default — triangular, mean and variance of the error
/// signal-independent), `"rpdf"` (uniform — zero-mean error but
/// signal-dependent variance), `"none"` (bare rounding, for
/// measurement only). `shaping` selects the error-feedback noise
/// transfer function: `"off"` (flat, default), `"first"`
/// (`1 - z⁻¹`, +6 dB/oct tilt), `"second"` (`(1 - z⁻¹)²`,
/// +12 dB/oct tilt — noise pushed out of the sensitive low/mid
/// band). Optional `seed` fixes the splitmix64 dither PRNG for
/// bit-reproducible output. The transparency counterpart to the
/// creative `bitcrusher` (which has no dither, no shaping, and an
/// aliasing sample-and-hold stage).
fn make_dither(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::{Dither, DitherMode, NoiseShaping};
    let p = params.as_object();
    let bits = p
        .and_then(|m| m.get("bits"))
        .and_then(|v| v.as_u64())
        .unwrap_or(16) as u8;
    let mode_str = p
        .and_then(|m| m.get("mode"))
        .and_then(|v| v.as_str())
        .unwrap_or("tpdf");
    let mode = match mode_str {
        "none" => DitherMode::None,
        "rpdf" => DitherMode::Rpdf,
        "tpdf" => DitherMode::Tpdf,
        other => {
            return Err(Error::invalid(format!(
                "job: filter 'dither' unknown mode '{other}' (expected none/rpdf/tpdf)"
            )));
        }
    };
    let shaping_str = p
        .and_then(|m| m.get("shaping"))
        .and_then(|v| v.as_str())
        .unwrap_or("off");
    let shaping = match shaping_str {
        "off" => NoiseShaping::Off,
        "first" => NoiseShaping::FirstOrder,
        "second" => NoiseShaping::SecondOrder,
        other => {
            return Err(Error::invalid(format!(
                "job: filter 'dither' unknown shaping '{other}' (expected off/first/second)"
            )));
        }
    };
    let flt = match p.and_then(|m| m.get("seed")).and_then(|v| v.as_u64()) {
        Some(seed) => Dither::with_seed(bits, mode, shaping, seed),
        None => Dither::with(bits, mode, shaping),
    };
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(flt),
        in_port,
        out_port,
    )))
}

fn make_stereo_balance_meter(params: &Value, inputs: &[PortSpec]) -> Result<Box<dyn StreamFilter>> {
    use crate::StereoBalanceMeter;
    let p = params.as_object();
    let window_ms = p
        .and_then(|m| m.get("window_ms"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(crate::stereo_balance_meter::SBM_DEFAULT_WINDOW_MS);
    let m = StereoBalanceMeter::with_window_ms(window_ms);
    let in_port = audio_in_port(inputs);
    let out_port = PortSpec {
        name: "audio".to_string(),
        ..in_port.clone()
    };
    Ok(Box::new(AudioFilterAdapter::new(
        Box::new(m),
        in_port,
        out_port,
    )))
}
