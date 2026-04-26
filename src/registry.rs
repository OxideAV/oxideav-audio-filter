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

use crate::AudioFilter;

/// Install Volume, NoiseGate, Echo, Resample, and Spectrogram into the
/// runtime context's filter registry. Idempotent — last write wins
/// per filter name.
pub fn register(ctx: &mut RuntimeContext) {
    ctx.filters.register("volume", Box::new(make_volume));
    ctx.filters
        .register("noise_gate", Box::new(make_noise_gate));
    ctx.filters.register("echo", Box::new(make_echo));
    ctx.filters.register("resample", Box::new(make_resample));
    ctx.filters
        .register("spectrogram", Box::new(make_spectrogram));
    ctx.filters.register("downmix", Box::new(make_downmix));
}

/// Wraps a legacy [`AudioFilter`] in the [`StreamFilter`] contract.
/// Single audio port in, single audio port out; both inherit params
/// from the upstream input port.
struct AudioFilterAdapter {
    inner: Box<dyn AudioFilter>,
    inp: [PortSpec; 1],
    outp: [PortSpec; 1],
}

impl AudioFilterAdapter {
    fn new(inner: Box<dyn AudioFilter>, in_port: PortSpec, out_port: PortSpec) -> Self {
        Self {
            inner,
            inp: [in_port],
            outp: [out_port],
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
        let outs = self.inner.process(a)?;
        for o in outs {
            ctx.emit(0, Frame::Audio(o))?;
        }
        Ok(())
    }
    fn flush(&mut self, ctx: &mut dyn FilterContext) -> Result<()> {
        let outs = self.inner.flush()?;
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
    let gate = NoiseGate::new(
        get_f64("threshold_db", -40.0) as f32,
        get_f64("attack_ms", 10.0) as f32,
        get_f64("release_ms", 100.0) as f32,
        get_f64("hold_ms", 50.0) as f32,
    );
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
