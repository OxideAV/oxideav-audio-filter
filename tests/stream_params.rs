//! `Frame::StreamParams` handling across the stream-filter surface.
//!
//! Covers the core `StreamFilter` contract for mid-stream audio shape
//! changes: rate-dependent filters re-plan for the new rate, the
//! resampler keeps a pinned output rate silent and announces a
//! following one, meters forward the event, per-channel state follows a
//! channel-count change, and fixed channel maps refuse with
//! `Error::Unsupported`.

use oxideav_audio_filter::{
    AudioFilter, AudioStreamParams, LoudnessITU, TruePeakDetector, register,
};
use oxideav_core::{
    AudioFrame, CodecId, CodecParameters, Error, FilterContext, Frame, PortParams, PortSpec,
    Result, RuntimeContext, SampleFormat, StreamFilter,
};
use serde_json::{Value, json};

// ───────────────────────────── helpers ─────────────────────────────

#[derive(Default)]
struct Collect {
    out: Vec<(usize, Frame)>,
}

impl FilterContext for Collect {
    fn emit(&mut self, port: usize, frame: Frame) -> Result<()> {
        self.out.push((port, frame));
        Ok(())
    }
}

impl Collect {
    fn events(&self, port: usize) -> Vec<&CodecParameters> {
        self.out
            .iter()
            .filter(|(p, _)| *p == port)
            .filter_map(|(_, f)| f.stream_params())
            .collect()
    }
    fn take(&mut self) -> Vec<(usize, Frame)> {
        std::mem::take(&mut self.out)
    }
}

fn ctx() -> RuntimeContext {
    let mut c = RuntimeContext::new();
    register(&mut c);
    c
}

fn make(
    name: &str,
    params: Value,
    rate: u32,
    channels: u16,
    fmt: SampleFormat,
) -> Box<dyn StreamFilter> {
    ctx()
        .filters
        .make(name, &params, &[PortSpec::audio("in", rate, channels, fmt)])
        .unwrap_or_else(|e| panic!("make {name}: {e}"))
}

fn event(rate: u32, channels: u16, fmt: SampleFormat) -> Frame {
    let mut cp = CodecParameters::audio(CodecId::new("pcm"));
    cp.sample_rate = Some(rate);
    cp.channels = Some(channels);
    cp.sample_format = Some(fmt);
    Frame::StreamParams(Box::new(cp))
}

/// Interleaved F32 frame, the same signal on every channel.
fn f32_frame(samples: &[f32], channels: u16) -> Frame {
    let mut data = Vec::with_capacity(samples.len() * channels as usize * 4);
    for s in samples {
        for _ in 0..channels {
            data.extend_from_slice(&s.to_le_bytes());
        }
    }
    Frame::Audio(AudioFrame {
        samples: samples.len() as u32,
        pts: None,
        data: vec![data],
    })
}

/// Interleaved S16 frame, the same signal on every channel.
fn s16_frame(samples: &[f32], channels: u16) -> Frame {
    let mut data = Vec::with_capacity(samples.len() * channels as usize * 2);
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        for _ in 0..channels {
            data.extend_from_slice(&v.to_le_bytes());
        }
    }
    Frame::Audio(AudioFrame {
        samples: samples.len() as u32,
        pts: None,
        data: vec![data],
    })
}

fn sine(freq: f32, rate: u32, n: usize, amp: f32) -> Vec<f32> {
    (0..n)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / rate as f32).sin())
        .collect()
}

/// Channel `ch` of an interleaved F32 frame.
fn f32_channel(f: &AudioFrame, channels: u16, ch: usize) -> Vec<f32> {
    f.data[0]
        .chunks_exact(4 * channels as usize)
        .map(|c| f32::from_le_bytes(c[ch * 4..ch * 4 + 4].try_into().unwrap()))
        .collect()
}

fn audio_frames(out: &[(usize, Frame)], port: usize) -> Vec<&AudioFrame> {
    out.iter()
        .filter(|(p, _)| *p == port)
        .filter_map(|(_, f)| match f {
            Frame::Audio(a) => Some(a),
            _ => None,
        })
        .collect()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn port_audio(p: &PortSpec) -> (u32, u16, SampleFormat) {
    match p.params {
        PortParams::Audio {
            sample_rate,
            channels,
            format,
        } => (sample_rate, channels, format),
        _ => panic!("not an audio port"),
    }
}

// ───────────────────────── 44.1 → 48 kHz: EQ ─────────────────────────

/// A narrow 1 kHz notch keeps notching 1 kHz after the switch only if
/// its coefficients were recomputed for 48 kHz (stale 44.1 kHz
/// coefficients put the notch at 1088 Hz and let most of 1 kHz pass).
#[test]
fn equalizer_recomputes_for_new_rate_and_forwards_event() {
    let params = json!({"bands": [{"kind": "notch", "freq_hz": 1000.0, "q": 8.0}]});
    let mut eq = make("equalizer", params, 44_100, 1, SampleFormat::F32);
    let mut c = Collect::default();
    eq.push(&mut c, 0, &f32_frame(&sine(1000.0, 44_100, 22_050, 0.5), 1))
        .unwrap();
    let ev = event(48_000, 1, SampleFormat::F32);
    eq.push(&mut c, 0, &ev).unwrap();
    // Shape-transparent: the event is forwarded verbatim.
    let evs = c.events(0);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0].sample_rate, Some(48_000));
    assert_eq!(evs[0].codec_id, CodecId::new("pcm"));
    assert_eq!(port_audio(&eq.output_ports()[0]).0, 48_000);
    assert_eq!(port_audio(&eq.input_ports()[0]).0, 48_000);
    c.take();
    eq.push(&mut c, 0, &f32_frame(&sine(1000.0, 48_000, 24_000, 0.5), 1))
        .unwrap();
    let out = audio_frames(&c.out, 0);
    let y = f32_channel(out[0], 1, 0);
    // Skip the switch transient (the notch memory carried over).
    let tail = rms(&y[12_000..]);
    assert!(
        tail < 0.01,
        "1 kHz leaks through the notch after the switch: rms {tail}"
    );
}

/// The single-section `biquad` filter follows the rate the same way.
#[test]
fn biquad_recomputes_for_new_rate() {
    let params = json!({"kind": "notch", "cutoff_hz": 1000.0, "q": 8.0});
    let mut bq = make("biquad", params, 44_100, 1, SampleFormat::F32);
    let mut c = Collect::default();
    bq.push(&mut c, 0, &f32_frame(&sine(1000.0, 44_100, 8_000, 0.5), 1))
        .unwrap();
    bq.push(&mut c, 0, &event(48_000, 1, SampleFormat::F32))
        .unwrap();
    c.take();
    bq.push(&mut c, 0, &f32_frame(&sine(1000.0, 48_000, 24_000, 0.5), 1))
        .unwrap();
    let y = f32_channel(audio_frames(&c.out, 0)[0], 1, 0);
    assert!(rms(&y[12_000..]) < 0.01);
}

// ──────────────────────── 44.1 → 48 kHz: resample ────────────────────────

/// Count positive-going zero crossings → frequency estimate.
fn est_freq(x: &[f32], rate: u32) -> f32 {
    let crossings = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
    crossings as f32 * rate as f32 / x.len() as f32
}

#[test]
fn resample_pinned_output_stays_silent_across_rate_change() {
    let mut rs = make(
        "resample",
        json!({"rate": 48_000}),
        44_100,
        1,
        SampleFormat::F32,
    );
    let mut c = Collect::default();
    rs.push(&mut c, 0, &f32_frame(&sine(1000.0, 44_100, 44_100, 0.5), 1))
        .unwrap();
    let before: usize = audio_frames(&c.out, 0)
        .iter()
        .map(|a| a.samples as usize)
        .sum();
    rs.push(&mut c, 0, &event(48_000, 1, SampleFormat::F32))
        .unwrap();
    // Output pinned at 48 kHz: no event on the output.
    assert!(
        c.events(0).is_empty(),
        "pinned resampler announced a change"
    );
    assert_eq!(
        port_audio(&rs.output_ports()[0]),
        (48_000, 1, SampleFormat::F32)
    );
    // The old segment's tail was drained at the boundary.
    let drained: usize = audio_frames(&c.out, 0)
        .iter()
        .map(|a| a.samples as usize)
        .sum();
    assert!(drained > before, "old-rate tail not drained");
    // ~1 s of 44.1 kHz input → ~48 000 output samples in total.
    assert!(
        (drained as i64 - 48_000).abs() < 64,
        "old segment length {drained}"
    );
    c.take();
    rs.push(&mut c, 0, &f32_frame(&sine(1000.0, 48_000, 48_000, 0.5), 1))
        .unwrap();
    let y: Vec<f32> = audio_frames(&c.out, 0)
        .iter()
        .flat_map(|a| f32_channel(a, 1, 0))
        .collect();
    // 48 → 48 now: the 1 kHz tone stays at 1 kHz on the 48 kHz output.
    let f = est_freq(&y[1_000..], 48_000);
    assert!((f - 1000.0).abs() < 5.0, "post-switch tone at {f} Hz");
    assert!(y.len() > 47_000);
}

#[test]
fn resample_following_ratio_announces_new_output_rate() {
    let mut rs = make(
        "resample",
        json!({"ratio": "2/1"}),
        44_100,
        1,
        SampleFormat::F32,
    );
    assert_eq!(port_audio(&rs.output_ports()[0]).0, 88_200);
    let mut c = Collect::default();
    rs.push(&mut c, 0, &f32_frame(&sine(1000.0, 44_100, 4_410, 0.5), 1))
        .unwrap();
    rs.push(&mut c, 0, &event(48_000, 1, SampleFormat::F32))
        .unwrap();
    rs.push(&mut c, 0, &f32_frame(&sine(1000.0, 48_000, 4_800, 0.5), 1))
        .unwrap();
    let evs = c.events(0);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0].sample_rate, Some(96_000));
    assert_eq!(evs[0].channels, Some(1));
    assert_eq!(port_audio(&rs.output_ports()[0]).0, 96_000);
    // Order: old-shape frames (incl. drained tail), then the event, then
    // new-shape frames.
    let ev_idx = c
        .out
        .iter()
        .position(|(_, f)| f.is_stream_params())
        .unwrap();
    assert!(ev_idx > 0);
    let after: usize = c.out[ev_idx + 1..]
        .iter()
        .filter_map(|(_, f)| match f {
            Frame::Audio(a) => Some(a.samples as usize),
            _ => None,
        })
        .sum();
    assert!(after > 9_000, "new segment produced {after} samples");
}

#[test]
fn resample_ratio_number_and_pair_forms() {
    let a = make(
        "resample",
        json!({"ratio": 2}),
        24_000,
        1,
        SampleFormat::F32,
    );
    assert_eq!(port_audio(&a.output_ports()[0]).0, 48_000);
    let b = make(
        "resample",
        json!({"ratio": [1, 2]}),
        96_000,
        1,
        SampleFormat::F32,
    );
    assert_eq!(port_audio(&b.output_ports()[0]).0, 48_000);
    let err = ctx()
        .filters
        .make(
            "resample",
            &json!({"ratio": "0/1"}),
            &[PortSpec::audio("in", 48_000, 1, SampleFormat::F32)],
        )
        .err()
        .expect("zero ratio accepted");
    assert!(matches!(err, Error::InvalidData(_)), "{err:?}");
}

/// A following ratio that would land on a non-integer rate refuses the
/// change and keeps the old plan.
#[test]
fn resample_ratio_refuses_non_integer_rate() {
    let mut rs = make(
        "resample",
        json!({"ratio": "1/2"}),
        48_000,
        1,
        SampleFormat::F32,
    );
    let mut c = Collect::default();
    let err = rs
        .push(&mut c, 0, &event(44_101, 1, SampleFormat::F32))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert_eq!(port_audio(&rs.output_ports()[0]).0, 24_000);
    rs.push(&mut c, 0, &f32_frame(&sine(500.0, 48_000, 4_800, 0.5), 1))
        .unwrap();
}

/// The `sample_format` converter pins the format: an input-format
/// change alone is silent, a rate change is announced (with the pinned
/// format).
#[test]
fn sample_format_pins_format_follows_rate() {
    let mut fc = make(
        "sample_format",
        json!({"format": "s16"}),
        44_100,
        2,
        SampleFormat::F32,
    );
    let mut c = Collect::default();
    fc.push(&mut c, 0, &event(44_100, 2, SampleFormat::S32))
        .unwrap();
    assert!(c.events(0).is_empty());
    fc.push(&mut c, 0, &event(48_000, 2, SampleFormat::S32))
        .unwrap();
    let evs = c.events(0);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0].sample_rate, Some(48_000));
    assert_eq!(evs[0].sample_format, Some(SampleFormat::S16));
}

// ───────────────────────── 44.1 → 48 kHz: meters ─────────────────────────

#[test]
fn meters_forward_event_and_pass_audio() {
    for name in [
        "true_peak_detector",
        "loudness_itu",
        "crest_factor_meter",
        "stereo_correlation_meter",
        "dc_offset_meter",
        "zero_crossing_rate",
        "envelope_follower",
        "silence_detector",
    ] {
        let mut m = make(name, json!({}), 44_100, 2, SampleFormat::F32);
        let mut c = Collect::default();
        let x = sine(997.0, 44_100, 4_410, 0.25);
        m.push(&mut c, 0, &f32_frame(&x, 2)).unwrap();
        m.push(&mut c, 0, &event(48_000, 2, SampleFormat::F32))
            .unwrap();
        let evs = c.events(0);
        assert_eq!(evs.len(), 1, "{name} did not forward the event");
        assert_eq!(evs[0].sample_rate, Some(48_000), "{name}");
        c.take();
        let x = sine(997.0, 48_000, 4_800, 0.25);
        m.push(&mut c, 0, &f32_frame(&x, 2)).unwrap();
        let out = audio_frames(&c.out, 0);
        if name == "loudness_itu" {
            // Sink-style meter: emits no audio, only the event.
            assert!(out.is_empty());
            continue;
        }
        assert_eq!(f32_channel(out[0], 2, 1), x, "{name} altered the audio");
    }
}

/// Integrated loudness is a time average: carried across the switch,
/// each second weighs the same on both sides of it.
#[test]
fn loudness_integrates_across_rate_change() {
    let p441 = AudioStreamParams::new(SampleFormat::F32, 1, 44_100);
    let p48 = p441.with_sample_rate(48_000);
    let frame = |x: &[f32]| match f32_frame(x, 1) {
        Frame::Audio(a) => a,
        _ => unreachable!(),
    };
    let loud = sine(1000.0, 44_100, 44_100, 0.9);
    let quiet = sine(1000.0, 48_000, 96_000, 0.05);
    let mut a = LoudnessITU::new();
    a.process(&frame(&loud), p441).unwrap();
    let la = a.integrated_lufs() as f64;
    let mut b = LoudnessITU::new();
    b.process(&frame(&quiet), p48).unwrap();
    let lb = b.integrated_lufs() as f64;
    // 1 s at la, 2 s at lb.
    let expect = 10.0 * ((10f64.powf(la / 10.0) * 1.0 + 10f64.powf(lb / 10.0) * 2.0) / 3.0).log10();

    let mut m = LoudnessITU::new();
    m.process(&frame(&loud), p441).unwrap();
    assert!(m.reconfigure(p441, p48).unwrap().is_empty());
    m.process(&frame(&quiet), p48).unwrap();
    let got = m.integrated_lufs() as f64;
    assert!((got - expect).abs() < 0.05, "integrated {got} vs {expect}");
    assert_eq!(m.samples_seen(), 44_100 + 96_000);
}

/// A true-peak reading taken before the switch survives it.
#[test]
fn true_peak_reading_carries_over() {
    let p441 = AudioStreamParams::new(SampleFormat::F32, 1, 44_100);
    let p48 = p441.with_sample_rate(48_000);
    let frame = |x: &[f32]| match f32_frame(x, 1) {
        Frame::Audio(a) => a,
        _ => unreachable!(),
    };
    let mut t = TruePeakDetector::new();
    t.process(&frame(&sine(1000.0, 44_100, 4_410, 0.9)), p441)
        .unwrap();
    let peak = t.max_dbtp();
    t.process(&frame(&sine(1000.0, 48_000, 4_800, 0.1)), p48)
        .unwrap();
    assert!(
        (t.max_dbtp() - peak).abs() < 1e-6,
        "{} vs {peak}",
        t.max_dbtp()
    );
    assert!(peak > -1.5);
}

// ───────────────────────────── mono → stereo ─────────────────────────────

#[test]
fn mono_to_stereo_resizes_per_channel_state() {
    let params = json!({"bands": [{"kind": "low_pass", "freq_hz": 2000.0}]});
    let mut eq = make("equalizer", params, 48_000, 1, SampleFormat::F32);
    let mut c = Collect::default();
    eq.push(&mut c, 0, &f32_frame(&sine(500.0, 48_000, 4_800, 0.5), 1))
        .unwrap();
    eq.push(&mut c, 0, &event(48_000, 2, SampleFormat::F32))
        .unwrap();
    assert_eq!(c.events(0)[0].channels, Some(2));
    assert_eq!(port_audio(&eq.output_ports()[0]).1, 2);
    c.take();
    eq.push(&mut c, 0, &f32_frame(&sine(500.0, 48_000, 4_800, 0.5), 2))
        .unwrap();
    let out = audio_frames(&c.out, 0)[0];
    assert_eq!(out.data[0].len(), 4_800 * 2 * 4);
    // Both channels start from fresh state, so they are identical.
    assert_eq!(f32_channel(out, 2, 0), f32_channel(out, 2, 1));
}

/// Shape-changing filters announce the new channel count they produce.
#[test]
fn mono_to_stereo_through_shape_changers() {
    // Crossover: 2× channels.
    let mut xo = make("crossover", json!({}), 48_000, 1, SampleFormat::F32);
    let mut c = Collect::default();
    xo.push(&mut c, 0, &event(48_000, 2, SampleFormat::F32))
        .unwrap();
    let evs = c.events(0);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0].channels, Some(4));
    assert_eq!(evs[0].channel_layout, None);
    // Pinned resampler: rate stays, channel count follows → announced.
    let mut rs = make(
        "resample",
        json!({"rate": 48_000}),
        44_100,
        1,
        SampleFormat::F32,
    );
    let mut c = Collect::default();
    rs.push(&mut c, 0, &f32_frame(&sine(440.0, 44_100, 4_410, 0.5), 1))
        .unwrap();
    rs.push(&mut c, 0, &event(44_100, 2, SampleFormat::F32))
        .unwrap();
    let evs = c.events(0);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0].sample_rate, Some(48_000));
    assert_eq!(evs[0].channels, Some(2));
    c.take();
    rs.push(&mut c, 0, &f32_frame(&sine(440.0, 44_100, 4_410, 0.5), 2))
        .unwrap();
    let out = audio_frames(&c.out, 0);
    assert!(
        out.iter()
            .all(|a| a.data[0].len() == a.samples as usize * 2 * 4)
    );
}

// ─────────────────────────────── refusals ───────────────────────────────

#[test]
fn fixed_channel_maps_refuse_channel_change() {
    // M/S transcoder is defined for exactly two channels.
    let mut ms = make("mid_side", json!({}), 48_000, 2, SampleFormat::F32);
    let mut c = Collect::default();
    let err = ms
        .push(&mut c, 0, &event(48_000, 1, SampleFormat::F32))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    assert!(c.out.is_empty());
    // Refused: the previous shape is kept and stereo still flows.
    assert_eq!(port_audio(&ms.input_ports()[0]).1, 2);
    ms.push(&mut c, 0, &f32_frame(&sine(440.0, 48_000, 480, 0.5), 2))
        .unwrap();
    // A rate change on the same layout is fine.
    ms.push(&mut c, 0, &event(44_100, 2, SampleFormat::F32))
        .unwrap();

    // Downmix matrix is built for one source layout.
    let mut dm = make(
        "downmix",
        json!({"to": "stereo"}),
        48_000,
        6,
        SampleFormat::F32,
    );
    let err = dm
        .push(&mut c, 0, &event(48_000, 2, SampleFormat::F32))
        .unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "{err:?}");
    // Rate change on the same 5.1 source: downmix output (stereo) moves
    // to the new rate → announced.
    let mut c = Collect::default();
    dm.push(&mut c, 0, &event(44_100, 6, SampleFormat::F32))
        .unwrap();
    let evs = c.events(0);
    assert_eq!(evs.len(), 1);
    assert_eq!(evs[0].channels, Some(2));
    assert_eq!(evs[0].sample_rate, Some(44_100));
}

// ───────────────────────── spectrogram (A + V) ─────────────────────────

#[test]
fn spectrogram_forwards_audio_event_video_shape_constant() {
    let mut s = make(
        "spectrogram",
        json!({"fps": 10, "width": 64, "height": 32, "fft_size": 256, "hop_size": 128}),
        44_100,
        1,
        SampleFormat::F32,
    );
    let mut c = Collect::default();
    // 0.25 s at 44.1 kHz → 2 video frames, 0.05 s carried toward the third.
    s.push(&mut c, 0, &f32_frame(&sine(1000.0, 44_100, 11_025, 0.5), 1))
        .unwrap();
    s.push(&mut c, 0, &event(48_000, 2, SampleFormat::F32))
        .unwrap();
    assert_eq!(c.events(0).len(), 1);
    assert!(c.events(1).is_empty(), "video shape does not change");
    // 0.25 s at 48 kHz → the third frame (0.05 s more) + 2 more.
    s.push(&mut c, 0, &f32_frame(&sine(1000.0, 48_000, 12_000, 0.5), 2))
        .unwrap();
    let pts: Vec<i64> = c
        .out
        .iter()
        .filter(|(p, _)| *p == 1)
        .filter_map(|(_, f)| f.pts())
        .collect();
    assert_eq!(pts.len(), 5, "{pts:?}");
    // Video time base stays 1/44100: frames every 4410 ticks throughout.
    for w in pts.windows(2) {
        assert_eq!(w[1] - w[0], 4_410, "{pts:?}");
    }
    assert_eq!(
        port_audio(&s.output_ports()[0]),
        (48_000, 2, SampleFormat::F32)
    );
}

// ──────────────────── every registered filter, generically ────────────────────

/// `(name, params, refuses_mono)`: every filter this crate registers.
fn every_filter() -> Vec<(&'static str, Value, bool)> {
    let simple = [
        "noise_gate",
        "echo",
        "compressor",
        "parallel_compressor",
        "limiter",
        "dc_blocker",
        "crossfeed",
        "stereo_widener",
        "reverb",
        "tremolo",
        "loudness_itu",
        "pitch_shift",
        "chorus",
        "flanger",
        "phaser",
        "white_noise",
        "pink_noise",
        "brown_noise",
        "silence_detector",
        "vibrato",
        "auto_pan",
        "bitcrusher",
        "tape_saturation",
        "hum_filter",
        "crossover",
        "envelope_follower",
        "de_esser",
        "wah",
        "octave_doubler",
        "adaptive_noise_gate",
        "exciter",
        "multiband_compressor",
        "stereo_imager",
        "talkbox",
        "transient_designer",
        "ducker",
        "gain_normalizer",
        "freq_shifter",
        "ring_modulator",
        "hard_clipper",
        "slew_limiter",
        "expander",
        "upward_compressor",
        "upward_expander",
        "true_peak_detector",
        "svf",
        "pre_emphasis",
        "de_emphasis",
        "median_filter",
        "crest_factor_meter",
        "stereo_correlation_meter",
        "zero_crossing_rate",
        "dc_offset_meter",
        "stereo_balance_meter",
        "dither",
    ];
    let mut v: Vec<(&'static str, Value, bool)> =
        simple.iter().map(|n| (*n, json!({}), false)).collect();
    v.push(("volume", json!({"gain_db": -3.0}), false));
    v.push(("resample", json!({"rate": 48_000}), false));
    v.push(("resample", json!({"ratio": 1}), false));
    v.push(("sample_format", json!({"format": "s32"}), false));
    v.push((
        "biquad",
        json!({"kind": "low_pass", "cutoff_hz": 3000.0}),
        false,
    ));
    v.push((
        "equalizer",
        json!({"bands": [{"kind": "peaking", "freq_hz": 1000, "gain_db": 6}]}),
        false,
    ));
    v.push(("comb_filter", json!({"delay_ms": 5.0, "gain": 0.5}), false));
    v.push((
        "spectrogram",
        json!({"fps": 25, "width": 32, "height": 16}),
        false,
    ));
    v.push(("mid_side", json!({}), true));
    v.push(("downmix", json!({"to": "mono"}), true));
    v
}

/// Every filter accepts a 44.1 kHz F32 stereo → 48 kHz S16 stereo
/// switch, then either adapts to mono or refuses it. Every frame and
/// event it emits after a change matches the output port it declares.
#[test]
fn every_filter_handles_rate_format_and_channel_changes() {
    for (name, params, refuses_mono) in every_filter() {
        let mut f = make(name, params.clone(), 44_100, 2, SampleFormat::F32);
        let mut c = Collect::default();
        let x = sine(700.0, 44_100, 2_205, 0.4);
        f.push(&mut c, 0, &f32_frame(&x, 2))
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        // Step 1: rate + format.
        f.push(&mut c, 0, &event(48_000, 2, SampleFormat::S16))
            .unwrap_or_else(|e| panic!("{name} rate/format change: {e}"));
        check_after_change(name, &mut *f, &mut c, 48_000, 2);

        // Step 2: stereo → mono.
        let r = f.push(&mut c, 0, &event(48_000, 1, SampleFormat::S16));
        if refuses_mono {
            let e = r.expect_err(name);
            assert!(matches!(e, Error::Unsupported(_)), "{name}: {e:?}");
            continue;
        }
        r.unwrap_or_else(|e| panic!("{name} channel change: {e}"));
        check_after_change(name, &mut *f, &mut c, 48_000, 1);

        // Step 3: mono → stereo at 44.1 kHz.
        f.push(&mut c, 0, &event(44_100, 2, SampleFormat::S16))
            .unwrap_or_else(|e| panic!("{name} back to stereo: {e}"));
        check_after_change(name, &mut *f, &mut c, 44_100, 2);
        f.flush(&mut c)
            .unwrap_or_else(|e| panic!("{name} flush: {e}"));
    }
}

fn check_after_change(
    name: &str,
    f: &mut dyn StreamFilter,
    c: &mut Collect,
    rate: u32,
    channels: u16,
) {
    let (in_rate, in_ch, in_fmt) = port_audio(&f.input_ports()[0]);
    assert_eq!(
        (in_rate, in_ch, in_fmt),
        (rate, channels, SampleFormat::S16),
        "{name} input port"
    );
    let (o_rate, o_ch, o_fmt) = port_audio(&f.output_ports()[0]);
    // Any event emitted on the audio port must announce exactly the
    // declared output shape.
    for ev in c.events(0) {
        assert_eq!(ev.sample_rate, Some(o_rate), "{name} event rate");
        assert_eq!(ev.channels, Some(o_ch), "{name} event channels");
        assert_eq!(ev.sample_format, Some(o_fmt), "{name} event format");
    }
    c.take();
    let x = sine(700.0, rate, (rate / 20) as usize, 0.4);
    f.push(c, 0, &s16_frame(&x, channels))
        .unwrap_or_else(|e| panic!("{name} process after change: {e}"));
    for a in audio_frames(&c.out, 0) {
        let expect = a.samples as usize * o_ch as usize * o_fmt.bytes_per_sample();
        let got: usize = a.data.iter().map(|p| p.len()).sum();
        assert_eq!(
            got, expect,
            "{name}: frame does not match the declared output shape"
        );
    }
    c.take();
}

/// A compressor's detector envelope is a level: it carries over the
/// switch, so gain reduction continues instead of the first new-rate
/// samples passing uncompressed.
#[test]
fn compressor_envelope_carries_over_rate_change() {
    use oxideav_audio_filter::Compressor;
    let p441 = AudioStreamParams::new(SampleFormat::F32, 1, 44_100);
    let p48 = p441.with_sample_rate(48_000);
    let frame = |x: &[f32]| match f32_frame(x, 1) {
        Frame::Audio(a) => a,
        _ => unreachable!(),
    };
    // Hard 10:1 above -20 dBFS, slow 200 ms attack so a fresh envelope
    // would need a long time to clamp down.
    let mut comp = Compressor::new(-20.0, 10.0, 200.0, 500.0, 0.0, 0.0);
    let loud = vec![0.9f32; 44_100];
    comp.process(&frame(&loud), p441).unwrap();
    assert!(comp.reconfigure(p441, p48).unwrap().is_empty());
    let out = comp.process(&frame(&vec![0.9f32; 480]), p48).unwrap();
    let y = f32_channel(&out[0], 1, 0);
    // Settled gain reduction keeps the first new-rate samples well
    // under the input level.
    assert!(
        y[0] < 0.3,
        "first post-switch sample {} not compressed",
        y[0]
    );
}
