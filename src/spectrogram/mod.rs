//! Streaming spectrogram renderer.
//!
//! Feeds [`AudioFrame`]s incrementally through a Hann/Hamming/Blackman
//! windowed STFT, accumulates per-FFT-column magnitudes, and finally renders
//! a `width x height` RGB image returned as a [`VideoFrame`] (encode it to
//! PNG/JPEG/etc. via the normal codec pipeline).
//!
//! Multi-channel input is mixed down to mono. Magnitudes are converted to
//! dBFS and clamped to `db_range`. Time downsampling uses max-pooling over
//! consecutive STFT columns; frequency mapping is linear by default.
//!
//! # Defaults
//!
//! * `fft_size`: 1024
//! * `hop_size`: 256
//! * `window`: Hann
//! * `db_range`: (-90.0, 0.0)
//! * `width`: 800
//! * `height`: 256
//! * `colormap`: Viridis

use crate::fft::real_fft;
use crate::sample_convert::decode_to_f32;
use oxideav_core::{
    AudioFrame, CodecParameters, Error, FilterContext, Frame, PixelFormat, PortParams, PortSpec,
    Result, SampleFormat, StreamFilter, TimeBase, VideoFrame, VideoPlane,
};

mod colormaps;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    Hann,
    Hamming,
    Blackman,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Colormap {
    Grayscale,
    Viridis,
    Magma,
}

/// Spectrogram configuration.
#[derive(Clone, Debug)]
pub struct SpectrogramOptions {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: Window,
    pub db_range: (f32, f32),
    pub width: u32,
    pub height: u32,
    pub colormap: Colormap,
}

impl Default for SpectrogramOptions {
    fn default() -> Self {
        Self {
            fft_size: 1024,
            hop_size: 256,
            window: Window::Hann,
            db_range: (-90.0, 0.0),
            width: 800,
            height: 256,
            colormap: Colormap::Viridis,
        }
    }
}

pub struct Spectrogram {
    opts: SpectrogramOptions,
    window: Vec<f32>,
    pending: Vec<f32>,
    /// Each column is `fft_size / 2 + 1` magnitudes (post-window FFT mag).
    columns: Vec<Vec<f32>>,
    /// Streaming-mode state — populated only when the filter is driven
    /// through [`StreamFilter::push`]. Finalize-mode callers
    /// (`feed` + `finalize_frame`) leave these at their defaults.
    stream: StreamState,
}

/// Per-stream runtime state for the [`StreamFilter`] driver path.
struct StreamState {
    /// Frozen after first push; derived from the upstream audio frame.
    input_sample_rate: u32,
    input_channels: u16,
    input_format: SampleFormat,
    /// Time-base copied from the first input audio frame. The emitted
    /// video stream inherits this time-base so A/V sync is a direct
    /// pts comparison downstream — no rescaling in the player.
    input_time_base: TimeBase,
    /// Frames per second of the emitted video stream.
    video_fps: u32,
    /// `input_sample_rate / video_fps`, computed once inputs land.
    samples_per_video_frame: u32,
    /// Cumulative mono-sample count since the first push.
    samples_elapsed: u64,
    /// Next `samples_elapsed` threshold at which a video frame is due.
    next_emit_at: u64,
    /// Video frame counter. Used to know which boundary we're at;
    /// the emitted pts is derived from this + `base_pts_in_tb` +
    /// `pts_step_in_tb`.
    emitted_video_frames: i64,
    /// pts of the first input audio frame, in `input_time_base` units.
    /// Captured on the first push; used as the anchor so non-zero
    /// starts (seek, midway join) are honoured rather than forcing
    /// video to restart at 0.
    base_pts_in_tb: i64,
    /// How far to advance the video pts per emission, in
    /// `input_time_base` units. Equals
    /// `samples_per_video_frame * input_time_base.den / (sample_rate * input_time_base.num)`.
    pts_step_in_tb: i64,
    /// Cached port descriptors; rebuilt after the first push lands the
    /// authoritative sample rate / channel count.
    input_ports: Vec<PortSpec>,
    output_ports: Vec<PortSpec>,
    /// True once we've configured the ports from the first push. Prior
    /// pushes would have used placeholders.
    initialized: bool,
}

impl StreamState {
    fn new(opts: &SpectrogramOptions) -> Self {
        let placeholder_tb = TimeBase::new(1, 48_000_i64);
        Self {
            input_sample_rate: 48_000,
            input_channels: 2,
            input_format: SampleFormat::F32,
            input_time_base: placeholder_tb,
            video_fps: 30,
            samples_per_video_frame: 48_000 / 30,
            samples_elapsed: 0,
            next_emit_at: 48_000 / 30,
            emitted_video_frames: 0,
            base_pts_in_tb: 0,
            pts_step_in_tb: (48_000 / 30) as i64,
            input_ports: vec![PortSpec::audio("audio", 48_000, 2, SampleFormat::F32)],
            output_ports: vec![
                PortSpec::audio("audio", 48_000, 2, SampleFormat::F32),
                PortSpec::video(
                    "video",
                    opts.width,
                    opts.height,
                    PixelFormat::Rgb24,
                    placeholder_tb,
                ),
            ],
            initialized: false,
        }
    }
}

impl Spectrogram {
    pub fn new(opts: SpectrogramOptions) -> Result<Self> {
        if !opts.fft_size.is_power_of_two() || opts.fft_size < 8 {
            return Err(Error::invalid(
                "spectrogram fft_size must be a power of two >= 8",
            ));
        }
        if opts.hop_size == 0 || opts.hop_size > opts.fft_size {
            return Err(Error::invalid(
                "spectrogram hop_size must be in (0, fft_size]",
            ));
        }
        if opts.width == 0 || opts.height == 0 {
            return Err(Error::invalid("spectrogram width/height must be non-zero"));
        }
        let window = build_window(opts.window, opts.fft_size);
        let stream = StreamState::new(&opts);
        Ok(Self {
            opts,
            window,
            pending: Vec::new(),
            columns: Vec::new(),
            stream,
        })
    }

    /// Anchor the per-stream cadence on the first input frame's pts.
    /// Stream-level shape (sample rate / channels / format / time_base)
    /// is no longer carried per-frame; configure it up-front via
    /// [`Spectrogram::with_audio_input`] or
    /// [`Spectrogram::with_codec_parameters`]. If neither is called the
    /// placeholder values from [`StreamState::new`] are used.
    fn init_stream_ports(&mut self, frame: &AudioFrame) {
        let ss = &mut self.stream;
        if ss.initialized {
            return;
        }
        ss.samples_elapsed = 0;
        ss.next_emit_at = ss.samples_per_video_frame as u64;
        ss.emitted_video_frames = 0;

        // Anchor the video stream on the audio timeline. `base_pts_in_tb`
        // starts at the audio frame's own pts (so non-zero starts from
        // seek / mid-stream join land correctly); `pts_step_in_tb`
        // converts one video-frame interval of samples into the audio
        // time_base's tick units so A/V sync is a direct pts compare.
        ss.base_pts_in_tb = frame.pts.unwrap_or(0);
        ss.pts_step_in_tb = pts_step_for_tb(
            ss.samples_per_video_frame as u64,
            ss.input_sample_rate as u64,
            ss.input_time_base,
        );
        ss.initialized = true;
    }

    /// Render a single RGB24 VideoFrame from the rolling-window column
    /// buffer, with 1-source-column-per-output-column mapping.
    ///
    /// Before the buffer has filled to `width` columns, the leftmost
    /// `width - columns.len()` pixels are left black (silent-floor
    /// colour) and the accumulated columns are right-aligned — this
    /// is the classic scrolling-waterfall look, not the "initial fat
    /// bands zooming in" behaviour that `finalize_rgb` produces when
    /// it stretches to fit.
    fn render_rolling_video_frame(&self) -> VideoFrame {
        let w = self.opts.width as usize;
        let h = self.opts.height as usize;
        let mut data = vec![0u8; w * h * 3];
        let n_cols = self.columns.len();
        let n_freq = self.opts.fft_size / 2 + 1;
        let ref_mag = self.opts.fft_size as f32 / 2.0;
        let (lo, hi) = self.opts.db_range;

        // Where the newest column should land. We draw column k of the
        // buffer at output x = `w - n_cols + k`, so col 0 is at the
        // leftmost occupied pixel and the last col is always at x=w-1.
        let left_skip = w.saturating_sub(n_cols);
        for col_idx in 0..n_cols {
            let x = left_skip + col_idx;
            let col = &self.columns[col_idx];
            for y in 0..h {
                let yy = h - 1 - y;
                let f0 = (yy * n_freq) / h;
                let f1 = (((yy + 1) * n_freq) / h).max(f0 + 1).min(n_freq);
                let mut max_mag = 0.0f32;
                for m in col.iter().take(f1).skip(f0) {
                    if *m > max_mag {
                        max_mag = *m;
                    }
                }
                let db = if max_mag <= 1.0e-12 {
                    -200.0
                } else {
                    20.0 * (max_mag / ref_mag).log10()
                };
                let t = ((db - lo) / (hi - lo)).clamp(0.0, 1.0);
                let idx = (t * 255.0) as u8;
                let (r, g, b) = colormap_lookup(self.opts.colormap, idx);
                let off = (y * w + x) * 3;
                data[off] = r;
                data[off + 1] = g;
                data[off + 2] = b;
            }
        }

        let stride = w * 3;
        let pts = self.stream.base_pts_in_tb
            + self.stream.emitted_video_frames * self.stream.pts_step_in_tb;
        // Stream-level video properties (format=Rgb24, width, height,
        // time_base) live on the output port spec — see `output_ports()`.
        VideoFrame {
            pts: Some(pts),
            planes: vec![VideoPlane { stride, data }],
        }
    }

    /// Trim `columns` so only the most recent `width` entries remain —
    /// older FFT frames scroll off the left edge of the display.
    fn trim_rolling_window(&mut self) {
        let cap = self.opts.width as usize;
        if self.columns.len() > cap {
            let excess = self.columns.len() - cap;
            self.columns.drain(..excess);
        }
    }

    /// Feed one audio frame. Multi-channel input is averaged to mono.
    /// Reads the cached `input_format` / `input_channels` (set up via
    /// [`Spectrogram::with_audio_input`] or
    /// [`Spectrogram::with_codec_parameters`]) since the frame itself
    /// no longer carries them.
    pub fn feed(&mut self, frame: &AudioFrame) -> Result<()> {
        let channels = decode_to_f32(frame, self.stream.input_format, self.stream.input_channels)?;
        let n_chan = channels.len();
        let n_samples = channels.first().map(|c| c.len()).unwrap_or(0);
        if n_chan == 0 || n_samples == 0 {
            return Ok(());
        }
        let inv_n = 1.0 / n_chan as f32;
        for s in 0..n_samples {
            let mut sum = 0.0;
            for ch in channels.iter().take(n_chan) {
                sum += ch[s];
            }
            self.pending.push(sum * inv_n);
        }

        // Drain windows
        while self.pending.len() >= self.opts.fft_size {
            let mut block = vec![0.0f32; self.opts.fft_size];
            for (i, slot) in block.iter_mut().enumerate().take(self.opts.fft_size) {
                *slot = self.pending[i] * self.window[i];
            }
            let bins = real_fft(&block);
            let mags: Vec<f32> = bins.iter().map(|c| c.magnitude()).collect();
            self.columns.push(mags);
            self.pending.drain(..self.opts.hop_size);
        }
        Ok(())
    }

    /// Render the accumulated columns into a `width * height * 3` RGB byte
    /// vector.
    pub fn finalize_rgb(&self) -> Vec<u8> {
        let w = self.opts.width as usize;
        let h = self.opts.height as usize;
        let mut out = vec![0u8; w * h * 3];
        if self.columns.is_empty() {
            return out;
        }
        let n_cols = self.columns.len();
        let n_freq = self.opts.fft_size / 2 + 1;

        // Time mapping (max-pool over column ranges)
        // For each output column x, source range is [x*n_cols/w, (x+1)*n_cols/w)
        // For each output row y (top = high freq), source range is similar
        for x in 0..w {
            let s0 = (x * n_cols) / w;
            let s1 = (((x + 1) * n_cols) / w).max(s0 + 1).min(n_cols);
            for y in 0..h {
                // y=0 is top row (high freq); flip so high freq at top
                let yy = h - 1 - y;
                let f0 = (yy * n_freq) / h;
                let f1 = (((yy + 1) * n_freq) / h).max(f0 + 1).min(n_freq);

                let mut max_mag = 0.0f32;
                for cx in s0..s1 {
                    let col = &self.columns[cx];
                    for m in col.iter().take(f1).skip(f0) {
                        if *m > max_mag {
                            max_mag = *m;
                        }
                    }
                }

                // Convert magnitude to dBFS. Reference: 0 dBFS == fft_size/2
                // (full-scale sine gives mag = N/2 in each bin).
                let ref_mag = self.opts.fft_size as f32 / 2.0;
                let db = if max_mag <= 1.0e-12 {
                    -200.0
                } else {
                    20.0 * (max_mag / ref_mag).log10()
                };
                let (lo, hi) = self.opts.db_range;
                let t = ((db - lo) / (hi - lo)).clamp(0.0, 1.0);
                let idx = (t * 255.0) as u8;
                let (r, g, b) = colormap_lookup(self.opts.colormap, idx);
                let off = (y * w + x) * 3;
                out[off] = r;
                out[off + 1] = g;
                out[off + 2] = b;
            }
        }
        out
    }

    /// Write the rendered spectrogram out as a sequence of RGB bytes.
    pub fn write_rgb<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let rgb = self.finalize_rgb();
        w.write_all(&rgb)
    }

    /// Render the accumulated columns into a single `PixelFormat::Rgb24`
    /// [`VideoFrame`]. Pipe this through any image/video encoder registered
    /// in the codec registry (PNG, JPEG, BMP, …) to write it to disk.
    /// Stream-level shape (Rgb24, width × height, identity time_base)
    /// must accompany the frame in a separate
    /// [`CodecParameters`](oxideav_core::CodecParameters); the
    /// downstream encoder reads them from there.
    pub fn finalize_frame(&self) -> VideoFrame {
        let rgb = self.finalize_rgb();
        let stride = self.opts.width as usize * 3;
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data: rgb }],
        }
    }

    /// Number of FFT columns accumulated so far.
    pub fn columns_recorded(&self) -> usize {
        self.columns.len()
    }

    /// Set the video output frame rate used when the spectrogram runs as
    /// a [`StreamFilter`]. Defaults to 30 fps. Must be called before the
    /// first `push`; takes effect at port initialisation time.
    pub fn with_video_fps(mut self, fps: u32) -> Self {
        self.stream.video_fps = fps.max(1);
        self
    }

    /// Pre-seed the audio input params so `input_ports()` /
    /// `output_ports()` return the correct `time_base` / sample_rate /
    /// channels / format BEFORE the first `push`. The pipeline reads
    /// these to synthesise `StreamInfo` for the sink's `start()` call,
    /// so getting them right up front prevents the engine from
    /// interpreting the video pts under a wrong time_base (which would
    /// break A/V sync with a large drift).
    pub fn with_audio_input(mut self, input: &PortSpec) -> Self {
        if let PortParams::Audio {
            sample_rate,
            channels,
            format,
        } = input.params
        {
            self.seed_input(format, channels, sample_rate);
        }
        self
    }

    /// Pre-seed the audio input params from a stream's
    /// [`CodecParameters`]. Equivalent to [`Self::with_audio_input`]
    /// but reads directly from the source-of-truth instead of an
    /// intermediate `PortSpec`. Frame-level reads (sample_rate /
    /// channels / format / time_base) used to populate this state from
    /// the first push; that pre-slim path is gone now.
    pub fn with_codec_parameters(mut self, params: &CodecParameters) -> Self {
        let format = params.sample_format.unwrap_or(SampleFormat::F32);
        let channels = params.resolved_channels().filter(|c| *c > 0).unwrap_or(2);
        let sample_rate = params.sample_rate.filter(|r| *r > 0).unwrap_or(48_000);
        self.seed_input(format, channels, sample_rate);
        self
    }

    fn seed_input(&mut self, format: SampleFormat, channels: u16, sample_rate: u32) {
        let ss = &mut self.stream;
        ss.input_sample_rate = sample_rate.max(1);
        ss.input_channels = channels.max(1);
        ss.input_format = format;
        // input_time_base is not carried on PortParams::Audio yet;
        // assume the common `(1, sample_rate)` convention used by the
        // executor's synthesised stream infos and every raw PCM decoder.
        // (The slimmed AudioFrame no longer carries a per-frame
        // override path.)
        ss.input_time_base = TimeBase::new(1, ss.input_sample_rate as i64);
        ss.samples_per_video_frame = ss.input_sample_rate.max(ss.video_fps) / ss.video_fps.max(1);
        ss.next_emit_at = ss.samples_per_video_frame as u64;
        ss.pts_step_in_tb = pts_step_for_tb(
            ss.samples_per_video_frame as u64,
            ss.input_sample_rate as u64,
            ss.input_time_base,
        );
        ss.input_ports = vec![PortSpec::audio(
            "audio",
            ss.input_sample_rate,
            ss.input_channels,
            ss.input_format,
        )];
        ss.output_ports = vec![
            PortSpec::audio(
                "audio",
                ss.input_sample_rate,
                ss.input_channels,
                ss.input_format,
            ),
            PortSpec::video(
                "video",
                self.opts.width,
                self.opts.height,
                PixelFormat::Rgb24,
                ss.input_time_base,
            ),
        ];
    }
}

impl StreamFilter for Spectrogram {
    fn input_ports(&self) -> &[PortSpec] {
        &self.stream.input_ports
    }

    fn output_ports(&self) -> &[PortSpec] {
        &self.stream.output_ports
    }

    fn push(&mut self, ctx: &mut dyn FilterContext, port: usize, frame: &Frame) -> Result<()> {
        if port != 0 {
            return Err(Error::invalid(format!(
                "spectrogram: unknown input port {port}"
            )));
        }
        let Frame::Audio(audio) = frame else {
            return Err(Error::invalid(
                "spectrogram: input port 0 only accepts audio frames",
            ));
        };

        self.init_stream_ports(audio);

        // Count mono-equivalent samples that passed through *before*
        // running the FFT accumulator so the video-frame cadence
        // reflects actual wall-clock input.
        let samples_this_frame = audio.samples as u64;
        self.stream.samples_elapsed += samples_this_frame;

        // Pass the audio through unchanged on port 0.
        ctx.emit(0, Frame::Audio(audio.clone()))?;

        // Accumulate FFT columns (shared with `feed`).
        self.feed(audio)?;
        self.trim_rolling_window();

        // Emit zero or more video frames (zero if the frame was shorter
        // than one video period, more if it was unusually long).
        while self.stream.samples_elapsed >= self.stream.next_emit_at {
            let vf = self.render_rolling_video_frame();
            ctx.emit(1, Frame::Video(vf))?;
            self.stream.emitted_video_frames += 1;
            self.stream.next_emit_at = (self.stream.emitted_video_frames as u64 + 1)
                * self.stream.samples_per_video_frame as u64;
        }
        Ok(())
    }

    fn flush(&mut self, ctx: &mut dyn FilterContext) -> Result<()> {
        // At EOF, emit one final rolling-window video frame if we've
        // ever been initialised and haven't emitted since the last
        // cadence boundary.
        if self.stream.initialized && !self.columns.is_empty() {
            let vf = self.render_rolling_video_frame();
            ctx.emit(1, Frame::Video(vf))?;
            self.stream.emitted_video_frames += 1;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Drop the rolling FFT-column buffer + any in-progress block of
        // pending audio samples; reset cadence + base-pts so the next
        // input frame after the seek anchors a fresh video timeline. The
        // port descriptors we already published stay valid (sample
        // rate / channels / format don't change across a seek), so we
        // keep `initialized=false` to force re-anchoring from the next
        // frame's pts.
        self.pending.clear();
        self.columns.clear();
        let opts = self.opts.clone();
        self.stream = StreamState::new(&opts);
        Ok(())
    }
}

/// Convert one video-frame interval (expressed in input audio samples)
/// into the audio time-base's tick count, so the video stream can
/// ride on the same clock as its audio source. Uses 64-bit math to
/// avoid overflow on the intermediate product for high-rate sources.
fn pts_step_for_tb(samples_per_frame: u64, sample_rate: u64, tb: TimeBase) -> i64 {
    let den = tb.0.den as u128;
    let num = (tb.0.num as u128).max(1);
    let rate = sample_rate.max(1) as u128;
    let step = (samples_per_frame as u128 * den) / (rate * num);
    step.min(i64::MAX as u128) as i64
}

fn build_window(kind: Window, n: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n];
    let denom = (n - 1) as f32;
    for (i, slot) in w.iter_mut().enumerate().take(n) {
        let phase = 2.0 * std::f32::consts::PI * i as f32 / denom;
        *slot = match kind {
            Window::Hann => 0.5 * (1.0 - phase.cos()),
            Window::Hamming => 0.54 - 0.46 * phase.cos(),
            Window::Blackman => 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos(),
        };
    }
    w
}

fn colormap_lookup(cm: Colormap, idx: u8) -> (u8, u8, u8) {
    match cm {
        Colormap::Grayscale => (idx, idx, idx),
        Colormap::Viridis => colormaps::VIRIDIS[idx as usize],
        Colormap::Magma => colormaps::MAGMA[idx as usize],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::{PortParams, SampleFormat, TimeBase};

    fn sine_frame(freq: f32, rate: u32, n: usize) -> AudioFrame {
        let mut bytes = Vec::with_capacity(n * 4);
        for i in 0..n {
            let t = i as f32 / rate as f32;
            let s = (2.0 * std::f32::consts::PI * freq * t).sin();
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![bytes],
        }
    }

    #[test]
    fn opts_default_is_sane() {
        let o = SpectrogramOptions::default();
        assert_eq!(o.fft_size, 1024);
        assert_eq!(o.hop_size, 256);
        assert_eq!(o.width, 800);
        assert_eq!(o.height, 256);
    }

    #[test]
    fn rgb_buffer_has_correct_size() {
        let opts = SpectrogramOptions {
            width: 32,
            height: 16,
            ..Default::default()
        };
        let s = Spectrogram::new(opts).unwrap();
        let rgb = s.finalize_rgb();
        assert_eq!(rgb.len(), 32 * 16 * 3);
    }

    #[test]
    fn finalize_frame_shape_matches_options() {
        let opts = SpectrogramOptions {
            width: 32,
            height: 16,
            ..Default::default()
        };
        let s = Spectrogram::new(opts).unwrap();
        let frame = s.finalize_frame();
        // Stream-level shape (Rgb24 / 32 × 16 / identity time_base) is
        // not on the frame any more — the contract is "stride = width * 3"
        // and "data = stride * height". Verify those.
        assert_eq!(frame.planes.len(), 1);
        assert_eq!(frame.planes[0].stride, 32 * 3);
        assert_eq!(frame.planes[0].data.len(), 32 * 16 * 3);
    }

    #[test]
    fn stream_filter_declares_audio_plus_video_ports() {
        let opts = SpectrogramOptions {
            width: 64,
            height: 32,
            ..Default::default()
        };
        let s = Spectrogram::new(opts).unwrap();
        let outs = s.output_ports();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].name, "audio");
        assert_eq!(outs[1].name, "video");
        assert!(matches!(
            &outs[1].params,
            PortParams::Video {
                width: 64,
                height: 32,
                format: PixelFormat::Rgb24,
                ..
            }
        ));
    }

    #[test]
    fn stream_filter_emits_audio_passthrough_and_video_at_cadence() {
        use oxideav_core::{CodecId, CodecParameters, Frame};

        struct Collect {
            out: Vec<(usize, Frame)>,
        }
        impl FilterContext for Collect {
            fn emit(&mut self, port: usize, frame: Frame) -> Result<()> {
                self.out.push((port, frame));
                Ok(())
            }
        }

        let opts = SpectrogramOptions {
            fft_size: 256,
            hop_size: 64,
            width: 32,
            height: 16,
            ..Default::default()
        };
        // Stream shape (sample_rate = 44.1k, mono, F32) used to be
        // sniffed off the first AudioFrame. With the slim it must be
        // seeded up front via with_codec_parameters / with_audio_input.
        let mut params = CodecParameters::audio(CodecId::new("pcm_f32le")).channels(1);
        params.sample_rate = Some(44_100);
        params.sample_format = Some(SampleFormat::F32);
        let mut s = Spectrogram::new(opts)
            .unwrap()
            .with_video_fps(30)
            .with_codec_parameters(&params);
        let mut ctx = Collect { out: Vec::new() };

        // Feed 1 second of 44100 Hz audio in 10 × 4410-sample chunks.
        // Expect: 10 audio frames passed through, and 30 video frames
        // emitted on port 1 (1 sec × 30 fps).
        for i in 0..10 {
            let frame = sine_frame(440.0, 44_100, 4_410);
            assert_eq!(frame.samples, 4_410);
            s.push(&mut ctx, 0, &Frame::Audio(frame)).unwrap();
            assert!(i < 1 || !ctx.out.is_empty());
        }

        let audio_count = ctx.out.iter().filter(|(p, _)| *p == 0).count();
        let video_count = ctx.out.iter().filter(|(p, _)| *p == 1).count();
        assert_eq!(audio_count, 10, "each input audio frame must pass through");
        assert_eq!(video_count, 30, "30 fps × 1 s must emit 30 video frames");

        // The video stream must ride on the input audio's time_base so
        // A/V sync is a direct pts compare downstream. With
        // sample_rate = 44_100 and fps = 30, one video-frame period is
        // 44_100 / 30 = 1470 sample-ticks under TimeBase(1, 44_100).
        // Stream-level video properties (Rgb24, 32 × 16, time_base)
        // live on the output port spec — checked separately below.
        let mut expected_pts = 0i64;
        for (p, f) in &ctx.out {
            if *p != 1 {
                continue;
            }
            let Frame::Video(vf) = f else {
                panic!("port 1 must carry video frames");
            };
            let pts = vf.pts.unwrap();
            assert_eq!(
                pts, expected_pts,
                "video pts must step by 1470 sample-ticks"
            );
            expected_pts += 1_470;
        }

        // Ports reflect the seeded input sample rate.
        let outs = s.output_ports();
        let PortParams::Audio { sample_rate, .. } = &outs[0].params else {
            panic!("expected audio on port 0")
        };
        assert_eq!(*sample_rate, 44_100);
        // Video output port carries the Rgb24 / width / height / tb
        // contract that was previously asserted on the frame itself.
        let PortParams::Video {
            width,
            height,
            format,
            time_base,
        } = &outs[1].params
        else {
            panic!("expected video on port 1")
        };
        assert_eq!(*width, 32);
        assert_eq!(*height, 16);
        assert_eq!(*format, PixelFormat::Rgb24);
        assert_eq!(*time_base, TimeBase::new(1, 44_100));
    }

    #[test]
    fn sine_produces_band_at_expected_frequency() {
        // 1 second of 440 Hz at 8000 Hz sample rate → bin = 440 / (Fs/N)
        // N = 1024, Fs = 8000 → bin width = 7.8125 Hz → 440 Hz at bin ~56
        let opts = SpectrogramOptions {
            fft_size: 1024,
            hop_size: 256,
            width: 64,
            height: 1024 / 2 + 1, // one row per bin so we can check directly
            db_range: (-90.0, 0.0),
            colormap: Colormap::Grayscale,
            window: Window::Hann,
        };
        // feed() reads the cached `input_format` / `input_channels`
        // values; seed them via with_codec_parameters so the F32 mono
        // 8 kHz fixture matches.
        use oxideav_core::{CodecId, CodecParameters};
        let mut params = CodecParameters::audio(CodecId::new("pcm_f32le")).channels(1);
        params.sample_rate = Some(8_000);
        params.sample_format = Some(SampleFormat::F32);
        let mut s = Spectrogram::new(opts.clone())
            .unwrap()
            .with_codec_parameters(&params);
        let frame = sine_frame(440.0, 8_000, 8_000);
        s.feed(&frame).unwrap();
        let rgb = s.finalize_rgb();
        let w = opts.width as usize;
        let h = opts.height as usize;
        // Expected bin
        let expected_bin = (440.0_f32 / (8000.0_f32 / 1024.0_f32)).round() as usize;
        // Expected row in image: high freq is at top (y=0), so y for bin b
        // is h - 1 - (b * h / nfreq) ; with h == nfreq this reduces to
        // y = h - 1 - b
        let target_y = h - 1 - expected_bin;
        // Take a sample column near the middle of the image
        let target_x = w / 2;
        let off = (target_y * w + target_x) * 3;
        let intensity = rgb[off] as i32;
        // Sample some other rows that should be much darker
        let mut peak_far = 0i32;
        for y in 0..h {
            if (y as i32 - target_y as i32).abs() < 5 {
                continue;
            }
            let off = (y * w + target_x) * 3;
            peak_far = peak_far.max(rgb[off] as i32);
        }
        assert!(
            intensity > peak_far + 50,
            "expected bright band at y={}, got {} but max-elsewhere = {}",
            target_y,
            intensity,
            peak_far
        );
    }
}
