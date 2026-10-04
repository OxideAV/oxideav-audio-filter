//! Sample-format conversion (`sample_format` filter).
//!
//! Converts every frame from the stream's sample format to a target
//! [`SampleFormat`] (any layout: interleaved or planar, integer or
//! float), keeping the sample rate and channel count. Conversion goes
//! through the crate's `f32` interchange ([`crate::sample_convert`]),
//! which clamps to the destination range, so a float stream that
//! overshoots full scale saturates instead of wrapping.
//!
//! The source layout is reconciled per frame ([`reconcile`]) so an
//! undeclared or mis-declared stream format still converts correctly.
//!
//! The pipeline inserts this stage automatically in front of an encoder
//! whose declared input formats do not include the running format
//! (e.g. an `F32` decoder feeding an `S16`-only encoder).
//!
//! # Parameters (`sample_format` filter)
//! * `format` — target format name: `u8`, `s8`, `s16`, `s24`, `s32`,
//!   `f32`, `f64`, `u8p`, `s16p`, `s32p`, `f32p`, `f64p`.

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, Error, Result, SampleFormat};

/// Every sample format paired with its filter-parameter name.
const NAMES: &[(&str, SampleFormat)] = &[
    ("u8", SampleFormat::U8),
    ("s8", SampleFormat::S8),
    ("s16", SampleFormat::S16),
    ("s24", SampleFormat::S24),
    ("s32", SampleFormat::S32),
    ("f32", SampleFormat::F32),
    ("f64", SampleFormat::F64),
    ("u8p", SampleFormat::U8P),
    ("s16p", SampleFormat::S16P),
    ("s32p", SampleFormat::S32P),
    ("f32p", SampleFormat::F32P),
    ("f64p", SampleFormat::F64P),
];

/// Parse a sample-format name (`"s16"`, `"f32p"`, …; case-insensitive).
pub fn parse_sample_format(name: &str) -> Option<SampleFormat> {
    let lower = name.to_ascii_lowercase();
    NAMES.iter().find(|(n, _)| *n == lower).map(|&(_, f)| f)
}

/// Canonical filter-parameter name of `fmt` (inverse of
/// [`parse_sample_format`]).
pub fn sample_format_name(fmt: SampleFormat) -> Option<&'static str> {
    NAMES.iter().find(|(_, f)| *f == fmt).map(|&(n, _)| n)
}

/// The layout `frame` is actually in, given the stream's `declared`
/// format. Streams do not always declare what their decoder emits (a
/// container that only knows "MP3 audio" leaves the format unset, and
/// the pipeline then assumes a default), so the frame's plane count and
/// byte length win when they contradict the declaration: one plane per
/// channel means planar, and the per-sample byte width follows from the
/// plane length. A 4-byte width keeps the declared float/integer
/// family.
pub fn reconcile(frame: &AudioFrame, declared: SampleFormat, channels: u16) -> SampleFormat {
    let ch = usize::from(channels.max(1));
    let n = frame.samples as usize;
    let Some(first) = frame.data.first() else {
        return declared;
    };
    if n == 0 {
        return declared;
    }
    let planar = if ch > 1 {
        frame.data.len() == ch
    } else {
        declared.is_planar()
    };
    let per_plane = if planar { n } else { n * ch };
    if first.len() % per_plane != 0 {
        return declared;
    }
    let bps = first.len() / per_plane;
    if bps == declared.bytes_per_sample() && planar == declared.is_planar() {
        return declared;
    }
    let float = declared.is_float();
    match (bps, planar) {
        (1, false) => SampleFormat::U8,
        (1, true) => SampleFormat::U8P,
        (2, false) => SampleFormat::S16,
        (2, true) => SampleFormat::S16P,
        (3, false) => SampleFormat::S24,
        (4, false) if float => SampleFormat::F32,
        (4, false) => SampleFormat::S32,
        (4, true) if float => SampleFormat::F32P,
        (4, true) => SampleFormat::S32P,
        (8, false) => SampleFormat::F64,
        (8, true) => SampleFormat::F64P,
        _ => declared,
    }
}

/// Converts frames to a fixed target sample format.
pub struct FormatConvert {
    target: SampleFormat,
}

impl FormatConvert {
    /// Build a converter emitting `target`.
    pub fn new(target: SampleFormat) -> Self {
        Self { target }
    }

    /// The format this converter emits.
    pub fn target(&self) -> SampleFormat {
        self.target
    }
}

impl AudioFilter for FormatConvert {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        if params.channels == 0 {
            return Err(Error::invalid("sample_format: zero channels"));
        }
        let source = reconcile(input, params.format, params.channels);
        if source == self.target {
            return Ok(vec![input.clone()]);
        }
        let data = decode_to_f32(input, source, params.channels)?;
        Ok(vec![encode_from_f32(
            self.target,
            params.channels,
            input,
            &data,
        )?])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(format: SampleFormat) -> AudioStreamParams {
        AudioStreamParams {
            format,
            channels: 2,
            sample_rate: 48_000,
        }
    }

    #[test]
    fn names_round_trip() {
        for &(n, f) in NAMES {
            assert_eq!(parse_sample_format(n), Some(f));
            assert_eq!(sample_format_name(f), Some(n));
        }
        assert_eq!(parse_sample_format("S16"), Some(SampleFormat::S16));
        assert_eq!(parse_sample_format("bogus"), None);
    }

    #[test]
    fn f32_to_s16_scales_and_clamps() {
        let samples: [f32; 4] = [0.5, -0.5, 2.0, -2.0];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame {
            samples: 2,
            pts: Some(7),
            data: vec![bytes],
        };
        let mut c = FormatConvert::new(SampleFormat::S16);
        let out = c.process(&frame, params(SampleFormat::F32)).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pts, Some(7));
        let got: Vec<i16> = out[0].data[0]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert!((got[0] as i32 - 16_384).abs() <= 1, "{got:?}");
        assert!((got[1] as i32 + 16_384).abs() <= 1, "{got:?}");
        assert_eq!(got[2], i16::MAX);
        assert!(got[3] <= -32_767);
    }

    #[test]
    fn s16_planar_to_interleaved() {
        let l: Vec<u8> = [1i16, 2].iter().flat_map(|s| s.to_le_bytes()).collect();
        let r: Vec<u8> = [-1i16, -2].iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame {
            samples: 2,
            pts: None,
            data: vec![l, r],
        };
        let mut c = FormatConvert::new(SampleFormat::S16);
        let out = c.process(&frame, params(SampleFormat::S16P)).unwrap();
        let got: Vec<i16> = out[0].data[0]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(got, vec![1, -1, 2, -2]);
    }

    #[test]
    fn undeclared_planar_s16_is_detected() {
        // Declared F32 (the pipeline default for an unknown format) but
        // the decoder emitted S16 planar: reconcile from the planes.
        let l: Vec<u8> = [100i16, 200].iter().flat_map(|s| s.to_le_bytes()).collect();
        let r: Vec<u8> = [-100i16, -200]
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let frame = AudioFrame {
            samples: 2,
            pts: None,
            data: vec![l, r],
        };
        assert_eq!(reconcile(&frame, SampleFormat::F32, 2), SampleFormat::S16P);
        let mut c = FormatConvert::new(SampleFormat::S16);
        let out = c.process(&frame, params(SampleFormat::F32)).unwrap();
        let got: Vec<i16> = out[0].data[0]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(got, vec![100, -100, 200, -200]);
    }

    #[test]
    fn consistent_declaration_is_trusted() {
        let frame = AudioFrame {
            samples: 2,
            pts: None,
            data: vec![vec![0; 16]],
        };
        assert_eq!(reconcile(&frame, SampleFormat::F32, 2), SampleFormat::F32);
        assert_eq!(reconcile(&frame, SampleFormat::S32, 2), SampleFormat::S32);
    }

    #[test]
    fn same_format_is_passthrough() {
        let frame = AudioFrame {
            samples: 1,
            pts: None,
            data: vec![vec![1, 2, 3, 4]],
        };
        let mut c = FormatConvert::new(SampleFormat::S16);
        let out = c.process(&frame, params(SampleFormat::S16)).unwrap();
        assert_eq!(out[0].data, frame.data);
    }
}
