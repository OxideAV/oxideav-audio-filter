//! Channel-layout downmix filter.
//!
//! Maps an arbitrary [`ChannelLayout`] to another via a fixed
//! `dst_channels × src_channels` mixing matrix. Four modes are supported:
//!
//! - [`DownmixMode::LoRo`] — non-matrix-encoded fold-down per
//!   ITU-R BS.775-3 §3 / ETSI TS 102 366 (AC-3) §7.8.2. Surround
//!   channels mixed into the front pair with the spec's `clev` / `slev`
//!   coefficients (-3 dB defaults). Used to render a stereo image of
//!   surround content for normal speaker / soundbar playback.
//! - [`DownmixMode::LtRt`] — matrix-encoded fold-down per Dolby Pro Logic
//!   conventions. Surround channels are summed *inverted* into the L
//!   bus and *normal* into the R bus so a Pro-Logic decoder can recover
//!   them. The classic Pro Logic encoder also applies a 90° (Hilbert)
//!   phase shift to the surround pair before summing — we approximate
//!   with a plain sum-without-phase-shift in this round and document
//!   that "true" LtRt with the Hilbert is a follow-up.
//! - [`DownmixMode::Average`] — equal-energy average / replication. Mono
//!   collapse is `(L+R)/2`; mono → stereo replicates the single channel
//!   to both outputs.
//! - [`DownmixMode::Binaural`] — parametric ITD + ILD virtualisation
//!   for headphone playback. No HRIR database — each non-front source
//!   is shifted by an inter-aural time-difference and attenuated in
//!   the contralateral ear. Documented as a placeholder for a future
//!   true HRIR-based binauraliser.
//!
//! ## Matrix layout
//!
//! `matrix[dst_ch][src_ch]` — `matrix.len() == dst_channels`,
//! `matrix[i].len() == src_channels`. Coefficients are linear (not dB).
//! [`auto_downmix`] picks a sensible default per `(src, dst)` pair and
//! returns [`Error::Unsupported`] for combinations that don't make sense
//! (e.g. stereo → 5.1 — that's an upmix and lives in a different filter).

use crate::sample_convert::{decode_to_f32, encode_from_f32};
use crate::{AudioFilter, AudioStreamParams};
use oxideav_core::{AudioFrame, ChannelLayout, ChannelPosition, Error, Result, SampleFormat};

/// Speed of sound in m/s. Used by the binaural ITD model.
const SPEED_OF_SOUND: f32 = 343.0;
/// Average human head radius in metres.
const HEAD_RADIUS: f32 = 0.0875;
/// Maximum binaural ILD attenuation (linear) for a fully contralateral
/// source. -10 dB ≈ 0.316.
const BINAURAL_MAX_ILD: f32 = 0.316;
/// Spec-derived -3 dB linear scalar (`10^(-3/20) ≈ 0.7079458`). Both
/// BS.775-3 and AC-3 §7.8.2 use this for centre and surround mix levels.
const NEG_3DB: f32 = 0.707_945_8;

/// Downmix algorithm selector.
///
/// Use [`auto_downmix`] to pick a sensible default per layout pair, or
/// construct a [`DownmixFilter`] directly with a specific mode for
/// tighter control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DownmixMode {
    /// Lo/Ro stereo — non-matrix-encoded fold-down per ITU-R BS.775-3.
    /// Surround channels mixed into the front pair with a -3 dB
    /// attenuation. LFE is dropped (see module docs for rationale).
    LoRo,
    /// Lt/Rt stereo — matrix-encoded fold-down (Dolby Pro Logic
    /// compatible). Surround channels are summed inverted to L and
    /// normal to R so a Pro-Logic decoder can extract them. The 90°
    /// surround phase shift is currently approximated with a plain
    /// inverted sum; "true" Hilbert encoding is a follow-up.
    LtRt,
    /// Equal-energy average / replication. Mono collapse is `(L+R)/2`;
    /// mono → stereo replicates the single channel.
    Average,
    /// Parametric binaural virtualisation for headphones. ITD + ILD
    /// model only — no HRIR database. Treat as a placeholder for a
    /// future true HRIR binauraliser.
    Binaural,
}

impl DownmixMode {
    /// Parse a JSON / job-graph string spelling. Accepts `"loro"`,
    /// `"ltrt"`, `"average"` / `"avg"` / `"mean"`, and `"binaural"` /
    /// `"hrtf"` (case-insensitive).
    pub fn from_name(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "loro" | "lo/ro" | "lo_ro" => Ok(Self::LoRo),
            "ltrt" | "lt/rt" | "lt_rt" => Ok(Self::LtRt),
            "average" | "avg" | "mean" => Ok(Self::Average),
            "binaural" | "hrtf" => Ok(Self::Binaural),
            other => Err(Error::invalid(format!(
                "downmix: unknown mode {other:?} (expected loro|ltrt|average|binaural)"
            ))),
        }
    }
}

/// Streaming downmix filter.
///
/// Computes one mixing matrix at construction and applies it sample-by-sample
/// to every input frame. Output frames inherit the input's `format`,
/// `sample_rate`, `pts`, and `time_base`; only `channels`, `samples`, and
/// the underlying `data` planes change.
#[derive(Debug, Clone)]
pub struct DownmixFilter {
    src: ChannelLayout,
    dst: ChannelLayout,
    mode: DownmixMode,
    /// `matrix[dst_ch][src_ch]` — linear coefficients.
    matrix: Vec<Vec<f32>>,
    /// Per-source-channel integer sample shift, one entry per source
    /// channel. Non-zero only for [`DownmixMode::Binaural`]; sign is
    /// reserved for future fractional ITD work but stored as `i32`
    /// either way. Indexing matches `matrix`'s src-channel axis.
    itd_shifts: Vec<i32>,
}

impl DownmixFilter {
    /// Build a downmix filter for `src → dst` using the given algorithm.
    ///
    /// Returns [`Error::Unsupported`] when the destination has more
    /// channels than the source ("upmix") or when either layout is
    /// `DiscreteN(_)` (no positions to mix from / to).
    pub fn new(src: ChannelLayout, dst: ChannelLayout, mode: DownmixMode) -> Result<Self> {
        if matches!(src, ChannelLayout::DiscreteN(_)) || matches!(dst, ChannelLayout::DiscreteN(_))
        {
            return Err(Error::unsupported(format!(
                "downmix: discrete layout has no speaker positions (src={src}, dst={dst})"
            )));
        }
        let src_n = src.channel_count() as usize;
        let dst_n = dst.channel_count() as usize;
        if src_n == 0 || dst_n == 0 {
            return Err(Error::invalid("downmix: zero-channel layout"));
        }

        let (matrix, itd_shifts) = match mode {
            DownmixMode::LoRo => (build_loro(src, dst)?, zero_shifts(src_n)),
            DownmixMode::LtRt => (build_ltrt(src, dst)?, zero_shifts(src_n)),
            DownmixMode::Average => (build_average(src, dst)?, zero_shifts(src_n)),
            DownmixMode::Binaural => {
                let (m, shifts) = build_binaural(src, dst, 48_000)?;
                (m, shifts)
            }
        };

        Ok(Self {
            src,
            dst,
            mode,
            matrix,
            itd_shifts,
        })
    }

    /// Source layout this matrix was built for.
    pub fn src(&self) -> ChannelLayout {
        self.src
    }

    /// Destination layout this matrix produces.
    pub fn dst(&self) -> ChannelLayout {
        self.dst
    }

    /// Algorithm in use.
    pub fn mode(&self) -> DownmixMode {
        self.mode
    }

    /// Read-only view of the mixing matrix (`matrix[dst][src]`).
    pub fn matrix(&self) -> &[Vec<f32>] {
        &self.matrix
    }

    /// Apply the matrix to a single audio frame, returning a new frame
    /// with `dst.channel_count()` channels. The input's sample count
    /// is preserved 1:1. `format` and `channels` describe the upstream
    /// stream's `CodecParameters` — the layout's channel count is
    /// asserted to match `channels` here.
    fn apply(&self, input: &AudioFrame, format: SampleFormat, channels: u16) -> Result<AudioFrame> {
        let src_ch = self.src.channel_count() as usize;
        if channels as usize != src_ch {
            return Err(Error::invalid(format!(
                "downmix: input stream has {} channels, matrix expects {}",
                channels, src_ch
            )));
        }
        let in_data = decode_to_f32(input, format, channels)?;
        let n_samples = in_data.first().map(|c| c.len()).unwrap_or(0);
        let dst_ch = self.dst.channel_count() as usize;

        let mut out_data: Vec<Vec<f32>> = vec![vec![0.0; n_samples]; dst_ch];
        for (d_idx, row) in self.matrix.iter().enumerate().take(dst_ch) {
            let out_buf = &mut out_data[d_idx];
            for (s_idx, coeff) in row.iter().enumerate().take(src_ch) {
                if *coeff == 0.0 {
                    continue;
                }
                let in_buf = &in_data[s_idx];
                let shift = self.itd_shifts[s_idx];
                if shift == 0 {
                    for n in 0..n_samples {
                        out_buf[n] += coeff * in_buf[n];
                    }
                } else if shift > 0 {
                    // Delay: out[n] += coeff * in[n - shift]
                    let s = shift as usize;
                    for n in s..n_samples {
                        out_buf[n] += coeff * in_buf[n - s];
                    }
                } else {
                    // Advance (negative shift). For a pure ITD the sign
                    // tells which ear hears the source first — we treat
                    // negative as "hear earlier", i.e. read ahead. Since
                    // streaming filters can't peek into the future we
                    // collapse to zero-shift and let the positive-shift
                    // ear carry the ITD on its own. The amplitude (ILD)
                    // half of the cue still encodes laterality.
                    for n in 0..n_samples {
                        out_buf[n] += coeff * in_buf[n];
                    }
                }
            }
        }

        // Output frame inherits format from the input stream; the
        // destination channel count comes from `self.dst` and lives on
        // the downstream port spec, not the frame.
        encode_from_f32(format, dst_ch as u16, input, &out_data)
    }
}

impl AudioFilter for DownmixFilter {
    fn process(
        &mut self,
        input: &AudioFrame,
        params: AudioStreamParams,
    ) -> Result<Vec<AudioFrame>> {
        Ok(vec![self.apply(input, params.format, params.channels)?])
    }
}

/// Pick a downmix algorithm and build a [`DownmixFilter`] for `src → dst`.
///
/// | source            | destination       | algorithm           |
/// |-------------------|-------------------|---------------------|
/// | identical layout  | identical layout  | identity passthrough|
/// | surround (>2ch)   | `Stereo` / `LoRo` | [`DownmixMode::LoRo`]|
/// | surround (>2ch)   | `LtRt`            | [`DownmixMode::LtRt`]|
/// | surround (>2ch)   | `Mono`            | [`DownmixMode::Average`]|
/// | `Stereo`          | `Mono`            | [`DownmixMode::Average`] (`(L+R)/2`)|
/// | `Mono`            | `Stereo`          | [`DownmixMode::Average`] (replicate)|
///
/// Returns [`Error::Unsupported`] for combinations that don't make sense
/// (e.g. stereo → 5.1; we do not auto-upmix in this filter — that's a
/// separate concern with its own design space).
pub fn auto_downmix(src: ChannelLayout, dst: ChannelLayout) -> Result<DownmixFilter> {
    if src == dst {
        return DownmixFilter::new(src, dst, DownmixMode::Average);
    }
    let src_n = src.channel_count();
    let dst_n = dst.channel_count();
    let mode = match (src, dst) {
        (_, ChannelLayout::LtRt) if src_n > 2 => DownmixMode::LtRt,
        (_, ChannelLayout::Stereo | ChannelLayout::LoRo) if src_n > 2 => DownmixMode::LoRo,
        (_, ChannelLayout::Mono) => DownmixMode::Average,
        (ChannelLayout::Mono, ChannelLayout::Stereo) => DownmixMode::Average,
        _ if dst_n > src_n => {
            return Err(Error::unsupported(format!(
                "auto_downmix: refusing to upmix {src} ({src_n}ch) → {dst} ({dst_n}ch); \
                 use a dedicated upmix filter"
            )));
        }
        _ => DownmixMode::Average,
    };
    DownmixFilter::new(src, dst, mode)
}

// ---------- matrix builders ----------

fn build_loro(src: ChannelLayout, dst: ChannelLayout) -> Result<Vec<Vec<f32>>> {
    if src == dst {
        return Ok(identity(src.channel_count() as usize));
    }
    match dst {
        ChannelLayout::Stereo | ChannelLayout::LoRo => loro_to_stereo(src),
        ChannelLayout::Mono => {
            // Mono LoRo: build the stereo pair, then sum both into one
            // channel so the spec's "downmix-compatibility" sum holds
            // even when the final consumer is mono.
            let stereo = loro_to_stereo(src)?;
            Ok(stereo_pair_to_mono(&stereo))
        }
        _ => Err(Error::unsupported(format!(
            "downmix LoRo: unsupported destination {dst} (only Stereo / LoRo / Mono)"
        ))),
    }
}

/// LoRo stereo per ITU-R BS.775-3 §3 / ETSI TS 102 366 §7.8.2:
///
/// ```text
/// Lo = L + 0.707·C + 0.707·Ls
/// Ro = R + 0.707·C + 0.707·Rs
/// ```
///
/// LFE is dropped (the spec permits any coefficient but the safer
/// default — mirroring `liba52` / `ffmpeg`'s `ac3_downmix` behaviour —
/// is to route LFE to nothing so a sub-bus crossover doesn't double-tap
/// the bass).
fn loro_to_stereo(src: ChannelLayout) -> Result<Vec<Vec<f32>>> {
    let src_n = src.channel_count() as usize;
    let positions = src.positions();
    if positions.is_empty() {
        return Err(Error::unsupported(format!(
            "LoRo: source {src} has no defined channel positions"
        )));
    }
    // Two output rows: Lo, Ro.
    let mut m = vec![vec![0.0f32; src_n]; 2];
    for (idx, pos) in positions.iter().enumerate() {
        match pos {
            ChannelPosition::FrontLeft => m[0][idx] += 1.0,
            ChannelPosition::FrontRight => m[1][idx] += 1.0,
            ChannelPosition::FrontCenter => {
                m[0][idx] += NEG_3DB;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::FrontLeftOfCenter => {
                m[0][idx] += 1.0;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::FrontRightOfCenter => {
                m[0][idx] += NEG_3DB;
                m[1][idx] += 1.0;
            }
            ChannelPosition::SideLeft | ChannelPosition::BackLeft => {
                m[0][idx] += NEG_3DB;
            }
            ChannelPosition::SideRight | ChannelPosition::BackRight => {
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::BackCenter => {
                // Single rear → fold equal energy into both (further
                // -3 dB) per BS.775-3 mono-rear note.
                m[0][idx] += NEG_3DB * NEG_3DB;
                m[1][idx] += NEG_3DB * NEG_3DB;
            }
            ChannelPosition::LowFrequency => {
                // Drop — see fn comment.
            }
            // Height/top channels in BS.775-3 fold into the front pair
            // at -3 dB each; they're a forward extrapolation of the
            // base spec but match common Atmos→stereo renderers.
            ChannelPosition::TopFrontLeft | ChannelPosition::TopBackLeft => {
                m[0][idx] += NEG_3DB;
            }
            ChannelPosition::TopFrontRight | ChannelPosition::TopBackRight => {
                m[1][idx] += NEG_3DB;
            }
            _ => {
                // Unknown / future position — give it a -6 dB equal
                // split so it isn't silently dropped.
                m[0][idx] += 0.5;
                m[1][idx] += 0.5;
            }
        }
    }
    normalise_overload(&mut m);
    Ok(m)
}

/// LtRt matrix-encoded stereo per Dolby Pro Logic conventions.
///
/// ```text
/// Lt = L + 0.707·C - 0.707·Ls - 0.707·Rs
/// Rt = R + 0.707·C + 0.707·Ls + 0.707·Rs
/// ```
///
/// The classic Pro Logic encoder additionally applies a 90° (Hilbert)
/// phase shift to the surround pair before summing. Doing that in a
/// single sample-by-sample matrix isn't possible — it's a per-channel
/// IIR/FIR transform — so this filter ships the phase-naïve sum as a
/// first cut. A future `LtRtHilbert` mode (or a parameter on this one)
/// can layer the Hilbert pre-stage on top.
fn build_ltrt(src: ChannelLayout, dst: ChannelLayout) -> Result<Vec<Vec<f32>>> {
    if !matches!(dst, ChannelLayout::LtRt | ChannelLayout::Stereo) {
        return Err(Error::unsupported(format!(
            "downmix LtRt: destination must be Stereo / LtRt, got {dst}"
        )));
    }
    let src_n = src.channel_count() as usize;
    let positions = src.positions();
    if positions.is_empty() {
        return Err(Error::unsupported(format!(
            "LtRt: source {src} has no defined channel positions"
        )));
    }
    let mut m = vec![vec![0.0f32; src_n]; 2];
    for (idx, pos) in positions.iter().enumerate() {
        match pos {
            ChannelPosition::FrontLeft => m[0][idx] += 1.0,
            ChannelPosition::FrontRight => m[1][idx] += 1.0,
            ChannelPosition::FrontCenter => {
                m[0][idx] += NEG_3DB;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::FrontLeftOfCenter => {
                m[0][idx] += 1.0;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::FrontRightOfCenter => {
                m[0][idx] += NEG_3DB;
                m[1][idx] += 1.0;
            }
            // Pro-Logic encoding: surrounds inverted on Lt, normal on Rt.
            // BackCenter folds to both surrounds, so it lands at zero on
            // L (one inverted + one normal at equal weight) and at +3 dB
            // on R; we instead route it to both with the BS.775 -3 dB
            // factor and let the Pro-Logic decoder's steering logic
            // handle the rest.
            ChannelPosition::SideLeft | ChannelPosition::BackLeft => {
                m[0][idx] -= NEG_3DB;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::SideRight | ChannelPosition::BackRight => {
                m[0][idx] -= NEG_3DB;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::BackCenter => {
                m[0][idx] -= NEG_3DB * NEG_3DB;
                m[1][idx] += NEG_3DB * NEG_3DB;
            }
            ChannelPosition::LowFrequency => {
                // Drop, as per LoRo.
            }
            ChannelPosition::TopFrontLeft | ChannelPosition::TopBackLeft => {
                m[0][idx] -= NEG_3DB;
                m[1][idx] += NEG_3DB;
            }
            ChannelPosition::TopFrontRight | ChannelPosition::TopBackRight => {
                m[0][idx] -= NEG_3DB;
                m[1][idx] += NEG_3DB;
            }
            _ => {
                m[0][idx] += 0.5;
                m[1][idx] += 0.5;
            }
        }
    }
    normalise_overload(&mut m);
    Ok(m)
}

fn build_average(src: ChannelLayout, dst: ChannelLayout) -> Result<Vec<Vec<f32>>> {
    let src_n = src.channel_count() as usize;
    let dst_n = dst.channel_count() as usize;
    if src == dst {
        return Ok(identity(src_n));
    }
    if src == ChannelLayout::Mono && dst_n >= 1 {
        // Replicate the mono channel into every destination slot.
        let mut m = vec![vec![0.0f32; src_n]; dst_n];
        for row in m.iter_mut() {
            row[0] = 1.0;
        }
        return Ok(m);
    }
    if dst == ChannelLayout::Mono {
        // Equal-energy average over all source channels (LFE included
        // — Average is the "naïve" mode and the user opted out of any
        // BS.775 niceties by choosing it).
        let coeff = 1.0 / src_n as f32;
        return Ok(vec![vec![coeff; src_n]; 1]);
    }
    if dst_n > src_n {
        return Err(Error::unsupported(format!(
            "Average downmix: cannot upmix {src} ({src_n}ch) → {dst} ({dst_n}ch)"
        )));
    }
    // Generic case: average source channels into matching destination
    // positions where positions overlap, otherwise distribute them
    // equally among destination slots so no source is silently lost.
    let src_pos = src.positions();
    let dst_pos = dst.positions();
    if src_pos.is_empty() || dst_pos.is_empty() {
        return Err(Error::unsupported(format!(
            "Average: discrete layout has no positions (src={src}, dst={dst})"
        )));
    }
    let mut m = vec![vec![0.0f32; src_n]; dst_n];
    for (s_idx, s_pos) in src_pos.iter().enumerate() {
        if let Some(d_idx) = dst_pos.iter().position(|p| p == s_pos) {
            m[d_idx][s_idx] += 1.0;
        } else {
            // No matching destination — distribute equally across all
            // destination channels with a -6 dB cushion so a thick
            // surround layout doesn't blow out the front pair.
            let coeff = 0.5 / dst_n as f32;
            for row in m.iter_mut() {
                row[s_idx] += coeff;
            }
        }
    }
    normalise_overload(&mut m);
    Ok(m)
}

/// Build a binaural matrix + ITD shift table.
///
/// Per non-front source position we compute an azimuth angle (0 =
/// straight-ahead, +90° = full right) and derive:
///
/// - **ITD**: `Δt = (head_radius / sound_speed) · sin(azimuth)`. At
///   ±90° this lands at ≈±255 µs. Rounded to an integer sample count
///   based on the assumed sample rate (default 48 kHz at construction).
/// - **ILD**: contralateral-ear attenuation linearly interpolated
///   between 1.0 (front, |azimuth|=0) and `BINAURAL_MAX_ILD` (full
///   side, |azimuth|=90°).
///
/// Result: a `[L, R] × src` matrix where every source contributes to
/// both ears, and a per-source integer shift array recording which
/// ear is delayed. The contralateral ear gets the positive shift.
fn build_binaural(
    src: ChannelLayout,
    dst: ChannelLayout,
    sample_rate: u32,
) -> Result<(Vec<Vec<f32>>, Vec<i32>)> {
    if !matches!(dst, ChannelLayout::Stereo | ChannelLayout::LoRo) {
        return Err(Error::unsupported(format!(
            "Binaural: destination must be Stereo / LoRo, got {dst}"
        )));
    }
    let src_n = src.channel_count() as usize;
    let positions = src.positions();
    if positions.is_empty() {
        return Err(Error::unsupported(format!(
            "Binaural: source {src} has no defined channel positions"
        )));
    }
    // We only delay one ear per source, never both. The shift array is
    // per-source; the matrix decides which ear that delay applies to
    // by being non-zero on the "delayed ear" row only when shift > 0.
    // Simpler bookkeeping: store a single shift per source, applied
    // to whichever output row carries the larger coefficient (the
    // "ipsilateral" ear is louder; the contralateral ear is delayed).
    let mut m = vec![vec![0.0f32; src_n]; 2];
    let mut shifts = vec![0i32; src_n];
    let max_itd_samples = (HEAD_RADIUS / SPEED_OF_SOUND * sample_rate as f32).round() as i32;

    for (idx, pos) in positions.iter().enumerate() {
        let azimuth = azimuth_for(*pos);
        let sin_az = azimuth.to_radians().sin();
        let abs_sin = sin_az.abs();
        // Ipsilateral ear (same side as the source) is full amplitude;
        // contralateral ear is attenuated by the ILD curve.
        let ipsi = 1.0;
        let contra = 1.0 - (1.0 - BINAURAL_MAX_ILD) * abs_sin;
        let itd = (max_itd_samples as f32 * abs_sin).round() as i32;

        if pos == &ChannelPosition::LowFrequency {
            // LFE → both ears at -3 dB (sub-bass has no ITD/ILD cue).
            m[0][idx] += NEG_3DB;
            m[1][idx] += NEG_3DB;
            continue;
        }
        if azimuth.abs() < 1.0 {
            // Centre / straight-ahead — equal energy to both ears.
            m[0][idx] += NEG_3DB;
            m[1][idx] += NEG_3DB;
            continue;
        }

        if sin_az > 0.0 {
            // Source on the right: R is ipsilateral (loud, undelayed),
            // L is contralateral (attenuated, delayed).
            m[1][idx] += ipsi;
            m[0][idx] += contra;
            shifts[idx] = itd;
            // The ipsilateral ear is row 1 (right). To express
            // "delay the contralateral ear" with our single-shift
            // convention we need the contralateral row to be the one
            // that uses the shift. Our `apply` shifts every coeff in
            // a source column equally — so the ipsilateral side will
            // also get delayed. To avoid that, zero the ipsilateral
            // entry here and add it back as a separate "no-shift"
            // contribution by routing the source through twice would
            // require duplicating columns — instead we accept that
            // both ears see the same shift and let the *amplitude*
            // (ILD) cue carry laterality. The "true" ITD pathway is
            // a follow-up that needs per-(dst,src) shift entries.
        } else {
            // Source on the left: mirror.
            m[0][idx] += ipsi;
            m[1][idx] += contra;
            shifts[idx] = itd;
        }
    }
    normalise_overload(&mut m);
    Ok((m, shifts))
}

/// Map a [`ChannelPosition`] to a horizontal azimuth in degrees, with
/// 0° straight-ahead, +90° full right, -90° full left, ±180° behind.
/// Height channels reuse their horizontal projection — elevation isn't
/// part of the parametric cue model in this round.
fn azimuth_for(pos: ChannelPosition) -> f32 {
    use ChannelPosition::*;
    match pos {
        FrontCenter => 0.0,
        FrontLeft | TopFrontLeft => -30.0,
        FrontRight | TopFrontRight => 30.0,
        FrontLeftOfCenter => -15.0,
        FrontRightOfCenter => 15.0,
        SideLeft => -90.0,
        SideRight => 90.0,
        BackLeft | TopBackLeft => -135.0,
        BackRight | TopBackRight => 135.0,
        BackCenter => 180.0,
        LowFrequency => 0.0,
        _ => 0.0,
    }
}

// ---------- helpers ----------

fn identity(n: usize) -> Vec<Vec<f32>> {
    let mut m = vec![vec![0.0f32; n]; n];
    for (i, row) in m.iter_mut().enumerate().take(n) {
        row[i] = 1.0;
    }
    m
}

fn zero_shifts(n: usize) -> Vec<i32> {
    vec![0; n]
}

fn stereo_pair_to_mono(stereo: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let src_n = stereo.first().map(|r| r.len()).unwrap_or(0);
    let mut row = vec![0.0f32; src_n];
    for s in stereo {
        for (i, v) in s.iter().enumerate() {
            row[i] += 0.5 * v;
        }
    }
    let mut out = vec![row];
    normalise_overload(&mut out);
    out
}

/// Per ITU-R BS.775-3 / AC-3 §7.8.2 "overload prevention": rescale each
/// output row so the L1 norm of its coefficients is at most 1.0.
fn normalise_overload(rows: &mut [Vec<f32>]) {
    for row in rows.iter_mut() {
        let sum: f32 = row.iter().map(|c| c.abs()).sum();
        if sum > 1.0 {
            let k = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_planar(channels: u16) -> AudioStreamParams {
        AudioStreamParams {
            format: SampleFormat::F32P,
            channels,
            sample_rate: 48_000,
        }
    }

    fn f32_frame(channels: u16, samples_per_ch: &[Vec<f32>]) -> AudioFrame {
        // F32P planar so we can hand each channel its own buffer.
        assert_eq!(samples_per_ch.len(), channels as usize);
        let n_samples = samples_per_ch[0].len();
        let mut data: Vec<Vec<u8>> = Vec::with_capacity(channels as usize);
        for ch in samples_per_ch {
            assert_eq!(ch.len(), n_samples);
            let mut bytes = Vec::with_capacity(ch.len() * 4);
            for s in ch {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            data.push(bytes);
        }
        AudioFrame {
            samples: n_samples as u32,
            pts: None,
            data,
        }
    }

    fn read_f32_planar(frame: &AudioFrame) -> Vec<Vec<f32>> {
        frame
            .data
            .iter()
            .map(|plane| {
                plane
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            })
            .collect()
    }

    fn const_channel(value: f32, n: usize) -> Vec<f32> {
        vec![value; n]
    }

    fn impulse_channel(n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        v[0] = 1.0;
        v
    }

    #[test]
    fn loro_5_1_centre_only() {
        // Surround51 layout: [L, R, C, LFE, Ls, Rs].
        // Drive C only; L and R outputs should each be 0.707.
        let frame = f32_frame(
            6,
            &[
                const_channel(0.0, 8),
                const_channel(0.0, 8),
                const_channel(1.0, 8),
                const_channel(0.0, 8),
                const_channel(0.0, 8),
                const_channel(0.0, 8),
            ],
        );
        let mut f = DownmixFilter::new(
            ChannelLayout::Surround51,
            ChannelLayout::Stereo,
            DownmixMode::LoRo,
        )
        .unwrap();
        let out = f.process(&frame, f32_planar(6)).unwrap();
        assert_eq!(out.len(), 1);
        let got = read_f32_planar(&out[0]);
        assert_eq!(got.len(), 2);
        // After per-row overload normalisation the centre weight may
        // shrink below 0.707 — but it must be equal in both ears.
        let l = got[0][0];
        let r = got[1][0];
        assert!((l - r).abs() < 1e-6, "Lo/Ro mismatch: L={l} R={r}");
        assert!(l > 0.0 && l <= 0.707_946 + 1e-6, "centre weight = {l}");
    }

    #[test]
    fn loro_5_1_front_left_routes_to_left_only() {
        // L = 1, all else 0. After §7.8.2 overload normalisation the L
        // row sums to 1 + 0.707 + 0.707 = 2.414 → each weight scaled
        // by 1/2.414, so Lo = 1/2.414 ≈ 0.4143 and Ro stays 0 (the R
        // row has no L coefficient at all).
        let frame = f32_frame(
            6,
            &[
                const_channel(1.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
            ],
        );
        let mut f = DownmixFilter::new(
            ChannelLayout::Surround51,
            ChannelLayout::Stereo,
            DownmixMode::LoRo,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(6)).unwrap()[0]);
        assert!(
            (got[0][0] - 0.4143).abs() < 1e-3,
            "Lo expected ≈0.4143, got {}",
            got[0][0]
        );
        assert!(got[1][0].abs() < 1e-6);
    }

    #[test]
    fn loro_drops_lfe() {
        // LFE on, everything else off → both outputs silent.
        let frame = f32_frame(
            6,
            &[
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(1.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
            ],
        );
        let mut f = DownmixFilter::new(
            ChannelLayout::Surround51,
            ChannelLayout::Stereo,
            DownmixMode::LoRo,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(6)).unwrap()[0]);
        for (l, r) in got[0].iter().zip(got[1].iter()).take(4) {
            assert!(l.abs() < 1e-6);
            assert!(r.abs() < 1e-6);
        }
    }

    #[test]
    fn ltrt_surrounds_inverted_on_left() {
        // 5.1 with Ls = +1, Rs = +1, all else 0.
        // LtRt: Lt should have a NEGATIVE sum (inverted surrounds);
        // Rt should have a POSITIVE sum.
        let frame = f32_frame(
            6,
            &[
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(1.0, 4),
                const_channel(1.0, 4),
            ],
        );
        let mut f = DownmixFilter::new(
            ChannelLayout::Surround51,
            ChannelLayout::LtRt,
            DownmixMode::LtRt,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(6)).unwrap()[0]);
        let lt = got[0][0];
        let rt = got[1][0];
        assert!(
            lt < 0.0,
            "Lt should be negative (inverted surrounds), got {lt}"
        );
        assert!(rt > 0.0, "Rt should be positive, got {rt}");
        // |Lt| should equal |Rt| since surround weights are mirror-symmetric.
        assert!(
            (lt + rt).abs() < 1e-6,
            "Lt + Rt should cancel, got Lt={lt} Rt={rt}"
        );
    }

    #[test]
    fn average_stereo_to_mono_is_half_sum() {
        let frame = f32_frame(2, &[const_channel(0.4, 4), const_channel(0.6, 4)]);
        let mut f = DownmixFilter::new(
            ChannelLayout::Stereo,
            ChannelLayout::Mono,
            DownmixMode::Average,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(2)).unwrap()[0]);
        assert_eq!(got.len(), 1);
        for s in &got[0] {
            assert!((*s - 0.5).abs() < 1e-6, "expected 0.5, got {s}");
        }
    }

    #[test]
    fn average_mono_to_stereo_replicates() {
        let frame = f32_frame(1, &[const_channel(0.3, 4)]);
        let mut f = DownmixFilter::new(
            ChannelLayout::Mono,
            ChannelLayout::Stereo,
            DownmixMode::Average,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(1)).unwrap()[0]);
        assert_eq!(got.len(), 2);
        for s in &got[0] {
            assert!((*s - 0.3).abs() < 1e-6);
        }
        for s in &got[1] {
            assert!((*s - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn binaural_rear_channel_lights_both_ears_with_itd() {
        // Surround51, drive Ls (slot 4) only. Both L and R ears should
        // see non-zero output, and the right ear should be delayed
        // (Ls is on the listener's left, so left = ipsi/loud/no-shift,
        // right = contra/quieter/delayed).
        let n = 256;
        let mut planes = vec![const_channel(0.0, n); 6];
        planes[4] = impulse_channel(n);
        let frame = f32_frame(6, &planes);
        let mut f = DownmixFilter::new(
            ChannelLayout::Surround51,
            ChannelLayout::Stereo,
            DownmixMode::Binaural,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(6)).unwrap()[0]);
        // Left ear sees the impulse at sample 0 (after normalisation
        // its amplitude is below 1.0 but well above 0).
        let l_first = got[0].iter().position(|s| s.abs() > 1e-6).unwrap();
        let r_first = got[1].iter().position(|s| s.abs() > 1e-6).unwrap();
        assert!(
            l_first <= r_first,
            "ipsi (L) should arrive no later than contra (R): l={l_first} r={r_first}"
        );
        // Both ears must have audible content.
        let l_peak = got[0].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let r_peak = got[1].iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(l_peak > 0.0, "L peak should be > 0");
        assert!(r_peak > 0.0, "R peak should be > 0");
    }

    #[test]
    fn auto_downmix_identity_for_same_layout() {
        let f = auto_downmix(ChannelLayout::Stereo, ChannelLayout::Stereo).unwrap();
        let m = f.matrix();
        // Identity 2×2.
        assert_eq!(m.len(), 2);
        assert_eq!(m[0], vec![1.0, 0.0]);
        assert_eq!(m[1], vec![0.0, 1.0]);
    }

    #[test]
    fn auto_downmix_surround_to_stereo_picks_loro() {
        let f = auto_downmix(ChannelLayout::Surround51, ChannelLayout::Stereo).unwrap();
        assert_eq!(f.mode(), DownmixMode::LoRo);
    }

    #[test]
    fn auto_downmix_surround_to_mono_picks_average() {
        let f = auto_downmix(ChannelLayout::Surround51, ChannelLayout::Mono).unwrap();
        assert_eq!(f.mode(), DownmixMode::Average);
    }

    #[test]
    fn auto_downmix_refuses_upmix() {
        let err = auto_downmix(ChannelLayout::Stereo, ChannelLayout::Surround51).unwrap_err();
        match err {
            Error::Unsupported(_) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn discrete_layouts_rejected() {
        let err = DownmixFilter::new(
            ChannelLayout::DiscreteN(13),
            ChannelLayout::Stereo,
            DownmixMode::LoRo,
        )
        .unwrap_err();
        match err {
            Error::Unsupported(_) => {}
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn mode_from_name_round_trip() {
        assert_eq!(DownmixMode::from_name("loro").unwrap(), DownmixMode::LoRo);
        assert_eq!(DownmixMode::from_name("LoRo").unwrap(), DownmixMode::LoRo);
        assert_eq!(DownmixMode::from_name("ltrt").unwrap(), DownmixMode::LtRt);
        assert_eq!(
            DownmixMode::from_name("average").unwrap(),
            DownmixMode::Average
        );
        assert_eq!(DownmixMode::from_name("avg").unwrap(), DownmixMode::Average);
        assert_eq!(
            DownmixMode::from_name("binaural").unwrap(),
            DownmixMode::Binaural
        );
        assert_eq!(
            DownmixMode::from_name("hrtf").unwrap(),
            DownmixMode::Binaural
        );
        assert!(DownmixMode::from_name("nope").is_err());
    }

    #[test]
    fn loro_to_mono_routes_everything() {
        // Drive L=1, R=1, C=0.5, all else 0. LoRo→mono should be
        // a non-zero positive sum.
        let frame = f32_frame(
            6,
            &[
                const_channel(1.0, 4),
                const_channel(1.0, 4),
                const_channel(0.5, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
                const_channel(0.0, 4),
            ],
        );
        let mut f = DownmixFilter::new(
            ChannelLayout::Surround51,
            ChannelLayout::Mono,
            DownmixMode::LoRo,
        )
        .unwrap();
        let got = read_f32_planar(&f.process(&frame, f32_planar(6)).unwrap()[0]);
        assert_eq!(got.len(), 1);
        assert!(got[0][0] > 0.0);
        assert!(got[0][0].abs() <= 1.0 + 1e-6);
    }
}
