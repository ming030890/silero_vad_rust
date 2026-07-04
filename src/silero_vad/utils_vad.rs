use crate::silero_vad::model::SileroVad16k;
use crate::silero_vad::{Result, SileroError};

use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

/// Timestamp describing the start/end of a detected speech segment.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechTimestamp {
    /// Start position (seconds or samples depending on context).
    pub start: f64,
    /// End position (seconds or samples depending on context).
    pub end: f64,
}

/// Configuration knobs mirroring the original Silero VAD Python helpers.
#[derive(Debug, Clone)]
pub struct VadParameters {
    /// Positive speech probability threshold.
    pub threshold: f32,
    /// Input sampling rate in Hz (8 kHz or 16 kHz).
    pub sampling_rate: u32,
    /// Minimum duration counted as speech (milliseconds).
    pub min_speech_duration_ms: u32,
    /// Maximum duration per speech chunk (seconds).
    pub max_speech_duration_s: f32,
    /// Minimum silence required to close an utterance (milliseconds).
    pub min_silence_duration_ms: u32,
    /// Number of milliseconds to pad before/after each segment.
    pub speech_pad_ms: u32,
    /// Convert timestamps to seconds instead of samples.
    pub return_seconds: bool,
    /// Decimal precision for returned timestamps (0 = integer).
    pub time_resolution: u32,
    /// Whether to keep intermediate probability traces for visualization.
    pub visualize_probs: bool,
    /// Optional custom negative threshold override.
    pub neg_threshold: Option<f32>,
    /// Optional override for model window size (samples).
    pub window_size_samples: Option<usize>,
    /// Silence required when a chunk hits `max_speech_duration_s` (milliseconds).
    pub min_silence_at_max_speech: u32,
    /// Whether to pick the longest possible silence window when splitting long speech.
    pub use_max_possible_silence: bool,
}

impl Default for VadParameters {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            sampling_rate: 16_000,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            return_seconds: false,
            time_resolution: 1,
            visualize_probs: false,
            neg_threshold: None,
            window_size_samples: None,
            min_silence_at_max_speech: 98,
            use_max_possible_silence: true,
        }
    }
}

fn make_error(message: impl Into<String>) -> SileroError {
    SileroError::Message(message.into())
}

/// Reads a mono or multi-channel audio file and decodes/resamples to `sampling_rate` mono f32 PCM.
pub fn read_audio<P: AsRef<Path>>(path: P, sampling_rate: u32) -> Result<Vec<f32>> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(make_error(format!(
            "Audio file not found: {}",
            path.display()
        )));
    }
    if sampling_rate == 0 {
        return Err(make_error("Target sampling rate must be greater than zero"));
    }

    let file = File::open(path).map_err(|e| make_error(format!("Failed to open file: {e}")))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let format_opts = FormatOptions::default();
    let metadata_opts = MetadataOptions::default();
    let decoder_opts = DecoderOptions::default();

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &format_opts, &metadata_opts)
        .map_err(|e| make_error(format!("Unsupported format: {e}")))?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .first()
        .ok_or_else(|| make_error("No audio track found in file"))?;

    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &decoder_opts)
        .map_err(|e| make_error(format!("Failed to create decoder: {e}")))?;

    let source_sr = track.codec_params.sample_rate
        .ok_or_else(|| make_error("Unknown source sample rate"))?;

    let mut samples = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(ref err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(err) => return Err(make_error(format!("Failed to read packet: {e}", e=err))),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                let channels = spec.channels.count();
                let mut sample_buf = SampleBuffer::<f32>::new(audio_buf.capacity() as u64, spec);
                sample_buf.copy_interleaved_ref(audio_buf);
                let interleaved = sample_buf.samples();

                if channels == 1 {
                    samples.extend_from_slice(interleaved);
                } else {
                    for frame in interleaved.chunks_exact(channels) {
                        let sum: f32 = frame.iter().sum();
                        samples.push(sum / channels as f32);
                    }
                }
            }
            Err(SymphoniaError::DecodeError(_)) => {
                // Decode errors on a packet are often skipped
                continue;
            }
            Err(err) => return Err(make_error(format!("Failed to decode packet: {err}"))),
        }
    }

    if source_sr != sampling_rate {
        samples = resample_linear(&samples, source_sr, sampling_rate)?;
    }

    Ok(samples)
}

/// Writes PCM samples to a mono WAV file.
pub fn save_audio<P: AsRef<Path>>(path: P, samples: &[f32], sampling_rate: u32) -> Result<()> {
    let path = path.as_ref();
    if sampling_rate == 0 {
        return Err(make_error("Sampling rate must be greater than zero"));
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sampling_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| make_error(format!("Failed to create WAV writer: {e}")))?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let quantized = (clamped * i16::MAX as f32).round() as i16;
        writer.write_sample(quantized)
            .map_err(|e| make_error(format!("Failed to write sample: {e}")))?;
    }

    writer.finalize().map_err(|e| make_error(format!("Failed to finalize WAV: {e}")))?;
    Ok(())
}

/// Concatenates speech chunks into a single buffer.
pub fn collect_chunks(
    timestamps: &[SpeechTimestamp],
    wav: &[f32],
    seconds: bool,
    sampling_rate: Option<u32>,
) -> Result<Vec<f32>> {
    if seconds && sampling_rate.is_none() {
        return Err(make_error(
            "sampling_rate must be provided when seconds is true",
        ));
    }

    if timestamps.is_empty() {
        return Ok(Vec::new());
    }

    let sr = if seconds { sampling_rate.unwrap() } else { 0 };
    let mut result = Vec::new();

    for ts in timestamps {
        let mut start = timestamp_to_index(ts.start, seconds, sr);
        let mut end = timestamp_to_index(ts.end, seconds, sr);
        if end <= start {
            continue;
        }
        start = start.min(wav.len());
        end = end.min(wav.len());
        if start < end {
            result.extend_from_slice(&wav[start..end]);
        }
    }

    Ok(result)
}

/// Removes speech chunks and keeps the background sections.
pub fn drop_chunks(
    timestamps: &[SpeechTimestamp],
    wav: &[f32],
    seconds: bool,
    sampling_rate: Option<u32>,
) -> Result<Vec<f32>> {
    if seconds && sampling_rate.is_none() {
        return Err(make_error(
            "sampling_rate must be provided when seconds is true",
        ));
    }

    if timestamps.is_empty() {
        return Ok(wav.to_vec());
    }

    let sr = if seconds { sampling_rate.unwrap() } else { 0 };
    let mut result = Vec::with_capacity(wav.len());
    let mut cursor = 0usize;

    for ts in timestamps {
        let start = timestamp_to_index(ts.start, seconds, sr).min(wav.len());
        let end = timestamp_to_index(ts.end, seconds, sr).min(wav.len());
        if start > cursor {
            result.extend_from_slice(&wav[cursor..start]);
        }
        cursor = cursor.max(end);
    }

    if cursor < wav.len() {
        result.extend_from_slice(&wav[cursor..]);
    }

    Ok(result)
}

/// Lightweight subset of [`VadParameters`] for streaming iterators.
#[derive(Debug, Clone)]
pub struct VadIteratorParams {
    /// Positive speech probability threshold.
    pub threshold: f32,
    /// Input sampling rate in Hz (8 kHz or 16 kHz).
    pub sampling_rate: u32,
    /// Minimum silence required to emit an `End` event (milliseconds).
    pub min_silence_duration_ms: u32,
    /// Padding applied to `Start`/`End` events (milliseconds).
    pub speech_pad_ms: u32,
}

impl Default for VadIteratorParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            sampling_rate: 16_000,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
        }
    }
}

/// Streaming VAD events emitted by [`VadIterator`].
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    /// Speech activity started at the provided position.
    Start(f64),
    /// Speech activity ended at the provided position.
    End(f64),
}

/// Iterator-style helper for reacting to speech start/end events.
pub struct VadIterator<'a> {
    /// Underlying Silero custom model used for inference.
    pub model: SileroVad16k<'a>,
    /// Iterator-level configuration knobs.
    pub params: VadIteratorParams,
    triggered: bool,
    temp_end: Option<usize>,
    current_sample: usize,
    min_silence_samples: f64,
    speech_pad_samples: f64,
}

impl<'a> std::fmt::Debug for VadIterator<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadIterator")
            .field("params", &self.params)
            .field("triggered", &self.triggered)
            .field("temp_end", &self.temp_end)
            .field("current_sample", &self.current_sample)
            .field("min_silence_samples", &self.min_silence_samples)
            .field("speech_pad_samples", &self.speech_pad_samples)
            .finish()
    }
}

impl<'a> VadIterator<'a> {
    /// Creates a new iterator with the given model and parameters.
    pub fn new(model: SileroVad16k<'a>, params: VadIteratorParams) -> Result<Self> {
        if params.sampling_rate != 16_000 {
            return Err(make_error(
                "VADIterator only supports 16000 Hz sampling rate",
            ));
        }

        let mut iterator = Self {
            model,
            params,
            triggered: false,
            temp_end: None,
            current_sample: 0,
            min_silence_samples: 0.0,
            speech_pad_samples: 0.0,
        };
        iterator.reset_states();
        Ok(iterator)
    }

    /// Resets the internal ONNX model state and iterator bookkeeping.
    pub fn reset_states(&mut self) {
        self.model.reset_states();
        self.triggered = false;
        self.temp_end = None;
        self.current_sample = 0;
        self.min_silence_samples =
            self.params.sampling_rate as f64 * self.params.min_silence_duration_ms as f64 / 1000.0;
        self.speech_pad_samples =
            self.params.sampling_rate as f64 * self.params.speech_pad_ms as f64 / 1000.0;
    }

    /// Consumes an audio chunk and optionally emits a [`VadEvent`].
    pub fn process_chunk(
        &mut self,
        chunk: &[f32],
        return_seconds: bool,
        time_resolution: u32,
    ) -> Result<Option<VadEvent>> {
        if chunk.is_empty() {
            return Ok(None);
        }

        let chunk_len = chunk.len();
        let speech_prob = self.model.predict_chunk(chunk)?;
        self.current_sample += chunk_len;

        if speech_prob >= self.params.threshold {
            if self.temp_end.is_some() {
                self.temp_end = None;
            }
            if !self.triggered {
                self.triggered = true;
                let pad = self.speech_pad_samples.floor() as isize;
                let mut start_index = self.current_sample as isize - pad - chunk_len as isize;
                if start_index < 0 {
                    start_index = 0;
                }
                let position = format_position(
                    start_index as usize,
                    return_seconds,
                    self.params.sampling_rate,
                    time_resolution,
                );
                return Ok(Some(VadEvent::Start(position)));
            }
        }

        let neg_threshold = self.params.threshold - 0.15;
        if speech_prob < neg_threshold && self.triggered {
            if self.temp_end.is_none() {
                self.temp_end = Some(self.current_sample);
            }

            let temp_end = self.temp_end.unwrap();
            let silence_duration = self.current_sample - temp_end;
            if (silence_duration as f64) < self.min_silence_samples {
                return Ok(None);
            }

            let pad = self.speech_pad_samples.floor() as isize;
            let mut end_index = temp_end as isize + pad - chunk_len as isize;
            if end_index < 0 {
                end_index = 0;
            }

            self.temp_end = None;
            self.triggered = false;

            let position = format_position(
                end_index as usize,
                return_seconds,
                self.params.sampling_rate,
                time_resolution,
            );
            return Ok(Some(VadEvent::End(position)));
        }

        Ok(None)
    }
}

/// Runs the classic Silero VAD post-processing pipeline over an audio buffer.
pub fn get_speech_timestamps(
    audio: &[f32],
    model: &mut SileroVad16k<'_>,
    params: &VadParameters,
) -> Result<Vec<SpeechTimestamp>> {
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    let mut audio_vec = audio.to_vec();
    let mut sampling_rate = params.sampling_rate;
    if sampling_rate == 0 {
        return Err(make_error("Sampling rate must be greater than zero"));
    }

    let mut step = 1usize;
    if sampling_rate > 16_000 && sampling_rate % 16_000 == 0 {
        let factor = (sampling_rate / 16_000) as usize;
        step = factor.max(1);
        sampling_rate = 16_000;
        audio_vec = audio_vec.into_iter().step_by(step).collect();
    }

    if sampling_rate != 16_000 {
        return Err(make_error(
            "Currently custom silero VAD model only supports 16000 (or multiply of 16000) sample rate",
        ));
    }

    let window_size_samples = 512usize;

    model.reset_states();
    let min_speech_samples = sampling_rate as f64 * params.min_speech_duration_ms as f64 / 1000.0;
    let speech_pad_samples = sampling_rate as f64 * params.speech_pad_ms as f64 / 1000.0;
    let max_speech_samples = if params.max_speech_duration_s.is_finite() {
        sampling_rate as f64 * params.max_speech_duration_s as f64
            - window_size_samples as f64
            - 2.0 * speech_pad_samples
    } else {
        f64::INFINITY
    };
    let min_silence_samples = sampling_rate as f64 * params.min_silence_duration_ms as f64 / 1000.0;
    let min_silence_samples_at_max_speech =
        sampling_rate as f64 * params.min_silence_at_max_speech as f64 / 1000.0;

    let audio_length_samples = audio_vec.len();
    let mut speech_probs = Vec::with_capacity(audio_length_samples.div_ceil(window_size_samples));

    let mut idx = 0usize;
    while idx < audio_length_samples {
        let end = (idx + window_size_samples).min(audio_length_samples);
        let mut chunk = vec![0.0_f32; window_size_samples];
        chunk[..(end - idx)].copy_from_slice(&audio_vec[idx..end]);
        let output = model.predict_chunk(&chunk)?;
        speech_probs.push(output);
        idx += window_size_samples;
    }

    let mut triggered = false;
    let mut current_start = 0usize;
    let mut has_current_start = false;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;
    let mut possible_ends: Vec<(usize, usize)> = Vec::new();
    let neg_threshold = params
        .neg_threshold
        .unwrap_or_else(|| (params.threshold - 0.15).max(0.01));

    let mut speeches: Vec<(usize, usize)> = Vec::new();

    for (i, &speech_prob) in speech_probs.iter().enumerate() {
        let cur_sample = window_size_samples * i;

        if speech_prob >= params.threshold && temp_end != 0 {
            let sil_dur = cur_sample.saturating_sub(temp_end);
            if (sil_dur as f64) > min_silence_samples_at_max_speech {
                possible_ends.push((temp_end, sil_dur));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur_sample;
            }
        }

        if speech_prob >= params.threshold && !triggered {
            triggered = true;
            current_start = cur_sample;
            has_current_start = true;
            continue;
        }

        if triggered
            && has_current_start
            && (cur_sample.saturating_sub(current_start) as f64) > max_speech_samples
        {
            if params.use_max_possible_silence && !possible_ends.is_empty() {
                let (best_end, dur) = possible_ends
                    .iter()
                    .cloned()
                    .max_by_key(|(_, dur)| *dur)
                    .unwrap();
                speeches.push((current_start, best_end));
                has_current_start = false;
                next_start = best_end + dur;
                if next_start < best_end + cur_sample {
                    current_start = next_start;
                    has_current_start = true;
                    triggered = true;
                } else {
                    triggered = false;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
                continue;
            } else if prev_end != 0 {
                speeches.push((current_start, prev_end));
                has_current_start = false;
                if next_start < prev_end {
                    triggered = false;
                } else {
                    current_start = next_start;
                    has_current_start = true;
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
                continue;
            } else {
                speeches.push((current_start, cur_sample));
                has_current_start = false;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        if speech_prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = cur_sample;
            }
            let sil_dur_now = cur_sample.saturating_sub(temp_end);

            if !params.use_max_possible_silence
                && (sil_dur_now as f64) > min_silence_samples_at_max_speech
            {
                prev_end = temp_end;
            }

            if (sil_dur_now as f64) < min_silence_samples {
                continue;
            } else {
                let end = temp_end;
                if has_current_start
                    && (end.saturating_sub(current_start) as f64) > min_speech_samples
                {
                    speeches.push((current_start, end));
                }
                triggered = false;
                has_current_start = false;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
                continue;
            }
        }
    }

    if has_current_start
        && (audio_length_samples.saturating_sub(current_start) as f64) > min_speech_samples
    {
        speeches.push((current_start, audio_length_samples));
    }

    if speeches.is_empty() {
        return Ok(Vec::new());
    }

    let mut segments = speeches;
    let speech_pad = speech_pad_samples.floor() as usize;

    for i in 0..segments.len() {
        if i == 0 {
            segments[i].0 = segments[i].0.saturating_sub(speech_pad);
        }

        if i != segments.len() - 1 {
            let silence_duration = segments[i + 1].0.saturating_sub(segments[i].1);
            if silence_duration < 2 * speech_pad {
                let adjust = silence_duration / 2;
                segments[i].1 = (segments[i].1 + adjust).min(audio_length_samples);
                segments[i + 1].0 = segments[i + 1].0.saturating_sub(adjust);
            } else {
                segments[i].1 = (segments[i].1 + speech_pad).min(audio_length_samples);
                segments[i + 1].0 = segments[i + 1].0.saturating_sub(speech_pad);
            }
        } else {
            segments[i].1 = (segments[i].1 + speech_pad).min(audio_length_samples);
        }
    }

    let mut result = Vec::with_capacity(segments.len());

    if params.return_seconds {
        let sr_f64 = sampling_rate as f64;
        let audio_length_seconds = audio_length_samples as f64 / sr_f64;
        for (start, end) in segments {
            if end <= start {
                continue;
            }
            let start_sec =
                round_with_resolution(start as f64 / sr_f64, params.time_resolution).max(0.0);
            let mut end_sec = round_with_resolution(end as f64 / sr_f64, params.time_resolution);
            if end_sec > audio_length_seconds {
                end_sec = audio_length_seconds;
            }
            if end_sec <= start_sec {
                continue;
            }
            result.push(SpeechTimestamp {
                start: start_sec,
                end: end_sec,
            });
        }
    } else {
        let scale = step as f64;
        for (start, end) in segments {
            if end <= start {
                continue;
            }
            result.push(SpeechTimestamp {
                start: start as f64 * scale,
                end: end as f64 * scale,
            });
        }
    }

    Ok(result)
}

pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(make_error("Sampling rate must be greater than zero"));
    }
    if samples.is_empty() || src_rate == dst_rate {
        return Ok(samples.to_vec());
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let mut output_len = ((samples.len() as f64) * ratio).round() as usize;
    if output_len == 0 {
        output_len = 1;
    }

    let mut output = Vec::with_capacity(output_len);
    let last_index = samples.len() - 1;

    for i in 0..output_len {
        let pos = (i as f64) / ratio;
        let base = pos.floor() as usize;
        let base = base.min(last_index);
        let frac = pos - base as f64;
        let next = if base >= last_index {
            last_index
        } else {
            base + 1
        };
        let v0 = samples[base];
        let v1 = samples[next];
        let frac_f32 = frac as f32;
        let interpolated = v0 * (1.0_f32 - frac_f32) + v1 * frac_f32;
        output.push(interpolated);
    }

    Ok(output)
}

fn timestamp_to_index(value: f64, seconds: bool, sampling_rate: u32) -> usize {
    if !value.is_finite() {
        return 0;
    }

    let scaled = if seconds {
        value * sampling_rate as f64
    } else {
        value
    };

    if scaled <= 0.0 {
        0
    } else {
        scaled.round() as usize
    }
}

fn format_position(index: usize, seconds: bool, sampling_rate: u32, time_resolution: u32) -> f64 {
    if seconds {
        let sr = sampling_rate as f64;
        round_with_resolution(index as f64 / sr, time_resolution)
    } else {
        index as f64
    }
}

fn round_with_resolution(value: f64, resolution: u32) -> f64 {
    if resolution == 0 {
        return value.round();
    }
    let factor = 10f64.powi(resolution as i32);
    (value * factor).round() / factor
}
