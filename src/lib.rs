pub mod silero_vad;

pub use silero_vad::model::{load_silero_vad, SileroVad16k};
pub use silero_vad::utils_vad::{
    collect_chunks, drop_chunks, get_speech_timestamps, read_audio, save_audio,
    SpeechTimestamp, VadEvent, VadIterator, VadIteratorParams, VadParameters,
};
pub use silero_vad::{Result, SileroError};
