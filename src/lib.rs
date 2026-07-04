pub mod silero_vad;

pub use silero_vad::model::{Detector, VadConfig, SpeechTimestamp};
pub use silero_vad::{Result, SileroError};
