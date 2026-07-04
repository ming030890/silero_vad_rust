pub mod silero_vad;

pub use silero_vad::model::{Detector, RawDetector, SpeechTimestamp, VadConfig};
pub use silero_vad::{Result, SileroError};
