pub mod data;
pub mod model;
pub mod utils_vad;
pub mod safetensors;
pub mod tensor;

use std::fmt;

/// Unified error type returned by Silero VAD helpers.
#[derive(Debug)]
pub enum SileroError {
    /// Arbitrary message produced by downstream crates or custom guards.
    Message(String),
}

impl std::error::Error for SileroError {}

impl fmt::Display for SileroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SileroError::Message(msg) => write!(f, "{}", msg),
        }
    }
}

/// Convenience alias for results returned by public Silero VAD functions.
pub type Result<T> = std::result::Result<T, SileroError>;

impl From<anyhow::Error> for SileroError {
    fn from(value: anyhow::Error) -> Self {
        Self::Message(value.to_string())
    }
}
