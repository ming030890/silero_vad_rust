use crate::silero_vad::safetensors::SafeTensors;
use crate::silero_vad::tensor::Tensor;
use crate::silero_vad::{Result, SileroError};

/// The supported chunk size for the 16 kHz Silero VAD model: 512 samples (32 ms).
///
/// The model was trained on 512, 1024, and 1536 sample windows at 16 kHz.
/// We support only 512 for streaming use, matching earshot's frame contract.
const CHUNK_SIZE: usize = 512;

pub(crate) struct SileroVad16k<'a> {
    stft_conv_w: Tensor<'a>,
    conv1_w: Tensor<'a>,
    conv1_b: Tensor<'a>,
    conv2_w: Tensor<'a>,
    conv2_b: Tensor<'a>,
    conv3_w: Tensor<'a>,
    conv3_b: Tensor<'a>,
    conv4_w: Tensor<'a>,
    conv4_b: Tensor<'a>,
    lstm_w_ih: Tensor<'a>,
    lstm_w_hh: Tensor<'a>,
    lstm_b_ih: Tensor<'a>,
    lstm_b_hh: Tensor<'a>,
    final_conv_w: Tensor<'a>,
    final_conv_b: Tensor<'a>,
    
    // States
    h: Tensor<'static>,
    c: Tensor<'static>,
    context: Vec<f32>,

    // Pre-allocated ping-pong memory buffers
    buf_a: Vec<f32>,
    buf_b: Vec<f32>,
    buf_gates: Vec<f32>,
}

impl<'a> SileroVad16k<'a> {
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Self> {
        let safe = SafeTensors::parse(bytes)
            .map_err(|e| SileroError::Message(format!("Failed to parse safetensors: {}", e)))?;
        
        let view = safe.get("stft_conv.weight")?;
        let stft_conv_w = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv1.weight")?;
        let conv1_w = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv1.bias")?;
        let conv1_b = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv2.weight")?;
        let conv2_w = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv2.bias")?;
        let conv2_b = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv3.weight")?;
        let conv3_w = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv3.bias")?;
        let conv3_b = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv4.weight")?;
        let conv4_w = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("conv4.bias")?;
        let conv4_b = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("lstm_cell.weight_ih")?;
        let lstm_w_ih = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("lstm_cell.weight_hh")?;
        let lstm_w_hh = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("lstm_cell.bias_ih")?;
        let lstm_b_ih = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("lstm_cell.bias_hh")?;
        let lstm_b_hh = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("final_conv.weight")?;
        let final_conv_w = Tensor { data: view.data, shape: view.shape };
        let view = safe.get("final_conv.bias")?;
        let final_conv_b = Tensor { data: view.data, shape: view.shape };

        let h = Tensor::new(vec![0.0f32; 128], vec![1, 128]);
        let c = Tensor::new(vec![0.0f32; 128], vec![1, 128]);
        let context = vec![0.0f32; 64];

        // Allocating reusable buffers once at load time
        let buf_a = vec![0.0f32; 1032];
        let buf_b = vec![0.0f32; 1032];
        let buf_gates = vec![0.0f32; 512];

        Ok(SileroVad16k {
            stft_conv_w,
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            conv3_w,
            conv3_b,
            conv4_w,
            conv4_b,
            lstm_w_ih,
            lstm_w_hh,
            lstm_b_ih,
            lstm_b_hh,
            final_conv_w,
            final_conv_b,
            h,
            c,
            context,
            buf_a,
            buf_b,
            buf_gates,
        })
    }

    pub fn reset_states(&mut self) {
        self.h = Tensor::new(vec![0.0f32; 128], vec![1, 128]);
        self.c = Tensor::new(vec![0.0f32; 128], vec![1, 128]);
        self.context = vec![0.0f32; 64];
        self.buf_a.fill(0.0);
        self.buf_b.fill(0.0);
        self.buf_gates.fill(0.0);
    }

    pub fn predict_chunk(&mut self, chunk: &[f32]) -> Result<f32> {
        assert_eq!(chunk.len(), CHUNK_SIZE, "Silero VAD 16k requires exactly {} samples per chunk", CHUNK_SIZE);

        // 1. Prepend 64-sample context
        let mut x_input = [0.0f32; 576];
        x_input[..64].copy_from_slice(&self.context);
        x_input[64..64 + CHUNK_SIZE].copy_from_slice(chunk);

        // 2. Reflect-pad right by 64
        let x_tensor = Tensor::from_borrowed(&x_input[..64 + CHUNK_SIZE], vec![1, 1, 64 + CHUNK_SIZE]);
        x_tensor.reflect_pad_1d_into(64, &mut self.buf_a[..128 + CHUNK_SIZE]);

        // 3. STFT conv (reads buf_a, writes buf_b)
        let out_seq_len = (CHUNK_SIZE - 128) / 128 + 1; // 4 for 512
        let padded_tensor = Tensor::from_borrowed(&self.buf_a[..128 + CHUNK_SIZE], vec![1, 1, 128 + CHUNK_SIZE]);
        padded_tensor.conv1d_into(&self.stft_conv_w, None, 128, 0, &mut self.buf_b[..258 * out_seq_len]);

        // 4. Magnitude extraction (reads buf_b, writes buf_a)
        let stft_tensor = Tensor::from_borrowed(&self.buf_b[..258 * out_seq_len], vec![1, 258, out_seq_len]);
        stft_tensor.magnitude_into(129, &mut self.buf_a[..129 * out_seq_len]);

        // 5. Conv stack
        // Conv1 (reads buf_a, writes buf_b)
        let mag_tensor = Tensor::from_borrowed(&self.buf_a[..129 * out_seq_len], vec![1, 129, out_seq_len]);
        mag_tensor.conv1d_into(&self.conv1_w, Some(&self.conv1_b), 1, 1, &mut self.buf_b[..128 * out_seq_len]);
        for val in &mut self.buf_b[..128 * out_seq_len] {
            *val = val.max(0.0);
        }

        // Conv2 (reads buf_b, writes buf_a)
        let out_seq_len2 = (out_seq_len - 1) / 2 + 1;
        let conv1_relu_tensor = Tensor::from_borrowed(&self.buf_b[..128 * out_seq_len], vec![1, 128, out_seq_len]);
        conv1_relu_tensor.conv1d_into(&self.conv2_w, Some(&self.conv2_b), 2, 1, &mut self.buf_a[..64 * out_seq_len2]);
        for val in &mut self.buf_a[..64 * out_seq_len2] {
            *val = val.max(0.0);
        }

        // Conv3 (reads buf_a, writes buf_b)
        let out_seq_len3 = (out_seq_len2 - 1) / 2 + 1;
        let conv2_relu_tensor = Tensor::from_borrowed(&self.buf_a[..64 * out_seq_len2], vec![1, 64, out_seq_len2]);
        conv2_relu_tensor.conv1d_into(&self.conv3_w, Some(&self.conv3_b), 2, 1, &mut self.buf_b[..64 * out_seq_len3]);
        for val in &mut self.buf_b[..64 * out_seq_len3] {
            *val = val.max(0.0);
        }

        // Conv4 (reads buf_b, writes buf_a)
        let conv3_relu_tensor = Tensor::from_borrowed(&self.buf_b[..64 * out_seq_len3], vec![1, 64, out_seq_len3]);
        conv3_relu_tensor.conv1d_into(&self.conv4_w, Some(&self.conv4_b), 1, 1, &mut self.buf_a[..128 * out_seq_len3]);
        for val in &mut self.buf_a[..128 * out_seq_len3] {
            *val = val.max(0.0);
        }

        // 6. LSTM Cell
        let lstm_in_tensor = Tensor::from_borrowed(&self.buf_a[..128], vec![1, 128]);
        let mut h_next_buf = [0.0f32; 128];
        let mut c_next_buf = [0.0f32; 128];
        lstm_in_tensor.lstm_cell_into(
            &self.lstm_w_ih,
            &self.lstm_w_hh,
            &self.lstm_b_ih,
            &self.lstm_b_hh,
            &self.h,
            &self.c,
            &mut self.buf_gates,
            &mut h_next_buf,
            &mut c_next_buf,
        );
        self.h.data.to_mut().copy_from_slice(&h_next_buf);
        self.c.data.to_mut().copy_from_slice(&c_next_buf);

        // 7. Update context
        self.context.copy_from_slice(&chunk[CHUNK_SIZE - 64..]);

        // 8. Decoder: ReLU(h) -> final_conv -> sigmoid
        self.buf_b[..128].copy_from_slice(&self.h.data);
        for val in &mut self.buf_b[..128] {
            *val = val.max(0.0);
        }
        let relu_h_tensor = Tensor::from_borrowed(&self.buf_b[..128], vec![1, 128, 1]);
        relu_h_tensor.conv1d_into(&self.final_conv_w, Some(&self.final_conv_b), 1, 0, &mut self.buf_a[..1]);

        let prob = 1.0 / (1.0 + (-self.buf_a[0]).exp());
        Ok(prob)
    }
}

impl SileroVad16k<'static> {
    fn load_embedded() -> Result<Self> {
        let bytes = include_bytes!("data/silero_vad_16k.safetensors");
        Self::from_bytes(bytes)
    }
}

// ---------------------------------------------------------------------------
// Public API: Detector
// ---------------------------------------------------------------------------

/// Streaming voice-activity detector backed by the 16 kHz Silero VAD model.
///
/// This is the sole public interface of the crate.  It is designed as a
/// drop-in replacement for `earshot::Detector`.
///
/// # Example
///
/// ```rust,no_run
/// use silero_vad_rust::Detector;
///
/// let mut detector = Detector::default();
///
/// // Feed 512-sample (32 ms) chunks of 16 kHz mono f32 PCM:
/// let chunk = vec![0.0f32; 512];
/// let score = detector.predict(&chunk);
/// if score >= 0.5 {
///     println!("Speech detected");
/// }
///
/// // Reset state between independent audio streams:
/// detector.reset();
/// ```
pub struct Detector {
    model: SileroVad16k<'static>,
}

impl Default for Detector {
    fn default() -> Self {
        Self::new().expect("Failed to load embedded Silero VAD weights")
    }
}

impl Detector {
    /// Create a new detector, loading the embedded 16 kHz Silero VAD weights.
    pub fn new() -> Result<Self> {
        let model = SileroVad16k::load_embedded()?;
        Ok(Self { model })
    }

    /// Run inference on a 512-sample chunk of 16 kHz mono `f32` PCM.
    ///
    /// Returns a speech probability in `[0.0, 1.0]`.
    pub fn predict(&mut self, chunk: &[f32]) -> f32 {
        self.model.predict_chunk(chunk).unwrap_or(0.0)
    }

    /// Reset all internal state (LSTM hidden/cell, audio context).
    ///
    /// Call this before processing a new, independent audio stream.
    pub fn reset(&mut self) {
        self.model.reset_states();
    }

    /// Read a 16 kHz mono WAV file and return per-chunk speech probabilities.
    ///
    /// Resets state before processing. The file must be 16 kHz mono PCM
    /// (integer or float).
    pub fn predict_wav<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(path.as_ref())
            .map_err(|e| SileroError::Message(format!("Failed to open WAV: {}", e)))?;
        let spec = reader.spec();
        if spec.sample_rate != 16000 || spec.channels != 1 {
            return Err(SileroError::Message(format!(
                "Expected 16 kHz mono WAV, got {} Hz {} ch",
                spec.sample_rate, spec.channels
            )));
        }

        let audio: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap()).collect()
            }
        };

        self.reset();
        let mut probs = Vec::with_capacity(audio.len() / CHUNK_SIZE + 1);
        for chunk in audio.chunks(CHUNK_SIZE) {
            if chunk.len() == CHUNK_SIZE {
                probs.push(self.predict(chunk));
            } else if !chunk.is_empty() {
                let mut padded = [0.0f32; CHUNK_SIZE];
                padded[..chunk.len()].copy_from_slice(chunk);
                probs.push(self.predict(&padded));
            }
        }
        Ok(probs)
    }
}

