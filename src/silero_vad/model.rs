use crate::silero_vad::safetensors::SafeTensors;
use crate::silero_vad::tensor::Tensor;
use crate::silero_vad::{Result, SileroError};

pub struct SileroVad16k<'a> {
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
        })
    }


    pub fn reset_states(&mut self) {
        self.h = Tensor::new(vec![0.0f32; 128], vec![1, 128]);
        self.c = Tensor::new(vec![0.0f32; 128], vec![1, 128]);
        self.context = vec![0.0f32; 64];
    }

    pub fn predict_chunk(&mut self, chunk: &[f32]) -> Result<f32> {
        if chunk.len() != 512 {
            return Err(SileroError::Message(format!(
                "predict_chunk expects exactly 512 samples, got {}",
                chunk.len()
            )));
        }

        // Construct 576-sample input: context (64) + chunk (512)
        let mut x_input = Vec::with_capacity(576);
        x_input.extend_from_slice(&self.context);
        x_input.extend_from_slice(chunk);

        // Pad right with reflect padding by 64 (results in 640 samples)
        let x_tensor = Tensor::new(x_input, vec![1, 1, 576]);
        let padded = x_tensor.reflect_pad_1d(64); // [1, 1, 640]

        // 1. stft_conv
        let x = padded.conv1d(&self.stft_conv_w, None, 128, 0); // [1, 258, 4]

        // 2. Magnitude extraction
        let x = x.magnitude(129); // [1, 129, 4]

        // 3. Conv stack
        let x = x.conv1d(&self.conv1_w, Some(&self.conv1_b), 1, 1).relu(); // [1, 128, 4]
        let x = x.conv1d(&self.conv2_w, Some(&self.conv2_b), 2, 1).relu(); // [1, 64, 2]
        let x = x.conv1d(&self.conv3_w, Some(&self.conv3_b), 2, 1).relu(); // [1, 64, 1]
        let x = x.conv1d(&self.conv4_w, Some(&self.conv4_b), 1, 1).relu(); // [1, 128, 1]

        // Squeeze last dimension to [1, 128]
        let x_squeezed = Tensor::new(x.data.into_owned(), vec![1, 128]);

        // 4. LSTM Cell update
        let (h_next, c_next) = x_squeezed.lstm_cell(
            &self.lstm_w_ih,
            &self.lstm_w_hh,
            &self.lstm_b_ih,
            &self.lstm_b_hh,
            &self.h,
            &self.c,
        );
        self.h = h_next;
        self.c = c_next;

        // 5. Update context with the last 64 samples of the current chunk
        self.context.copy_from_slice(&chunk[448..512]);

        // 6. Decoder
        let h_unsqueezed = Tensor::new(self.h.data.to_vec(), vec![1, 128, 1]);
        let decoded = h_unsqueezed.relu();
        let prob_tensor = decoded.conv1d(&self.final_conv_w, Some(&self.final_conv_b), 1, 0).sigmoid(); // [1, 1, 1]

        // prob_tensor is [1, 1, 1], mean over sequence is just the value
        let prob = prob_tensor.data[0];

        Ok(prob)
    }
}

impl SileroVad16k<'static> {
    pub fn load_embedded() -> Result<Self> {
        let bytes = include_bytes!("data/silero_vad_16k.safetensors");
        Self::from_bytes(bytes)
    }
}

pub fn load_silero_vad() -> Result<SileroVad16k<'static>> {
    SileroVad16k::load_embedded()
}
