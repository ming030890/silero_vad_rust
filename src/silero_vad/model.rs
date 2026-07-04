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
        if chunk.len() != 512 {
            return Err(SileroError::Message(format!(
                "predict_chunk expects exactly 512 samples, got {}",
                chunk.len()
            )));
        }

        // 1. Stack-allocated input sequence of 576 samples
        let mut x_input = [0.0f32; 576];
        x_input[..64].copy_from_slice(&self.context);
        x_input[64..].copy_from_slice(chunk);

        // 2. Pad right with reflect padding by 64 (results in 640 samples in buf_a)
        let x_tensor = Tensor::from_borrowed(&x_input, vec![1, 1, 576]);
        x_tensor.reflect_pad_1d_into(64, &mut self.buf_a[..640]);

        // 3. stft_conv (reads buf_a[..640], writes buf_b[..1032])
        let padded_tensor = Tensor::from_borrowed(&self.buf_a[..640], vec![1, 1, 640]);
        padded_tensor.conv1d_into(&self.stft_conv_w, None, 128, 0, &mut self.buf_b[..1032]);

        // 4. Magnitude extraction (reads buf_b[..1032], writes buf_a[..516])
        let stft_tensor = Tensor::from_borrowed(&self.buf_b[..1032], vec![1, 258, 4]);
        stft_tensor.magnitude_into(129, &mut self.buf_a[..516]);

        // 5. Conv stack
        // Conv1 (reads buf_a[..516], writes buf_b[..512])
        let mag_tensor = Tensor::from_borrowed(&self.buf_a[..516], vec![1, 129, 4]);
        mag_tensor.conv1d_into(&self.conv1_w, Some(&self.conv1_b), 1, 1, &mut self.buf_b[..512]);
        // In-place ReLU on buf_b
        for val in &mut self.buf_b[..512] {
            *val = val.max(0.0);
        }

        // Conv2 (reads buf_b[..512], writes buf_a[..128])
        let conv1_relu_tensor = Tensor::from_borrowed(&self.buf_b[..512], vec![1, 128, 4]);
        conv1_relu_tensor.conv1d_into(&self.conv2_w, Some(&self.conv2_b), 2, 1, &mut self.buf_a[..128]);
        // In-place ReLU on buf_a
        for val in &mut self.buf_a[..128] {
            *val = val.max(0.0);
        }

        // Conv3 (reads buf_a[..128], writes buf_b[..64])
        let conv2_relu_tensor = Tensor::from_borrowed(&self.buf_a[..128], vec![1, 64, 2]);
        conv2_relu_tensor.conv1d_into(&self.conv3_w, Some(&self.conv3_b), 2, 1, &mut self.buf_b[..64]);
        // In-place ReLU on buf_b
        for val in &mut self.buf_b[..64] {
            *val = val.max(0.0);
        }

        // Conv4 (reads buf_b[..64], writes buf_a[..128])
        let conv3_relu_tensor = Tensor::from_borrowed(&self.buf_b[..64], vec![1, 64, 1]);
        conv3_relu_tensor.conv1d_into(&self.conv4_w, Some(&self.conv4_b), 1, 1, &mut self.buf_a[..128]);
        // In-place ReLU on buf_a
        for val in &mut self.buf_a[..128] {
            *val = val.max(0.0);
        }

        // 6. LSTM Cell (reads input from buf_a[..128])
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

        // Copy outputs back to self.h and self.c without allocating
        self.h.data.to_mut().copy_from_slice(&h_next_buf);
        self.c.data.to_mut().copy_from_slice(&c_next_buf);

        // 7. Update context with the last 64 samples of the current chunk
        self.context.copy_from_slice(&chunk[448..512]);

        // 8. Decoder
        // h has shape [1, 128, 1]. Relu it into buf_b[..128]
        self.buf_b[..128].copy_from_slice(&self.h.data);
        for val in &mut self.buf_b[..128] {
            *val = val.max(0.0);
        }

        // Conv1d from buf_b[..128] into buf_a[..1]
        let relu_h_tensor = Tensor::from_borrowed(&self.buf_b[..128], vec![1, 128, 1]);
        relu_h_tensor.conv1d_into(&self.final_conv_w, Some(&self.final_conv_b), 1, 0, &mut self.buf_a[..1]);

        // In-place sigmoid on buf_a[..1]
        self.buf_a[0] = 1.0 / (1.0 + (-self.buf_a[0]).exp());

        let prob = self.buf_a[0];
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
