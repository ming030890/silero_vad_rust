use std::borrow::Cow;

#[cfg(all(feature = "openblas", target_vendor = "apple"))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        transa: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

#[cfg(all(feature = "openblas", not(target_vendor = "apple")))]
unsafe extern "C" {
    fn cblas_sgemv(
        order: i32,
        transa: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
}

#[cfg(feature = "openblas")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(feature = "openblas")]
const CBLAS_NO_TRANS: i32 = 111;

#[derive(Clone, Debug)]
pub struct Tensor<'a> {
    pub data: Cow<'a, [f32]>,
    pub shape: Vec<usize>,
}

impl<'a> Tensor<'a> {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor {
            data: Cow::Owned(data),
            shape,
        }
    }

    pub fn from_borrowed(data: &'a [f32], shape: Vec<usize>) -> Self {
        Tensor {
            data: Cow::Borrowed(data),
            shape,
        }
    }

    pub fn reflect_pad_1d(&self, pad_right: usize) -> Tensor<'static> {
        assert_eq!(self.shape.len(), 3, "Input to reflect_pad_1d must be 3D [batch, channels, seq]");
        let batch = self.shape[0];
        let channels = self.shape[1];
        let seq_len = self.shape[2];

        let new_seq_len = seq_len + pad_right;
        let mut padded = vec![0.0f32; batch * channels * new_seq_len];

        for b in 0..batch {
            for c in 0..channels {
                for t in 0..new_seq_len {
                    let src_t = if t < seq_len {
                        t
                    } else {
                        // Reflect index
                        2 * seq_len - 2 - t
                    };
                    let src_idx = b * (channels * seq_len) + c * seq_len + src_t;
                    let dst_idx = b * (channels * new_seq_len) + c * new_seq_len + t;
                    padded[dst_idx] = self.data[src_idx];
                }
            }
        }

        Tensor::new(padded, vec![batch, channels, new_seq_len])
    }

    pub fn conv1d(&self, weight: &Tensor<'_>, bias: Option<&Tensor<'_>>, stride: usize, padding: usize) -> Tensor<'static> {
        assert_eq!(self.shape.len(), 3, "Conv1d input must be 3D [batch, in_channels, seq_len]");
        assert_eq!(weight.shape.len(), 3, "Conv1d weight must be 3D [out_channels, in_channels, kernel_size]");

        let batch = self.shape[0];
        let in_channels = self.shape[1];
        let seq_len = self.shape[2];

        let out_channels = weight.shape[0];
        let w_in_channels = weight.shape[1];
        let kernel_size = weight.shape[2];

        assert_eq!(in_channels, w_in_channels, "Conv1d input channels must match weight in_channels");

        let out_seq_len = (seq_len + 2 * padding - kernel_size) / stride + 1;
        let mut out_data = vec![0.0f32; batch * out_channels * out_seq_len];

        for b in 0..batch {
            for oc in 0..out_channels {
                let w_oc_base = oc * in_channels * kernel_size;
                for t in 0..out_seq_len {
                    let mut sum = 0.0f32;
                    for ic in 0..in_channels {
                        let in_ic_base = b * (in_channels * seq_len) + ic * seq_len;
                        let w_ic_base = w_oc_base + ic * kernel_size;
                        
                        let t_start = (t * stride) as isize - padding as isize;

                        if t_start >= 0 && (t_start + kernel_size as isize) <= seq_len as isize {
                            // FAST PATH: contiguous slice dot product (fully auto-vectorized by LLVM)
                            let start_idx = in_ic_base + t_start as usize;
                            let in_slice = &self.data[start_idx .. start_idx + kernel_size];
                            let w_slice = &weight.data[w_ic_base .. w_ic_base + kernel_size];
                            
                            let mut channel_sum = 0.0f32;
                            for k in 0..kernel_size {
                                unsafe {
                                    channel_sum += *in_slice.get_unchecked(k) * *w_slice.get_unchecked(k);
                                }
                            }
                            sum += channel_sum;
                        } else {
                            // SLOW PATH: fallback boundary checks (padding)
                            for k in 0..kernel_size {
                                let t_in = t_start + k as isize;
                                if t_in >= 0 && t_in < seq_len as isize {
                                    unsafe {
                                        let val = *self.data.get_unchecked(in_ic_base + t_in as usize);
                                        let w_val = *weight.data.get_unchecked(w_ic_base + k);
                                        sum += val * w_val;
                                    }
                                }
                            }
                        }
                    }
                    if let Some(ref b_val) = bias {
                        sum += b_val.data[oc];
                    }
                    let out_idx = b * (out_channels * out_seq_len) + oc * out_seq_len + t;
                    out_data[out_idx] = sum;
                }
            }
        }

        Tensor::new(out_data, vec![batch, out_channels, out_seq_len])
    }

    pub fn relu(&self) -> Tensor<'static> {
        let data = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sigmoid(&self) -> Tensor<'static> {
        let data = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn sqrt(&self) -> Tensor<'static> {
        let data = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor::new(data, self.shape.clone())
    }

    pub fn lstm_cell(
        &self,
        weight_ih: &Tensor<'_>,
        weight_hh: &Tensor<'_>,
        bias_ih: &Tensor<'_>,
        bias_hh: &Tensor<'_>,
        h: &Tensor<'_>,
        c: &Tensor<'_>,
    ) -> (Tensor<'static>, Tensor<'static>) {
        let batch = self.shape[0];
        let input_size = self.shape[1];
        let hidden_size = h.shape[1];

        assert_eq!(batch, 1, "LSTMCell batch size must be 1");
        assert_eq!(weight_ih.shape[0], 4 * hidden_size);
        assert_eq!(weight_ih.shape[1], input_size);
        assert_eq!(weight_hh.shape[0], 4 * hidden_size);
        assert_eq!(weight_hh.shape[1], hidden_size);

        let mut gates = vec![0.0f32; 4 * hidden_size];

        #[cfg(feature = "openblas")]
        {
            for g in 0..(4 * hidden_size) {
                unsafe {
                    *gates.get_unchecked_mut(g) = *bias_ih.data.get_unchecked(g) + *bias_hh.data.get_unchecked(g);
                }
            }
            unsafe {
                cblas_sgemv(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    (4 * hidden_size) as i32,
                    input_size as i32,
                    1.0,
                    weight_ih.data.as_ptr(),
                    input_size as i32,
                    self.data.as_ptr(),
                    1,
                    1.0,
                    gates.as_mut_ptr(),
                    1,
                );
                cblas_sgemv(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    (4 * hidden_size) as i32,
                    hidden_size as i32,
                    1.0,
                    weight_hh.data.as_ptr(),
                    hidden_size as i32,
                    h.data.as_ptr(),
                    1,
                    1.0,
                    gates.as_mut_ptr(),
                    1,
                );
            }
        }

        #[cfg(not(feature = "openblas"))]
        {
            for g in 0..(4 * hidden_size) {
                let mut val = unsafe {
                    *bias_ih.data.get_unchecked(g) + *bias_hh.data.get_unchecked(g)
                };
                let w_ih_base = g * input_size;
                for j in 0..input_size {
                    unsafe {
                        val += *self.data.get_unchecked(j) * *weight_ih.data.get_unchecked(w_ih_base + j);
                    }
                }
                let w_hh_base = g * hidden_size;
                for j in 0..hidden_size {
                    unsafe {
                        val += *h.data.get_unchecked(j) * *weight_hh.data.get_unchecked(w_hh_base + j);
                    }
                }
                gates[g] = val;
            }
        }

        let mut h_next = vec![0.0f32; hidden_size];
        let mut c_next = vec![0.0f32; hidden_size];

        for j in 0..hidden_size {
            let i = 1.0 / (1.0 + (-gates[j]).exp());
            let f = 1.0 / (1.0 + (-gates[hidden_size + j]).exp());
            let g = gates[2 * hidden_size + j].tanh();
            let o = 1.0 / (1.0 + (-gates[3 * hidden_size + j]).exp());

            c_next[j] = f * c.data[j] + i * g;
            h_next[j] = o * c_next[j].tanh();
        }

        (
            Tensor::new(h_next, vec![1, hidden_size]),
            Tensor::new(c_next, vec![1, hidden_size]),
        )
    }

    pub fn magnitude(&self, cutoff: usize) -> Tensor<'static> {
        // self shape: [batch, channels, seq]
        let batch = self.shape[0];
        let channels = self.shape[1];
        let seq_len = self.shape[2];

        assert_eq!(channels, 258);
        assert_eq!(cutoff, 129);

        let mut mag = vec![0.0f32; batch * cutoff * seq_len];

        for b in 0..batch {
            for c in 0..cutoff {
                for t in 0..seq_len {
                    let r_idx = b * (channels * seq_len) + c * seq_len + t;
                    let i_idx = b * (channels * seq_len) + (c + cutoff) * seq_len + t;
                    let val = (self.data[r_idx].powi(2) + self.data[i_idx].powi(2)).sqrt();
                    let dst_idx = b * (cutoff * seq_len) + c * seq_len + t;
                    mag[dst_idx] = val;
                }
            }
        }

        Tensor::new(mag, vec![batch, cutoff, seq_len])
    }

    pub fn reflect_pad_1d_into(&self, pad_right: usize, out: &mut [f32]) {
        let batch = self.shape[0];
        let channels = self.shape[1];
        let seq_len = self.shape[2];
        let new_seq_len = seq_len + pad_right;
        assert_eq!(out.len(), batch * channels * new_seq_len);

        for b in 0..batch {
            for c in 0..channels {
                for t in 0..new_seq_len {
                    let src_t = if t < seq_len { t } else { 2 * seq_len - 2 - t };
                    let src_idx = b * (channels * seq_len) + c * seq_len + src_t;
                    let dst_idx = b * (channels * new_seq_len) + c * new_seq_len + t;
                    out[dst_idx] = self.data[src_idx];
                }
            }
        }
    }

    pub fn conv1d_into(&self, weight: &Tensor<'_>, bias: Option<&Tensor<'_>>, stride: usize, padding: usize, out: &mut [f32]) {
        let batch = self.shape[0];
        let in_channels = self.shape[1];
        let seq_len = self.shape[2];
        let out_channels = weight.shape[0];
        let _w_in_channels = weight.shape[1];
        let kernel_size = weight.shape[2];
        let out_seq_len = (seq_len + 2 * padding - kernel_size) / stride + 1;
        
        assert_eq!(out.len(), batch * out_channels * out_seq_len);

        for b in 0..batch {
            for oc in 0..out_channels {
                let w_oc_base = oc * in_channels * kernel_size;
                for t in 0..out_seq_len {
                    let mut sum = 0.0f32;
                    for ic in 0..in_channels {
                        let in_ic_base = b * (in_channels * seq_len) + ic * seq_len;
                        let w_ic_base = w_oc_base + ic * kernel_size;
                        
                        let t_start = (t * stride) as isize - padding as isize;

                        if t_start >= 0 && (t_start + kernel_size as isize) <= seq_len as isize {
                            // FAST PATH: contiguous slice dot product (fully auto-vectorized by LLVM)
                            let start_idx = in_ic_base + t_start as usize;
                            let in_slice = &self.data[start_idx .. start_idx + kernel_size];
                            let w_slice = &weight.data[w_ic_base .. w_ic_base + kernel_size];
                            
                            let mut channel_sum = 0.0f32;
                            for k in 0..kernel_size {
                                unsafe {
                                    channel_sum += *in_slice.get_unchecked(k) * *w_slice.get_unchecked(k);
                                }
                            }
                            sum += channel_sum;
                        } else {
                            // SLOW PATH: fallback boundary checks (padding)
                            for k in 0..kernel_size {
                                let t_in = t_start + k as isize;
                                if t_in >= 0 && t_in < seq_len as isize {
                                    unsafe {
                                        let val = *self.data.get_unchecked(in_ic_base + t_in as usize);
                                        let w_val = *weight.data.get_unchecked(w_ic_base + k);
                                        sum += val * w_val;
                                    }
                                }
                            }
                        }
                    }
                    if let Some(ref b_val) = bias {
                        sum += b_val.data[oc];
                    }
                    let out_idx = b * (out_channels * out_seq_len) + oc * out_seq_len + t;
                    out[out_idx] = sum;
                }
            }
        }
    }

    pub fn relu_into(&self, out: &mut [f32]) {
        assert_eq!(self.data.len(), out.len());
        for i in 0..self.data.len() {
            unsafe {
                *out.get_unchecked_mut(i) = self.data.get_unchecked(i).max(0.0);
            }
        }
    }

    pub fn sigmoid_into(&self, out: &mut [f32]) {
        assert_eq!(self.data.len(), out.len());
        for i in 0..self.data.len() {
            unsafe {
                *out.get_unchecked_mut(i) = 1.0 / (1.0 + (-self.data.get_unchecked(i)).exp());
            }
        }
    }

    pub fn magnitude_into(&self, cutoff: usize, out: &mut [f32]) {
        let batch = self.shape[0];
        let channels = self.shape[1];
        let seq_len = self.shape[2];
        assert_eq!(out.len(), batch * cutoff * seq_len);

        for b in 0..batch {
            for c in 0..cutoff {
                for t in 0..seq_len {
                    let r_idx = b * (channels * seq_len) + c * seq_len + t;
                    let i_idx = b * (channels * seq_len) + (c + cutoff) * seq_len + t;
                    let val = (self.data[r_idx].powi(2) + self.data[i_idx].powi(2)).sqrt();
                    let dst_idx = b * (cutoff * seq_len) + c * seq_len + t;
                    out[dst_idx] = val;
                }
            }
        }
    }

    pub fn lstm_cell_into(
        &self,
        weight_ih: &Tensor<'_>,
        weight_hh: &Tensor<'_>,
        bias_ih: &Tensor<'_>,
        bias_hh: &Tensor<'_>,
        h: &Tensor<'_>,
        c: &Tensor<'_>,
        gates: &mut [f32],
        h_next: &mut [f32],
        c_next: &mut [f32],
    ) {
        let batch = self.shape[0];
        let input_size = self.shape[1];
        let hidden_size = h.shape[1];

        assert_eq!(batch, 1);
        assert_eq!(weight_ih.shape[0], 4 * hidden_size);
        assert_eq!(weight_ih.shape[1], input_size);
        assert_eq!(weight_hh.shape[0], 4 * hidden_size);
        assert_eq!(weight_hh.shape[1], hidden_size);
        assert_eq!(gates.len(), 4 * hidden_size);
        assert_eq!(h_next.len(), hidden_size);
        assert_eq!(c_next.len(), hidden_size);

        #[cfg(feature = "openblas")]
        {
            for g in 0..(4 * hidden_size) {
                unsafe {
                    *gates.get_unchecked_mut(g) = *bias_ih.data.get_unchecked(g) + *bias_hh.data.get_unchecked(g);
                }
            }
            unsafe {
                cblas_sgemv(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    (4 * hidden_size) as i32,
                    input_size as i32,
                    1.0,
                    weight_ih.data.as_ptr(),
                    input_size as i32,
                    self.data.as_ptr(),
                    1,
                    1.0,
                    gates.as_mut_ptr(),
                    1,
                );
                cblas_sgemv(
                    CBLAS_ROW_MAJOR,
                    CBLAS_NO_TRANS,
                    (4 * hidden_size) as i32,
                    hidden_size as i32,
                    1.0,
                    weight_hh.data.as_ptr(),
                    hidden_size as i32,
                    h.data.as_ptr(),
                    1,
                    1.0,
                    gates.as_mut_ptr(),
                    1,
                );
            }
        }

        #[cfg(not(feature = "openblas"))]
        {
            for g in 0..(4 * hidden_size) {
                let mut val = unsafe {
                    *bias_ih.data.get_unchecked(g) + *bias_hh.data.get_unchecked(g)
                };
                let w_ih_base = g * input_size;
                for j in 0..input_size {
                    unsafe {
                        val += *self.data.get_unchecked(j) * *weight_ih.data.get_unchecked(w_ih_base + j);
                    }
                }
                let w_hh_base = g * hidden_size;
                for j in 0..hidden_size {
                    unsafe {
                        val += *h.data.get_unchecked(j) * *weight_hh.data.get_unchecked(w_hh_base + j);
                    }
                }
                gates[g] = val;
            }
        }

        for j in 0..hidden_size {
            let i = 1.0 / (1.0 + (-gates[j]).exp());
            let f = 1.0 / (1.0 + (-gates[hidden_size + j]).exp());
            let g = gates[2 * hidden_size + j].tanh();
            let o = 1.0 / (1.0 + (-gates[3 * hidden_size + j]).exp());

            c_next[j] = f * c.data[j] + i * g;
            h_next[j] = o * c_next[j].tanh();
        }
    }
}
