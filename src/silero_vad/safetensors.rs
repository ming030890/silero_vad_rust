use std::borrow::Cow;
use std::collections::HashMap;
use serde::Deserialize;
use anyhow::{Result, anyhow};

#[derive(Deserialize, Debug)]
struct TensorMetadata {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Clone, Debug)]
pub struct TensorView<'a> {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Cow<'a, [f32]>,
}

pub struct SafeTensors<'a> {
    pub tensors: HashMap<String, TensorView<'a>>,
}

impl<'a> SafeTensors<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(anyhow!("File too short: length is {}", bytes.len()));
        }

        let header_size = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
        if bytes.len() < 8 + header_size {
            return Err(anyhow!("File corrupted: header size {} exceeds file length {}", header_size, bytes.len()));
        }

        let header_json = std::str::from_utf8(&bytes[8..8 + header_size])
            .map_err(|e| anyhow!("Failed to parse header as UTF-8: {}", e))?;

        let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_json)
            .map_err(|e| anyhow!("Failed to parse JSON header: {}", e))?;

        let data_start = 8 + header_size;
        let mut tensors = HashMap::new();

        for (name, val) in header {
            if name == "__metadata__" {
                continue;
            }

            let meta: TensorMetadata = serde_json::from_value(val)
                .map_err(|e| anyhow!("Invalid tensor metadata for '{}': {}", name, e))?;

            if meta.dtype != "F32" {
                return Err(anyhow!("Unsupported datatype '{}' for tensor '{}'. Only F32 is supported.", meta.dtype, name));
            }

            let start = data_start + meta.data_offsets[0];
            let end = data_start + meta.data_offsets[1];

            if end > bytes.len() {
                return Err(anyhow!("Tensor '{}' offset out of bounds: end is {} but file size is {}", name, end, bytes.len()));
            }

            let byte_slice = &bytes[start..end];
            if byte_slice.len() % 4 != 0 {
                return Err(anyhow!("Tensor '{}' data size is not a multiple of 4 bytes", name));
            }

            let expected_size = meta.shape.iter().product::<usize>();
            if byte_slice.len() / 4 != expected_size {
                return Err(anyhow!("Tensor '{}' shape product is {} but data has {} elements", name, expected_size, byte_slice.len() / 4));
            }

            // Perform aligned zero-copy cast if possible, otherwise copy
            let (prefix, floats, suffix) = unsafe { byte_slice.align_to::<f32>() };
            let data = if prefix.is_empty() && suffix.is_empty() {
                Cow::Borrowed(floats)
            } else {
                let mut vec = vec![0.0f32; byte_slice.len() / 4];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        byte_slice.as_ptr(),
                        vec.as_mut_ptr() as *mut u8,
                        byte_slice.len(),
                    );
                }
                Cow::Owned(vec)
            };

            tensors.insert(name, TensorView {
                dtype: meta.dtype,
                shape: meta.shape,
                data,
            });
        }

        Ok(SafeTensors { tensors })
    }

    pub fn get(&self, name: &str) -> Result<TensorView<'a>> {
        let view = self.tensors.get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in weights", name))?;
        Ok(TensorView {
            dtype: view.dtype.clone(),
            shape: view.shape.clone(),
            data: view.data.clone(),
        })
    }
}
