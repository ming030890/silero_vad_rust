# Silero VAD Rust

A lightweight, pure Rust inference engine for the 16 kHz [Silero VAD](https://github.com/snakers4/silero-vad) model.

This crate runs Silero VAD without PyTorch, ONNX Runtime, tinygrad, libtorch, or any general-purpose machine learning runtime. The required neural network operators are implemented directly in Rust for this one fixed model.

The project is intentionally narrow in scope: it supports the 16 kHz Silero VAD model, processes audio in 512-sample chunks, maintains streaming state, and returns speech probabilities.

## Features

* **Pure Rust inference**

  * No PyTorch, ONNX Runtime, tinygrad, libtorch, or C++ runtime dependency.
  * The Silero-specific operators are implemented directly in Rust.

* **Embedded model weights**

  * The 16 kHz `.safetensors` weights are bundled with the crate using `include_bytes!`.

* **Zero-copy weight data**

  * The safetensors payload is mapped directly into `&[f32]` slices where possible.
  * Weight tensor data is not copied into a separate runtime format.

* **Streaming VAD**

  * Processes 16 kHz mono audio in 512-sample chunks.
  * Maintains the model context and LSTM state across chunks.
  * Supports explicit state reset between independent audio streams.

* **Small deployment footprint**

  * No external inference runtime or dynamic library setup.
  * Suitable for embedding in latency-sensitive audio applications.

* **Reference-tested**

  * Integration tests compare the Rust implementation against the official PyTorch reference output.

## Scope

This is not a general ONNX runtime or a general neural network framework.

It only implements the operators needed by the Silero VAD 16 kHz model:

* 1D convolution
* Reflect padding
* ReLU
* Sigmoid
* LSTM cell
* Elementwise arithmetic
* Mean reduction
* Learned STFT-style frontend used by Silero VAD

## Limitations

* Only the **16 kHz** Silero VAD model is supported.
* Audio must be mono `f32` PCM at **16 kHz**.
* Inference is performed in fixed **512-sample chunks**.
* The crate does not currently implement the full Silero timestamp post-processing pipeline.
* This is a model-specific inference path, not a replacement for ONNX Runtime.

## Quick Start

Add the crate to your project:

```toml
[dependencies]
silero-vad-rust = "0.1"
```

Then run chunk-by-chunk inference:

```rust
use silero_vad_rust::{load_silero_vad, read_audio};

fn main() -> anyhow::Result<()> {
    // Load embedded model weights.
    let mut model = load_silero_vad()?;

    // Read and decode an audio file, resampled to 16 kHz mono PCM.
    let mut audio = read_audio("speech.wav", 16_000)?;

    // Pad to a multiple of 512 samples.
    let remainder = audio.len() % 512;
    if remainder != 0 {
        audio.resize(audio.len() + 512 - remainder, 0.0);
    }

    // Run streaming inference.
    for chunk in audio.chunks_exact(512) {
        let speech_prob = model.predict_chunk(chunk)?;

        if speech_prob >= 0.5 {
            println!("Speech detected: {speech_prob:.4}");
        }
    }

    Ok(())
}
```

## Streaming API

The model is stateful. It keeps both:

* a 64-sample audio context buffer
* LSTM hidden and cell state

Call `reset_state()` before processing a new independent audio stream:

```rust
let mut model = load_silero_vad()?;

// First stream
for chunk in first_audio.chunks_exact(512) {
    let prob = model.predict_chunk(chunk)?;
}

// New unrelated stream
model.reset_state();

for chunk in second_audio.chunks_exact(512) {
    let prob = model.predict_chunk(chunk)?;
}
```

## Demo CLI

Run the demo on a WAV file:

```bash
cargo run --release -- <path-to-wav-file>
```

The CLI prints per-chunk speech probabilities and simple speech/silence decisions.

Example output:

```text
chunk=0000 prob=0.0123 speech=false
chunk=0001 prob=0.0187 speech=false
chunk=0002 prob=0.7321 speech=true
chunk=0003 prob=0.8914 speech=true
```

## Benchmarks

Benchmarks were run on macOS with Apple Silicon using the same 60-second WAV file.

The comparison includes:

1. this custom Rust implementation
2. this implementation with optional BLAS acceleration
3. the official Silero `rust-example` baseline using the ONNX Runtime `ort` crate

| Metric                      |              Pure Rust | With OpenBLAS / Accelerate | Official Rust ORT Baseline |
| :-------------------------- | ---------------------: | -------------------------: | -------------------------: |
| Peak memory usage / Max RSS |            **9.63 MB** |               **10.99 MB** |               **42.24 MB** |
| Total command duration      |                 0.65 s |                 **0.18 s** |                     0.47 s |
| VAD Loop Real-time factor   |                 128.8x |                **1444.1x** |                     387.7x |
| Avg Chunk Latency           |                 229 µs |                **22.1 µs** |                    82.5 µs |
| ONNX Runtime dependency     |                     No |                         No |                        Yes |
| External runtime required   |                     No |               BLAS backend |           `libonnxruntime` |
| Weight loading              | Zero-copy weight views |     Zero-copy weight views |  ONNX Runtime session init |

Our GEMV-optimized, zero-allocation custom Rust engine is **3.72x faster in raw VAD inference** than the official single-threaded ONNX Runtime baseline (22.1 µs vs 82.5 µs chunk latency). In addition, because of the zero-copy weight memory layout, it initializes 55x faster (0.9 ms vs 56.5 ms), reducing total command duration to **0.18 seconds** (2.6x faster than ORT) while maintaining a 4x smaller memory footprint.

Run the benchmarks:

```bash
# Pure Rust implementation
cargo run --release --bin benchmark

# With BLAS acceleration
cargo run --release --bin benchmark --features openblas
```

## Testing

Run the test suite:

```bash
cargo test
```

The integration tests compare Rust inference outputs against reference probabilities generated from the official PyTorch implementation.

The default tolerance is:

```text
1e-5
```

The tests cover:

* silence
* synthetic audio
* real speech fixtures
* recurrent state behavior
* chunk-by-chunk streaming parity

## Design Notes

Silero VAD is a good fit for a bespoke inference implementation because the model is small, fixed, and streaming-oriented.

A general inference runtime is useful when an application needs broad model support, dynamic graphs, many operators, hardware execution providers, and cross-framework compatibility. This crate takes a different approach: it implements only the exact computation needed for one tiny VAD model.

That makes the deployment tradeoff different:

* no ONNX parser
* no graph executor
* no dynamic kernel registry
* no execution provider abstraction
* no external runtime library
* no framework dependency

The result is a small, auditable inference path specialized for one model.

## License

Check the upstream Silero VAD license before redistributing model weights.

This repository’s source code license should be listed here.
