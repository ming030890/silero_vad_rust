# Silero VAD Rust

A lightweight, zero-dependency Rust inference engine for the 16 kHz [Silero VAD](https://github.com/snakers4/silero-vad) model. Drop-in replacement for [`earshot`](https://crates.io/crates/earshot).

No PyTorch, ONNX Runtime, tinygrad, or libtorch — the neural network operators are implemented directly in Rust for this one fixed model.

## Benchmarks

60-second WAV on macOS Apple Silicon, single-threaded:

| Metric               | Pure Rust |  + BLAS (Accelerate) | Official ORT Baseline |
| :------------------- | --------: | -------------------: | --------------------: |
| Peak RSS             |   9.63 MB |            10.99 MB  |             42.24 MB  |
| Total command time   |    0.65 s |           **0.18 s** |               0.47 s  |
| VAD loop (60 s file) |    465 ms |           **41.5 ms**|              154.7 ms |
| Avg chunk latency    |    229 µs |          **22.1 µs** |              82.5 µs  |
| Real-time factor     |    128.8x |         **1444.1x**  |              387.7x  |
| Weight init          |    0.9 ms |              0.9 ms  |             56.5 ms  |
| ONNX Runtime needed  |        No |                  No  |                 Yes  |
| External libs        |      None |        BLAS backend  |     `libonnxruntime` |

```bash
cargo run --release --bin benchmark                    # pure Rust
cargo run --release --bin benchmark --features openblas # with BLAS
```

## Quick Start

```rust
use silero_vad_rust::Detector;

let mut detector = Detector::default();

// Stream 512-sample (32 ms) chunks of 16 kHz mono f32 PCM:
let chunk = vec![0.0f32; 512];
let score = detector.predict(&chunk);
if score >= 0.5 {
    println!("Speech detected: {score:.4}");
}

// Reset between independent audio streams:
detector.reset();
```

## Speech Timestamps (File-Level)

Process an entire WAV file with the full Silero post-processing pipeline — onset/offset hysteresis, min-speech filtering, max-segment splitting, and speech padding:

```rust
use silero_vad_rust::{Detector, VadConfig};

let config = VadConfig {
    min_silence_duration_ms: 200,
    ..Default::default()
};
let mut detector = Detector::with_config(config).unwrap();
let timestamps = detector.predict_wav("speech.wav").unwrap();
for ts in &timestamps {
    println!("{:.2}s – {:.2}s", ts.start, ts.end);
}
```

### VadConfig Parameters

| Parameter                    | Default  | Description                                              |
| :--------------------------- | :------- | :------------------------------------------------------- |
| `threshold`                  | `0.5`    | Speech onset probability threshold                       |
| `neg_threshold`              | `0.35`   | Speech offset threshold (hysteresis to avoid toggling)   |
| `min_speech_duration_ms`     | `250`    | Discard segments shorter than this                        |
| `max_speech_duration_s`      | `∞`      | Force-split segments exceeding this at longest silence    |
| `min_silence_duration_ms`    | `100`    | Silence duration required to close a segment             |
| `speech_pad_ms`              | `30`     | Padding added to start/end of each segment               |
| `min_silence_at_max_speech_ms` | `98`   | Min silence gap considered when splitting at max duration |

## Demo CLI

```bash
cargo run --release -- <path-to-wav-file>
```

```text
Detected 3 speech segment(s):
  [  0]    1.234s –    4.567s  (3.333s)
  [  1]    6.789s –   10.123s  (3.334s)
  [  2]   12.456s –   15.789s  (3.333s)
```

## Testing

```bash
cargo test
```

Tests verify inference output against golden probabilities from the official PyTorch implementation (tolerance `1e-5`), plus timestamp extraction and config behavior.

## Design

This crate implements only the operators needed by Silero VAD 16 kHz: 1D convolution, reflect padding, ReLU, sigmoid, LSTM cell, and a learned STFT frontend. No ONNX parser, graph executor, or kernel registry.

The model weights are embedded via `include_bytes!` and parsed with zero-copy `&[f32]` slices — no allocation, no session init.

Optionally accelerated via BLAS (`cblas_sgemv`) by reformulating convolutions as GEMV operations.

## Limitations

* Only the **16 kHz** Silero VAD model is supported
* Audio must be mono `f32` PCM at **16 kHz**
* Fixed **512-sample** (32 ms) chunk size
* Model-specific inference path, not a general ONNX runtime

## License

Check the upstream [Silero VAD license](https://github.com/snakers4/silero-vad/blob/master/LICENSE) before redistributing model weights.
