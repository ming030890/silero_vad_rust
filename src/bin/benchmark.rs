use silero_vad_rust::{load_silero_vad, read_audio};
use std::time::Instant;
use std::process::Command;

fn main() {
    println!("============================================================");
    println!("           SILERO VAD LATENCY & THROUGHPUT BENCHMARK         ");
    println!("============================================================");

    // 1. Rust latency measurement
    println!("Loading Rust custom Silero VAD model...");
    let mut model = load_silero_vad().expect("Failed to load Rust model");

    println!("Loading benchmark audio file (tests/data/test.wav)...");
    let mut audio = read_audio("tests/data/test.wav", 16000).expect("Failed to load audio");
    
    // Pad end of audio to a multiple of 512
    let remainder = audio.len() % 512;
    if remainder != 0 {
        audio.resize(audio.len() + 512 - remainder, 0.0);
    }
    
    let audio_duration_s = audio.len() as f64 / 16000.0;
    let total_chunks = audio.len() / 512;

    println!("Running Rust inference ({} chunks of 32 ms each)...", total_chunks);
    let start_rust = Instant::now();
    let mut dummy_sum = 0.0f32;
    for chunk in audio.chunks_exact(512) {
        let prob = model.predict_chunk(chunk).expect("Inference failed");
        dummy_sum += prob;
    }
    let rust_elapsed = start_rust.elapsed();
    let rust_elapsed_ms = rust_elapsed.as_secs_f64() * 1000.0;
    let rust_per_chunk_us = (rust_elapsed_ms * 1000.0) / total_chunks as f64;
    let rust_rtf = audio_duration_s / rust_elapsed.as_secs_f64();

    println!("Rust Inference finished (checksum: {:.4}).", dummy_sum);

    // 2. Python baseline measurement
    println!("\nSpawning Python process to measure PyTorch/ONNX baseline latency...");
    let python_cmd = r#"
import torch
import torch.nn as nn
import numpy as np
import wave
import struct
import time
from safetensors.torch import load_file

class PyTorchSileroVAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_conv = nn.Conv1d(1, 258, kernel_size=256, stride=128, padding=0, bias=False)
        self.conv1 = nn.Conv1d(129, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.lstm_cell = nn.LSTMCell(128, 128)
        self.final_conv = nn.Conv1d(128, 1, 1)

    def forward(self, x, state=None):
        x = torch.nn.functional.pad(x.unsqueeze(1), (0, 64), mode='reflect')
        x = self.stft_conv(x)
        x = torch.sqrt(x[:, :129, :]**2 + x[:, 129:, :]**2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x)).squeeze(-1)
        if state is None:
            h = torch.zeros(x.shape[0], 128)
            c = torch.zeros(x.shape[0], 128)
        else:
            h, c = state
        h, c = self.lstm_cell(x, (h, c))
        state = (h, c)
        x = h.unsqueeze(-1)
        x = torch.relu(x)
        x = torch.sigmoid(self.final_conv(x))
        x = x.squeeze(1).mean(dim=1, keepdim=True)
        return x, state

weights = load_file('/tmp/silero-vad-repo/src/silero_vad/data/silero_vad_16k.safetensors')
model = PyTorchSileroVAD()
model.load_state_dict(weights)
model.eval()

with wave.open('/Users/tony/Documents/keyboard/silero_vad_rust/tests/data/test.wav', 'rb') as w:
    frames = w.readframes(w.getnframes())
    samples = struct.unpack(f'<{w.getnframes()}h', frames)
    audio = torch.tensor(samples, dtype=torch.float32) / 32768.0

wav = audio.unsqueeze(0)
num_samples = 512
context_size = 64
padded_wav = torch.nn.functional.pad(wav, (context_size, 0))

state = None
start_time = time.perf_counter()
for i in range(context_size, padded_wav.shape[1], num_samples):
    chunk = padded_wav[:, i-context_size:i+num_samples]
    if chunk.shape[1] < context_size + num_samples:
        chunk = torch.nn.functional.pad(chunk, (0, context_size + num_samples - chunk.shape[1]))
    out, state = model(chunk, state)
    val = out.item()
elapsed = time.perf_counter() - start_time
print(f"RESULT_MS:{elapsed * 1000:.3f}")
"#;

    let output = Command::new("/opt/homebrew/Caskroom/miniconda/base/envs/asr/bin/python")
        .arg("-c")
        .arg(python_cmd)
        .output()
        .expect("Failed to execute python baseline command");

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let py_ms = stdout_str
        .lines()
        .find(|l| l.starts_with("RESULT_MS:"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|v| v.parse::<f64>().ok())
        .expect("Failed to parse python benchmark latency output");

    let py_per_chunk_us = (py_ms * 1000.0) / total_chunks as f64;
    let py_rtf = audio_duration_s / (py_ms / 1000.0);

    // 3. Output results comparison
    println!("\n============================================================");
    println!("                      BENCHMARK COMPARISON                  ");
    println!("============================================================");
    println!("Metric                 | Custom Rust VAD | PyTorch Baseline");
    println!("-----------------------|-----------------|------------------");
    println!("Audio Duration         | {:<15.3}s | {:<15.3}s", audio_duration_s, audio_duration_s);
    println!("Total Execution Time   | {:<15.3}ms| {:<15.3}ms", rust_elapsed_ms, py_ms);
    println!("Average Chunk Latency  | {:<15.3}µs| {:<15.3}µs", rust_per_chunk_us, py_per_chunk_us);
    println!("Real-time Factor (RTF) | {:<15.2}x | {:<15.2}x", rust_rtf, py_rtf);
    println!("Speedup Factor         | {:<15.2}x | 1.00x (Baseline)", rust_rtf / py_rtf);
    println!("============================================================");
}
