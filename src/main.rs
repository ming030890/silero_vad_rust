use silero_vad_rust::{load_silero_vad, read_audio};
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-wav-file>", args[0]);
        process::exit(1);
    }

    let wav_path = &args[1];
    println!("Loading custom Silero VAD model...");
    let mut model = match load_silero_vad() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error loading model: {}", e);
            process::exit(1);
        }
    };

    println!("Loading and resampling audio file: {} ...", wav_path);
    let mut audio = match read_audio(wav_path, 16000) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error reading audio: {}", e);
            process::exit(1);
        }
    };

    // Pad end of audio to a multiple of 512
    let remainder = audio.len() % 512;
    if remainder != 0 {
        audio.resize(audio.len() + 512 - remainder, 0.0);
    }

    let threshold = 0.5f32;
    let total_chunks = audio.len() / 512;
    println!("Processing {} chunks of 512 samples (32 ms each)...", total_chunks);
    println!("------------------------------------------------------------");
    println!("| Chunk  | Timestamp (s) | Prob     | Speech Decision (>0.5) |");
    println!("------------------------------------------------------------");

    for (i, chunk) in audio.chunks_exact(512).enumerate() {
        let prob = match model.predict_chunk(chunk) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("Error predicting chunk {}: {}", i, e);
                process::exit(1);
            }
        };

        let timestamp_s = (i * 512) as f64 / 16000.0;
        let is_speech = if prob >= threshold { "SPEECH" } else { "silence" };
        println!("| {:<6} | {:<13.3} | {:<8.4} | {:<22} |", i, timestamp_s, prob, is_speech);
    }
    println!("------------------------------------------------------------");
    println!("Inference completed successfully!");
}
