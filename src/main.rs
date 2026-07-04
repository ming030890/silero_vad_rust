use silero_vad_rust::Detector;
use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-wav-file>", args[0]);
        process::exit(1);
    }

    let wav_path = &args[1];
    println!("Loading Silero VAD model...");
    let mut detector = Detector::default();

    println!("Processing: {} ...", wav_path);
    let probs = match detector.predict_wav(wav_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    let threshold = 0.5f32;
    println!("------------------------------------------------------------");
    println!("| Chunk  | Timestamp (s) | Prob     | Speech Decision (>0.5) |");
    println!("------------------------------------------------------------");

    for (i, &prob) in probs.iter().enumerate() {
        let timestamp_s = (i * 512) as f64 / 16000.0;
        let is_speech = if prob >= threshold { "SPEECH" } else { "silence" };
        println!("| {:<6} | {:<13.3} | {:<8.4} | {:<22} |", i, timestamp_s, prob, is_speech);
    }
    println!("------------------------------------------------------------");
    println!("Inference completed successfully!");
}
