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
    let mut detector = Detector::default();

    let timestamps = match detector.predict_wav(wav_path) {
        Ok(ts) => ts,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    println!("Detected {} speech segment(s):", timestamps.len());
    for (i, ts) in timestamps.iter().enumerate() {
        println!("  [{:>3}] {:>8.3}s – {:>8.3}s  ({:.3}s)", i, ts.start, ts.end, ts.end - ts.start);
    }
}
