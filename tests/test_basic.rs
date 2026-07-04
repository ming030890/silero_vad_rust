use silero_vad_rust::{Detector, VadConfig};
use std::fs::File;
use std::io::Read;

#[test]
fn test_regression_probs() {
    // Use predict() directly so we can compare raw probs against the fixture
    let mut detector = Detector::default();
    let mut reader = hound::WavReader::open("tests/data/test.wav").expect("Failed to open test.wav");
    let spec = reader.spec();
    let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
    let audio: Vec<f32> = reader.samples::<i32>().map(|s| s.unwrap() as f32 / max_val).collect();

    let mut file = File::open("tests/data/expected_probs.json").expect("Failed to open expected_probs.json");
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).expect("Failed to read expected_probs.json");
    let expected_probs: Vec<f32> = serde_json::from_str(&json_str).expect("Failed to parse expected_probs.json");

    let mut probs = Vec::new();
    for chunk in audio.chunks_exact(512) {
        probs.push(detector.predict(chunk));
    }

    assert!(probs.len() >= expected_probs.len(), "Computed fewer chunks than expected: {} < {}", probs.len(), expected_probs.len());

    let tolerance = 1e-5;
    for (i, &expected) in expected_probs.iter().enumerate() {
        let actual = probs[i];
        let diff = (actual - expected).abs();
        assert!(diff < tolerance, "Chunk {} mismatch: expected {}, got {} (diff {})", i, expected, actual, diff);
    }

    println!("All {} regression test chunks matched within tolerance {}!", expected_probs.len(), tolerance);
}

#[test]
fn test_detector_api() {
    let mut detector = Detector::default();

    // Raw streaming predict
    let chunk = vec![0.0f32; 512];
    let prob = detector.predict(&chunk);
    assert!(prob >= 0.0 && prob <= 1.0, "Probability out of range: {}", prob);
    detector.reset();
}

#[test]
fn test_predict_wav_returns_timestamps() {
    let mut detector = Detector::default();
    let timestamps = detector.predict_wav("tests/data/test.wav").expect("predict_wav failed");

    // All segments must be ordered and non-empty
    for (i, ts) in timestamps.iter().enumerate() {
        assert!(ts.start < ts.end, "Segment {i} has start >= end: {:?}", ts);
        if i > 0 {
            assert!(timestamps[i].start >= timestamps[i - 1].end, "Segment {i} overlaps previous");
        }
    }
    println!("Detected {} speech segments", timestamps.len());
    for ts in &timestamps {
        println!("  {:.3}s – {:.3}s", ts.start, ts.end);
    }
}

#[test]
fn test_vad_config() {
    // Tighter silence threshold should produce fewer / shorter segments
    let strict = VadConfig { min_silence_duration_ms: 500, ..Default::default() };
    let mut detector = Detector::with_config(strict).unwrap();
    let timestamps = detector.predict_wav("tests/data/test.wav").expect("predict_wav failed");
    println!("With 500ms silence threshold: {} segments", timestamps.len());
}
