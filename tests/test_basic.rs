use silero_vad_rust::{load_silero_vad, read_audio};
use std::fs::File;
use std::io::Read;

#[test]
fn test_regression_probs() {
    let mut model = load_silero_vad().expect("Failed to load embedded model");
    let mut audio = read_audio("tests/data/test.wav", 16000).expect("Failed to load test.wav");
    
    // Pad end to a multiple of 512 (matching python/tinygrad padding)
    let remainder = audio.len() % 512;
    if remainder != 0 {
        audio.resize(audio.len() + 512 - remainder, 0.0);
    }

    // Load expected probabilities from JSON fixture
    let mut file = File::open("tests/data/expected_probs.json").expect("Failed to open expected_probs.json");
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).expect("Failed to read expected_probs.json");
    let expected_probs: Vec<f32> = serde_json::from_str(&json_str).expect("Failed to parse expected_probs.json");

    let mut probs = Vec::new();
    for chunk in audio.chunks_exact(512) {
        let prob = model.predict_chunk(chunk).expect("Failed to run predict_chunk");
        probs.push(prob);
    }

    // Assert that the length is at least what we expected
    assert!(probs.len() >= expected_probs.len(), "Computed fewer chunks than expected: {} < {}", probs.len(), expected_probs.len());

    // Check tolerance
    let tolerance = 1e-5;
    for (i, &expected) in expected_probs.iter().enumerate() {
        let actual = probs[i];
        let diff = (actual - expected).abs();
        assert!(
            diff < tolerance,
            "Chunk {} probability mismatch: expected {}, got {} (diff {})",
            i, expected, actual, diff
        );
    }

    println!("All {} regression test chunks matched successfully within tolerance of {}!", expected_probs.len(), tolerance);
}
