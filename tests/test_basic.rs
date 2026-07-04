use silero_vad_rust::Detector;
use std::fs::File;
use std::io::Read;

#[test]
fn test_regression_probs() {
    let mut detector = Detector::default();
    let probs = detector.predict_wav("tests/data/test.wav").expect("Failed to run predict_wav");

    // Load expected probabilities from JSON fixture
    let mut file = File::open("tests/data/expected_probs.json").expect("Failed to open expected_probs.json");
    let mut json_str = String::new();
    file.read_to_string(&mut json_str).expect("Failed to read expected_probs.json");
    let expected_probs: Vec<f32> = serde_json::from_str(&json_str).expect("Failed to parse expected_probs.json");

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

#[test]
fn test_detector_api() {
    let mut detector = Detector::default();
    
    // Predict a 512-sample chunk of silence
    let chunk = vec![0.0f32; 512];
    let prob = detector.predict(&chunk);
    assert!(prob >= 0.0 && prob <= 1.0, "Probability must be in range [0, 1], got {}", prob);

    // Verify reset works
    detector.reset();
}
