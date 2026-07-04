use silero_vad_rust::RawDetector;
use std::time::Instant;

#[cfg(feature = "benchmark_ort")]
use ndarray::{Array, Array1, ArrayD};
#[cfg(feature = "benchmark_ort")]
use ort::session::Session;
#[cfg(feature = "benchmark_ort")]
use ort::value::Value;
#[cfg(feature = "benchmark_ort")]
use std::mem::take;

#[cfg(feature = "benchmark_ort")]
struct OrtSilero {
    session: Session,
    sample_rate: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 1]>>,
    state: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
    context: Array1<f32>,
    context_size: usize,
}

#[cfg(feature = "benchmark_ort")]
impl OrtSilero {
    fn new(model_path: &str) -> Result<Self, ort::Error> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;
        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        let context_size = 64;
        let context = Array1::<f32>::zeros(context_size);
        let sample_rate = Array::from_shape_vec([1], vec![16000i64]).unwrap();
        Ok(Self {
            session,
            sample_rate,
            state,
            context,
            context_size,
        })
    }

    fn reset(&mut self) {
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());
        self.context = Array1::<f32>::zeros(self.context_size);
    }

    fn calc_level(&mut self, data: &[f32]) -> Result<f32, ort::Error> {
        let mut input_with_context = Vec::with_capacity(self.context_size + data.len());
        input_with_context.extend_from_slice(self.context.as_slice().unwrap());
        input_with_context.extend_from_slice(data);

        let frame = ndarray::Array2::<f32>::from_shape_vec(
            [1, input_with_context.len()],
            input_with_context,
        )
        .unwrap();

        let frame_value = Value::from_array(frame)?;
        let state_value = Value::from_array(take(&mut self.state))?;
        let sr_value = Value::from_array(self.sample_rate.clone())?;

        let res = self.session.run([
            (&frame_value).into(),
            (&state_value).into(),
            (&sr_value).into(),
        ])?;

        let (shape, state_data) = res["stateN"].try_extract_tensor::<f32>()?;
        let shape_usize: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        self.state = ArrayD::from_shape_vec(shape_usize.as_slice(), state_data.to_vec()).unwrap();

        if data.len() >= self.context_size {
            self.context = Array1::from_vec(data[data.len() - self.context_size..].to_vec());
        }

        let prob = *res["output"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .1
            .first()
            .unwrap();
        Ok(prob)
    }
}

fn read_wav_f32(path: &str) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).expect("Failed to open WAV file");
    let spec = reader.spec();
    assert_eq!(
        spec.sample_rate, 16000,
        "Expected 16 kHz WAV, got {} Hz",
        spec.sample_rate
    );
    assert_eq!(
        spec.channels, 1,
        "Expected mono WAV, got {} channels",
        spec.channels
    );

    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    }
}

fn main() {
    println!("============================================================");
    println!("           SILERO VAD LATENCY & THROUGHPUT BENCHMARK         ");
    println!("============================================================");

    // 1. Rust profiling
    println!("Profiling Custom Rust VAD...");

    let start_load = Instant::now();
    let mut detector = RawDetector::default();
    let rust_load_ms = start_load.elapsed().as_secs_f64() * 1000.0;

    let start_decode = Instant::now();
    let mut audio = read_wav_f32("tests/data/test.wav");
    let rust_decode_ms = start_decode.elapsed().as_secs_f64() * 1000.0;

    let remainder = audio.len() % 512;
    if remainder != 0 {
        audio.resize(audio.len() + 512 - remainder, 0.0);
    }

    let audio_duration_s = audio.len() as f64 / 16000.0;
    let total_chunks = audio.len() / 512;

    let mut rust_runs = Vec::new();
    for _ in 0..10 {
        detector.reset();
        let start_inference = Instant::now();
        for chunk in audio.chunks_exact(512) {
            let _prob = detector.predict_f32(chunk).expect("Rust inference failed");
        }
        rust_runs.push(start_inference.elapsed().as_secs_f64() * 1000.0);
    }
    let rust_inf_min = rust_runs.iter().copied().fold(f64::INFINITY, f64::min);
    let rust_inf_avg = rust_runs.iter().sum::<f64>() / rust_runs.len() as f64;

    // 2. ORT Rust profiling
    #[cfg(feature = "benchmark_ort")]
    let (ort_load_ms, ort_inf_min, ort_inf_avg) = {
        println!("Profiling Official Rust ORT Baseline...");

        let start_ort_load = Instant::now();
        let mut ort_model =
            OrtSilero::new("/tmp/silero-vad-repo/src/silero_vad/data/silero_vad.onnx")
                .expect("Failed to load ORT model");
        let ort_load_ms = start_ort_load.elapsed().as_secs_f64() * 1000.0;

        let mut ort_runs = Vec::new();
        for _ in 0..10 {
            ort_model.reset();
            let start_inference = Instant::now();
            for chunk in audio.chunks_exact(512) {
                let _prob = ort_model.calc_level(chunk).expect("ORT Inference failed");
            }
            ort_runs.push(start_inference.elapsed().as_secs_f64() * 1000.0);
        }
        let ort_inf_min = ort_runs.iter().copied().fold(f64::INFINITY, f64::min);
        let ort_inf_avg = ort_runs.iter().sum::<f64>() / ort_runs.len() as f64;
        (ort_load_ms, ort_inf_min, ort_inf_avg)
    };

    #[cfg(not(feature = "benchmark_ort"))]
    let (ort_load_ms, ort_inf_min, ort_inf_avg) = (0.0, 0.0, 0.0);

    // 3. Output results comparison
    println!("\n============================================================");
    println!("                DETAILED PROFILING COMPARISON                ");
    println!("============================================================");
    println!("Phase / Metric          | Custom Rust VAD | Official ORT Rust");
    println!("------------------------|-----------------|------------------");
    println!(
        "Model Load & Init       | {:<15.3}ms| {:<15.3}ms",
        rust_load_ms, ort_load_ms
    );
    println!(
        "Audio Load & Decode     | {:<15.3}ms| {:<15.3}ms",
        rust_decode_ms, rust_decode_ms
    );
    println!(
        "Min Inference (60s loop)| {:<15.3}ms| {:<15.3}ms",
        rust_inf_min, ort_inf_min
    );
    println!(
        "Avg Inference (60s loop)| {:<15.3}ms| {:<15.3}ms",
        rust_inf_avg, ort_inf_avg
    );
    println!(
        "Avg Chunk Latency       | {:<15.3}µs| {:<15.3}µs",
        (rust_inf_avg * 1000.0) / total_chunks as f64,
        (ort_inf_avg * 1000.0) / total_chunks as f64
    );
    println!(
        "Avg Real-time Factor    | {:<15.2}x | {:<15.2}x",
        audio_duration_s / (rust_inf_avg / 1000.0),
        audio_duration_s / (ort_inf_avg / 1000.0)
    );
    #[cfg(feature = "benchmark_ort")]
    println!(
        "Speedup Factor          | {:<15.2}x | 1.00x (Baseline)",
        ort_inf_avg / rust_inf_avg
    );
    #[cfg(not(feature = "benchmark_ort"))]
    println!(
        "Speedup Factor          | [Run with --features benchmark_ort to see comparative metrics]"
    );
    println!("============================================================");
}
