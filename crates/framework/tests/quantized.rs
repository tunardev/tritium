use tritium_framework::operations::{bit_linear, embedding, int8_linear, relu_squared, sum};
use tritium_framework::quantization::{quantize_activations, ternarize};
use tritium_framework::{Tensor, Tokenizer};

#[test]
fn bit_linear_matches_integer_product() {
    let input = vec![300.0, -40.0, 1000.0, 7.0, -650.0, 12.0];
    let weights = vec![0.3, -0.02, -0.5, 0.25, 0.0, 0.6];
    let output =
        bit_linear(&Tensor::new(input.clone(), &[2, 3]), &Tensor::new(weights.clone(), &[2, 3]));
    let (trits, scale) = ternarize(&weights);
    let mut expected = Vec::new();
    for row in input.chunks(3) {
        let quantized =
            quantize_activations(&row.iter().map(|&value| value as i32).collect::<Vec<_>>());
        for weight_row in trits.chunks(3) {
            let dot: i32 =
                quantized.iter().zip(weight_row).map(|(&a, &w)| i32::from(a) * i32::from(w)).sum();
            expected.push(dot as f32 * scale);
        }
    }
    let actual = output.to_vec();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
    }
}

#[test]
fn straight_through_gradients_reach_weights() {
    let input = Tensor::parameter(vec![100.0, -50.0, 25.0, 80.0], &[1, 4]);
    let weight = Tensor::parameter(vec![0.1, -0.2, 0.3, 0.05, 0.4, -0.1, 0.0, 0.2], &[2, 4]);
    sum(&bit_linear(&input, &weight)).backward();
    assert!(weight.gradient().unwrap().iter().any(|&value| value != 0.0));
    assert!(input.gradient().unwrap().iter().any(|&value| value != 0.0));
    let head = Tensor::parameter(vec![0.5, -0.25, 0.75, 0.1], &[1, 4]);
    sum(&int8_linear(&input, &head)).backward();
    assert!(head.gradient().unwrap().iter().all(|&value| value != 0.0));
}

#[test]
fn embedding_rows_are_int8_quantized() {
    let table = Tensor::parameter(vec![0.5, -0.25, 1.0, 0.004, 0.0, -2.0], &[3, 2]);
    let rows = embedding(&table, &[2, 0, 2]);
    assert_eq!(rows.shape(), &[3, 2]);
    let first = rows.to_vec();
    assert!(first[0] == 0.0 && (first[1] + 2.0).abs() < 1e-5, "{first:?}");
    sum(&rows).backward();
    assert_eq!(table.gradient().unwrap(), vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0]);
}

#[test]
fn relu_squared_clamps_its_input() {
    let output = relu_squared(&Tensor::new(vec![-3.0, 2.4, 40000.0], &[3]));
    assert_eq!(output.to_vec(), vec![0.0, 4.0, 32767.0 * 32767.0]);
}

#[test]
fn tokenizer_round_trips_text() {
    let text = "Once upon a time, a little cat sat.<|endoftext|>Once upon a time, a dog ran! 42";
    let tokenizer = Tokenizer::train(&text.repeat(20), 300);
    let tokens = tokenizer.encode(text);
    assert_eq!(tokenizer.decode(&tokens), text);
    assert!(tokens.len() < text.len() / 2, "{} tokens", tokens.len());
    assert!(tokens.contains(&tokenizer.end_of_text()));
    let restored = Tokenizer::from_text(&tokenizer.to_text()).unwrap();
    assert_eq!(restored, tokenizer);
}

#[test]
fn tensor_files_round_trip() {
    let path = std::env::temp_dir().join(format!("tritium-tensors-{}.bin", std::process::id()));
    let tensors = vec![
        ("weight".to_string(), Tensor::new(vec![1.5, -2.0, 3.25, 0.0], &[2, 2])),
        ("bias".to_string(), Tensor::new(vec![7.0], &[1])),
    ];
    tritium_framework::storage::save_tensors(&path, &tensors).unwrap();
    let loaded = tritium_framework::storage::load_tensors(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(loaded.len(), 2);
    assert_eq!((loaded[0].0.as_str(), loaded[0].1.shape()), ("weight", &[2, 2][..]));
    assert_eq!(loaded[0].1.to_vec(), vec![1.5, -2.0, 3.25, 0.0]);
}

#[test]
fn batches_pair_inputs_with_successors() {
    let path = std::env::temp_dir().join(format!("tritium-tokens-{}.bin", std::process::id()));
    let bytes: Vec<u8> = (0..100u16).flat_map(u16::to_le_bytes).collect();
    std::fs::write(&path, bytes).unwrap();
    let dataset = tritium_framework::TokenDataset::open(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let (inputs, targets) = dataset.batch(3, 8, &mut tritium_framework::Random::new(1));
    assert_eq!(inputs.len(), 24);
    assert!(inputs.iter().zip(&targets).all(|(input, target)| target - input == 1));
}
