use tritium_framework::Tensor;
use tritium_framework::operations::{bit_linear, embedding, int8_linear, relu_squared, sum};
use tritium_framework::quantization::{quantize_activations, ternarize};

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
