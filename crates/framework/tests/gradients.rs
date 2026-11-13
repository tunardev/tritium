use tritium_framework::operations::{
    add, causal_attention, cross_entropy, linear, matmul, mean, multiply, reshape, scale, sum,
};
use tritium_framework::{AdamW, Tensor, set_gradient_enabled};

fn values(seed: u32, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn assert_gradient_matches(parameter: &Tensor, loss: impl Fn() -> Tensor) {
    parameter.zero_gradient();
    loss().backward();
    let analytic = parameter.gradient().unwrap();
    let original = parameter.to_vec();
    for index in (0..original.len()).step_by(original.len().div_ceil(12)) {
        let step = 1e-2;
        let mut shifted = original.clone();
        shifted[index] += step;
        parameter.assign(&shifted);
        let above = loss().to_vec()[0];
        shifted[index] -= 2.0 * step;
        parameter.assign(&shifted);
        let below = loss().to_vec()[0];
        parameter.assign(&original);
        let numeric = (above - below) / (2.0 * step);
        let tolerance = 2e-2 * numeric.abs().max(analytic[index].abs()).max(1.0);
        assert!(
            (numeric - analytic[index]).abs() < tolerance,
            "index {index}: {numeric} vs {}",
            analytic[index]
        );
    }
}

#[test]
fn linear_gradients_match_differences() {
    let input = Tensor::parameter(values(1, 12), &[3, 4]);
    let weight = Tensor::parameter(values(2, 20), &[5, 4]);
    let loss = || sum(&multiply(&linear(&input, &weight), &linear(&input, &weight)));
    assert_gradient_matches(&input, loss);
    assert_gradient_matches(&weight, loss);
}

#[test]
fn matmul_gradients_match_differences() {
    let left = Tensor::parameter(values(3, 6), &[2, 3]);
    let right = Tensor::parameter(values(4, 12), &[3, 4]);
    let loss = || mean(&scale(&matmul(&left, &right), 3.0));
    assert_gradient_matches(&left, loss);
    assert_gradient_matches(&right, loss);
}

#[test]
fn attention_gradients_match_differences() {
    let (batch, sequence, width, heads) = (2, 5, 8, 2);
    let shape = [batch * sequence, width];
    let query = Tensor::parameter(values(5, batch * sequence * width), &shape);
    let key = Tensor::parameter(values(6, batch * sequence * width), &shape);
    let value = Tensor::parameter(values(7, batch * sequence * width), &shape);
    let target = Tensor::new(values(8, batch * sequence * width), &shape);
    let loss = || {
        let output = causal_attention(&query, &key, &value, batch, heads);
        sum(&multiply(&output, &target))
    };
    assert_gradient_matches(&query, loss);
    assert_gradient_matches(&key, loss);
    assert_gradient_matches(&value, loss);
}

#[test]
fn attention_ignores_future_positions() {
    let shape = [4, 4];
    let query = Tensor::new(values(9, 16), &shape);
    let key = Tensor::new(values(10, 16), &shape);
    let value = Tensor::new(values(11, 16), &shape);
    let mut changed = value.to_vec();
    changed[12..16].copy_from_slice(&[9.0; 4]);
    let first = causal_attention(&query, &key, &value, 1, 1).to_vec();
    let second = causal_attention(&query, &key, &Tensor::new(changed, &shape), 1, 1).to_vec();
    assert_eq!(first[..12], second[..12]);
}

#[test]
fn cross_entropy_gradients_match_differences() {
    let logits = Tensor::parameter(values(12, 15), &[3, 5]);
    assert_gradient_matches(&logits, || cross_entropy(&logits, &[0, 4, 2]));
    let uniform = Tensor::new(vec![0.0; 10], &[2, 5]);
    assert!((cross_entropy(&uniform, &[1, 3]).to_vec()[0] - 5f32.ln()).abs() < 1e-6);
}

#[test]
fn shared_inputs_accumulate_gradients() {
    let input = Tensor::parameter(vec![1.0, 2.0, 3.0], &[3]);
    sum(&add(&reshape(&input, &[3, 1]), &reshape(&scale(&input, 2.0), &[3, 1]))).backward();
    assert_eq!(input.gradient().unwrap(), vec![3.0; 3]);
}

#[test]
fn disabled_gradients_build_no_graph() {
    let weight = Tensor::parameter(vec![1.0; 4], &[2, 2]);
    let previous = set_gradient_enabled(false);
    let output = linear(&Tensor::new(vec![1.0; 2], &[1, 2]), &weight);
    set_gradient_enabled(previous);
    assert!(!output.requires_gradient());
}

#[test]
fn adamw_reduces_a_quadratic_loss() {
    let weight = Tensor::parameter(values(13, 8), &[2, 4]);
    let target = Tensor::new(values(14, 8), &[2, 4]);
    let mut optimizer = AdamW::new(vec![weight.clone()], 0.05, 0.0);
    let loss = || {
        let difference = add(&weight, &scale(&target, -1.0));
        sum(&multiply(&difference, &difference))
    };
    let initial = loss().to_vec()[0];
    for _ in 0..200 {
        optimizer.zero_gradients();
        loss().backward();
        optimizer.step();
    }
    assert!(loss().to_vec()[0] < initial * 1e-3);
}
