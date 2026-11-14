pub const ACTIVATION_LIMIT: i32 = 127;
pub const RESIDUAL_ONE: f32 = 256.0;
pub const FIXED_ONE: f32 = 65536.0;
const ACTIVATION_BITS: i32 = 7;

pub fn activation_shift(absolute_maximum: i32) -> i32 {
    if absolute_maximum == 0 {
        return 0;
    }
    (32 - absolute_maximum.unsigned_abs().leading_zeros() as i32) - ACTIVATION_BITS
}

pub fn quantize_activation(value: i32, shift: i32) -> i32 {
    let shifted = if shift <= 0 { value << -shift } else { (value + (1 << (shift - 1))) >> shift };
    shifted.clamp(-ACTIVATION_LIMIT, ACTIVATION_LIMIT)
}

pub fn quantize_activations(values: &[i32]) -> Vec<i8> {
    let absolute_maximum = values.iter().map(|value| value.saturating_abs()).max().unwrap_or(0);
    let shift = activation_shift(absolute_maximum);
    values.iter().map(|&value| quantize_activation(value, shift) as i8).collect()
}

pub fn ternary_scale(weights: &[f32]) -> f32 {
    weights.iter().map(|weight| weight.abs()).sum::<f32>() / weights.len() as f32 + 1e-8
}

pub fn ternarize(weights: &[f32]) -> (Vec<i8>, f32) {
    let scale = ternary_scale(weights);
    let trits =
        weights.iter().map(|weight| (weight / scale).round().clamp(-1.0, 1.0) as i8).collect();
    (trits, scale)
}

pub fn quantize_rows(weights: &[f32], columns: usize) -> (Vec<i8>, Vec<f32>) {
    let mut values = Vec::with_capacity(weights.len());
    let mut scales = Vec::with_capacity(weights.len() / columns);
    for row in weights.chunks_exact(columns) {
        let maximum = row.iter().fold(0f32, |maximum, weight| maximum.max(weight.abs()));
        let scale = maximum / ACTIVATION_LIMIT as f32 + 1e-8;
        values.extend(row.iter().map(|weight| (weight / scale).round().clamp(-127.0, 127.0) as i8));
        scales.push(scale);
    }
    (values, scales)
}

pub fn to_fixed(value: f32) -> i32 {
    (f64::from(value) * f64::from(FIXED_ONE))
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shift_keeps_seven_bit_maximum() {
        for maximum in [1, 63, 64, 127, 128, 255, 256, 70_000] {
            let shift = activation_shift(maximum);
            let quantized = quantize_activation(maximum, shift);
            assert!((64..=127).contains(&quantized), "{maximum} -> {quantized}");
        }
        assert_eq!(activation_shift(0), 0);
    }

    #[test]
    fn rounding_never_exceeds_the_limit() {
        assert_eq!(quantize_activations(&[255, -255, 1]), vec![127, -127, 1]);
        assert_eq!(quantize_activations(&[3, -2, 1]), vec![96, -64, 32]);
    }

    #[test]
    fn ternarized_weights_use_absmean_scale() {
        let (trits, scale) = ternarize(&[0.5, -0.1, 0.0, -0.9]);
        assert!((scale - 0.375).abs() < 1e-6);
        assert_eq!(trits, vec![1, 0, 0, -1]);
    }
}
