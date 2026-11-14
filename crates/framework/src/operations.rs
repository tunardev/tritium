use rayon::prelude::*;

use crate::quantization::{ACTIVATION_LIMIT, activation_shift, quantize_rows, ternary_scale};
use crate::tensor::Tensor;

pub const RELU_SQUARED_LIMIT: f32 = 32767.0;

#[allow(clippy::too_many_arguments)]
fn multiply_into(
    rows: usize,
    inner: usize,
    columns: usize,
    left: &[f32],
    is_left_transposed: bool,
    right: &[f32],
    is_right_transposed: bool,
    output: &mut [f32],
) {
    let (left_row_stride, left_column_stride) =
        if is_left_transposed { (1, rows as isize) } else { (inner as isize, 1) };
    let (right_row_stride, right_column_stride) =
        if is_right_transposed { (1, inner as isize) } else { (columns as isize, 1) };
    unsafe {
        matrixmultiply::sgemm(
            rows,
            inner,
            columns,
            1.0,
            left.as_ptr(),
            left_row_stride,
            left_column_stride,
            right.as_ptr(),
            right_row_stride,
            right_column_stride,
            1.0,
            output.as_mut_ptr(),
            columns as isize,
            1,
        );
    }
}

fn matrix_product(
    rows: usize,
    inner: usize,
    columns: usize,
    left: &[f32],
    is_left_transposed: bool,
    right: &[f32],
    is_right_transposed: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; rows * columns];
    multiply_into(
        rows,
        inner,
        columns,
        left,
        is_left_transposed,
        right,
        is_right_transposed,
        &mut output,
    );
    output
}

fn matrix_shape(tensor: &Tensor, name: &str) -> (usize, usize) {
    match *tensor.shape() {
        [rows, columns] => (rows, columns),
        ref shape => panic!("{name} must be a matrix, found shape {shape:?}"),
    }
}

fn elementwise(
    left: &Tensor,
    right: &Tensor,
    forward: fn(f32, f32) -> f32,
    backward: fn(f32, f32, f32) -> (f32, f32),
) -> Tensor {
    assert_eq!(left.shape(), right.shape(), "elementwise operands differ in shape");
    let (left_data, right_data) = (left.to_vec(), right.to_vec());
    let data = left_data.iter().zip(&right_data).map(|(&a, &b)| forward(a, b)).collect();
    Tensor::from_operation(
        data,
        left.shape(),
        &[left, right],
        Box::new(move |gradient| {
            let (left_gradient, right_gradient): (Vec<f32>, Vec<f32>) = gradient
                .iter()
                .zip(left_data.iter().zip(&right_data))
                .map(|(&output, (&a, &b))| backward(output, a, b))
                .unzip();
            vec![Some(left_gradient), Some(right_gradient)]
        }),
    )
}

pub fn add(left: &Tensor, right: &Tensor) -> Tensor {
    elementwise(left, right, |a, b| a + b, |gradient, _, _| (gradient, gradient))
}

pub fn subtract(left: &Tensor, right: &Tensor) -> Tensor {
    elementwise(left, right, |a, b| a - b, |gradient, _, _| (gradient, -gradient))
}

pub fn multiply(left: &Tensor, right: &Tensor) -> Tensor {
    elementwise(left, right, |a, b| a * b, |gradient, a, b| (gradient * b, gradient * a))
}

pub fn scale(input: &Tensor, factor: f32) -> Tensor {
    let data = input.data().iter().map(|value| value * factor).collect();
    Tensor::from_operation(
        data,
        input.shape(),
        &[input],
        Box::new(move |gradient| vec![Some(gradient.iter().map(|value| value * factor).collect())]),
    )
}

pub fn sum(input: &Tensor) -> Tensor {
    let total = input.data().iter().sum();
    let length = input.len();
    Tensor::from_operation(
        vec![total],
        &[],
        &[input],
        Box::new(move |gradient| vec![Some(vec![gradient[0]; length])]),
    )
}

pub fn mean(input: &Tensor) -> Tensor {
    scale(&sum(input), 1.0 / input.len() as f32)
}

pub fn reshape(input: &Tensor, shape: &[usize]) -> Tensor {
    assert_eq!(input.len(), shape.iter().product::<usize>(), "cannot reshape to {shape:?}");
    Tensor::from_operation(
        input.to_vec(),
        shape,
        &[input],
        Box::new(|gradient| vec![Some(gradient.to_vec())]),
    )
}

pub fn matmul(left: &Tensor, right: &Tensor) -> Tensor {
    let (rows, inner) = matrix_shape(left, "the left operand");
    let (right_inner, columns) = matrix_shape(right, "the right operand");
    assert_eq!(inner, right_inner, "matmul inner dimensions differ");
    let (left_data, right_data) = (left.to_vec(), right.to_vec());
    let data = matrix_product(rows, inner, columns, &left_data, false, &right_data, false);
    Tensor::from_operation(
        data,
        &[rows, columns],
        &[left, right],
        Box::new(move |gradient| {
            let left_gradient =
                matrix_product(rows, columns, inner, gradient, false, &right_data, true);
            let right_gradient =
                matrix_product(inner, rows, columns, &left_data, true, gradient, false);
            vec![Some(left_gradient), Some(right_gradient)]
        }),
    )
}

pub fn linear(input: &Tensor, weight: &Tensor) -> Tensor {
    let (rows, inner) = matrix_shape(input, "the input");
    let (outputs, weight_inner) = matrix_shape(weight, "the weight");
    assert_eq!(
        inner, weight_inner,
        "linear input has {inner} features, weight expects {weight_inner}"
    );
    let (input_data, weight_data) = (input.to_vec(), weight.to_vec());
    let data = matrix_product(rows, inner, outputs, &input_data, false, &weight_data, true);
    Tensor::from_operation(
        data,
        &[rows, outputs],
        &[input, weight],
        Box::new(move |gradient| {
            let input_gradient =
                matrix_product(rows, outputs, inner, gradient, false, &weight_data, false);
            let weight_gradient =
                matrix_product(outputs, rows, inner, gradient, true, &input_data, false);
            vec![Some(input_gradient), Some(weight_gradient)]
        }),
    )
}

fn quantize_rows_of_activations(data: &[f32], columns: usize) -> (Vec<f32>, Vec<f32>) {
    let mut quantized = Vec::with_capacity(data.len());
    let mut steps = Vec::with_capacity(data.len() / columns);
    for row in data.chunks_exact(columns) {
        let rounded: Vec<i32> = row.iter().map(|value| value.round() as i32).collect();
        let maximum = rounded.iter().map(|value| value.saturating_abs()).max().unwrap_or(0);
        let step = 2f32.powi(activation_shift(maximum));
        let limit = ACTIVATION_LIMIT as f32;
        quantized.extend(row.iter().map(|value| (value / step).round().clamp(-limit, limit)));
        steps.push(step);
    }
    (quantized, steps)
}

fn quantized_linear(
    input: &Tensor,
    weight: &Tensor,
    effective_weight: Vec<f32>,
    row_scales: Vec<f32>,
) -> Tensor {
    let (rows, inner) = matrix_shape(input, "the input");
    let (outputs, weight_inner) = matrix_shape(weight, "the weight");
    assert_eq!(inner, weight_inner, "input has {inner} features, weight expects {weight_inner}");
    let (quantized, steps) = quantize_rows_of_activations(&input.data(), inner);
    let mut data = matrix_product(rows, inner, outputs, &quantized, false, &effective_weight, true);
    for row in data.chunks_exact_mut(outputs) {
        for (value, scale) in row.iter_mut().zip(&row_scales) {
            *value *= scale;
        }
    }
    Tensor::from_operation(
        data,
        &[rows, outputs],
        &[input, weight],
        Box::new(move |gradient| {
            let mut scaled = gradient.to_vec();
            for row in scaled.chunks_exact_mut(outputs) {
                for (value, scale) in row.iter_mut().zip(&row_scales) {
                    *value *= scale;
                }
            }
            let mut input_gradient =
                matrix_product(rows, outputs, inner, &scaled, false, &effective_weight, false);
            for (row, step) in input_gradient.chunks_exact_mut(inner).zip(&steps) {
                row.iter_mut().for_each(|value| *value /= step);
            }
            let mut weight_gradient =
                matrix_product(outputs, rows, inner, &scaled, true, &quantized, false);
            for (row, scale) in weight_gradient.chunks_exact_mut(inner).zip(&row_scales) {
                row.iter_mut().for_each(|value| *value /= scale);
            }
            vec![Some(input_gradient), Some(weight_gradient)]
        }),
    )
}

pub fn fan_in_factor(inner: usize) -> f32 {
    1.0 / (inner as f32).sqrt()
}

pub fn bit_linear(input: &Tensor, weight: &Tensor) -> Tensor {
    let (outputs, inner) = matrix_shape(weight, "the weight");
    let weight_data = weight.to_vec();
    let scale = ternary_scale(&weight_data);
    let trits = weight_data.iter().map(|value| (value / scale).round().clamp(-1.0, 1.0)).collect();
    quantized_linear(input, weight, trits, vec![scale * fan_in_factor(inner); outputs])
}

pub fn int8_linear(input: &Tensor, weight: &Tensor) -> Tensor {
    let (_, columns) = matrix_shape(weight, "the weight");
    let (values, scales) = quantize_rows(&weight.data(), columns);
    let values = values.into_iter().map(f32::from).collect();
    let scales = scales.into_iter().map(|scale| scale * fan_in_factor(columns)).collect();
    quantized_linear(input, weight, values, scales)
}

pub fn embedding(table: &Tensor, indices: &[usize]) -> Tensor {
    let (rows, columns) = matrix_shape(table, "the embedding table");
    let (values, scales) = quantize_rows(&table.data(), columns);
    let mut data = Vec::with_capacity(indices.len() * columns);
    for &index in indices {
        assert!(index < rows, "index {index} is outside an embedding of {rows} rows");
        let row = &values[index * columns..][..columns];
        data.extend(row.iter().map(|&value| f32::from(value) * scales[index]));
    }
    let indices = indices.to_vec();
    Tensor::from_operation(
        data,
        &[indices.len(), columns],
        &[table],
        Box::new(move |gradient| {
            let mut table_gradient = vec![0.0; rows * columns];
            for (row, &index) in gradient.chunks_exact(columns).zip(&indices) {
                let target = &mut table_gradient[index * columns..][..columns];
                target.iter_mut().zip(row).for_each(|(total, value)| *total += value);
            }
            vec![Some(table_gradient)]
        }),
    )
}

pub fn relu_squared(input: &Tensor) -> Tensor {
    let input_data = input.to_vec();
    let data = input_data
        .iter()
        .map(|value| {
            let clamped = value.round().clamp(0.0, RELU_SQUARED_LIMIT);
            clamped * clamped
        })
        .collect();
    Tensor::from_operation(
        data,
        input.shape(),
        &[input],
        Box::new(move |gradient| {
            vec![Some(
                gradient
                    .iter()
                    .zip(&input_data)
                    .map(|(gradient, &value)| {
                        if value > 0.0 && value < RELU_SQUARED_LIMIT {
                            2.0 * value * gradient
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            )]
        }),
    )
}

pub fn causal_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    batch: usize,
    heads: usize,
) -> Tensor {
    let (rows, width) = matrix_shape(query, "the query");
    assert_eq!(key.shape(), query.shape(), "key and query differ in shape");
    assert_eq!(value.shape(), query.shape(), "value and query differ in shape");
    assert!(rows % batch == 0 && width % heads == 0, "attention shape does not divide");
    let sequence = rows / batch;
    let head_width = width / heads;
    let factor = 1.0 / (head_width as f32).sqrt();
    let (query_data, key_data, value_data) = (query.to_vec(), key.to_vec(), value.to_vec());
    let at = move |data: &[f32], position: usize, head: usize, offset: usize| {
        data[position * width + head * head_width + offset]
    };

    let pairs: Vec<(usize, usize)> =
        (0..batch).flat_map(|item| (0..heads).map(move |head| (item, head))).collect();
    let probabilities: Vec<Vec<f32>> = pairs
        .par_iter()
        .map(|&(item, head)| {
            let mut weights = vec![0.0; sequence * sequence];
            for target in 0..sequence {
                let row = &mut weights[target * sequence..][..sequence];
                let query_position = item * sequence + target;
                let mut maximum = f32::NEG_INFINITY;
                for (source, weight) in row[..=target].iter_mut().enumerate() {
                    let key_position = item * sequence + source;
                    let score: f32 = (0..head_width)
                        .map(|offset| {
                            at(&query_data, query_position, head, offset)
                                * at(&key_data, key_position, head, offset)
                        })
                        .sum::<f32>()
                        * factor;
                    *weight = score;
                    maximum = maximum.max(score);
                }
                let mut total = 0.0;
                for weight in &mut row[..=target] {
                    *weight = (*weight - maximum).exp();
                    total += *weight;
                }
                row[..=target].iter_mut().for_each(|weight| *weight /= total);
            }
            weights
        })
        .collect();

    let mut data = vec![0.0; rows * width];
    for (&(item, head), weights) in pairs.iter().zip(&probabilities) {
        for target in 0..sequence {
            let output =
                &mut data[(item * sequence + target) * width + head * head_width..][..head_width];
            for source in 0..=target {
                let weight = weights[target * sequence + source];
                for (offset, total) in output.iter_mut().enumerate() {
                    *total += weight * at(&value_data, item * sequence + source, head, offset);
                }
            }
        }
    }

    Tensor::from_operation(
        data,
        &[rows, width],
        &[query, key, value],
        Box::new(move |gradient| {
            let partials: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = pairs
                .par_iter()
                .zip(&probabilities)
                .map(|(&(item, head), weights)| {
                    let size = sequence * head_width;
                    let (mut query_gradient, mut key_gradient, mut value_gradient) =
                        (vec![0.0; size], vec![0.0; size], vec![0.0; size]);
                    for target in 0..sequence {
                        let target_position = item * sequence + target;
                        let output_gradient =
                            &gradient[target_position * width + head * head_width..][..head_width];
                        let row = &weights[target * sequence..][..sequence];
                        let mut weight_gradients = vec![0.0; target + 1];
                        for source in 0..=target {
                            let source_position = item * sequence + source;
                            for offset in 0..head_width {
                                value_gradient[source * head_width + offset] +=
                                    row[source] * output_gradient[offset];
                                weight_gradients[source] += output_gradient[offset]
                                    * at(&value_data, source_position, head, offset);
                            }
                        }
                        let dot: f32 =
                            (0..=target).map(|source| weight_gradients[source] * row[source]).sum();
                        for source in 0..=target {
                            let score_gradient =
                                row[source] * (weight_gradients[source] - dot) * factor;
                            let source_position = item * sequence + source;
                            for offset in 0..head_width {
                                query_gradient[target * head_width + offset] +=
                                    score_gradient * at(&key_data, source_position, head, offset);
                                key_gradient[source * head_width + offset] +=
                                    score_gradient * at(&query_data, target_position, head, offset);
                            }
                        }
                    }
                    (query_gradient, key_gradient, value_gradient)
                })
                .collect();
            let mut query_gradient = vec![0.0; rows * width];
            let mut key_gradient = vec![0.0; rows * width];
            let mut value_gradient = vec![0.0; rows * width];
            for (&(item, head), (partial_query, partial_key, partial_value)) in
                pairs.iter().zip(&partials)
            {
                for position in 0..sequence {
                    let start = (item * sequence + position) * width + head * head_width;
                    let source = position * head_width..(position + 1) * head_width;
                    query_gradient[start..start + head_width]
                        .copy_from_slice(&partial_query[source.clone()]);
                    key_gradient[start..start + head_width]
                        .copy_from_slice(&partial_key[source.clone()]);
                    value_gradient[start..start + head_width]
                        .copy_from_slice(&partial_value[source]);
                }
            }
            vec![Some(query_gradient), Some(key_gradient), Some(value_gradient)]
        }),
    )
}

pub fn cross_entropy(logits: &Tensor, targets: &[usize]) -> Tensor {
    let (rows, classes) = matrix_shape(logits, "the logits");
    assert_eq!(rows, targets.len(), "{rows} rows of logits but {} targets", targets.len());
    let mut probabilities = logits.to_vec();
    let mut loss = 0.0f64;
    for (row, &target) in probabilities.chunks_exact_mut(classes).zip(targets) {
        assert!(target < classes, "target {target} is outside {classes} classes");
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for value in row.iter_mut() {
            *value = (*value - maximum).exp();
            total += *value;
        }
        row.iter_mut().for_each(|value| *value /= total);
        loss -= f64::from(row[target].max(1e-30)).ln();
    }
    let targets = targets.to_vec();
    Tensor::from_operation(
        vec![(loss / rows as f64) as f32],
        &[],
        &[logits],
        Box::new(move |gradient| {
            let factor = gradient[0] / rows as f32;
            let mut logits_gradient = probabilities.clone();
            for (row, &target) in logits_gradient.chunks_exact_mut(classes).zip(&targets) {
                row[target] -= 1.0;
                row.iter_mut().for_each(|value| *value *= factor);
            }
            vec![Some(logits_gradient)]
        }),
    )
}
