use std::collections::HashMap;
use std::fmt;

use tritium_framework::Tensor;
use tritium_framework::quantization::{quantize_rows, ternarize, to_fixed};
use tritium_isa::{TRITS_PER_WORD, exp2, pack_trits, rsqrt, saturate_to_int8, scale, unpack_word};

pub const SHARED_VECTOR_BYTES: usize = 640;
pub const MAXIMUM_CONTEXT: usize = 128;
pub const MAXIMUM_HEAD_INPUT: usize = 256;
const RESIDUAL_ONE: f32 = 256.0;
const ACTIVATION_BITS: i32 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelConfig {
    pub vocabulary_size: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub hidden: usize,
    pub context: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelError(pub String);

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ModelError {}

impl ModelConfig {
    pub fn head_width(&self) -> usize {
        self.width / self.heads
    }

    pub fn validate(&self) -> Result<(), ModelError> {
        let fail = |message: String| Err(ModelError(message));
        if self.width % 64 != 0 || self.hidden % 64 != 0 || self.vocabulary_size % 64 != 0 {
            return fail("width, hidden and vocabulary size must be multiples of 64".into());
        }
        if self.heads == 0 || self.width % self.heads != 0 || !self.head_width().is_power_of_two() {
            return fail("the head width must be a power of two that divides the width".into());
        }
        if self.head_width() > 64 || self.width > MAXIMUM_HEAD_INPUT {
            return fail(format!("width must be at most {MAXIMUM_HEAD_INPUT} with heads of 64"));
        }
        if self.hidden.div_ceil(TRITS_PER_WORD) * TRITS_PER_WORD > SHARED_VECTOR_BYTES {
            return fail(format!("hidden must fit {SHARED_VECTOR_BYTES} padded activations"));
        }
        if self.context == 0 || self.context > MAXIMUM_CONTEXT {
            return fail(format!("context must be between 1 and {MAXIMUM_CONTEXT}"));
        }
        if self.vocabulary_size > 32768 || self.layers == 0 {
            return fail(
                "the vocabulary must fit in 32768 tokens and there must be a layer".into(),
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projection {
    pub rows: usize,
    pub words_per_row: usize,
    pub packed: Vec<u8>,
    pub trits: Vec<i8>,
    pub factor: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Int8Rows {
    pub columns: usize,
    pub values: Vec<i8>,
    pub factors: Vec<i32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer {
    pub query: Projection,
    pub key: Projection,
    pub value: Projection,
    pub output: Projection,
    pub up: Projection,
    pub down: Projection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntegerModel {
    pub config: ModelConfig,
    pub token_embedding: Int8Rows,
    pub position_embedding: Int8Rows,
    pub layers: Vec<Layer>,
    pub head: Int8Rows,
    pub attention_factor: i32,
}

fn projection(weight: &Tensor, rows: usize, inner: usize) -> Projection {
    let data = weight.data();
    let (trits, weight_scale) = ternarize(&data);
    let words_per_row = inner.div_ceil(TRITS_PER_WORD);
    let mut packed = Vec::with_capacity(rows * words_per_row * 4);
    let mut padded = Vec::with_capacity(rows * words_per_row * TRITS_PER_WORD);
    for row in trits.chunks_exact(inner) {
        packed.extend(pack_trits(row));
        padded.extend_from_slice(row);
        padded.resize(padded.len() + words_per_row * TRITS_PER_WORD - inner, 0);
    }
    let factor = to_fixed(weight_scale / (inner as f32).sqrt() * RESIDUAL_ONE);
    Projection { rows, words_per_row, packed, trits: padded, factor }
}

fn int8_rows(weight: &Tensor, columns: usize, factor_of: impl Fn(f32) -> f32) -> Int8Rows {
    let (values, scales) = quantize_rows(&weight.data(), columns);
    let factors = scales.into_iter().map(|scale| to_fixed(factor_of(scale))).collect();
    Int8Rows { columns, values, factors }
}

impl IntegerModel {
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
    ) -> Result<IntegerModel, ModelError> {
        config.validate()?;
        let get = |name: &str, shape: [usize; 2]| -> Result<&Tensor, ModelError> {
            let tensor = tensors.get(name).ok_or_else(|| ModelError(format!("missing {name}")))?;
            if tensor.shape() != shape {
                return Err(ModelError(format!(
                    "{name} has shape {:?}, expected {shape:?}",
                    tensor.shape()
                )));
            }
            Ok(tensor)
        };
        let (width, hidden) = (config.width, config.hidden);
        let mut layers = Vec::with_capacity(config.layers);
        for index in 0..config.layers {
            let name = |part: &str| format!("blocks.{index}.{part}.weight");
            let square = |part: &str| -> Result<Projection, ModelError> {
                Ok(projection(get(&name(part), [width, width])?, width, width))
            };
            layers.push(Layer {
                query: square("query")?,
                key: square("key")?,
                value: square("value")?,
                output: square("output")?,
                up: projection(get(&name("up"), [hidden, width])?, hidden, width),
                down: projection(get(&name("down"), [width, hidden])?, width, hidden),
            });
        }
        let embedding_factor = |scale: f32| scale * RESIDUAL_ONE;
        let head_factor = |scale: f32| scale / (width as f32).sqrt() * RESIDUAL_ONE;
        Ok(IntegerModel {
            config,
            token_embedding: int8_rows(
                get("token_embedding.weight", [config.vocabulary_size, width])?,
                width,
                embedding_factor,
            ),
            position_embedding: int8_rows(
                get("position_embedding.weight", [config.context, width])?,
                width,
                embedding_factor,
            ),
            layers,
            head: int8_rows(
                get("head.weight", [config.vocabulary_size, width])?,
                width,
                head_factor,
            ),
            attention_factor: to_fixed(
                std::f32::consts::LOG2_E / (config.head_width() as f32).sqrt(),
            ),
        })
    }

    pub fn ternary_weight_bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| {
                [&layer.query, &layer.key, &layer.value, &layer.output, &layer.up, &layer.down]
            })
            .map(|projection| projection.packed.len())
            .sum()
    }
}

pub fn quantize(values: &[i32], is_squared: bool) -> Vec<i8> {
    let prepare = |value: i32| {
        if is_squared {
            let rectified = value.clamp(0, 32767);
            rectified * rectified
        } else {
            value
        }
    };
    let largest = values.iter().fold(0, |largest: i32, &value| {
        let value = prepare(value);
        largest.max(value.max(value.wrapping_neg()))
    });
    let bits = 32 - largest.leading_zeros() as i32;
    let shift = if largest == 0 { 0 } else { bits - ACTIVATION_BITS };
    values
        .iter()
        .map(|&value| {
            let value = prepare(value);
            if shift <= 0 {
                value.wrapping_shl((-shift) as u32) as i8
            } else {
                saturate_to_int8(value.wrapping_add(1 << (shift - 1)) >> shift) as i8
            }
        })
        .collect()
}

fn project(projection: &Projection, input: &[i8]) -> Vec<i32> {
    let mut activations = input.to_vec();
    activations.resize(projection.words_per_row * TRITS_PER_WORD, 0);
    projection
        .packed
        .chunks_exact(4 * projection.words_per_row)
        .map(|row| {
            let sum = row.chunks_exact(4).enumerate().fold(0i32, |sum, (word, bytes)| {
                let trits = unpack_word(u32::from_le_bytes(bytes.try_into().unwrap()));
                let window = &activations[word * TRITS_PER_WORD..][..TRITS_PER_WORD];
                trits.iter().zip(window).fold(sum, |sum, (&trit, &activation)| {
                    sum.wrapping_add(i32::from(trit) * i32::from(activation))
                })
            });
            scale(sum, projection.factor)
        })
        .collect()
}

fn multiply_halfwords(left: i32, right: i32) -> i32 {
    i32::from(left as i16) * i32::from(right as i16)
}

pub struct ReferenceSession {
    model: IntegerModel,
    keys: Vec<Vec<i32>>,
    values: Vec<Vec<i32>>,
    position: usize,
}

impl ReferenceSession {
    pub fn new(model: IntegerModel) -> ReferenceSession {
        let cache = vec![vec![0; model.config.context * model.config.width]; model.config.layers];
        ReferenceSession { model, keys: cache.clone(), values: cache, position: 0 }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn decode(&mut self, token: usize) -> Result<Vec<i32>, ModelError> {
        let model = &self.model;
        let config = model.config;
        if token >= config.vocabulary_size {
            return Err(ModelError(format!("token {token} is outside the vocabulary")));
        }
        if self.position >= config.context {
            return Err(ModelError(format!("the context of {} tokens is full", config.context)));
        }
        let (width, position) = (config.width, self.position);
        let embed = |rows: &Int8Rows, row: usize| -> Vec<i32> {
            rows.values[row * width..][..width]
                .iter()
                .map(|&value| scale(i32::from(value), rows.factors[row]))
                .collect()
        };
        let mut residual: Vec<i32> = embed(&model.token_embedding, token)
            .iter()
            .zip(embed(&model.position_embedding, position))
            .map(|(a, b)| a.wrapping_add(b))
            .collect();
        for (index, layer) in model.layers.iter().enumerate() {
            let input = quantize(&residual, false);
            let query = project(&layer.query, &input);
            self.keys[index][position * width..][..width]
                .copy_from_slice(&project(&layer.key, &input));
            self.values[index][position * width..][..width]
                .copy_from_slice(&project(&layer.value, &input));
            let attended = self.attend(index, &query);
            add_into(&mut residual, &project(&layer.output, &quantize(&attended, false)));
            let hidden = project(&layer.up, &quantize(&residual, false));
            add_into(&mut residual, &project(&layer.down, &quantize(&hidden, true)));
        }
        let input: Vec<i32> = quantize(&residual, false).into_iter().map(i32::from).collect();
        let head = &model.head;
        let logits = head
            .values
            .chunks_exact(width)
            .zip(&head.factors)
            .map(|(row, &factor)| {
                let sum = row.iter().zip(&input).fold(0i32, |sum, (&weight, &value)| {
                    sum.wrapping_add(multiply_halfwords(i32::from(weight), value))
                });
                scale(sum, factor)
            })
            .collect();
        self.position += 1;
        Ok(logits)
    }

    fn attend(&self, layer: usize, query: &[i32]) -> Vec<i32> {
        let config = self.model.config;
        let (width, head_width, position) = (config.width, config.head_width(), self.position);
        let clamp = |value: i32| value.clamp(-32767, 32767);
        let mut output = vec![0; width];
        for head in 0..config.heads {
            let base = head * head_width;
            let scores: Vec<i32> = (0..=position)
                .map(|source| {
                    let key = &self.keys[layer][source * width + base..][..head_width];
                    let dot = query[base..base + head_width]
                        .iter()
                        .zip(key)
                        .fold(0i32, |sum, (&q, &k)| {
                            sum.wrapping_add(multiply_halfwords(clamp(q), clamp(k)))
                        });
                    scale(dot, self.model.attention_factor)
                })
                .collect();
            let peak = scores.iter().copied().fold(i32::MIN, i32::max);
            let weights: Vec<i32> =
                scores.iter().map(|&score| exp2(score.wrapping_sub(peak))).collect();
            let total = weights.iter().fold(0i32, |total, &weight| total.wrapping_add(weight));
            let root = rsqrt(total);
            let inverse = scale(root, root);
            for offset in 0..head_width {
                output[base + offset] = (0..=position).fold(0i32, |sum, source| {
                    let weight = scale(weights[source], inverse);
                    let value = self.values[layer][source * width + base + offset];
                    sum.wrapping_add(scale(value, weight))
                });
            }
        }
        output
    }
}

fn add_into(total: &mut [i32], values: &[i32]) {
    for (total, value) in total.iter_mut().zip(values) {
        *total = total.wrapping_add(*value);
    }
}
