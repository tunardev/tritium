use std::collections::HashMap;

use tritium_framework::{Random, Tensor};
use tritium_model::{IntegerModel, ModelConfig};

pub const SMALL: ModelConfig =
    ModelConfig { vocabulary_size: 64, width: 64, layers: 2, heads: 2, hidden: 128, context: 8 };

pub fn random_model(config: ModelConfig, seed: u64) -> IntegerModel {
    let mut random = Random::new(seed);
    let mut tensor = |rows: usize, columns: usize, deviation: f32| {
        let data = (0..rows * columns).map(|_| random.normal() * deviation).collect();
        Tensor::new(data, &[rows, columns])
    };
    let (width, hidden) = (config.width, config.hidden);
    let mut tensors = HashMap::new();
    tensors
        .insert("token_embedding.weight".to_string(), tensor(config.vocabulary_size, width, 0.5));
    tensors.insert("position_embedding.weight".to_string(), tensor(config.context, width, 0.5));
    tensors.insert("head.weight".to_string(), tensor(config.vocabulary_size, width, 0.1));
    for layer in 0..config.layers {
        for (part, rows, columns) in [
            ("query", width, width),
            ("key", width, width),
            ("value", width, width),
            ("output", width, width),
            ("up", hidden, width),
            ("down", width, hidden),
        ] {
            tensors.insert(format!("blocks.{layer}.{part}.weight"), tensor(rows, columns, 0.05));
        }
    }
    IntegerModel::from_tensors(config, &tensors).unwrap()
}
