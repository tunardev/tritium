use tritium_compiler::{Kernel, compile};
use tritium_runtime::{Backend, BackendKind, LaunchConfig};

use crate::integer::{IntegerModel, ModelError, Projection, SHARED_VECTOR_BYTES};

const BLOCK_THREADS: u32 = 64;

struct Kernels {
    embed: Kernel,
    quantize: Kernel,
    project: Kernel,
    attention: Kernel,
    head: Kernel,
}

impl Kernels {
    fn compile() -> Kernels {
        let mut kernels = compile(crate::KERNELS).expect("the model kernels compile").into_iter();
        let mut next = || kernels.next().expect("five model kernels");
        Kernels {
            embed: next(),
            quantize: next(),
            project: next(),
            attention: next(),
            head: next(),
        }
    }
}

#[derive(Clone, Copy)]
struct PlacedProjection {
    address: u32,
    rows: u32,
    words_per_row: u32,
    factor: i32,
}

struct PlacedLayer {
    query: PlacedProjection,
    key: PlacedProjection,
    value: PlacedProjection,
    output: PlacedProjection,
    up: PlacedProjection,
    down: PlacedProjection,
    keys: u32,
    values: u32,
}

struct Layout {
    token_values: u32,
    token_factors: u32,
    position_values: u32,
    position_factors: u32,
    layers: Vec<PlacedLayer>,
    head_values: u32,
    head_factors: u32,
    residual: u32,
    query: u32,
    attended: u32,
    hidden: u32,
    quantized: u32,
    logits: u32,
}

struct Memory {
    bytes: Vec<u8>,
}

impl Memory {
    fn place(&mut self, data: &[u8]) -> u32 {
        let address = self.reserve(data.len());
        self.bytes[address as usize..][..data.len()].copy_from_slice(data);
        address
    }

    fn reserve(&mut self, length: usize) -> u32 {
        let address = self.bytes.len().next_multiple_of(4);
        self.bytes.resize(address + length.next_multiple_of(4), 0);
        address as u32
    }

    fn place_words(&mut self, words: &[i32]) -> u32 {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        self.place(&bytes)
    }

    fn place_projection(&mut self, projection: &Projection) -> PlacedProjection {
        PlacedProjection {
            address: self.place(&projection.packed),
            rows: projection.rows as u32,
            words_per_row: projection.words_per_row as u32,
            factor: projection.factor,
        }
    }
}

fn bytes_of(values: &[i8]) -> Vec<u8> {
    values.iter().map(|&value| value as u8).collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Statistics {
    pub tokens: u64,
    pub launches: u64,
    pub warp_instructions: u64,
    pub cycles: Option<u64>,
}

pub struct DeviceSession {
    backend: Box<dyn Backend>,
    kernels: Kernels,
    layout: Layout,
    width: u32,
    hidden: u32,
    heads: u32,
    head_width: u32,
    vocabulary_size: usize,
    context: usize,
    attention_factor: i32,
    position: usize,
    statistics: Statistics,
}

impl DeviceSession {
    pub fn new(model: &IntegerModel, kind: BackendKind) -> DeviceSession {
        let config = model.config;
        let mut memory = Memory { bytes: Vec::new() };
        let token_values = memory.place(&bytes_of(&model.token_embedding.values));
        let token_factors = memory.place_words(&model.token_embedding.factors);
        let position_values = memory.place(&bytes_of(&model.position_embedding.values));
        let position_factors = memory.place_words(&model.position_embedding.factors);
        let cache_bytes = config.context * config.width * 4;
        let layers = model
            .layers
            .iter()
            .map(|layer| PlacedLayer {
                query: memory.place_projection(&layer.query),
                key: memory.place_projection(&layer.key),
                value: memory.place_projection(&layer.value),
                output: memory.place_projection(&layer.output),
                up: memory.place_projection(&layer.up),
                down: memory.place_projection(&layer.down),
                keys: memory.reserve(cache_bytes),
                values: memory.reserve(cache_bytes),
            })
            .collect();
        let head_values = memory.place(&bytes_of(&model.head.values));
        let head_factors = memory.place_words(&model.head.factors);
        let layout = Layout {
            token_values,
            token_factors,
            position_values,
            position_factors,
            layers,
            head_values,
            head_factors,
            residual: memory.reserve(config.width * 4),
            query: memory.reserve(config.width * 4),
            attended: memory.reserve(config.width * 4),
            hidden: memory.reserve(config.hidden * 4),
            quantized: memory.reserve(SHARED_VECTOR_BYTES),
            logits: memory.reserve(config.vocabulary_size * 4),
        };
        let mut backend = kind.open(memory.bytes.len());
        backend.global_mut().copy_from_slice(&memory.bytes);
        DeviceSession {
            backend,
            kernels: Kernels::compile(),
            layout,
            width: config.width as u32,
            hidden: config.hidden as u32,
            heads: config.heads as u32,
            head_width: config.head_width() as u32,
            vocabulary_size: config.vocabulary_size,
            context: config.context,
            attention_factor: model.attention_factor,
            position: 0,
            statistics: Statistics::default(),
        }
    }

    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    pub fn global_bytes(&self) -> usize {
        self.backend.global().len()
    }

    fn launch(
        &mut self,
        kernel: fn(&Kernels) -> &Kernel,
        grid_blocks: u32,
        arguments: &[u32],
    ) -> Result<(), ModelError> {
        let kernel = kernel(&self.kernels);
        let config = LaunchConfig {
            grid_blocks,
            block_threads: BLOCK_THREADS,
            shared_bytes: kernel.shared_bytes,
        };
        let report = self
            .backend
            .launch(&kernel.program, config, arguments)
            .map_err(|fault| ModelError(format!("{} faulted: {fault}", kernel.name)))?;
        self.statistics.launches += 1;
        self.statistics.warp_instructions += report.warp_instructions;
        if let Some(cycles) = self.backend.cycles() {
            *self.statistics.cycles.get_or_insert(0) += cycles;
        }
        Ok(())
    }

    fn quantize(&mut self, input: u32, length: u32, is_squared: bool) -> Result<(), ModelError> {
        let arguments = [input, self.layout.quantized, length, u32::from(is_squared)];
        self.launch(|kernels| &kernels.quantize, 1, &arguments)
    }

    fn project(
        &mut self,
        projection: PlacedProjection,
        output: u32,
        is_accumulating: bool,
    ) -> Result<(), ModelError> {
        let arguments = [
            self.layout.quantized,
            projection.address,
            output,
            projection.words_per_row,
            projection.factor as u32,
            u32::from(is_accumulating),
        ];
        self.launch(|kernels| &kernels.project, projection.rows / BLOCK_THREADS, &arguments)
    }

    pub fn decode(&mut self, token: usize) -> Result<Vec<i32>, ModelError> {
        if token >= self.vocabulary_size {
            return Err(ModelError(format!("token {token} is outside the vocabulary")));
        }
        if self.position >= self.context {
            return Err(ModelError(format!("the context of {} tokens is full", self.context)));
        }
        let (width, position) = (self.width, self.position as u32);
        let layout_residual = self.layout.residual;
        let embed_arguments = [
            self.layout.token_values,
            self.layout.token_factors,
            self.layout.position_values,
            self.layout.position_factors,
            layout_residual,
            token as u32,
            position,
            width,
        ];
        self.launch(|kernels| &kernels.embed, width / BLOCK_THREADS, &embed_arguments)?;
        for index in 0..self.layout.layers.len() {
            let layer = &self.layout.layers[index];
            let (query, key, value, output, up, down) =
                (layer.query, layer.key, layer.value, layer.output, layer.up, layer.down);
            let slot = position * width * 4;
            let (keys, values) = (layer.keys, layer.values);
            self.quantize(layout_residual, width, false)?;
            self.project(query, self.layout.query, false)?;
            self.project(key, keys + slot, false)?;
            self.project(value, values + slot, false)?;
            let attention_arguments = [
                self.layout.query,
                keys,
                values,
                self.layout.attended,
                position,
                width,
                self.head_width,
                self.attention_factor as u32,
            ];
            self.launch(|kernels| &kernels.attention, self.heads, &attention_arguments)?;
            self.quantize(self.layout.attended, width, false)?;
            self.project(output, layout_residual, true)?;
            self.quantize(layout_residual, width, false)?;
            self.project(up, self.layout.hidden, false)?;
            self.quantize(self.layout.hidden, self.hidden, true)?;
            self.project(down, layout_residual, true)?;
        }
        self.quantize(layout_residual, width, false)?;
        let head_arguments = [
            self.layout.quantized,
            self.layout.head_values,
            self.layout.head_factors,
            self.layout.logits,
            width,
        ];
        let blocks = self.vocabulary_size as u32 / BLOCK_THREADS;
        self.launch(|kernels| &kernels.head, blocks, &head_arguments)?;
        self.position += 1;
        self.statistics.tokens += 1;
        let start = self.layout.logits as usize;
        Ok(self.backend.global()[start..start + 4 * self.vocabulary_size]
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    }
}
