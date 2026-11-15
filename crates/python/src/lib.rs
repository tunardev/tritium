use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::{PyIOError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use tritium_framework::operations;
use tritium_framework::{Random, TokenDataset, storage};
use tritium_model::{
    DeviceSession, IntegerModel, ModelConfig, ModelError, ReferenceSession, Statistics, sample,
};
use tritium_runtime::BackendKind;

type CoreTensor = tritium_framework::Tensor;

#[pyclass(name = "Tensor", unsendable)]
#[derive(Clone)]
pub struct Tensor(pub CoreTensor);

fn shape_of(shape: &[usize]) -> Vec<usize> {
    shape.to_vec()
}

fn checked_shape(length: usize, shape: &[usize]) -> PyResult<()> {
    if shape.iter().product::<usize>() == length {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{length} values do not fill shape {shape:?}")))
    }
}

enum Operand {
    Tensor(CoreTensor),
    Number(f32),
}

fn operand(value: &Bound<'_, PyAny>) -> PyResult<Operand> {
    if let Ok(tensor) = value.extract::<PyRef<'_, Tensor>>() {
        return Ok(Operand::Tensor(tensor.0.clone()));
    }
    value
        .extract::<f32>()
        .map(Operand::Number)
        .map_err(|_| PyTypeError::new_err("expected a Tensor or a number"))
}

fn same_shape(left: &CoreTensor, right: &CoreTensor) -> PyResult<()> {
    if left.shape() == right.shape() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "shapes {:?} and {:?} differ",
            left.shape(),
            right.shape()
        )))
    }
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (data, shape, requires_grad = false))]
    fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> PyResult<Tensor> {
        checked_shape(data.len(), &shape)?;
        Ok(Tensor(if requires_grad {
            CoreTensor::parameter(data, &shape)
        } else {
            CoreTensor::new(data, &shape)
        }))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        shape_of(self.0.shape())
    }

    #[getter]
    fn requires_grad(&self) -> bool {
        self.0.requires_gradient()
    }

    #[getter]
    fn grad(&self) -> Option<Tensor> {
        self.0.gradient().map(|gradient| Tensor(CoreTensor::new(gradient, self.0.shape())))
    }

    fn zero_grad(&self) {
        self.0.zero_gradient();
    }

    fn backward(&self) -> PyResult<()> {
        if self.0.len() != 1 {
            return Err(PyValueError::new_err("backward needs a tensor with one element"));
        }
        self.0.backward();
        Ok(())
    }

    fn tolist(&self) -> Vec<f32> {
        self.0.to_vec()
    }

    fn item(&self) -> PyResult<f32> {
        match self.0.data().as_slice() {
            [value] => Ok(*value),
            _ => Err(PyValueError::new_err("item needs a tensor with one element")),
        }
    }

    fn detach(&self) -> Tensor {
        Tensor(self.0.detach())
    }

    fn copy_(&self, source: &Tensor) -> PyResult<()> {
        same_shape(&self.0, &source.0)?;
        self.0.assign(&source.0.data());
        Ok(())
    }

    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Tensor> {
        checked_shape(self.0.len(), &shape)?;
        Ok(Tensor(operations::reshape(&self.0, &shape)))
    }

    fn sum(&self) -> Tensor {
        Tensor(operations::sum(&self.0))
    }

    fn mean(&self) -> Tensor {
        Tensor(operations::mean(&self.0))
    }

    fn __len__(&self) -> usize {
        self.0.shape().first().copied().unwrap_or(1)
    }

    fn __repr__(&self) -> String {
        let data = self.0.data();
        let preview: Vec<String> = data.iter().take(6).map(|value| format!("{value:.4}")).collect();
        let more = if data.len() > 6 { ", ..." } else { "" };
        format!("tensor([{}{more}], shape={:?})", preview.join(", "), self.0.shape())
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        match operand(other)? {
            Operand::Tensor(other) => {
                same_shape(&self.0, &other)?;
                Ok(Tensor(operations::add(&self.0, &other)))
            }
            Operand::Number(number) => {
                let constant = CoreTensor::new(vec![number; self.0.len()], self.0.shape());
                Ok(Tensor(operations::add(&self.0, &constant)))
            }
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        match operand(other)? {
            Operand::Tensor(other) => {
                same_shape(&self.0, &other)?;
                Ok(Tensor(operations::subtract(&self.0, &other)))
            }
            Operand::Number(number) => {
                let constant = CoreTensor::new(vec![number; self.0.len()], self.0.shape());
                Ok(Tensor(operations::subtract(&self.0, &constant)))
            }
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        match operand(other)? {
            Operand::Tensor(other) => {
                same_shape(&self.0, &other)?;
                Ok(Tensor(operations::multiply(&self.0, &other)))
            }
            Operand::Number(number) => Ok(Tensor(operations::scale(&self.0, number))),
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<Tensor> {
        self.__mul__(other)
    }

    fn __neg__(&self) -> Tensor {
        Tensor(operations::scale(&self.0, -1.0))
    }

    fn __matmul__(&self, other: &Tensor) -> PyResult<Tensor> {
        match (self.0.shape(), other.0.shape()) {
            ([_, inner], [other_inner, _]) if inner == other_inner => {
                Ok(Tensor(operations::matmul(&self.0, &other.0)))
            }
            (left, right) => {
                Err(PyValueError::new_err(format!("cannot multiply {left:?} by {right:?}")))
            }
        }
    }
}

fn matrix(tensor: &Tensor, name: &str) -> PyResult<(usize, usize)> {
    match *tensor.0.shape() {
        [rows, columns] => Ok((rows, columns)),
        ref shape => Err(PyValueError::new_err(format!("{name} must be 2-D, found {shape:?}"))),
    }
}

fn matching_features(input: &Tensor, weight: &Tensor) -> PyResult<()> {
    let (_, features) = matrix(input, "input")?;
    let (_, weight_features) = matrix(weight, "weight")?;
    if features == weight_features {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "input has {features} features but the weight expects {weight_features}"
        )))
    }
}

#[pyfunction]
fn linear(input: &Tensor, weight: &Tensor) -> PyResult<Tensor> {
    matching_features(input, weight)?;
    Ok(Tensor(operations::linear(&input.0, &weight.0)))
}

#[pyfunction]
fn bit_linear(input: &Tensor, weight: &Tensor) -> PyResult<Tensor> {
    matching_features(input, weight)?;
    Ok(Tensor(operations::bit_linear(&input.0, &weight.0)))
}

#[pyfunction]
fn int8_linear(input: &Tensor, weight: &Tensor) -> PyResult<Tensor> {
    matching_features(input, weight)?;
    Ok(Tensor(operations::int8_linear(&input.0, &weight.0)))
}

#[pyfunction]
fn embedding(table: &Tensor, indices: Vec<usize>) -> PyResult<Tensor> {
    let (rows, _) = matrix(table, "the embedding table")?;
    if let Some(&index) = indices.iter().find(|&&index| index >= rows) {
        return Err(PyValueError::new_err(format!("index {index} exceeds {rows} rows")));
    }
    Ok(Tensor(operations::embedding(&table.0, &indices)))
}

#[pyfunction]
fn relu_squared(input: &Tensor) -> Tensor {
    Tensor(operations::relu_squared(&input.0))
}

#[pyfunction]
fn causal_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    batch: usize,
    heads: usize,
) -> PyResult<Tensor> {
    let (rows, width) = matrix(query, "query")?;
    same_shape(&query.0, &key.0)?;
    same_shape(&query.0, &value.0)?;
    if batch == 0 || heads == 0 || rows % batch != 0 || width % heads != 0 {
        return Err(PyValueError::new_err("batch and heads must divide the query shape"));
    }
    Ok(Tensor(operations::causal_attention(&query.0, &key.0, &value.0, batch, heads)))
}

#[pyfunction]
fn cross_entropy(logits: &Tensor, targets: Vec<usize>) -> PyResult<Tensor> {
    let (rows, classes) = matrix(logits, "logits")?;
    if rows != targets.len() || targets.iter().any(|&target| target >= classes) {
        return Err(PyValueError::new_err("targets must give one class per row of logits"));
    }
    Ok(Tensor(operations::cross_entropy(&logits.0, &targets)))
}

#[pyfunction]
#[pyo3(signature = (shape, standard_deviation, seed, requires_grad = false))]
fn normal(shape: Vec<usize>, standard_deviation: f32, seed: u64, requires_grad: bool) -> Tensor {
    let mut random = Random::new(seed);
    let data = (0..shape.iter().product::<usize>())
        .map(|_| random.normal() * standard_deviation)
        .collect();
    Tensor(if requires_grad {
        CoreTensor::parameter(data, &shape)
    } else {
        CoreTensor::new(data, &shape)
    })
}

#[pyfunction]
fn set_grad_enabled(is_enabled: bool) -> bool {
    tritium_framework::set_gradient_enabled(is_enabled)
}

#[pyfunction]
fn is_grad_enabled() -> bool {
    tritium_framework::is_gradient_enabled()
}

#[pyfunction]
fn save_tensors(path: PathBuf, tensors: Vec<(String, PyRef<'_, Tensor>)>) -> PyResult<()> {
    let tensors: Vec<(String, CoreTensor)> =
        tensors.into_iter().map(|(name, tensor)| (name, tensor.0.clone())).collect();
    storage::save_tensors(&path, &tensors).map_err(|error| PyIOError::new_err(error.to_string()))
}

#[pyfunction]
fn load_tensors(path: PathBuf) -> PyResult<Vec<(String, Tensor)>> {
    let tensors =
        storage::load_tensors(&path).map_err(|error| PyIOError::new_err(error.to_string()))?;
    Ok(tensors.into_iter().map(|(name, tensor)| (name, Tensor(tensor))).collect())
}

#[pyclass(name = "AdamW", unsendable)]
pub struct AdamW(tritium_framework::AdamW);

#[pymethods]
impl AdamW {
    #[new]
    #[pyo3(signature = (parameters, lr = 1e-3, weight_decay = 0.01))]
    fn new(parameters: Vec<PyRef<'_, Tensor>>, lr: f32, weight_decay: f32) -> AdamW {
        let parameters = parameters.iter().map(|parameter| parameter.0.clone()).collect();
        AdamW(tritium_framework::AdamW::new(parameters, lr, weight_decay))
    }

    #[getter]
    fn lr(&self) -> f32 {
        self.0.learning_rate
    }

    #[setter]
    fn set_lr(&mut self, learning_rate: f32) {
        self.0.learning_rate = learning_rate;
    }

    fn step(&mut self) {
        self.0.step();
    }

    fn zero_grad(&self) {
        self.0.zero_gradients();
    }
}

#[pyclass(name = "Tokenizer")]
pub struct Tokenizer(pub tritium_framework::Tokenizer);

#[pymethods]
impl Tokenizer {
    #[staticmethod]
    fn train(text: &str, vocab_size: usize) -> PyResult<Tokenizer> {
        if vocab_size <= 257 {
            return Err(PyValueError::new_err("vocab_size must exceed 257"));
        }
        Ok(Tokenizer(tritium_framework::Tokenizer::train(text, vocab_size)))
    }

    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Tokenizer> {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| PyIOError::new_err(error.to_string()))?;
        tritium_framework::Tokenizer::from_text(&text).map(Tokenizer).map_err(PyValueError::new_err)
    }

    fn save(&self, path: PathBuf) -> PyResult<()> {
        std::fs::write(path, self.0.to_text())
            .map_err(|error| PyIOError::new_err(error.to_string()))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.0.vocabulary_size()
    }

    #[getter]
    fn end_of_text(&self) -> u32 {
        self.0.end_of_text()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.0.encode(text)
    }

    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        if let Some(token) =
            tokens.iter().find(|&&token| token as usize >= self.0.vocabulary_size())
        {
            return Err(PyValueError::new_err(format!("token {token} is outside the vocabulary")));
        }
        Ok(self.0.decode(&tokens))
    }

    fn encode_file(&self, text_path: PathBuf, token_path: PathBuf) -> PyResult<usize> {
        TokenDataset::encode_file(&self.0, &text_path, &token_path)
            .map_err(|error| PyIOError::new_err(error.to_string()))
    }
}

#[pyclass(name = "TokenDataset", unsendable)]
pub struct Dataset {
    dataset: TokenDataset,
    random: Random,
}

#[pymethods]
impl Dataset {
    #[new]
    #[pyo3(signature = (path, seed = 0))]
    fn new(path: PathBuf, seed: u64) -> PyResult<Dataset> {
        let dataset =
            TokenDataset::open(&path).map_err(|error| PyIOError::new_err(error.to_string()))?;
        Ok(Dataset { dataset, random: Random::new(seed) })
    }

    fn __len__(&self) -> usize {
        self.dataset.len()
    }

    fn batch(&mut self, batch: usize, sequence: usize) -> PyResult<(Vec<usize>, Vec<usize>)> {
        if self.dataset.len() <= sequence + 1 {
            return Err(PyValueError::new_err("the dataset is shorter than one sequence"));
        }
        Ok(self.dataset.batch(batch, sequence, &mut self.random))
    }
}

enum Session {
    Reference(Box<ReferenceSession>),
    Device(Box<DeviceSession>),
}

#[pyclass(name = "DeviceModel", unsendable)]
pub struct DeviceModel {
    model: IntegerModel,
    backend: String,
    session: Session,
}

fn start_session(model: &IntegerModel, backend: &str) -> PyResult<Session> {
    if backend == "reference" {
        return Ok(Session::Reference(Box::new(ReferenceSession::new(model.clone()))));
    }
    let kind = backend.parse::<BackendKind>().map_err(|_| {
        PyValueError::new_err(format!(
            "unknown backend {backend}; expected reference, simulator or rtl"
        ))
    })?;
    Ok(Session::Device(Box::new(DeviceSession::new(model, kind))))
}

fn model_error(error: ModelError) -> PyErr {
    PyValueError::new_err(error.0)
}

#[pymethods]
impl DeviceModel {
    #[new]
    fn new(
        config: HashMap<String, usize>,
        state: Vec<(String, PyRef<'_, Tensor>)>,
        backend: &str,
    ) -> PyResult<DeviceModel> {
        let field = |name: &str| {
            config
                .get(name)
                .copied()
                .ok_or_else(|| PyValueError::new_err(format!("config lacks {name}")))
        };
        let config = ModelConfig {
            vocabulary_size: field("vocab_size")?,
            width: field("width")?,
            layers: field("layers")?,
            heads: field("heads")?,
            hidden: field("hidden")?,
            context: field("context")?,
        };
        let tensors = state.into_iter().map(|(name, tensor)| (name, tensor.0.clone())).collect();
        let model = IntegerModel::from_tensors(config, &tensors).map_err(model_error)?;
        let session = start_session(&model, backend)?;
        Ok(DeviceModel { model, backend: backend.to_string(), session })
    }

    #[getter]
    fn backend(&self) -> &str {
        &self.backend
    }

    #[getter]
    fn context(&self) -> usize {
        self.model.config.context
    }

    #[getter]
    fn ternary_weight_bytes(&self) -> usize {
        self.model.ternary_weight_bytes()
    }

    #[getter]
    fn head_weight_bytes(&self) -> usize {
        self.model.head.values.len()
    }

    fn reset(&mut self) -> PyResult<()> {
        self.session = start_session(&self.model, &self.backend)?;
        Ok(())
    }

    fn decode(&mut self, token: usize) -> PyResult<Vec<i32>> {
        match &mut self.session {
            Session::Reference(session) => session.decode(token),
            Session::Device(session) => session.decode(token),
        }
        .map_err(model_error)
    }

    #[pyo3(signature = (prompt, max_new_tokens, temperature = 0.0, seed = 0))]
    fn generate(
        &mut self,
        prompt: Vec<usize>,
        max_new_tokens: usize,
        temperature: f32,
        seed: u64,
    ) -> PyResult<Vec<usize>> {
        if prompt.is_empty() {
            return Err(PyValueError::new_err("the prompt needs at least one token"));
        }
        self.reset()?;
        let mut random = Random::new(seed);
        let mut logits = Vec::new();
        for &token in &prompt {
            logits = self.decode(token)?;
        }
        let mut generated = Vec::new();
        let room = self.model.config.context - prompt.len().min(self.model.config.context);
        for index in 0..max_new_tokens.min(room + 1) {
            let token = sample(&logits, temperature, &mut random);
            generated.push(token);
            if index + 1 < max_new_tokens.min(room + 1) {
                logits = self.decode(token)?;
            }
        }
        Ok(generated)
    }

    fn statistics(&self) -> HashMap<&'static str, Option<u64>> {
        let statistics = match &self.session {
            Session::Reference(_) => Statistics::default(),
            Session::Device(session) => session.statistics(),
        };
        HashMap::from([
            ("tokens", Some(statistics.tokens)),
            ("launches", Some(statistics.launches)),
            ("warp_instructions", Some(statistics.warp_instructions)),
            ("cycles", statistics.cycles),
        ])
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Tensor>()?;
    module.add_class::<AdamW>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<Dataset>()?;
    module.add_class::<DeviceModel>()?;
    for function in [
        wrap_pyfunction!(linear, module)?,
        wrap_pyfunction!(bit_linear, module)?,
        wrap_pyfunction!(int8_linear, module)?,
        wrap_pyfunction!(embedding, module)?,
        wrap_pyfunction!(relu_squared, module)?,
        wrap_pyfunction!(causal_attention, module)?,
        wrap_pyfunction!(cross_entropy, module)?,
        wrap_pyfunction!(normal, module)?,
        wrap_pyfunction!(set_grad_enabled, module)?,
        wrap_pyfunction!(is_grad_enabled, module)?,
        wrap_pyfunction!(save_tensors, module)?,
        wrap_pyfunction!(load_tensors, module)?,
    ] {
        module.add_function(function)?;
    }
    Ok(())
}
