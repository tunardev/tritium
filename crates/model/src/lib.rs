mod device;
mod integer;
mod sampling;

pub use device::{DeviceSession, Statistics};
pub use integer::{
    Int8Rows, IntegerModel, Layer, ModelConfig, ModelError, Projection, ReferenceSession,
    WeightFormat, quantize,
};
pub use sampling::sample;

pub const KERNELS: &str = include_str!("kernels.tk");
