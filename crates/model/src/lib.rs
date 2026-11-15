mod device;
mod integer;

pub use device::{DeviceSession, Statistics};
pub use integer::{
    Int8Rows, IntegerModel, Layer, ModelConfig, ModelError, Projection, ReferenceSession, quantize,
};

pub const KERNELS: &str = include_str!("kernels.tk");
