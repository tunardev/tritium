use std::fs;
use std::io;
use std::path::Path;

use crate::tensor::Tensor;

const MAGIC: &[u8; 8] = b"TRITIUM1";

pub fn save_tensors(path: &Path, tensors: &[(String, Tensor)]) -> io::Result<()> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for (name, tensor) in tensors {
        bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(tensor.shape().len() as u32).to_le_bytes());
        for &dimension in tensor.shape() {
            bytes.extend_from_slice(&(dimension as u64).to_le_bytes());
        }
        for value in tensor.data().iter() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    fs::write(path, bytes)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn take(&mut self, count: usize) -> io::Result<&[u8]> {
        let end = self.offset.checked_add(count).filter(|&end| end <= self.bytes.len());
        let end =
            end.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "truncated tensor file"))?;
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

pub fn load_tensors(path: &Path) -> io::Result<Vec<(String, Tensor)>> {
    let bytes = fs::read(path)?;
    let mut reader = Reader { bytes: &bytes, offset: 0 };
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a tritium tensor file"));
    }
    let count = reader.u32()?;
    let mut tensors = Vec::new();
    for _ in 0..count {
        let length = reader.u32()? as usize;
        let name = String::from_utf8(reader.take(length)?.to_vec())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "tensor name is not UTF-8"))?;
        let dimensions = reader.u32()? as usize;
        let mut shape = Vec::with_capacity(dimensions);
        for _ in 0..dimensions {
            shape.push(u64::from_le_bytes(reader.take(8)?.try_into().unwrap()) as usize);
        }
        let elements: usize = shape.iter().product();
        let data = reader
            .take(elements * 4)?
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        tensors.push((name, Tensor::new(data, &shape)));
    }
    Ok(tensors)
}
