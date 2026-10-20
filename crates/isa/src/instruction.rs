use std::fmt;

pub const WARP_LANES: usize = 16;
pub const SHARED_BYTES: usize = 16384;
pub const MAX_ARGUMENTS: usize = 8;
pub const REGISTER_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Register(u8);

impl Register {
    pub const ZERO: Register = Register(0);

    pub fn new(index: u8) -> Option<Register> {
        (usize::from(index) < REGISTER_COUNT).then_some(Register(index))
    }

    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

impl fmt::Display for Register {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "r{}", self.0)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_beyond_31_are_invalid() {
        assert!(Register::new(31).is_some());
        assert!(Register::new(32).is_none());
    }
}
