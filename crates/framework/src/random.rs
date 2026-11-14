pub struct Random {
    state: u64,
    increment: u64,
}

impl Random {
    pub fn new(seed: u64) -> Random {
        let mut random = Random { state: 0, increment: (seed << 1) | 1 };
        random.next_u32();
        random.state = random.state.wrapping_add(seed ^ 0x853C_49E6_748F_EA9B);
        random.next_u32();
        random
    }

    pub fn next_u32(&mut self) -> u32 {
        let previous = self.state;
        self.state = previous.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(self.increment);
        let shifted = (((previous >> 18) ^ previous) >> 27) as u32;
        shifted.rotate_right((previous >> 59) as u32)
    }

    pub fn uniform(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn below(&mut self, bound: usize) -> usize {
        ((u64::from(self.next_u32()) * bound as u64) >> 32) as usize
    }

    pub fn normal(&mut self) -> f32 {
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        radius * (std::f32::consts::TAU * self.uniform()).cos()
    }
}
