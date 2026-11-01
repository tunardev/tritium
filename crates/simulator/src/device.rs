use tritium_isa::{Fault, LaunchConfig, LaunchReport, WARP_LANES, validate_launch};

use crate::warp::{Warp, WarpStatus};

const DEFAULT_STEP_LIMIT: u64 = 100_000_000;

pub struct Device {
    global: Vec<u8>,
    step_limit: u64,
}

impl Device {
    pub fn new(global_bytes: usize) -> Device {
        Device { global: vec![0; global_bytes], step_limit: DEFAULT_STEP_LIMIT }
    }

    pub fn global(&self) -> &[u8] {
        &self.global
    }

    pub fn global_mut(&mut self) -> &mut [u8] {
        &mut self.global
    }

    pub fn set_step_limit(&mut self, steps: u64) {
        self.step_limit = steps;
    }

    pub fn launch(
        &mut self,
        program: &[u32],
        config: LaunchConfig,
        arguments: &[u32],
    ) -> Result<LaunchReport, Fault> {
        validate_launch(config, arguments)?;
        let warp_count = config.block_threads / WARP_LANES as u32;
        let mut steps = 0u64;
        for block in 0..config.grid_blocks {
            let mut shared = vec![0u8; config.shared_bytes as usize];
            let mut warps: Vec<Warp> =
                (0..warp_count).map(|warp| Warp::new(block, warp, config, arguments)).collect();
            loop {
                for warp in &mut warps {
                    while warp.status() == WarpStatus::Running {
                        if steps >= self.step_limit {
                            return Err(Fault::StepLimitExceeded);
                        }
                        steps += 1;
                        warp.step(program, &mut self.global, &mut shared)?;
                    }
                }
                let is_waiting = warps.iter().any(|warp| warp.status() == WarpStatus::AtBarrier);
                if !is_waiting {
                    break;
                }
                if warps.iter().any(|warp| warp.status() == WarpStatus::Exited) {
                    return Err(Fault::BarrierDeadlock { block });
                }
                warps.iter_mut().for_each(Warp::release_barrier);
            }
        }
        Ok(LaunchReport { warp_instructions: steps })
    }
}
