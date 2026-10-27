use std::fmt;

use tritium_isa::{MAX_ARGUMENTS, SHARED_BYTES, WARP_LANES};

use crate::warp::{Warp, WarpStatus};

const DEFAULT_STEP_LIMIT: u64 = 100_000_000;
const MAX_BLOCK_WARPS: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchConfig {
    pub grid_blocks: u32,
    pub block_threads: u32,
    pub shared_bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LaunchReport {
    pub warp_instructions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    InvalidLaunch(String),
    InvalidInstruction { block: u32, warp: u32, program_counter: u32, word: u32 },
    ProgramCounterOutOfRange { block: u32, warp: u32, program_counter: u32 },
    MisalignedAccess { block: u32, warp: u32, program_counter: u32, address: u32 },
    AddressOutOfRange { block: u32, warp: u32, program_counter: u32, address: u32 },
    DivergentBranch { block: u32, warp: u32, program_counter: u32 },
    BarrierDeadlock { block: u32 },
    StepLimitExceeded,
}

impl fmt::Display for Fault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::InvalidLaunch(reason) => write!(formatter, "invalid launch: {reason}"),
            Fault::InvalidInstruction { block, warp, program_counter, word } => write!(
                formatter,
                "block {block} warp {warp} at {program_counter}: invalid instruction {word:#010x}"
            ),
            Fault::ProgramCounterOutOfRange { block, warp, program_counter } => write!(
                formatter,
                "block {block} warp {warp}: program counter {program_counter} is past the program"
            ),
            Fault::MisalignedAccess { block, warp, program_counter, address } => write!(
                formatter,
                "block {block} warp {warp} at {program_counter}: misaligned access to {address:#x}"
            ),
            Fault::AddressOutOfRange { block, warp, program_counter, address } => write!(
                formatter,
                "block {block} warp {warp} at {program_counter}: address {address:#x} is out of range"
            ),
            Fault::DivergentBranch { block, warp, program_counter } => write!(
                formatter,
                "block {block} warp {warp} at {program_counter}: lanes disagree on a branch"
            ),
            Fault::BarrierDeadlock { block } => {
                write!(formatter, "block {block}: a warp waits at a barrier no other warp reaches")
            }
            Fault::StepLimitExceeded => formatter.write_str("the step limit was exceeded"),
        }
    }
}

impl std::error::Error for Fault {}

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
                let live = warps.iter().filter(|warp| warp.status() != WarpStatus::Exited).count();
                if live == 1 && warps.len() > 1 {
                    return Err(Fault::BarrierDeadlock { block });
                }
                warps.iter_mut().for_each(Warp::release_barrier);
            }
        }
        Ok(LaunchReport { warp_instructions: steps })
    }
}

fn validate_launch(config: LaunchConfig, arguments: &[u32]) -> Result<(), Fault> {
    let invalid = |reason: String| Err(Fault::InvalidLaunch(reason));
    let lanes = WARP_LANES as u32;
    if config.grid_blocks == 0 {
        return invalid("the grid has no blocks".into());
    }
    if config.block_threads == 0
        || config.block_threads % lanes != 0
        || config.block_threads > MAX_BLOCK_WARPS * lanes
    {
        return invalid(format!(
            "block threads must be 16, 32, 48 or 64, not {}",
            config.block_threads
        ));
    }
    if config.shared_bytes as usize > SHARED_BYTES {
        return invalid(format!("{} shared bytes exceed {SHARED_BYTES}", config.shared_bytes));
    }
    if arguments.len() > MAX_ARGUMENTS {
        return invalid(format!("{} arguments exceed {MAX_ARGUMENTS}", arguments.len()));
    }
    Ok(())
}
