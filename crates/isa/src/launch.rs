use std::fmt;

use crate::{MAX_ARGUMENTS, SHARED_BYTES, WARP_LANES};

pub const MAX_BLOCK_WARPS: u32 = 4;

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

pub fn validate_launch(config: LaunchConfig, arguments: &[u32]) -> Result<(), Fault> {
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
