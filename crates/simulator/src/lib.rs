use std::fmt;

use tritium_isa::{
    BranchCondition, ImmediateOperation, Instruction, LoadWidth, MAX_ARGUMENTS, MemorySpace,
    REGISTER_COUNT, Register, RegisterOperation, SHARED_BYTES, Special, StoreWidth, TRITS_PER_WORD,
    WARP_LANES, exp2, rsqrt, saturate_to_int8, scale, unpack_word,
};

const DEFAULT_STEP_LIMIT: u64 = 100_000_000;

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
        let mut steps = 0u64;
        for block in 0..config.grid_blocks {
            let mut shared = vec![0u8; config.shared_bytes as usize];
            let mut warp = Warp::new(block, 0, config, arguments);
            while warp.status() == WarpStatus::Running {
                if steps >= self.step_limit {
                    return Err(Fault::StepLimitExceeded);
                }
                steps += 1;
                warp.step(program, &mut self.global, &mut shared)?;
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
    if config.block_threads != lanes {
        return invalid(format!("block threads must be 16, not {}", config.block_threads));
    }
    if config.shared_bytes as usize > SHARED_BYTES {
        return invalid(format!("{} shared bytes exceed {SHARED_BYTES}", config.shared_bytes));
    }
    if arguments.len() > MAX_ARGUMENTS {
        return invalid(format!("{} arguments exceed {MAX_ARGUMENTS}", arguments.len()));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WarpStatus {
    Running,
    Exited,
}

struct Warp {
    block: u32,
    index: u32,
    config: LaunchConfig,
    registers: [[u32; REGISTER_COUNT]; WARP_LANES],
    program_counter: u32,
    status: WarpStatus,
}

impl Warp {
    fn new(block: u32, index: u32, config: LaunchConfig, arguments: &[u32]) -> Warp {
        let mut registers = [[0u32; REGISTER_COUNT]; WARP_LANES];
        for lane_registers in &mut registers {
            lane_registers[1..=arguments.len()].copy_from_slice(arguments);
        }
        Warp { block, index, config, registers, program_counter: 0, status: WarpStatus::Running }
    }

    fn status(&self) -> WarpStatus {
        self.status
    }

    fn step(&mut self, program: &[u32], global: &mut [u8], shared: &mut [u8]) -> Result<(), Fault> {
        let program_counter = self.program_counter;
        let word =
            *program.get(program_counter as usize).ok_or(Fault::ProgramCounterOutOfRange {
                block: self.block,
                warp: self.index,
                program_counter,
            })?;
        let instruction = Instruction::decode(word).map_err(|_| Fault::InvalidInstruction {
            block: self.block,
            warp: self.index,
            program_counter,
            word,
        })?;
        let mut next = program_counter.wrapping_add(1);
        match instruction {
            Instruction::Register { operation, destination, source, second_source } => {
                for lane in 0..WARP_LANES {
                    let value = self.register_operation(
                        operation,
                        lane,
                        destination,
                        source,
                        second_source,
                        global,
                        shared,
                    )?;
                    self.write(lane, destination, value);
                }
            }
            Instruction::Immediate { operation, destination, source, immediate } => {
                for lane in 0..WARP_LANES {
                    let value = immediate_operation(operation, self.read(lane, source), immediate);
                    self.write(lane, destination, value);
                }
            }
            Instruction::Load { width, space, destination, base, offset } => {
                for lane in 0..WARP_LANES {
                    let address = self.read(lane, base).wrapping_add(offset as i32 as u32);
                    let memory = match space {
                        MemorySpace::Global => &*global,
                        MemorySpace::Shared => &*shared,
                    };
                    let value = self.load(memory, width, address)?;
                    self.write(lane, destination, value);
                }
            }
            Instruction::Store { width, space, value, base, offset } => {
                for lane in 0..WARP_LANES {
                    let address = self.read(lane, base).wrapping_add(offset as i32 as u32);
                    let memory = match space {
                        MemorySpace::Global => &mut *global,
                        MemorySpace::Shared => &mut *shared,
                    };
                    self.store(memory, width, address, self.read(lane, value))?;
                }
            }
            Instruction::Branch { condition, source, second_source, offset } => {
                let is_taken = self.uniform(|warp, lane| {
                    let (left, right) =
                        (warp.read(lane, source) as i32, warp.read(lane, second_source) as i32);
                    match condition {
                        BranchCondition::Equal => left == right,
                        BranchCondition::NotEqual => left != right,
                        BranchCondition::LessThan => left < right,
                        BranchCondition::GreaterOrEqual => left >= right,
                    }
                })?;
                if is_taken {
                    next = program_counter.wrapping_add(offset as i32 as u32);
                }
            }
            Instruction::LoadUpper { destination, immediate } => {
                for lane in 0..WARP_LANES {
                    self.write(lane, destination, u32::from(immediate) << 16);
                }
            }
            Instruction::ReadSpecial { destination, special } => {
                for lane in 0..WARP_LANES {
                    let thread = self.index * WARP_LANES as u32 + lane as u32;
                    let value = match special {
                        Special::ThreadIndex => thread,
                        Special::BlockIndex => self.block,
                        Special::BlockThreads => self.config.block_threads,
                        Special::GridBlocks => self.config.grid_blocks,
                        Special::LaneIndex => lane as u32,
                        Special::WarpIndex => self.index,
                    };
                    self.write(lane, destination, value);
                }
            }
            Instruction::JumpAndLink { destination, offset } => {
                for lane in 0..WARP_LANES {
                    self.write(lane, destination, next);
                }
                next = program_counter.wrapping_add(offset as u32);
            }
            Instruction::JumpAndLinkRegister { destination, base, offset } => {
                let target = self.uniform(|warp, lane| {
                    warp.read(lane, base).wrapping_add(offset as i32 as u32)
                })?;
                for lane in 0..WARP_LANES {
                    self.write(lane, destination, next);
                }
                next = target;
            }
            Instruction::Barrier => {}
            Instruction::Exit => self.status = WarpStatus::Exited,
        }
        self.program_counter = next;
        Ok(())
    }

    fn read(&self, lane: usize, register: Register) -> u32 {
        self.registers[lane][register.index()]
    }

    fn write(&mut self, lane: usize, register: Register, value: u32) {
        if register != Register::ZERO {
            self.registers[lane][register.index()] = value;
        }
    }

    fn uniform<T: PartialEq>(&self, value: impl Fn(&Warp, usize) -> T) -> Result<T, Fault> {
        let first = value(self, 0);
        if (1..WARP_LANES).all(|lane| value(self, lane) == first) {
            Ok(first)
        } else {
            Err(Fault::DivergentBranch {
                block: self.block,
                warp: self.index,
                program_counter: self.program_counter,
            })
        }
    }

    fn checked_range(&self, length: usize, address: u32, size: usize) -> Result<usize, Fault> {
        if address as usize % size != 0 {
            return Err(Fault::MisalignedAccess {
                block: self.block,
                warp: self.index,
                program_counter: self.program_counter,
                address,
            });
        }
        let start = address as usize;
        if start.checked_add(size).is_none_or(|end| end > length) {
            return Err(Fault::AddressOutOfRange {
                block: self.block,
                warp: self.index,
                program_counter: self.program_counter,
                address,
            });
        }
        Ok(start)
    }

    fn load(&self, memory: &[u8], width: LoadWidth, address: u32) -> Result<u32, Fault> {
        let size = match width {
            LoadWidth::Word => 4,
            LoadWidth::Half | LoadWidth::HalfUnsigned => 2,
            LoadWidth::Byte | LoadWidth::ByteUnsigned => 1,
        };
        let start = self.checked_range(memory.len(), address, size)?;
        let bytes = &memory[start..start + size];
        Ok(match width {
            LoadWidth::Word => u32::from_le_bytes(bytes.try_into().unwrap()),
            LoadWidth::Half => u32::from(u16::from_le_bytes(bytes.try_into().unwrap())),
            LoadWidth::HalfUnsigned => u32::from(u16::from_le_bytes(bytes.try_into().unwrap())),
            LoadWidth::Byte => bytes[0] as i8 as i32 as u32,
            LoadWidth::ByteUnsigned => u32::from(bytes[0]),
        })
    }

    fn store(
        &self,
        memory: &mut [u8],
        width: StoreWidth,
        address: u32,
        value: u32,
    ) -> Result<(), Fault> {
        let size = match width {
            StoreWidth::Word => 4,
            StoreWidth::Half => 2,
            StoreWidth::Byte => 1,
        };
        let start = self.checked_range(memory.len(), address, size)?;
        memory[start..start + size].copy_from_slice(&value.to_le_bytes()[..size]);
        Ok(())
    }

    fn register_operation(
        &self,
        operation: RegisterOperation,
        lane: usize,
        destination: Register,
        source: Register,
        second_source: Register,
        global: &[u8],
        shared: &[u8],
    ) -> Result<u32, Fault> {
        let left = self.read(lane, source);
        let right = self.read(lane, second_source);
        Ok(match operation {
            RegisterOperation::Add => left.wrapping_add(right),
            RegisterOperation::Subtract => left.wrapping_sub(right),
            RegisterOperation::And => left & right,
            RegisterOperation::Or => left | right,
            RegisterOperation::Xor => left ^ right,
            RegisterOperation::ShiftLeft => left << (right & 31),
            RegisterOperation::ShiftRightLogical => left >> (right & 31),
            RegisterOperation::ShiftRightArithmetic => ((left as i32) >> (right & 31)) as u32,
            RegisterOperation::SetLessThan => u32::from((left as i32) < (right as i32)),
            RegisterOperation::SetLessThanUnsigned => u32::from(left < right),
            RegisterOperation::Minimum => (left as i32).min(right as i32) as u32,
            RegisterOperation::Maximum => (left as i32).max(right as i32) as u32,
            RegisterOperation::MultiplyBytes => {
                (i32::from(left as u8 as i8) * i32::from(right as u8 as i8)) as u32
            }
            RegisterOperation::Select => {
                if self.read(lane, destination) != 0 {
                    left
                } else {
                    right
                }
            }
            RegisterOperation::TernaryDot => {
                let weight_start = self.checked_range(global.len(), left, 4)?;
                let word =
                    u32::from_le_bytes(global[weight_start..weight_start + 4].try_into().unwrap());
                let activation_start = self.checked_range(shared.len(), right, 1)?;
                if activation_start + TRITS_PER_WORD > shared.len() {
                    return Err(Fault::AddressOutOfRange {
                        block: self.block,
                        warp: self.index,
                        program_counter: self.program_counter,
                        address: right,
                    });
                }
                let sum: i32 = unpack_word(word)
                    .iter()
                    .zip(&shared[activation_start..activation_start + TRITS_PER_WORD])
                    .map(|(&trit, &activation)| i32::from(trit) * i32::from(activation as i8))
                    .sum();
                self.read(lane, destination).wrapping_add(sum as u32)
            }
            RegisterOperation::Scale => scale(left as i32, right as i32) as u32,
            RegisterOperation::Exp2 => exp2(left as i32) as u32,
            RegisterOperation::ReciprocalSquareRoot => rsqrt(left as i32) as u32,
            RegisterOperation::SaturateInt8 => saturate_to_int8(left as i32) as u32,
        })
    }
}

fn immediate_operation(operation: ImmediateOperation, value: u32, immediate: i16) -> u32 {
    let signed = immediate as i32 as u32;
    let unsigned = u32::from(immediate as u16);
    match operation {
        ImmediateOperation::AddImmediate => value.wrapping_add(signed),
        ImmediateOperation::AndImmediate => value & unsigned,
        ImmediateOperation::OrImmediate => value | unsigned,
        ImmediateOperation::XorImmediate => value ^ unsigned,
        ImmediateOperation::ShiftLeftImmediate => value << (unsigned & 31),
        ImmediateOperation::ShiftRightLogicalImmediate => value >> (unsigned & 31),
        ImmediateOperation::ShiftRightArithmeticImmediate => {
            ((value as i32) >> (unsigned & 31)) as u32
        }
        ImmediateOperation::SetLessThanImmediate => u32::from((value as i32) < (signed as i32)),
    }
}
