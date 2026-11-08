use std::ptr::NonNull;

use tritium_isa::{DEFAULT_STEP_LIMIT, Fault, LaunchConfig, LaunchReport, validate_launch};

const PROGRAM_WORDS: usize = 4096;
const CYCLES_PER_STEP_LIMIT: u64 = 4096;
const MEMORY_LATENCY: u32 = 4;

const HOST_STATUS: u32 = 0;
const HOST_GRID_BLOCKS: u32 = 1;
const HOST_BLOCK_THREADS: u32 = 2;
const HOST_SHARED_BYTES: u32 = 3;
const HOST_ARGUMENT_COUNT: u32 = 4;
const HOST_GLOBAL_BYTES: u32 = 5;
const HOST_PROGRAM_LENGTH: u32 = 6;
const HOST_STEP_LIMIT_LOW: u32 = 7;
const HOST_STEP_LIMIT_HIGH: u32 = 8;
const HOST_PROGRAM_ADDRESS: u32 = 9;
const HOST_PROGRAM_DATA: u32 = 10;
const HOST_ARGUMENTS: u32 = 16;
const HOST_FAULT_BLOCK: u32 = 24;
const HOST_FAULT_WARP: u32 = 25;
const HOST_FAULT_PROGRAM_COUNTER: u32 = 26;
const HOST_FAULT_VALUE: u32 = 27;
const HOST_STEPS_LOW: u32 = 28;
const HOST_STEPS_HIGH: u32 = 29;

const STATUS_DONE: u32 = 2;

#[repr(C)]
struct Harness {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn tritium_harness_create(global_bytes: usize, latency: u32) -> *mut Harness;
    fn tritium_harness_destroy(harness: *mut Harness);
    fn tritium_harness_global(harness: *mut Harness) -> *mut u8;
    fn tritium_harness_write_register(harness: *mut Harness, address: u32, value: u32);
    fn tritium_harness_read_register(harness: *mut Harness, address: u32) -> u32;
    fn tritium_harness_run(harness: *mut Harness, cycle_limit: u64) -> u64;
}

pub struct RtlDevice {
    harness: NonNull<Harness>,
    global_bytes: usize,
    step_limit: u64,
    cycles: u64,
}

impl RtlDevice {
    pub fn new(global_bytes: usize) -> RtlDevice {
        let harness = NonNull::new(unsafe { tritium_harness_create(global_bytes, MEMORY_LATENCY) })
            .expect("the Verilator harness could not be created");
        RtlDevice { harness, global_bytes, step_limit: DEFAULT_STEP_LIMIT, cycles: 0 }
    }

    pub fn global(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                tritium_harness_global(self.harness.as_ptr()),
                self.global_bytes,
            )
        }
    }

    pub fn global_mut(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                tritium_harness_global(self.harness.as_ptr()),
                self.global_bytes,
            )
        }
    }

    pub fn set_step_limit(&mut self, steps: u64) {
        self.step_limit = steps;
    }

    pub fn cycles(&self) -> u64 {
        self.cycles
    }

    pub fn launch(
        &mut self,
        program: &[u32],
        config: LaunchConfig,
        arguments: &[u32],
    ) -> Result<LaunchReport, Fault> {
        validate_launch(config, arguments)?;
        if program.len() > PROGRAM_WORDS {
            return Err(Fault::InvalidLaunch(format!(
                "{} instructions exceed the {PROGRAM_WORDS}-word program memory",
                program.len()
            )));
        }
        self.write(HOST_PROGRAM_ADDRESS, 0);
        for &word in program {
            self.write(HOST_PROGRAM_DATA, word);
        }
        self.write(HOST_PROGRAM_LENGTH, program.len() as u32);
        self.write(HOST_GRID_BLOCKS, config.grid_blocks);
        self.write(HOST_BLOCK_THREADS, config.block_threads);
        self.write(HOST_SHARED_BYTES, config.shared_bytes);
        self.write(HOST_GLOBAL_BYTES, self.global_bytes as u32);
        self.write(HOST_ARGUMENT_COUNT, arguments.len() as u32);
        for (index, &argument) in arguments.iter().enumerate() {
            self.write(HOST_ARGUMENTS + index as u32, argument);
        }
        self.write(HOST_STEP_LIMIT_LOW, self.step_limit as u32);
        self.write(HOST_STEP_LIMIT_HIGH, (self.step_limit >> 32) as u32);
        self.write(HOST_STATUS, 1);

        let cycle_limit = self.step_limit.saturating_mul(CYCLES_PER_STEP_LIMIT);
        self.cycles = unsafe { tritium_harness_run(self.harness.as_ptr(), cycle_limit) };
        let status = self.read(HOST_STATUS);
        assert!(status & STATUS_DONE != 0, "the RTL model stopped responding");
        self.fault((status >> 4) & 0xF)?;
        let steps =
            u64::from(self.read(HOST_STEPS_LOW)) | u64::from(self.read(HOST_STEPS_HIGH)) << 32;
        Ok(LaunchReport { warp_instructions: steps })
    }

    fn fault(&self, kind: u32) -> Result<(), Fault> {
        let block = self.read(HOST_FAULT_BLOCK);
        let warp = self.read(HOST_FAULT_WARP);
        let program_counter = self.read(HOST_FAULT_PROGRAM_COUNTER);
        let value = self.read(HOST_FAULT_VALUE);
        Err(match kind {
            0 => return Ok(()),
            1 => Fault::InvalidInstruction { block, warp, program_counter, word: value },
            2 => Fault::ProgramCounterOutOfRange { block, warp, program_counter },
            3 => Fault::MisalignedAccess { block, warp, program_counter, address: value },
            4 => Fault::AddressOutOfRange { block, warp, program_counter, address: value },
            5 => Fault::DivergentBranch { block, warp, program_counter },
            6 => Fault::BarrierDeadlock { block },
            7 => Fault::StepLimitExceeded,
            _ => panic!("the RTL model reported unknown fault {kind}"),
        })
    }

    fn write(&mut self, address: u32, value: u32) {
        unsafe { tritium_harness_write_register(self.harness.as_ptr(), address, value) }
    }

    fn read(&self, address: u32) -> u32 {
        unsafe { tritium_harness_read_register(self.harness.as_ptr(), address) }
    }
}

impl Drop for RtlDevice {
    fn drop(&mut self) {
        unsafe { tritium_harness_destroy(self.harness.as_ptr()) }
    }
}
