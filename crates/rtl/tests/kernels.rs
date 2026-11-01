use tritium_assembler::assemble;
use tritium_isa::{Fault, LaunchConfig};
use tritium_rtl::RtlDevice;

#[test]
fn exit_retires_once_per_warp() {
    let mut device = RtlDevice::new(64);
    let config = LaunchConfig { grid_blocks: 2, block_threads: 64, shared_bytes: 0 };
    let report = device.launch(&assemble("exit").unwrap(), config, &[]).unwrap();
    assert_eq!(report.warp_instructions, 8);
}

#[test]
fn unknown_opcode_faults_with_word() {
    let mut device = RtlDevice::new(64);
    let config = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };
    let fault = device.launch(&[0x3F << 26], config, &[]).unwrap_err();
    assert_eq!(
        fault,
        Fault::InvalidInstruction { block: 0, warp: 0, program_counter: 0, word: 0x3F << 26 }
    );
}

#[test]
fn empty_program_faults_at_zero() {
    let mut device = RtlDevice::new(64);
    let config = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };
    let fault = device.launch(&[], config, &[]).unwrap_err();
    assert_eq!(fault, Fault::ProgramCounterOutOfRange { block: 0, warp: 0, program_counter: 0 });
}
