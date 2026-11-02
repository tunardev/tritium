use tritium_assembler::assemble;
use tritium_isa::{Fault, LaunchConfig, LaunchReport};
use tritium_rtl::RtlDevice;
use tritium_simulator::Device;

const ONE_WARP: LaunchConfig = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };

fn pseudo_random_bytes(seed: u32, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

fn assert_matches_simulator(
    source: &str,
    config: LaunchConfig,
    arguments: &[u32],
    memory: &[u8],
) -> Result<LaunchReport, Fault> {
    let program = assemble(source).unwrap();
    let mut simulator = Device::new(memory.len());
    simulator.global_mut().copy_from_slice(memory);
    let mut rtl = RtlDevice::new(memory.len());
    rtl.global_mut().copy_from_slice(memory);

    let expected = simulator.launch(&program, config, arguments);
    let actual = rtl.launch(&program, config, arguments);

    assert_eq!(actual, expected);
    if expected.is_ok() {
        assert!(rtl.global() == simulator.global(), "global memory differs");
    }
    actual
}

#[test]
fn exit_retires_once_per_warp() {
    let config = LaunchConfig { grid_blocks: 2, block_threads: 64, shared_bytes: 0 };
    let report = assert_matches_simulator("exit", config, &[], &[0; 64]).unwrap();
    assert_eq!(report.warp_instructions, 8);
}

#[test]
fn unknown_opcode_faults_with_word() {
    let mut device = RtlDevice::new(64);
    let fault = device.launch(&[0x3F << 26], ONE_WARP, &[]).unwrap_err();
    assert_eq!(
        fault,
        Fault::InvalidInstruction { block: 0, warp: 0, program_counter: 0, word: 0x3F << 26 }
    );
}

#[test]
fn empty_program_faults_at_zero() {
    let mut device = RtlDevice::new(64);
    let fault = device.launch(&[], ONE_WARP, &[]).unwrap_err();
    assert_eq!(fault, Fault::ProgramCounterOutOfRange { block: 0, warp: 0, program_counter: 0 });
}

#[test]
fn vector_add_matches_simulator() {
    let source = "sreg r10, block_index\nsreg r11, block_threads\nmul16 r10, r10, r11\n\
                  sreg r12, thread_index\nadd r10, r10, r12\nslli r10, r10, 2\n\
                  add r13, r1, r10\nlw r14, 0(r13)\nadd r13, r2, r10\nlw r15, 0(r13)\n\
                  add r14, r14, r15\nadd r13, r3, r10\nsw r14, 0(r13)\nexit";
    let config = LaunchConfig { grid_blocks: 3, block_threads: 48, shared_bytes: 0 };
    let memory = pseudo_random_bytes(1, 3072);
    assert_matches_simulator(source, config, &[0, 1024, 2048], &memory).unwrap();
}

#[test]
fn special_registers_match_simulator() {
    let source = "sreg r10, thread_index\nsreg r11, block_index\nsreg r12, block_threads\n\
                  sreg r13, block_threads\nmul16 r13, r13, r11\nadd r13, r13, r10\n\
                  slli r13, r13, 5\nsw r10, 0(r13)\nsw r11, 4(r13)\nsw r12, 8(r13)\n\
                  sreg r14, grid_blocks\nsw r14, 12(r13)\nsreg r14, lane_index\nsw r14, 16(r13)\n\
                  sreg r14, warp_index\nsw r14, 20(r13)\nsw r1, 24(r13)\nsw r8, 28(r13)\nexit";
    let config = LaunchConfig { grid_blocks: 2, block_threads: 32, shared_bytes: 0 };
    let arguments = [11, 12, 13, 14, 15, 16, 17, 18];
    assert_matches_simulator(source, config, &arguments, &[0; 2048]).unwrap();
}

#[test]
fn alu_operations_match_simulator() {
    let operations = ["add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu"]
        .into_iter()
        .chain(["min", "max", "mul16", "select", "sat8"]);
    let mut source = String::from(
        "sreg r10, lane_index\nslli r10, r10, 3\nadd r11, r1, r10\nlw r12, 0(r11)\n\
         lw r13, 4(r11)\nslli r20, r10, 4\nadd r20, r2, r20\n",
    );
    for (index, operation) in operations.enumerate() {
        source += &format!("mov r14, r12\n{operation} r14, r12, r13\nsw r14, {}(r20)\n", 4 * index);
    }
    source += "addi r14, r12, -5\nsw r14, 60(r20)\nslti r14, r12, 7\nsw r14, 64(r20)\n\
               xori r14, r12, 0xFFFF\nsw r14, 68(r20)\nsrai r14, r12, 7\nsw r14, 72(r20)\n\
               lui r14, 0xBEEF\nsw r14, 76(r20)\nexit";
    let mut memory = pseudo_random_bytes(2, 4096);
    memory[..8].copy_from_slice(&[0, 0, 0, 0, 1, 0, 0, 0]);
    assert_matches_simulator(&source, ONE_WARP, &[0, 1024], &memory).unwrap();
}

#[test]
fn subword_accesses_match_simulator() {
    let source = "sreg r10, lane_index\nadd r11, r1, r10\nlb r12, 0(r11)\nlbu r13, 0(r11)\n\
                  slli r14, r10, 1\nadd r14, r1, r14\nlh r15, 0(r14)\nlhu r16, 0(r14)\n\
                  slli r17, r10, 4\nadd r17, r2, r17\nsw r12, 0(r17)\nsw r13, 4(r17)\n\
                  sw r15, 8(r17)\nsh r16, 12(r17)\nadd r18, r3, r10\nsb r15, 0(r18)\nexit";
    let memory = pseudo_random_bytes(3, 1024);
    assert_matches_simulator(source, ONE_WARP, &[0, 256, 768], &memory).unwrap();
}

#[test]
fn misaligned_word_load_faults() {
    let source = "sreg r10, lane_index\nslli r10, r10, 2\naddi r10, r10, 2\nlw r11, 0(r10)\nexit";
    let fault = assert_matches_simulator(source, ONE_WARP, &[], &[0; 256]).unwrap_err();
    assert!(matches!(fault, Fault::MisalignedAccess { address: 2, .. }));
}

#[test]
fn store_past_memory_end_faults() {
    let source = "sreg r10, lane_index\nslli r10, r10, 4\nsw r10, 0(r10)\nexit";
    let fault = assert_matches_simulator(source, ONE_WARP, &[], &[0; 130]).unwrap_err();
    assert!(matches!(fault, Fault::AddressOutOfRange { address: 128, .. }));
}
