mod support;

use support::{ONE_WARP, assert_matches_simulator};
use tritium_assembler::assemble;
use tritium_isa::{Fault, LaunchConfig};
use tritium_rtl::{MemoryTiming, RtlDevice};

fn pseudo_random_bytes(seed: u32, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
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
fn backward_jumps_repeat_loops() {
    let source = "li r10, 0\nli r11, 5\nloop:\nbeq r10, r11, done\naddi r10, r10, 1\nj loop\n\
                  done:\nsreg r12, lane_index\nslli r12, r12, 2\nsw r10, 0(r12)\nexit";
    assert_matches_simulator(source, ONE_WARP, &[], &[0; 64]).unwrap();
}

#[test]
fn calls_return_through_link_register() {
    let source = "li r10, 3\njal r31, triple\njal r31, triple\nsreg r12, lane_index\n\
                  slli r12, r12, 2\nsw r10, 0(r12)\nexit\n\
                  triple:\nadd r11, r10, r10\nadd r10, r10, r11\njalr r0, r31, 0";
    assert_matches_simulator(source, ONE_WARP, &[], &[0; 64]).unwrap();
}

#[test]
fn signed_branches_compare_negative_values() {
    let source = "li r10, -1\nli r11, 1\nli r12, 0\nblt r10, r11, less\naddi r12, r12, 1\n\
                  less:\nbge r10, r11, greater\naddi r12, r12, 2\ngreater:\n\
                  bne r10, r11, different\naddi r12, r12, 4\ndifferent:\n\
                  sreg r13, lane_index\nslli r13, r13, 2\nsw r12, 0(r13)\nexit";
    assert_matches_simulator(source, ONE_WARP, &[], &[0; 64]).unwrap();
}

#[test]
fn barrier_synchronizes_four_warps() {
    let source = "sreg r10, thread_index\nslli r11, r10, 2\naddi r12, r10, 100\nsw r12, 0(r11)\n\
                  bar\naddi r13, r10, 16\nandi r13, r13, 63\nslli r13, r13, 2\nlw r14, 0(r13)\n\
                  add r14, r14, r12\nsw r14, 256(r11)\nexit";
    let config = LaunchConfig { grid_blocks: 1, block_threads: 64, shared_bytes: 0 };
    assert_matches_simulator(source, config, &[], &[0; 512]).unwrap();
}

#[test]
fn shared_memory_reduction_uses_barrier() {
    let source = "sreg r10, thread_index\nsreg r11, block_index\nsreg r12, block_threads\n\
                  mul16 r13, r11, r12\nadd r13, r13, r10\nslli r13, r13, 2\nadd r13, r1, r13\n\
                  lw r14, 0(r13)\nslli r15, r10, 2\nsws r14, 0(r15)\nbar\nli r16, 32\n\
                  loop:\nlws r17, 0(r15)\nadd r18, r10, r16\nandi r18, r18, 63\nslli r18, r18, 2\n\
                  lws r19, 0(r18)\nbar\nadd r19, r17, r19\nslt r20, r10, r16\n\
                  select r20, r19, r17\nsws r20, 0(r15)\nbar\nsrli r16, r16, 1\n\
                  bne r16, r0, loop\nlws r21, 0(r0)\nslli r22, r11, 2\nadd r22, r2, r22\n\
                  sw r21, 0(r22)\nexit";
    let config = LaunchConfig { grid_blocks: 4, block_threads: 64, shared_bytes: 256 };
    let memory = pseudo_random_bytes(4, 2048);
    assert_matches_simulator(source, config, &[0, 1024], &memory).unwrap();
}

#[test]
fn shared_subword_accesses_match_simulator() {
    let source = "sreg r10, lane_index\nslli r11, r10, 2\nadd r12, r1, r11\nlw r13, 0(r12)\n\
                  sws r13, 0(r11)\nadd r14, r11, r10\nshs r10, 2(r11)\nsbs r10, 64(r10)\n\
                  lbs r15, 1(r11)\nlbus r16, 3(r11)\nlhs r17, 0(r11)\nlhus r18, 2(r11)\n\
                  lws r19, 64(r0)\nslli r20, r10, 5\nadd r20, r2, r20\nsw r15, 0(r20)\n\
                  sw r16, 4(r20)\nsw r17, 8(r20)\nsw r18, 12(r20)\nsw r19, 16(r20)\nexit";
    let config = LaunchConfig { shared_bytes: 128, ..ONE_WARP };
    let memory = pseudo_random_bytes(5, 1024);
    assert_matches_simulator(source, config, &[0, 256], &memory).unwrap();
}

#[test]
fn shared_memory_starts_zeroed() {
    let source = "sreg r10, block_index\nsreg r11, lane_index\nslli r11, r11, 2\n\
                  lws r12, 0(r11)\nslli r13, r10, 6\nadd r13, r13, r11\nsw r12, 0(r13)\n\
                  addi r12, r10, 7\nsws r12, 0(r11)\nexit";
    let config = LaunchConfig { grid_blocks: 3, block_threads: 16, shared_bytes: 4096 };
    assert_matches_simulator(source, config, &[], &[0xAA; 256]).unwrap();
}

#[test]
fn ternary_dot_matches_simulator() {
    let source = "sreg r10, lane_index\nslli r11, r10, 3\nadd r12, r2, r11\nlw r13, 0(r12)\n\
                  lw r14, 4(r12)\nsws r13, 0(r11)\nsws r14, 4(r11)\nbar\nslli r15, r10, 2\n\
                  add r16, r1, r15\nmov r17, r16\ntdot r17, r16, r15\naddi r18, r15, 44\n\
                  tdot r17, r16, r18\nadd r19, r3, r15\nsw r17, 0(r19)\nexit";
    let config = LaunchConfig { shared_bytes: 128, ..ONE_WARP };
    let memory = pseudo_random_bytes(6, 1024);
    assert_matches_simulator(source, config, &[0, 256, 512], &memory).unwrap();
}

#[test]
fn ternary_dot_matches_unaligned_activations() {
    let source = "sreg r10, lane_index\nslli r11, r10, 3\nadd r12, r2, r11\nlw r13, 0(r12)\n\
                  lw r14, 4(r12)\nsws r13, 0(r11)\nsws r14, 4(r11)\nbar\nslli r15, r10, 2\n\
                  add r16, r1, r15\nmov r17, r16\ntdot r17, r16, r10\naddi r18, r10, 90\n\
                  tdot r17, r16, r18\nadd r19, r3, r15\nsw r17, 0(r19)\nexit";
    let config = LaunchConfig { shared_bytes: 128, ..ONE_WARP };
    let memory = pseudo_random_bytes(6, 1024);
    assert_matches_simulator(source, config, &[0, 256, 512], &memory).unwrap();
}

#[test]
fn int8_dot_matches_unaligned_activations() {
    let source = "sreg r10, lane_index\nslli r11, r10, 3\nadd r12, r2, r11\nlw r13, 0(r12)\n\
                  lw r14, 4(r12)\nsws r13, 0(r11)\nsws r14, 4(r11)\nbar\nslli r15, r10, 2\n\
                  add r16, r1, r15\nmov r17, r16\ndot8 r17, r16, r10\naddi r18, r10, 101\n\
                  dot8 r17, r16, r18\nadd r19, r3, r15\nsw r17, 0(r19)\nexit";
    let config = LaunchConfig { shared_bytes: 128, ..ONE_WARP };
    let memory = pseudo_random_bytes(9, 1024);
    assert_matches_simulator(source, config, &[0, 256, 512], &memory).unwrap();
}

#[test]
fn fixed_point_operations_match_simulator() {
    let source = "sreg r10, lane_index\nslli r11, r10, 3\nadd r11, r1, r11\nlw r12, 0(r11)\n\
                  lw r13, 4(r11)\nslli r20, r10, 4\nadd r20, r2, r20\nscale r14, r12, r13\n\
                  sw r14, 0(r20)\nexp2 r14, r12, r0\nsw r14, 4(r20)\nrsqrt r14, r12, r0\n\
                  sw r14, 8(r20)\nsrai r15, r12, 12\nexp2 r14, r15, r0\nsw r14, 12(r20)\nexit";
    let mut memory = pseudo_random_bytes(7, 512);
    let extremes: [u32; 12] = [0, 0, 1, 1, 0x8000_0000, 0x8000_0000, 0x7FFF_FFFF, 0x7FFF_FFFF]
        .into_iter()
        .chain([0xFFFF_0000, 65536, 0xFFFF_FFFF, 0x0001_0000])
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    for (index, value) in extremes.iter().enumerate() {
        memory[4 * index..][..4].copy_from_slice(&value.to_le_bytes());
    }
    assert_matches_simulator(source, ONE_WARP, &[0, 256], &memory).unwrap();
}

#[test]
fn two_blocks_run_in_parallel() {
    let program =
        assemble("li r10, 200\nloop:\naddi r10, r10, -1\nbne r10, r0, loop\nexit").unwrap();
    let cycles = |grid_blocks| {
        let mut device = RtlDevice::new(64);
        let config = LaunchConfig { grid_blocks, ..ONE_WARP };
        device.launch(&program, config, &[]).unwrap();
        device.cycles()
    };
    let (one, two) = (cycles(1), cycles(2));
    assert!(two < one + one / 10, "one block {one} cycles, two blocks {two}");
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

#[test]
fn memory_timing_changes_only_cycles() {
    let program = assemble(
        "sreg r10, thread_index\nslli r10, r10, 2\nlw r11, 0(r10)\naddi r11, r11, 1\n\
         sw r11, 256(r10)\nexit",
    )
    .unwrap();
    let config = LaunchConfig { grid_blocks: 1, block_threads: 64, shared_bytes: 0 };
    let memory = pseudo_random_bytes(8, 512);
    let mut cycles = Vec::new();
    for (latency, bytes_per_cycle) in [(1, 4), (12, 2), (40, 1)] {
        let mut device =
            RtlDevice::with_memory_timing(512, MemoryTiming { latency, bytes_per_cycle });
        device.global_mut().copy_from_slice(&memory);
        device.launch(&program, config, &[]).unwrap();
        let words: Vec<u32> = (0..64)
            .map(|index| u32::from_le_bytes(memory[4 * index..][..4].try_into().unwrap()))
            .collect();
        for (index, word) in words.iter().enumerate() {
            let stored = &device.global()[256 + 4 * index..][..4];
            assert_eq!(u32::from_le_bytes(stored.try_into().unwrap()), word.wrapping_add(1));
        }
        cycles.push(device.cycles());
    }
    assert!(cycles[0] < cycles[1] && cycles[1] < cycles[2], "{cycles:?}");
}
