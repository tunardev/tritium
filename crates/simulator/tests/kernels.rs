use tritium_assembler::assemble;
use tritium_isa::{TRITS_PER_WORD, pack_trits};
use tritium_simulator::{Device, LaunchConfig};

fn read_words(device: &Device, address: usize, count: usize) -> Vec<u32> {
    device.global()[address..address + 4 * count]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn write_words(device: &mut Device, address: usize, words: &[u32]) {
    for (index, word) in words.iter().enumerate() {
        device.global_mut()[address + 4 * index..][..4].copy_from_slice(&word.to_le_bytes());
    }
}

fn pseudo_random(seed: u32, count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

#[test]
fn vector_add_covers_every_thread() {
    let program = assemble(
        "sreg r10, block_index\nsreg r11, block_threads\nmul8 r10, r10, r11\n\
         sreg r12, thread_index\nadd r10, r10, r12\nslli r10, r10, 2\n\
         add r13, r1, r10\nlw r14, 0(r13)\nadd r13, r2, r10\nlw r15, 0(r13)\n\
         add r14, r14, r15\nadd r13, r3, r10\nsw r14, 0(r13)\nexit",
    )
    .unwrap();
    let mut device = Device::new(4096);
    let (first, second) = (pseudo_random(1, 256), pseudo_random(2, 256));
    write_words(&mut device, 0, &first);
    write_words(&mut device, 1024, &second);

    let config = LaunchConfig { grid_blocks: 16, block_threads: 16, shared_bytes: 0 };
    device.launch(&program, config, &[0, 1024, 2048]).unwrap();

    let expected: Vec<u32> =
        first.iter().zip(&second).map(|(left, right)| left.wrapping_add(*right)).collect();
    assert_eq!(read_words(&device, 2048, 256), expected);
}

#[test]
fn shared_memory_reduction_uses_barrier() {
    let program = assemble(
        "sreg r10, thread_index\nsreg r11, block_index\nsreg r12, block_threads\n\
         mul8 r13, r11, r12\nadd r13, r13, r10\nslli r13, r13, 2\nadd r13, r1, r13\n\
         lw r14, 0(r13)\nslli r15, r10, 2\nsws r14, 0(r15)\nbar\nli r16, 32\n\
         loop:\nlws r17, 0(r15)\nadd r18, r10, r16\nandi r18, r18, 63\nslli r18, r18, 2\n\
         lws r19, 0(r18)\nbar\nadd r19, r17, r19\nslt r20, r10, r16\nselect r20, r19, r17\n\
         sws r20, 0(r15)\nbar\nsrli r16, r16, 1\nbne r16, r0, loop\n\
         lws r21, 0(r0)\nslli r22, r11, 2\nadd r22, r2, r22\nsw r21, 0(r22)\nexit",
    )
    .unwrap();
    let mut device = Device::new(4096);
    let values: Vec<u32> = pseudo_random(3, 256).iter().map(|value| value % 1000).collect();
    write_words(&mut device, 0, &values);

    let config = LaunchConfig { grid_blocks: 4, block_threads: 64, shared_bytes: 256 };
    device.launch(&program, config, &[0, 2048]).unwrap();

    let expected: Vec<u32> = values.chunks(64).map(|block| block.iter().sum()).collect();
    assert_eq!(read_words(&device, 2048, 4), expected);
}

#[test]
fn ternary_dot_matches_reference_sum() {
    let words_per_row = 3;
    let row_bytes = 4 * words_per_row;
    let length = words_per_row * TRITS_PER_WORD;
    let randoms = pseudo_random(4, 16 * length + length);
    let trits: Vec<Vec<i8>> = (0..16)
        .map(|row| (0..length).map(|index| (randoms[row * length + index] % 3) as i8 - 1).collect())
        .collect();
    let activations: Vec<i8> =
        randoms[16 * length..].iter().map(|value| (value % 255) as i8).collect();

    let mut device = Device::new(4096);
    for (row, row_trits) in trits.iter().enumerate() {
        device.global_mut()[row * row_bytes..][..row_bytes].copy_from_slice(&pack_trits(row_trits));
    }
    for (index, &activation) in activations.iter().enumerate() {
        device.global_mut()[1024 + index] = activation as u8;
    }
    let mut source = format!(
        "sreg r10, lane_index\nmov r12, r10\nli r11, {copy_passes}\ncopy:\nadd r13, r2, r12\n\
         lb r14, 0(r13)\nsbs r14, 0(r12)\naddi r12, r12, 16\naddi r11, r11, -1\n\
         bne r11, r0, copy\nli r19, {row_bytes}\nmul8 r20, r10, r19\nadd r20, r1, r20\n\
         li r21, 0\n",
        copy_passes = length.div_ceil(16)
    );
    for word in 0..words_per_row {
        source += &format!(
            "addi r22, r20, {}\nli r23, {}\ntdot r21, r22, r23\n",
            4 * word,
            word * TRITS_PER_WORD
        );
    }
    source += "slli r24, r10, 2\nadd r24, r3, r24\nsw r21, 0(r24)\nexit";
    let program = assemble(&source).unwrap();

    let config = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 256 };
    device.launch(&program, config, &[0, 1024, 2048]).unwrap();

    let expected: Vec<u32> = trits
        .iter()
        .map(|row| {
            row.iter()
                .zip(&activations)
                .map(|(&trit, &activation)| i32::from(trit) * i32::from(activation))
                .sum::<i32>() as u32
        })
        .collect();
    assert_eq!(read_words(&device, 2048, 16), expected);
}

#[test]
fn special_registers_report_launch_shape() {
    let program = assemble(
        "sreg r10, block_index\nsreg r11, block_threads\nmul8 r12, r10, r11\n\
         sreg r13, thread_index\nadd r12, r12, r13\nli r14, 24\nmul8 r12, r12, r14\n\
         add r12, r1, r12\nsw r13, 0(r12)\nsw r10, 4(r12)\nsw r11, 8(r12)\n\
         sreg r15, grid_blocks\nsw r15, 12(r12)\nsreg r15, lane_index\nsw r15, 16(r12)\n\
         sreg r15, warp_index\nsw r15, 20(r12)\nexit",
    )
    .unwrap();
    let mut device = Device::new(4096);

    let config = LaunchConfig { grid_blocks: 2, block_threads: 48, shared_bytes: 0 };
    device.launch(&program, config, &[0]).unwrap();

    for block in 0..2u32 {
        for thread in 0..48u32 {
            let global_thread = (block * 48 + thread) as usize;
            assert_eq!(
                read_words(&device, global_thread * 24, 6),
                vec![thread, block, 48, 2, thread % 16, thread / 16]
            );
        }
    }
}

#[test]
fn select_picks_lanes_without_branching() {
    let program = assemble(
        "sreg r10, lane_index\nslli r10, r10, 2\nadd r11, r1, r10\nlw r12, 0(r11)\n\
         add r11, r2, r10\nlw r13, 0(r11)\nslt r14, r12, r13\nselect r14, r13, r12\n\
         add r11, r3, r10\nsw r14, 0(r11)\nexit",
    )
    .unwrap();
    let mut device = Device::new(1024);
    let first: Vec<u32> = pseudo_random(5, 16).iter().map(|value| value % 100).collect();
    let second: Vec<u32> = pseudo_random(6, 16).iter().map(|value| value % 100).collect();
    write_words(&mut device, 0, &first);
    write_words(&mut device, 256, &second);

    let config = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };
    device.launch(&program, config, &[0, 256, 512]).unwrap();

    let expected: Vec<u32> =
        first.iter().zip(&second).map(|(left, right)| *left.max(right)).collect();
    assert_eq!(read_words(&device, 512, 16), expected);
}

#[test]
fn backward_jumps_repeat_loops() {
    let program = assemble(
        "li r10, 0\nli r11, 5\nloop:\nbeq r10, r11, done\naddi r10, r10, 1\nj loop\n\
         done:\nsreg r12, lane_index\nslli r12, r12, 2\nadd r12, r1, r12\nsw r10, 0(r12)\nexit",
    )
    .unwrap();
    let mut device = Device::new(256);

    let config = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };
    device.launch(&program, config, &[0]).unwrap();

    assert_eq!(read_words(&device, 0, 16), vec![5; 16]);
}
