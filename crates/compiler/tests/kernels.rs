use tritium_compiler::{compile, compile_kernel};
use tritium_isa::{TRITS_PER_WORD, pack_trits};
use tritium_simulator::Device;

fn read_words(device: &Device, address: usize, count: usize) -> Vec<i32> {
    device.global()[address..address + 4 * count]
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn write_words(device: &mut Device, address: usize, words: &[i32]) {
    for (index, word) in words.iter().enumerate() {
        device.global_mut()[address + 4 * index..][..4].copy_from_slice(&word.to_le_bytes());
    }
}

fn values(seed: i32, count: usize) -> Vec<i32> {
    (0..count as i32).map(|index| (index * 7919 + seed * 104_729) % 2001 - 1000).collect()
}

#[test]
fn vector_add_covers_every_thread() {
    let kernel = compile_kernel(
        "kernel vector_add(first: *i32, second: *i32, output: *i32) {
             let index = block_index() * block_threads() + thread_index();
             output[index] = first[index] + second[index];
         }",
        "vector_add",
    )
    .unwrap();
    let mut device = Device::new(4096);
    let (first, second) = (values(1, 256), values(2, 256));
    write_words(&mut device, 0, &first);
    write_words(&mut device, 1024, &second);

    device.launch(&kernel.program, kernel.launch_config(4, 64), &[0, 1024, 2048]).unwrap();

    let expected: Vec<i32> = first.iter().zip(&second).map(|(a, b)| a + b).collect();
    assert_eq!(read_words(&device, 2048, 256), expected);
}

#[test]
fn shared_reduction_sums_each_block() {
    let kernel = compile_kernel(
        "kernel block_sum(input: *i32, output: *i32) {
             shared partial: [i32; 64];
             let thread = thread_index();
             partial[thread] = input[block_index() * 64 + thread];
             barrier();
             let stride = 32;
             while stride > 0 {
                 let other = partial[(thread + stride) & 63];
                 barrier();
                 partial[thread] = select(thread < stride, partial[thread] + other, partial[thread]);
                 barrier();
                 stride = stride >> 1;
             }
             if block_threads() == 64 {
                 output[block_index()] = partial[0];
             }
         }",
        "block_sum",
    )
    .unwrap();
    assert_eq!(kernel.shared_bytes, 256);
    let mut device = Device::new(4096);
    let input = values(3, 256);
    write_words(&mut device, 0, &input);

    device.launch(&kernel.program, kernel.launch_config(4, 64), &[0, 2048]).unwrap();

    let expected: Vec<i32> = input.chunks(64).map(|block| block.iter().sum()).collect();
    assert_eq!(read_words(&device, 2048, 4), expected);
}

#[test]
fn ternary_rows_match_reference_dot() {
    let kernel = compile_kernel(
        "kernel ternary_rows(weights: *i32, activations: *i8, output: *i32, words: i32) {
             shared vector: [i8; 64];
             for index in 0..4 {
                 vector[index * 16 + lane_index()] = activations[index * 16 + lane_index()];
             }
             barrier();
             let row = thread_index();
             let sum = 0;
             for word in 0..words {
                 sum = tdot(sum, weights + row * words + word, vector, word * 20);
             }
             output[row] = sum;
         }",
        "ternary_rows",
    )
    .unwrap();
    let words = 3;
    let length = words * TRITS_PER_WORD;
    let trits: Vec<Vec<i8>> = (0..16)
        .map(|row| (0..length).map(|index| ((row * 31 + index * 7) % 3) as i8 - 1).collect())
        .collect();
    let activations: Vec<i8> = (0..64).map(|index| (index * 37 % 255) as i8).collect();
    let mut device = Device::new(4096);
    for (row, row_trits) in trits.iter().enumerate() {
        device.global_mut()[row * 4 * words..][..4 * words].copy_from_slice(&pack_trits(row_trits));
    }
    for (index, &activation) in activations.iter().enumerate() {
        device.global_mut()[1024 + index] = activation as u8;
    }

    let arguments = [0, 1024, 2048, words as u32];
    device.launch(&kernel.program, kernel.launch_config(1, 16), &arguments).unwrap();

    let expected: Vec<i32> = trits
        .iter()
        .map(|row| {
            row.iter()
                .zip(&activations)
                .map(|(&trit, &value)| i32::from(trit) * i32::from(value))
                .sum()
        })
        .collect();
    assert_eq!(read_words(&device, 2048, 16), expected);
}

#[test]
fn subword_pointers_load_and_store() {
    let kernel = compile_kernel(
        "kernel widen(bytes: *i8, halves: *u16, output: *i32, narrow: *i16) {
             let index = thread_index();
             output[index] = bytes[index] * 1000 + halves[index];
             narrow[index] = -bytes[index];
         }",
        "widen",
    )
    .unwrap();
    let mut device = Device::new(1024);
    for index in 0..16 {
        device.global_mut()[index] = (index as i8 - 8) as u8;
        device.global_mut()[64 + 2 * index..][..2]
            .copy_from_slice(&(60000 - index as u16).to_le_bytes());
    }

    device.launch(&kernel.program, kernel.launch_config(1, 16), &[0, 64, 256, 512]).unwrap();

    let expected: Vec<i32> = (0..16).map(|index| (index - 8) * 1000 + 60000 - index).collect();
    assert_eq!(read_words(&device, 256, 16), expected);
    let narrow = i16::from_le_bytes(device.global()[512..514].try_into().unwrap());
    assert_eq!(narrow, 8);
}

#[test]
fn control_flow_and_operators_compute() {
    let kernel = compile_kernel(
        "kernel arithmetic(output: *i32) {
             let lane = lane_index();
             let total = 0;
             for step in 0..10 {
                 if step == 3 || step >= 8 {
                     total += 100;
                 } else if step < 2 && !(step != 1) {
                     total -= 7;
                 } else {
                     total = total + step * 3 - (step << 1) ^ 5;
                 }
             }
             output[lane] = select(lane & 1, sat8(total + lane), max(min(lane, 3), -total));
         }",
        "arithmetic",
    )
    .unwrap();
    let mut device = Device::new(256);
    device.launch(&kernel.program, kernel.launch_config(1, 16), &[0]).unwrap();

    let mut total = 0i32;
    for step in 0..10 {
        if step == 3 || step >= 8 {
            total += 100;
        } else if step < 2 && step == 1 {
            total -= 7;
        } else {
            total = (total + step * 3 - (step << 1)) ^ 5;
        }
    }
    let expected: Vec<i32> =
        (0..16)
            .map(|lane| {
                if lane & 1 != 0 {
                    (total + lane).clamp(-127, 127)
                } else {
                    lane.min(3).max(-total)
                }
            })
            .collect();
    assert_eq!(read_words(&device, 0, 16), expected);
}

#[test]
fn fixed_point_builtins_match_isa() {
    let kernel = compile_kernel(
        "kernel fixed(input: *i32, output: *i32) {
             let index = thread_index();
             let value = input[index];
             output[index * 3] = exp2(-value);
             output[index * 3 + 1] = rsqrt(value);
             output[index * 3 + 2] = scale(value, 98304);
         }",
        "fixed",
    )
    .unwrap();
    let mut device = Device::new(1024);
    let inputs: Vec<i32> = (0..16).map(|index| index * 40_000 + 1).collect();
    write_words(&mut device, 0, &inputs);

    device.launch(&kernel.program, kernel.launch_config(1, 16), &[0, 256]).unwrap();

    let expected: Vec<i32> = inputs
        .iter()
        .flat_map(|&value| {
            [tritium_isa::exp2(-value), tritium_isa::rsqrt(value), tritium_isa::scale(value, 98304)]
        })
        .collect();
    assert_eq!(read_words(&device, 256, 48), expected);
}

#[test]
fn unknown_names_report_their_position() {
    let error = compile("kernel broken(output: *i32) {\n    output[0] = missing;\n}").unwrap_err();
    assert_eq!(error.to_string(), "2:17: unknown name missing");
}

#[test]
fn duplicate_kernels_are_rejected() {
    let error = compile("kernel twice() { }\nkernel twice() { }").unwrap_err();
    assert_eq!(error.line, 2);
}

#[test]
fn pointer_arithmetic_scales_by_element() {
    let kernel = compile_kernel(
        "kernel offsets(words: *i32, halves: *i16, output: *i32) {
             let lane = lane_index();
             let word = words + 2;
             let half = halves + 3;
             output[lane] = word[lane] + half[lane];
         }",
        "offsets",
    )
    .unwrap();
    let mut device = Device::new(512);
    let words: Vec<i32> = (0..32).collect();
    write_words(&mut device, 0, &words);
    for index in 0..32u16 {
        device.global_mut()[128 + 2 * index as usize..][..2]
            .copy_from_slice(&(index * 10).to_le_bytes());
    }

    device.launch(&kernel.program, kernel.launch_config(1, 16), &[0, 128, 256]).unwrap();

    let expected: Vec<i32> = (0..16).map(|lane| (lane + 2) + (lane + 3) * 10).collect();
    assert_eq!(read_words(&device, 256, 16), expected);
}
