use tritium_assembler::assemble;
use tritium_runtime::{BackendKind, LaunchConfig};

const MEMORY_BYTES: usize = 4096;
const DATA: u32 = 0;
const SCRATCH: u32 = 2048;
const OUTPUT: u32 = 3072;
const WORKING_REGISTERS: u32 = 16;
const FIRST_WORKING_REGISTER: u32 = 10;
const SHARED_BYTES: u32 = 128;

struct Random(u64);

impl Random {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn below(&mut self, bound: u32) -> u32 {
        self.next() % bound
    }

    fn register(&mut self) -> String {
        format!("r{}", FIRST_WORKING_REGISTER + self.below(WORKING_REGISTERS))
    }

    fn word(&mut self) -> u32 {
        match self.below(8) {
            0 => [0, 1, u32::MAX, 0x8000_0000, 0x7FFF_FFFF, 65536][self.below(6) as usize],
            1 => self.below(256).wrapping_sub(128),
            2 => self.below(1 << 20),
            _ => self.next(),
        }
    }
}

fn random_instruction(random: &mut Random) -> String {
    let (destination, left, right) = (random.register(), random.register(), random.register());
    match random.below(10) {
        0..=2 => {
            let operations = [
                "add", "sub", "and", "or", "xor", "sll", "srl", "sra", "slt", "sltu", "min", "max",
                "mul16", "select", "sat8", "scale", "exp2", "rsqrt",
            ];
            let operation = operations[random.below(operations.len() as u32) as usize];
            format!("{operation} {destination}, {left}, {right}")
        }
        3 => {
            let immediate = random.below(65536) as i32 - 32768;
            let operation = ["addi", "slti"][random.below(2) as usize];
            format!("{operation} {destination}, {left}, {immediate}")
        }
        4 => {
            let operation = ["andi", "ori", "xori"][random.below(3) as usize];
            format!("{operation} {destination}, {left}, {}", random.below(65536))
        }
        5 => {
            let operation = ["slli", "srli", "srai"][random.below(3) as usize];
            format!("{operation} {destination}, {left}, {}", random.below(32))
        }
        6 => {
            let operation = ["tdot", "dot8"][random.below(2) as usize];
            format!(
                "andi r26, {left}, 252\nadd r26, r26, r1\nandi r27, {right}, {}\n\
                 {operation} {destination}, r26, r27",
                SHARED_BYTES - 20 - 1
            )
        }
        7 => {
            let (operation, mask) =
                [("lw", 1020), ("lh", 1022), ("lhu", 1022), ("lb", 1023), ("lbu", 1023)]
                    [random.below(5) as usize];
            format!("andi r26, {left}, {mask}\nadd r26, r26, r3\n{operation} {destination}, 0(r26)")
        }
        8 => {
            let (operation, mask) =
                [("sw", 1020), ("sh", 1022), ("sb", 1023)][random.below(3) as usize];
            format!("andi r26, {left}, {mask}\nadd r26, r26, r3\n{operation} {right}, 0(r26)")
        }
        _ => {
            let (operation, mask) =
                [("lws", 124), ("lhs", 126), ("lbus", 127), ("sws", 124), ("sbs", 127)]
                    [random.below(5) as usize];
            let value = if operation.starts_with('l') { &destination } else { &right };
            format!("andi r26, {left}, {mask}\n{operation} {value}, 0(r26)")
        }
    }
}

fn random_program(random: &mut Random, length: usize) -> String {
    let mut source = String::from(
        "sreg r29, lane_index\nslli r30, r29, 6\nadd r30, r1, r30\nslli r31, r29, 3\n\
         add r27, r1, r31\nlw r26, 1024(r27)\nsws r26, 0(r31)\nlw r26, 1028(r27)\n\
         sws r26, 4(r31)\nbar\n",
    );
    for index in 0..WORKING_REGISTERS {
        source += &format!("lw r{}, {}(r30)\n", FIRST_WORKING_REGISTER + index, 4 * index);
    }
    for _ in 0..length {
        source += &random_instruction(random);
        source.push('\n');
    }
    source += "slli r28, r29, 6\nadd r28, r2, r28\n";
    for index in 0..WORKING_REGISTERS {
        source += &format!("sw r{}, {}(r28)\n", FIRST_WORKING_REGISTER + index, 4 * index);
    }
    source + "exit"
}

#[test]
fn random_programs_match_across_backends() {
    let config = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: SHARED_BYTES };
    for seed in 1..=40u64 {
        let mut random = Random(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let source = random_program(&mut random, 60);
        let program = assemble(&source).unwrap();
        let memory: Vec<u8> =
            (0..MEMORY_BYTES / 4).flat_map(|_| random.word().to_le_bytes()).collect();

        let results: Vec<_> = BackendKind::ALL
            .into_iter()
            .map(|kind| {
                let mut backend = kind.open(MEMORY_BYTES);
                backend.global_mut().copy_from_slice(&memory);
                let report = backend.launch(&program, config, &[DATA, OUTPUT, SCRATCH]);
                (report, backend.global().to_vec())
            })
            .collect();

        let (expected, actual) = (&results[0], &results[1]);
        assert_eq!(actual.0, expected.0, "seed {seed}\n{source}");
        if let Some(address) = actual.1.iter().zip(&expected.1).position(|(a, b)| a != b) {
            panic!("seed {seed}: memory differs at {address:#x}\n{source}");
        }
    }
}
