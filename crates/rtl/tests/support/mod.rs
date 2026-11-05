use tritium_assembler::assemble;
use tritium_isa::{Fault, LaunchConfig, LaunchReport};
use tritium_rtl::RtlDevice;
use tritium_simulator::Device;

pub const ONE_WARP: LaunchConfig =
    LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };

const STEP_LIMIT: u64 = 20_000;

pub fn assert_matches_simulator(
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
    simulator.set_step_limit(STEP_LIMIT);
    rtl.set_step_limit(STEP_LIMIT);

    let expected = simulator.launch(&program, config, arguments);
    let actual = rtl.launch(&program, config, arguments);

    assert_eq!(actual, expected);
    if expected.is_ok() {
        let difference = rtl.global().iter().zip(simulator.global()).position(|(a, b)| a != b);
        if let Some(address) = difference {
            let word = address & !3;
            panic!(
                "global memory differs at {address:#x}: rtl {:02x?}, simulator {:02x?}",
                &rtl.global()[word..(word + 4).min(memory.len())],
                &simulator.global()[word..(word + 4).min(memory.len())]
            );
        }
    }
    actual
}
