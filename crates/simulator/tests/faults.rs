use tritium_assembler::assemble;
use tritium_simulator::{Device, Fault, LaunchConfig};

const ONE_WARP: LaunchConfig = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };

fn run(source: &str, config: LaunchConfig) -> Result<(), Fault> {
    let mut device = Device::new(256);
    device.set_step_limit(1000);
    device.launch(&assemble(source).unwrap(), config, &[]).map(|_| ())
}

#[test]
fn divergent_warp_branch_faults() {
    let fault = run("sreg r1, lane_index\nbeq r1, r0, done\ndone:\nexit", ONE_WARP).unwrap_err();
    assert!(matches!(fault, Fault::DivergentBranch { program_counter: 1, .. }), "{fault:?}");
}

#[test]
fn misaligned_word_load_faults() {
    let fault = run("li r1, 2\nlw r2, 0(r1)\nexit", ONE_WARP).unwrap_err();
    assert!(matches!(fault, Fault::MisalignedAccess { address: 2, .. }), "{fault:?}");
}

#[test]
fn out_of_bounds_access_faults() {
    let fault = run("li r1, 256\nsw r0, 0(r1)\nexit", ONE_WARP).unwrap_err();
    assert!(matches!(fault, Fault::AddressOutOfRange { address: 256, .. }), "{fault:?}");
}

#[test]
fn shared_access_past_allocation_faults() {
    let config = LaunchConfig { shared_bytes: 64, ..ONE_WARP };
    let fault = run("li r1, 64\nlws r2, 0(r1)\nexit", config).unwrap_err();
    assert!(matches!(fault, Fault::AddressOutOfRange { address: 64, .. }), "{fault:?}");
}

#[test]
fn running_past_program_end_faults() {
    let fault = run("add r1, r1, r1", ONE_WARP).unwrap_err();
    assert!(
        matches!(fault, Fault::ProgramCounterOutOfRange { program_counter: 1, .. }),
        "{fault:?}"
    );
}

#[test]
fn endless_loop_stops_at_limit() {
    let fault = run("loop:\nj loop", ONE_WARP).unwrap_err();
    assert_eq!(fault, Fault::StepLimitExceeded);
}

#[test]
fn barrier_without_partners_deadlocks() {
    let config = LaunchConfig { block_threads: 32, ..ONE_WARP };
    let fault = run("sreg r1, warp_index\nbeq r1, r0, done\nbar\ndone:\nexit", config).unwrap_err();
    assert!(matches!(fault, Fault::BarrierDeadlock { block: 0 }), "{fault:?}");
}

#[test]
fn invalid_launch_shapes_are_rejected() {
    let mut device = Device::new(256);
    let program = assemble("exit").unwrap();
    for config in [
        LaunchConfig { block_threads: 20, ..ONE_WARP },
        LaunchConfig { block_threads: 80, ..ONE_WARP },
        LaunchConfig { grid_blocks: 0, ..ONE_WARP },
        LaunchConfig { shared_bytes: 16385, ..ONE_WARP },
    ] {
        assert!(matches!(device.launch(&program, config, &[]), Err(Fault::InvalidLaunch(_))));
    }
    assert!(matches!(device.launch(&program, ONE_WARP, &[0; 9]), Err(Fault::InvalidLaunch(_))));
}

#[test]
fn undecodable_instruction_word_faults() {
    let mut device = Device::new(256);
    let fault = device.launch(&[0xFFFF_FFFF], ONE_WARP, &[]).unwrap_err();
    assert!(matches!(fault, Fault::InvalidInstruction { word: 0xFFFF_FFFF, .. }), "{fault:?}");
}

#[test]
fn barrier_after_warp_exit_deadlocks() {
    let config = LaunchConfig { block_threads: 48, ..ONE_WARP };
    let fault = run("sreg r1, warp_index\nbeq r1, r0, done\nbar\ndone:\nexit", config).unwrap_err();
    assert!(matches!(fault, Fault::BarrierDeadlock { block: 0 }), "{fault:?}");
}
