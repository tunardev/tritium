mod support;

use support::{ONE_WARP, assert_matches_simulator};
use tritium_isa::{Fault, LaunchConfig};
use tritium_rtl::RtlDevice;

fn run(source: &str, config: LaunchConfig) -> Fault {
    assert_matches_simulator(source, config, &[], &[0; 256]).unwrap_err()
}

#[test]
fn divergent_warp_branch_faults() {
    let fault = run("sreg r1, lane_index\nbeq r1, r0, done\ndone:\nexit", ONE_WARP);
    assert!(matches!(fault, Fault::DivergentBranch { program_counter: 1, .. }), "{fault:?}");
}

#[test]
fn divergent_jump_register_faults() {
    let fault = run("sreg r1, lane_index\njalr r2, r1, 4\nexit", ONE_WARP);
    assert!(matches!(fault, Fault::DivergentBranch { program_counter: 1, .. }), "{fault:?}");
}

#[test]
fn running_past_program_end_faults() {
    let fault = run("add r1, r1, r1", ONE_WARP);
    assert!(matches!(fault, Fault::ProgramCounterOutOfRange { program_counter: 1, .. }));
}

#[test]
fn jump_far_past_end_faults() {
    let fault = run("li r1, -8\njalr r0, r1, 0\nexit", ONE_WARP);
    assert!(matches!(fault, Fault::ProgramCounterOutOfRange { program_counter: 0xFFFF_FFF8, .. }));
}

#[test]
fn endless_loop_stops_at_limit() {
    assert_eq!(run("loop:\nj loop", ONE_WARP), Fault::StepLimitExceeded);
}

#[test]
fn barrier_without_partners_deadlocks() {
    let config = LaunchConfig { block_threads: 32, ..ONE_WARP };
    let fault = run("sreg r1, warp_index\nbeq r1, r0, done\nbar\ndone:\nexit", config);
    assert_eq!(fault, Fault::BarrierDeadlock { block: 0 });
}

#[test]
fn barrier_after_warp_exit_deadlocks() {
    let config = LaunchConfig { grid_blocks: 2, block_threads: 48, shared_bytes: 0 };
    let fault = run("sreg r1, warp_index\nbne r1, r0, wait\nexit\nwait:\nbar\nexit", config);
    assert_eq!(fault, Fault::BarrierDeadlock { block: 0 });
}

#[test]
fn shared_access_past_allocation_faults() {
    let config = LaunchConfig { shared_bytes: 64, ..ONE_WARP };
    let fault =
        run("sreg r1, lane_index\nslli r1, r1, 2\naddi r1, r1, 4\nlws r2, 0(r1)\nexit", config);
    assert!(matches!(fault, Fault::AddressOutOfRange { address: 64, .. }), "{fault:?}");
}

#[test]
fn misaligned_dot_weights_fault() {
    let config = LaunchConfig { shared_bytes: 64, ..ONE_WARP };
    let fault = run("li r1, 6\ntdot r2, r1, r0\nexit", config);
    assert!(matches!(fault, Fault::MisalignedAccess { address: 6, .. }), "{fault:?}");
}

#[test]
fn dot_activations_past_allocation_fault() {
    let config = LaunchConfig { shared_bytes: 64, ..ONE_WARP };
    let fault = run("li r1, 45\ntdot r2, r0, r1\nexit", config);
    assert!(matches!(fault, Fault::AddressOutOfRange { address: 45, .. }), "{fault:?}");
}

#[test]
fn unknown_special_register_faults() {
    let word = (0x0A << 26) | (1 << 21) | 6;
    let mut device = RtlDevice::new(64);
    let fault = device.launch(&[word], ONE_WARP, &[]).unwrap_err();
    assert_eq!(fault, Fault::InvalidInstruction { block: 0, warp: 0, program_counter: 0, word });
}
