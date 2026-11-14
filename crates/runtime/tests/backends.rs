use tritium_assembler::assemble;
use tritium_runtime::{BackendKind, Fault, LaunchConfig};

const ONE_WARP: LaunchConfig = LaunchConfig { grid_blocks: 1, block_threads: 16, shared_bytes: 0 };

#[test]
fn backends_parse_from_names() {
    for kind in BackendKind::ALL {
        assert_eq!(kind.name().parse::<BackendKind>(), Ok(kind));
    }
    assert!("gpu".parse::<BackendKind>().is_err());
}

#[test]
fn every_backend_runs_lane_stores() {
    let program = assemble("sreg r10, lane_index\nslli r11, r10, 2\nsw r10, 0(r11)\nexit").unwrap();
    for kind in BackendKind::ALL {
        let mut backend = kind.open(64);
        let report = backend.launch(&program, ONE_WARP, &[]).unwrap();
        assert_eq!(report.warp_instructions, 4, "{kind}");
        assert_eq!(backend.global()[60..64], [15, 0, 0, 0], "{kind}");
        assert_eq!(backend.cycles().is_some(), kind == BackendKind::Rtl, "{kind}");
    }
}

#[test]
fn every_backend_honors_step_limit() {
    let program = assemble("loop:\nj loop").unwrap();
    for kind in BackendKind::ALL {
        let mut backend = kind.open(64);
        backend.set_step_limit(50);
        assert_eq!(
            backend.launch(&program, ONE_WARP, &[]),
            Err(Fault::StepLimitExceeded),
            "{kind}"
        );
    }
}
