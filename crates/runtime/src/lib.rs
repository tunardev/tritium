use std::fmt;
use std::str::FromStr;

pub use tritium_isa::{Fault, LaunchConfig, LaunchReport};
use tritium_rtl::RtlDevice;
use tritium_simulator::Device;

pub trait Backend {
    fn global(&self) -> &[u8];
    fn global_mut(&mut self) -> &mut [u8];
    fn set_step_limit(&mut self, steps: u64);
    fn cycles(&self) -> Option<u64>;
    fn launch(
        &mut self,
        program: &[u32],
        config: LaunchConfig,
        arguments: &[u32],
    ) -> Result<LaunchReport, Fault>;
}

impl Backend for Device {
    fn global(&self) -> &[u8] {
        Device::global(self)
    }

    fn global_mut(&mut self) -> &mut [u8] {
        Device::global_mut(self)
    }

    fn set_step_limit(&mut self, steps: u64) {
        Device::set_step_limit(self, steps);
    }

    fn cycles(&self) -> Option<u64> {
        None
    }

    fn launch(
        &mut self,
        program: &[u32],
        config: LaunchConfig,
        arguments: &[u32],
    ) -> Result<LaunchReport, Fault> {
        Device::launch(self, program, config, arguments)
    }
}

impl Backend for RtlDevice {
    fn global(&self) -> &[u8] {
        RtlDevice::global(self)
    }

    fn global_mut(&mut self) -> &mut [u8] {
        RtlDevice::global_mut(self)
    }

    fn set_step_limit(&mut self, steps: u64) {
        RtlDevice::set_step_limit(self, steps);
    }

    fn cycles(&self) -> Option<u64> {
        Some(RtlDevice::cycles(self))
    }

    fn launch(
        &mut self,
        program: &[u32],
        config: LaunchConfig,
        arguments: &[u32],
    ) -> Result<LaunchReport, Fault> {
        RtlDevice::launch(self, program, config, arguments)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Simulator,
    Rtl,
}

impl BackendKind {
    pub const ALL: [BackendKind; 2] = [BackendKind::Simulator, BackendKind::Rtl];

    pub fn open(self, global_bytes: usize) -> Box<dyn Backend> {
        match self {
            BackendKind::Simulator => Box::new(Device::new(global_bytes)),
            BackendKind::Rtl => Box::new(RtlDevice::new(global_bytes)),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            BackendKind::Simulator => "simulator",
            BackendKind::Rtl => "rtl",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

impl FromStr for BackendKind {
    type Err = String;

    fn from_str(text: &str) -> Result<BackendKind, String> {
        BackendKind::ALL
            .into_iter()
            .find(|kind| kind.name() == text)
            .ok_or_else(|| format!("unknown backend {text}; expected simulator or rtl"))
    }
}
