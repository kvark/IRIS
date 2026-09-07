#![warn(trivial_numeric_casts, unused_extern_crates)]

//! Dreamer learning with causal feature prediction and native visual perception.
//!
//! One shared learner can collect independent environments through batched
//! LeVJEPA inference. The single-stream DINOv3 path remains a measured control.
//! Environment types, configuration, metrics and encoder boundaries stay explicit.

pub mod dreamer;
pub mod env;
pub mod vision;

pub use dreamer::{
    ActionMode, BehaviorMetrics, DreamerAgent, DreamerConfig, DreamerCore, FrameFlags,
    InputWriteTiming, LearnReport, LearnTiming, LossScales, ModelProvenance, ModelSize,
    ReadbackTiming, Reward, VectorDreamerAgent, WorldMetrics,
};
pub use env::{Environment, RgbFrame, Transition};

/// GPU adapter metadata recorded with measured runs.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuDeviceInfo {
    pub device_name: String,
    pub driver_name: String,
    pub driver_info: String,
    pub is_software_emulated: bool,
    pub requested_device_id: Option<String>,
}

/// Initialize the GPU selected by Meganeura's explicit environment options.
///
/// In particular, `MEGANEURA_DEVICE_ID` selects a backend-reported numeric
/// device ID when more than one Vulkan adapter is present. With no override,
/// this retains Blade's default-adapter behavior.
pub fn init_gpu_context() -> Result<blade_graphics::Context, blade_graphics::NotSupportedError> {
    meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env())
}

fn gpu_device_info(information: &blade_graphics::DeviceInformation) -> GpuDeviceInfo {
    GpuDeviceInfo {
        device_name: information.device_name.clone(),
        driver_name: information.driver_name.clone(),
        driver_info: information.driver_info.clone(),
        is_software_emulated: information.is_software_emulated,
        requested_device_id: std::env::var("MEGANEURA_DEVICE_ID").ok(),
    }
}
