//! IRIS: a continually self-training RL agent built on meganeura.
//!
//! The agent starts from a cold network, trains perpetually from experience,
//! and derives reward from three frozen primitives: surprise, novelty, and
//! homeostatic balance.

pub mod agent;
pub mod buffer;
pub mod credit;
pub mod encoder;
pub mod env;
pub mod policy;
pub mod reward;
pub mod world_model;

pub use agent::{Agent, AgentConfig};
pub use buffer::ExperienceBuffer;
pub use env::{Action, Environment, HomeostaticProvider, Observation};
pub use reward::RewardCircuit;

/// Controls whether meganeura's e-graph optimizer is used.
#[derive(Clone, Copy, Debug, Default)]
pub enum OptLevel {
    /// Skip full-graph e-graph optimization (forward graph is still optimized).
    None,
    /// Default meganeura optimization pipeline.
    #[default]
    Full,
}
