#![allow(
    clippy::match_like_matches_macro,
    clippy::redundant_pattern_matching,
    clippy::needless_lifetimes,
    clippy::new_without_default,
    clippy::single_match,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::collapsible_match
)]
#![warn(
    trivial_numeric_casts,
    unused_extern_crates,
    clippy::pattern_type_mismatch
)]

//! kindle: a continually self-training RL agent built on meganeura.
//!
//! The agent starts from a cold network, trains perpetually from experience,
//! and derives reward from four frozen primitives: surprise, novelty,
//! homeostatic balance, and order. To kindle is to start a fire from
//! nothing — this crate is the ignition.

pub mod adapter;
pub mod agent;
pub mod buffer;
pub mod credit;
pub mod encoder;
pub mod env;
pub mod option;
pub mod outcome;
pub mod policy;
pub mod reward;
pub mod world_model;

pub use adapter::{EnvAdapter, GenericAdapter, MAX_ACTION_DIM, OBS_TOKEN_DIM};
pub use agent::{Agent, AgentConfig};
pub use buffer::ExperienceBuffer;
pub use env::{Action, ActionKind, Environment, HomeostaticProvider, Observation};
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
