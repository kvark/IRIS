//! DreamerV3 learner with frozen visual perception and causal feature prediction.
//!
//! The implementation follows the pinned upstream DreamerV3 algorithm for
//! replay alignment, categorical RSSM state, KL balancing, distributional
//! reward/value heads, and imagined actor-critic learning. The deliberate
//! perception exception lives in [`crate::vision`].

mod agent;
mod behavior;
mod checkpoint;
mod config;
mod cpu;
mod device_copy;
mod distributions;
mod intrinsic;
mod networks;
mod readback;
mod replay;
mod runtime;
mod world;

pub use agent::{
    ActionMode, BehaviorMetrics, DreamerAgent, DreamerCore, LearnReport, LearnTiming,
    ModelProvenance, VectorDreamerAgent, WorldMetrics,
};
pub use config::{DreamerConfig, LossScales, ModelSize, NetworkSize};
pub use replay::{FrameFlags, Reward};

/// Upstream DreamerV3 revision used as the behavioral contract.
pub const DREAMERV3_UPSTREAM_REV: &str = "e3f02248693a79dc8b0ebd62c93683888ddaccfe";
/// Meganeura revision used to compile and optimize the baseline graphs.
pub const MEGANEURA_REV: &str = "a7e2efd9d2d1c14e654658fa1b736582d96e16cd";
/// Exact published Blade package providing the shared graphics runtime.
pub const BLADE_REV: &str = "crates.io:blade-graphics@0.9.0#6f50161de1b828487e321d0df36cba06666e624a6a943293485f0dd0e97ef6ea";

#[cfg(test)]
mod tests {
    use super::{BLADE_REV, MEGANEURA_REV};

    #[test]
    fn reported_backend_revisions_match_every_dependency_lock() {
        const KINDLE_MANIFEST: &str = include_str!("../../Cargo.toml");
        const WORKSPACE_LOCK: &str = include_str!("../../../Cargo.lock");
        const PYTHON_LOCK: &str = include_str!("../../../python/Cargo.lock");

        let manifest_pin =
            format!("git = \"https://github.com/kvark/meganeura\", rev = \"{MEGANEURA_REV}\"");
        assert!(KINDLE_MANIFEST.contains(&manifest_pin));
        let (blade_version, blade_checksum) = BLADE_REV
            .strip_prefix("crates.io:blade-graphics@")
            .unwrap()
            .split_once('#')
            .unwrap();
        assert!(KINDLE_MANIFEST.contains(&format!("blade-graphics = \"={blade_version}\"")));

        for (name, lock) in [("workspace", WORKSPACE_LOCK), ("Python", PYTHON_LOCK)] {
            let packages = |wanted: &str| {
                lock.split("[[package]]")
                    .filter(|package| package.lines().any(|line| line == wanted))
                    .collect::<Vec<_>>()
            };
            let meganeura = packages("name = \"meganeura\"");
            assert_eq!(
                meganeura.len(),
                1,
                "{name} has conflicting Meganeura sources"
            );
            assert!(meganeura[0].contains(&format!(
                "git+https://github.com/kvark/meganeura?rev={MEGANEURA_REV}#{MEGANEURA_REV}"
            )));
            let blade = packages("name = \"blade-graphics\"");
            assert_eq!(blade.len(), 1, "{name} has conflicting GPU context types");
            for identity in [
                format!("version = \"{blade_version}\""),
                "source = \"registry+https://github.com/rust-lang/crates.io-index\"".into(),
                format!("checksum = \"{blade_checksum}\""),
            ] {
                assert!(
                    blade[0].contains(&identity),
                    "{name} mismatches {BLADE_REV}"
                );
            }
        }
    }
}
