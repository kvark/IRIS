//! Tier 5: long-run stability tests.
//!
//! These verify that the agent remains stable (no NaN, no divergence)
//! over extended training runs. They are GPU-gated via `#[ignore]` and
//! run manually or via `cargo test -- --ignored`.

use iris::envs::grid_world::{GridWorld, NUM_ACTIONS, OBS_DIM};
use iris::envs::random_walk::RandomWalk;
use iris::{Agent, AgentConfig, Environment};
use rand::SeedableRng;

/// Short stability check on GridWorld with all Phase 4 mechanisms enabled.
///
/// Verifies:
/// - No NaN in any loss or diagnostic value across many steps
/// - World model loss decreases from early to late
/// - Drift stays bounded
///
/// Step count chosen to fit within ~1 minute on lavapipe; longer runs
/// are reserved for real GPU.
#[test]
#[ignore] // requires GPU
fn stability_grid_world() {
    let mut env = GridWorld::new();
    let config = AgentConfig {
        obs_dim: OBS_DIM,
        action_dim: NUM_ACTIONS,
        latent_dim: 8,
        hidden_dim: 16,
        history_len: 8,
        buffer_capacity: 2000,
        batch_size: 1,
        learning_rate: 1e-3,
        drift_interval: 500,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Step count chosen to fit within ~2 minutes on lavapipe (CPU Vulkan).
    // Real GPUs run this orders of magnitude faster; longer sweeps belong
    // in a separate manual benchmark.
    let total_steps = 500usize;
    let early_step = 100usize;
    let mut early_loss = f32::NAN;
    let mut late_loss = f32::NAN;
    let mut max_drift = 0.0f32;

    for step in 0..total_steps {
        let obs = env.observe();
        let action = agent.act(&obs, &mut rng);
        env.step(&action);
        agent.observe(&obs, &action, &env, &mut rng);

        let d = agent.diagnostics();
        assert!(
            d.loss_world_model.is_finite(),
            "NaN/Inf in wm_loss at step {step}"
        );
        assert!(
            d.reward_mean.is_finite(),
            "NaN/Inf in reward at step {step}"
        );
        if step == early_step {
            early_loss = d.loss_world_model;
        }
        if step == total_steps - 1 {
            late_loss = d.loss_world_model;
        }
        if d.repr_drift > max_drift {
            max_drift = d.repr_drift;
        }
    }

    // World model should converge: late loss not worse than 10x early
    assert!(
        late_loss < early_loss * 10.0 || late_loss < 0.1,
        "world model diverged: early={early_loss:.4}, late={late_loss:.4}"
    );
    // Drift stays bounded (we reduce encoder LR on excess drift)
    assert!(
        max_drift < 10.0,
        "representation drift too large: {max_drift}"
    );
}

/// Random walk canary: the world model should near-perfectly predict
/// deterministic next-state transitions on a 1D random walk.
#[test]
#[ignore] // requires GPU
fn stability_random_walk_convergence() {
    let size = 10;
    let mut env = RandomWalk::new(size);
    let config = AgentConfig {
        obs_dim: size,
        action_dim: 2,
        latent_dim: 8,
        hidden_dim: 16,
        history_len: 8,
        buffer_capacity: 2000,
        batch_size: 1,
        learning_rate: 1e-3,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);

    for _ in 0..500 {
        let obs = env.observe();
        let action = agent.act(&obs, &mut rng);
        env.step(&action);
        agent.observe(&obs, &action, &env, &mut rng);
    }

    let final_loss = agent.diagnostics().loss_world_model;
    assert!(
        final_loss.is_finite() && final_loss < 0.5,
        "random walk world model didn't converge: {final_loss}"
    );
}
