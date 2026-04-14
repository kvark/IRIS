//! Reward signal verification.
//!
//! Verify that the reward circuit produces non-zero, varying signals
//! across different states and transitions.

use iris::buffer::{ExperienceBuffer, Transition};
use iris::env::HomeostaticVariable;
use iris::reward::{RewardCircuit, RewardWeights};

/// Reward signals should be non-zero and vary across different states.
#[test]
fn reward_varies_across_states() {
    let rc = RewardCircuit::new(RewardWeights::default());

    // Different surprise levels
    let r1 = rc.compute(0.1, 1.0, 0.0, 0.0);
    let r2 = rc.compute(0.9, 1.0, 0.0, 0.0);
    assert!(
        (r1 - r2).abs() > 0.1,
        "reward should vary with surprise: r1={r1}, r2={r2}"
    );

    // Different novelty levels
    let r3 = rc.compute(0.5, 1.0, 0.0, 0.0);
    let r4 = rc.compute(0.5, 0.1, 0.0, 0.0);
    assert!(
        (r3 - r4).abs() > 0.1,
        "reward should vary with novelty: r3={r3}, r4={r4}"
    );

    // Different homeostatic levels
    let r5 = rc.compute(0.5, 0.5, 0.0, 0.0);
    let r6 = rc.compute(0.5, 0.5, -1.0, 0.0);
    assert!(
        (r5 - r6).abs() > 0.1,
        "reward should vary with homeostatic: r5={r5}, r6={r6}"
    );
}

/// Novelty should decay as the same state is revisited.
#[test]
fn novelty_decays_with_visits() {
    let mut buf = ExperienceBuffer::new(100, 0.1);
    let latent = vec![0.5, 0.5];

    let n0 = RewardCircuit::novelty(buf.visit_count(&latent));
    assert_eq!(n0, 1.0, "first visit should have max novelty");

    // Visit the same state repeatedly
    for _ in 0..10 {
        buf.push(Transition {
            latent: latent.clone(),
            ..Default::default()
        });
    }

    let n10 = RewardCircuit::novelty(buf.visit_count(&latent));
    assert!(n10 < n0, "novelty should decay: n0={n0}, n10={n10}");
    // 1/sqrt(10) ≈ 0.316
    assert!(
        (n10 - 1.0 / (10.0f32).sqrt()).abs() < 1e-5,
        "novelty should be 1/sqrt(N): n10={n10}"
    );
}

/// Homeostatic reward should be zero when in range, negative when out.
#[test]
fn homeostatic_penalty_gradient() {
    let in_range = vec![HomeostaticVariable {
        value: 0.5,
        target: 0.5,
        tolerance: 0.1,
    }];
    assert_eq!(RewardCircuit::homeostatic(&in_range), 0.0);

    // Gradually move out of range
    let mut penalties = Vec::new();
    for offset in [0.0, 0.2, 0.5, 1.0] {
        let vars = vec![HomeostaticVariable {
            value: 0.5 + offset,
            target: 0.5,
            tolerance: 0.1,
        }];
        penalties.push(RewardCircuit::homeostatic(&vars));
    }

    // Should be monotonically decreasing
    for i in 1..penalties.len() {
        assert!(
            penalties[i] <= penalties[i - 1],
            "homeostatic penalty should increase with deviation: {:?}",
            penalties
        );
    }
}

/// Combined reward should be non-zero for a typical transition.
#[test]
fn combined_reward_nonzero() {
    let rc = RewardCircuit::new(RewardWeights::default());
    let r = rc.compute(0.5, 0.8, -0.1, 0.0);
    assert!(r.abs() > 0.01, "combined reward should be non-zero: {r}");
    assert!(r.is_finite(), "reward must be finite");
}

/// Order reward should prefer concentrated observations over uniform ones
/// when the order weight is enabled.
#[test]
fn order_prefers_concentrated_observations() {
    let rc = RewardCircuit::new(RewardWeights {
        surprise: 0.0,
        novelty: 0.0,
        homeostatic: 0.0,
        order: 1.0,
    });
    let concentrated = RewardCircuit::order(&[5.0, 0.1, 0.1, 0.1]);
    let chaotic = RewardCircuit::order(&[1.0; 4]);
    let r_concentrated = rc.compute(0.0, 0.0, 0.0, concentrated);
    let r_chaotic = rc.compute(0.0, 0.0, 0.0, chaotic);
    assert!(
        r_concentrated > r_chaotic,
        "order reward should prefer concentrated obs: {r_concentrated} vs {r_chaotic}"
    );
}
