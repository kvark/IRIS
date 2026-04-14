//! Frozen Reward Circuit.
//!
//! Computes `r_t` as a weighted sum of four primitive signals:
//! - **Surprise**: world model prediction error
//! - **Novelty**: inverse-sqrt visit count
//! - **Homeostatic balance**: deviation from target ranges
//! - **Order**: negative Shannon entropy of the observation (reduces chaos)
//!
//! This circuit is intentionally frozen — it receives no gradient updates.
//! All computation happens on CPU.

use crate::env::HomeostaticVariable;

/// Weights for the four reward components.
#[derive(Clone, Debug)]
pub struct RewardWeights {
    pub surprise: f32,
    pub novelty: f32,
    pub homeostatic: f32,
    /// Weight for the order primitive. Default 0 — only envs that actually
    /// benefit from entropy reduction (e.g. ARC-AGI-ish patterning tasks)
    /// should enable this.
    pub order: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        // Default: homeostatic dominates, surprise secondary, novelty tertiary,
        // order off.
        Self {
            surprise: 1.0,
            novelty: 0.5,
            homeostatic: 2.0,
            order: 0.0,
        }
    }
}

/// The frozen reward circuit.
pub struct RewardCircuit {
    pub weights: RewardWeights,
}

impl RewardCircuit {
    pub fn new(weights: RewardWeights) -> Self {
        Self { weights }
    }

    /// Surprise: the L2 norm of world model prediction error.
    /// `pred_error` is `||W(z_{t-1}, a_{t-1}) - z_t||_2`, already computed.
    pub fn surprise(pred_error: f32) -> f32 {
        pred_error
    }

    /// Novelty: `1 / sqrt(N(z_t))` where N is the visit count.
    pub fn novelty(visit_count: u32) -> f32 {
        if visit_count == 0 {
            1.0
        } else {
            1.0 / (visit_count as f32).sqrt()
        }
    }

    /// Homeostatic balance: negative penalty for deviation from target ranges.
    /// `r_homeo = -sum_i max(0, |h_i - target_i| - tolerance_i)`
    pub fn homeostatic(variables: &[HomeostaticVariable]) -> f32 {
        let mut penalty = 0.0f32;
        for var in variables {
            let deviation = (var.value - var.target).abs() - var.tolerance;
            penalty += deviation.max(0.0);
        }
        -penalty
    }

    /// Order: negative Shannon entropy of the observation treated as a
    /// distribution over its own components.
    ///
    /// The observation is normalized so that `|o_i| / Σ|o_j|` forms a
    /// probability distribution. Low-entropy distributions (one or a few
    /// dominant components) score near zero — "ordered". High-entropy
    /// distributions (mass spread evenly) score negative — "chaotic".
    ///
    /// The reward is `−H`, so maximizing it drives the agent toward
    /// states where observations are more concentrated. Useful for
    /// pattern-fixing tasks (ARC-AGI) where success = regularity.
    pub fn order(obs: &[f32]) -> f32 {
        const EPS: f32 = 1e-8;
        if obs.is_empty() {
            return 0.0;
        }
        let abs_sum: f32 = obs.iter().map(|x| x.abs()).sum::<f32>() + EPS;
        let mut entropy = 0.0f32;
        for &o in obs {
            let p = (o.abs() / abs_sum) + EPS;
            entropy -= p * p.ln();
        }
        -entropy
    }

    /// Combined reward signal.
    pub fn compute(&self, surprise: f32, novelty: f32, homeostatic: f32, order: f32) -> f32 {
        self.weights.surprise * surprise
            + self.weights.novelty * novelty
            + self.weights.homeostatic * homeostatic
            + self.weights.order * order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surprise_passes_through() {
        assert_eq!(RewardCircuit::surprise(0.5), 0.5);
    }

    #[test]
    fn novelty_first_visit() {
        assert_eq!(RewardCircuit::novelty(0), 1.0);
        assert_eq!(RewardCircuit::novelty(1), 1.0);
        assert!((RewardCircuit::novelty(4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn homeostatic_in_range() {
        let vars = vec![HomeostaticVariable {
            value: 0.5,
            target: 0.5,
            tolerance: 0.1,
        }];
        assert_eq!(RewardCircuit::homeostatic(&vars), 0.0);
    }

    #[test]
    fn homeostatic_out_of_range() {
        let vars = vec![HomeostaticVariable {
            value: 1.0,
            target: 0.5,
            tolerance: 0.1,
        }];
        // deviation = |1.0 - 0.5| - 0.1 = 0.4
        assert!((RewardCircuit::homeostatic(&vars) - (-0.4)).abs() < 1e-6);
    }

    #[test]
    fn order_one_hot_is_maximal() {
        // One-hot observation = fully concentrated = high order (near 0)
        let obs = vec![0.0, 0.0, 1.0, 0.0, 0.0];
        let r = RewardCircuit::order(&obs);
        assert!(r > -0.001, "one-hot should have order ≈ 0, got {r}");
    }

    #[test]
    fn order_uniform_is_minimal() {
        // Uniform observation = maximum entropy = low order (negative)
        let obs = vec![1.0; 8];
        let r = RewardCircuit::order(&obs);
        // Max entropy for 8 components is ln(8) ≈ 2.079
        assert!(r < -2.0, "uniform obs should have low order, got {r}");
    }

    #[test]
    fn order_monotonic_in_concentration() {
        // Concentrating mass onto fewer components should increase order
        let spread = vec![1.0, 1.0, 1.0, 1.0];
        let mid = vec![3.0, 1.0, 1.0, 1.0];
        let peaked = vec![10.0, 0.5, 0.5, 0.5];
        let r_spread = RewardCircuit::order(&spread);
        let r_mid = RewardCircuit::order(&mid);
        let r_peaked = RewardCircuit::order(&peaked);
        assert!(r_spread < r_mid, "{r_spread} should be < {r_mid}");
        assert!(r_mid < r_peaked, "{r_mid} should be < {r_peaked}");
    }

    #[test]
    fn order_empty_is_zero() {
        let obs: Vec<f32> = vec![];
        assert_eq!(RewardCircuit::order(&obs), 0.0);
    }

    #[test]
    fn combined_reward() {
        let rc = RewardCircuit::new(RewardWeights {
            surprise: 1.0,
            novelty: 0.5,
            homeostatic: 2.0,
            order: 0.0,
        });
        let r = rc.compute(0.2, 0.8, -0.1, 0.0);
        // 1.0*0.2 + 0.5*0.8 + 2.0*(-0.1) + 0.0*0.0 = 0.4
        assert!((r - 0.4).abs() < 1e-6);
    }

    #[test]
    fn combined_reward_with_order() {
        let rc = RewardCircuit::new(RewardWeights {
            surprise: 0.0,
            novelty: 0.0,
            homeostatic: 0.0,
            order: 1.0,
        });
        let r_peaked = rc.compute(0.0, 0.0, 0.0, RewardCircuit::order(&[10.0, 0.1, 0.1, 0.1]));
        let r_uniform = rc.compute(0.0, 0.0, 0.0, RewardCircuit::order(&[1.0; 4]));
        assert!(
            r_peaked > r_uniform,
            "peaked ({r_peaked}) should reward more than uniform ({r_uniform})"
        );
    }
}
