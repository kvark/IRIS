//! Frozen Reward Circuit.
//!
//! Computes `r_t` as a weighted sum of three primitive signals:
//! - **Surprise**: world model prediction error
//! - **Novelty**: inverse-sqrt visit count
//! - **Homeostatic balance**: deviation from target ranges
//!
//! This circuit is intentionally frozen — it receives no gradient updates.
//! All computation happens on CPU.

use crate::env::HomeostaticVariable;

/// Weights for the three reward components.
#[derive(Clone, Debug)]
pub struct RewardWeights {
    pub surprise: f32,
    pub novelty: f32,
    pub homeostatic: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        // Default: homeostatic dominates, surprise secondary, novelty tertiary
        Self {
            surprise: 1.0,
            novelty: 0.5,
            homeostatic: 2.0,
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

    /// Combined reward signal.
    pub fn compute(&self, surprise: f32, novelty: f32, homeostatic: f32) -> f32 {
        self.weights.surprise * surprise
            + self.weights.novelty * novelty
            + self.weights.homeostatic * homeostatic
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
    fn combined_reward() {
        let rc = RewardCircuit::new(RewardWeights {
            surprise: 1.0,
            novelty: 0.5,
            homeostatic: 2.0,
        });
        let r = rc.compute(0.2, 0.8, -0.1);
        // 1.0*0.2 + 0.5*0.8 + 2.0*(-0.1) = 0.2 + 0.4 - 0.2 = 0.4
        assert!((r - 0.4).abs() < 1e-6);
    }
}
