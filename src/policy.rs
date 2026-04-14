//! Policy and Value Head.
//!
//! - **Policy** `π(z_t) → action distribution`, updated by credit-weighted
//!   policy gradient with entropy bonus.
//! - **Value Head** `V(z_t) → V̂`, trained via TD using the credit-adjusted
//!   reward signal, serving as a variance-reduction baseline.

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// Stochastic policy network for discrete action spaces.
pub struct Policy {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
}

impl Policy {
    pub fn new(g: &mut Graph, latent_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            fc1: nn::Linear::new(g, "policy.fc1", latent_dim, hidden_dim),
            fc2: nn::Linear::no_bias(g, "policy.fc2", hidden_dim, action_dim),
        }
    }

    /// Forward pass: `[batch, latent_dim] → [batch, action_dim]` (logits).
    pub fn forward(&self, g: &mut Graph, z: NodeId) -> NodeId {
        let h = self.fc1.forward(g, z);
        let h = g.relu(h);
        self.fc2.forward(g, h)
    }
}

/// Value head: estimates cumulative future reward from the current state.
pub struct ValueHead {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
}

impl ValueHead {
    pub fn new(g: &mut Graph, latent_dim: usize, hidden_dim: usize) -> Self {
        Self {
            fc1: nn::Linear::new(g, "value.fc1", latent_dim, hidden_dim),
            fc2: nn::Linear::no_bias(g, "value.fc2", hidden_dim, 1),
        }
    }

    /// Forward pass: `[batch, latent_dim] → [batch, 1]`.
    pub fn forward(&self, g: &mut Graph, z: NodeId) -> NodeId {
        let h = self.fc1.forward(g, z);
        let h = g.relu(h);
        self.fc2.forward(g, h)
    }
}

/// Build the policy + value training graph.
///
/// Inputs:
/// - `"z"`: `[1, latent_dim]` — latent from encoder (detached, fed as input)
/// - `"action"`: `[1, action_dim]` — one-hot taken action
/// - `"value_target"`: `[1, 1]` — TD target for value head
///
/// Outputs:
/// - `[0]`: combined loss (policy cross-entropy + value MSE)
/// - `[1]`: logits `[1, action_dim]` — for action sampling
/// - `[2]`: value `[1, 1]` — for advantage computation
///
/// The policy loss is `cross_entropy(logits, action_onehot)`. Advantage
/// weighting is applied by scaling the learning rate on the CPU side:
/// `lr_effective = lr_policy * advantage`.
pub fn build_policy_graph(
    latent_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[1, latent_dim]);
    let action = g.input("action", &[1, action_dim]);
    let value_target = g.input("value_target", &[1, 1]);

    let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
    let logits = policy.forward(&mut g, z);

    let value_head = ValueHead::new(&mut g, latent_dim, hidden_dim);
    let value = value_head.forward(&mut g, z);

    // Policy loss: cross-entropy with one-hot action selects -log π(a|s)
    let policy_loss = g.cross_entropy_loss(logits, action);

    // Value loss: MSE(V(z), target_value)
    let value_loss = g.mse_loss(value, value_target);

    // Combined loss
    let total_loss = g.add(policy_loss, value_loss);

    g.set_outputs(vec![total_loss, logits, value]);
    g
}

/// Compute softmax probabilities from logits.
pub fn softmax_probs(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Sample an action from logits using the Gumbel-max trick.
pub fn sample_action<R: rand::Rng>(logits: &[f32], rng: &mut R) -> usize {
    let probs = softmax_probs(logits);
    let u: f32 = rng.gen_range(0.0..1.0);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if u < cumulative {
            return i;
        }
    }
    probs.len() - 1
}

/// Compute policy entropy: `H[π] = -Σ π_i · log π_i`.
pub fn entropy(logits: &[f32]) -> f32 {
    let probs = softmax_probs(logits);
    -probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}
