//! Policy and Value Head.
//!
//! - **Policy** `π(z_t) → action distribution`, updated by credit-weighted
//!   policy gradient with entropy bonus.
//! - **Value Head** `V(z_t) → V̂`, trained via TD using the credit-adjusted
//!   reward signal, serving as a variance-reduction baseline.

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// `scale · tanh(x / scale)`: an element-wise soft clamp that saturates
/// smoothly to ±scale. Implemented via full-shape constant tensors
/// because meganeura's `Op::Mul` requires matching shapes (no scalar
/// broadcast).
fn scaled_tanh(
    g: &mut Graph,
    x: NodeId,
    scale: f32,
    batch_size: usize,
    last_dim: usize,
) -> NodeId {
    let n = batch_size * last_dim;
    let scale_full = g.constant(vec![scale; n], &[batch_size, last_dim]);
    let inv_scale_full = g.constant(vec![1.0 / scale; n], &[batch_size, last_dim]);
    let shrunk = g.mul(x, inv_scale_full);
    let squashed = g.tanh(shrunk);
    g.mul(squashed, scale_full)
}

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

/// Build the discrete policy + value training graph.
///
/// Inputs:
/// - `"z"`: `[batch_size, latent_dim]` — latent from encoder (detached, fed as input)
/// - `"action"`: `[batch_size, action_dim]` — one-hot taken action (or
///   advantage-weighted one-hot for REINFORCE-style per-row weighting, see
///   `agent.rs::policy_step_batched`).
/// - `"value_target"`: `[batch_size, 1]` — TD target for value head
///
/// Outputs:
/// - `[0]`: combined loss (policy cross-entropy + value MSE, optionally
///   minus `entropy_beta · H(π)`; mean over batch)
/// - `[1]`: logits `[batch_size, action_dim]` — for action sampling
/// - `[2]`: value `[batch_size, 1]` — for advantage computation
///
/// `entropy_beta` controls a shannon-entropy regularizer on the policy:
///
///   H(π) = -∑_k softmax(ℓ)_k · log softmax(ℓ)_k
///
/// The loss subtracts `entropy_beta · H(π)` (after mean-reduction across
/// the batch and action dims). For `β > 0` this *encourages* higher
/// entropy — the standard exploration bonus in policy-gradient methods.
/// For `β < 0` it *discourages* entropy and drives the softmax to
/// collapse toward a delta distribution (useful when a policy is stuck
/// at uniform and needs to commit). At `β = 0` the entropy term is
/// elided entirely and the graph is identical to the pre-regularization
/// policy graph — parity guarantee.
pub fn build_policy_graph(
    latent_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
    batch_size: usize,
    entropy_beta: f32,
    num_options: usize,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let action = g.input("action", &[batch_size, action_dim]);
    let value_target = g.input("value_target", &[batch_size, 1]);

    let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
    let trunk_logits = policy.forward(&mut g, z);

    // Phase G v2: per-option direct-to-logits bias head. Each option
    // has its own learned `[action_dim]` bias vector; the bias is
    // selected per-lane by matmul with a one-hot option encoding.
    // This bypasses the shared trunk so the option-conditional signal
    // is not attenuated by Xavier-init trunk weights or crowded out
    // by the base-reward gradient through z. Only the active option's
    // row receives gradient on each step, preventing cross-option
    // contamination.
    let raw_logits = if num_options > 1 {
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let option_bias =
            nn::Linear::no_bias(&mut g, "policy.option_bias", num_options, action_dim);
        let bias_out = option_bias.forward(&mut g, option_onehot);
        g.add(trunk_logits, bias_out)
    } else {
        trunk_logits
    };

    // No soft-clamp on the logits themselves — a tanh bound here
    // saturates the gradient as the policy commits and prevents
    // useful confident policies from forming (Taxi: entropy 0.20 →
    // 1.78 under a scale=50 clamp). The value head's MSE was the
    // dominant long-run overflow source; clamping value output alone
    // (below) is sufficient to keep policy-loss finite over 1M-step
    // runs.
    let logits = raw_logits;

    // Value head conditions on z only — option-agnostic baseline so
    // `advantage = reward − value` retains the option-conditional
    // bonus signal instead of seeing it cancelled. A symmetric tanh
    // soft-clamp keeps the value output bounded in ±100 so MSE can't
    // drive the head into an overflow loop.
    let value_head = ValueHead::new(&mut g, latent_dim, hidden_dim);
    let raw_value = value_head.forward(&mut g, z);
    let value = scaled_tanh(&mut g, raw_value, 200.0, batch_size, 1);

    // Policy loss: cross-entropy with one-hot action selects -log π(a|s)
    let policy_loss = g.cross_entropy_loss(logits, action);
    let value_loss = g.mse_loss(value, value_target);
    let base_loss = g.add(policy_loss, value_loss);

    // Entropy regularizer: subtract β · H(π) from the loss.
    //
    //   mean_all(p · log p) = -<H>/K   (negative scalar, K = action_dim)
    //
    // so `loss += β · mean_all(p · log p)` adds a negative term for β > 0
    // (encouraging entropy, since minimizing a more-negative value lets
    // |H| grow) and a positive term for β < 0. Elide entirely at β = 0 so
    // we don't add a no-op `softmax→log_softmax→mul→mean→scalar→mul→add`
    // chain that the optimizer would then have to prove away.
    let total_loss = if entropy_beta == 0.0 {
        base_loss
    } else {
        // Entropy regularizer using the fused `log_softmax` — numerically
        // stable (uses `ℓ − log_sum_exp(ℓ)` internally, no log-of-zero) and
        // with a proper backward kernel as of meganeura commit 7c3bdcf.
        let sm = g.softmax(logits);
        let lsm = g.log_softmax(logits);
        let p_log_p = g.mul(sm, lsm);
        let mean_ent = g.mean_all(p_log_p);
        let beta_node = g.scalar(entropy_beta);
        let ent_penalty = g.mul(mean_ent, beta_node);
        g.add(base_loss, ent_penalty)
    };

    g.set_outputs(vec![total_loss, logits, value]);
    g
}

/// Build the continuous policy + value training graph for a diagonal
/// Gaussian with fixed unit variance.
///
/// Inputs:
/// - `"z"`: `[batch_size, latent_dim]` — latent from encoder
/// - `"action"`: `[batch_size, action_dim]` — the taken action vector
/// - `"value_target"`: `[batch_size, 1]` — TD target for value head
///
/// Outputs:
/// - `[0]`: combined loss (mean MSE + value MSE, mean over batch)
/// - `[1]`: action mean `[batch_size, action_dim]` — sampled by adding Gaussian noise
/// - `[2]`: value `[batch_size, 1]`
///
/// For a fixed-variance Gaussian, the negative log-likelihood of the taken
/// action is `0.5·(a − μ)² / σ² + const`. With σ² = 1 this reduces to the
/// MSE between predicted mean and taken action, up to a constant — the
/// same advantage-weighted LR trick applies.
pub fn build_continuous_policy_graph(
    latent_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
    batch_size: usize,
    num_options: usize,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let action = g.input("action", &[batch_size, action_dim]);
    let value_target = g.input("value_target", &[batch_size, 1]);

    // The "Policy" struct outputs [1, action_dim] logits; for continuous
    // actions we reinterpret this as the Gaussian mean μ.
    let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
    let trunk_mean = policy.forward(&mut g, z);

    // Phase G v2: per-option direct-to-mean bias head. See discrete
    // variant for the rationale; here the bias acts on the Gaussian
    // mean, so each option chooses a different centroid for the action
    // distribution while sharing the trunk's state-conditioned part.
    let raw_mean = if num_options > 1 {
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let option_bias =
            nn::Linear::no_bias(&mut g, "policy.option_bias", num_options, action_dim);
        let bias_out = option_bias.forward(&mut g, option_onehot);
        g.add(trunk_mean, bias_out)
    } else {
        trunk_mean
    };
    // No soft-clamp on the Gaussian mean either — see the discrete
    // build_policy_graph for rationale.
    let mean = raw_mean;

    let value_head = ValueHead::new(&mut g, latent_dim, hidden_dim);
    let raw_value = value_head.forward(&mut g, z);
    let value = scaled_tanh(&mut g, raw_value, 200.0, batch_size, 1);

    // Policy loss: MSE(μ, taken_action) ≡ Gaussian NLL with σ² = 1
    let policy_loss = g.mse_loss(mean, action);
    let value_loss = g.mse_loss(value, value_target);
    let total_loss = g.add(policy_loss, value_loss);

    g.set_outputs(vec![total_loss, mean, value]);
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
    let u: f32 = rng.random_range(0.0..1.0);
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

/// Sample from a diagonal Gaussian with mean `mu` and fixed std `scale`.
/// Uses the Box–Muller transform.
pub fn sample_gaussian_action<R: rand::Rng>(mu: &[f32], scale: f32, rng: &mut R) -> Vec<f32> {
    use std::f32::consts::TAU;
    mu.iter()
        .map(|&m| {
            let u1: f32 = rng.random_range(1e-7..1.0);
            let u2: f32 = rng.random_range(0.0..1.0);
            let noise = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
            m + scale * noise
        })
        .collect()
}
