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
fn scaled_tanh(g: &mut Graph, x: NodeId, scale: f32, batch_size: usize, last_dim: usize) -> NodeId {
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

/// Phase G Tier-3: per-option fc2 heads. The shared trunk (fc1) still
/// produces a state-conditioned hidden vector `h`, but each option gets
/// its *own* `[hidden_dim → action_dim]` linear head. The active option's
/// head is selected per-lane via a onehot-weighted sum, so only that
/// head sees gradient on a given step — capacity is silo'd per option.
///
/// Without broadcast-mul or gather, the selection is spelled out as:
///   col_o    = onehot @ e_o           (e_o ∈ R^{N×1}, 1 at row o)
///   mask_o   = col_o @ ones_row       (ones_row ∈ R^{1×A})
///   masked_o = logits_o * mask_o
///   combined = Σ_o masked_o
fn per_option_fc2(
    g: &mut Graph,
    h: NodeId,
    option_onehot: NodeId,
    num_options: usize,
    hidden_dim: usize,
    action_dim: usize,
) -> NodeId {
    let ones_row = g.constant(vec![1.0f32; action_dim], &[1, action_dim]);
    let mut sum: Option<NodeId> = None;
    for o in 0..num_options {
        let head_o = nn::Linear::new(g, &format!("policy.fc2_opt{o}"), hidden_dim, action_dim);
        let logits_o = head_o.forward(g, h);

        let mut selector = vec![0.0f32; num_options];
        selector[o] = 1.0;
        let sel_col = g.constant(selector, &[num_options, 1]);
        let col_o = g.matmul(option_onehot, sel_col);
        let mask_full = g.matmul(col_o, ones_row);

        let masked_o = g.mul(logits_o, mask_full);
        sum = Some(match sum {
            None => masked_o,
            Some(prev) => g.add(prev, masked_o),
        });
    }
    sum.expect("num_options >= 1")
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
    per_option_heads: bool,
    value_loss_coef: f32,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let action = g.input("action", &[batch_size, action_dim]);
    let value_target = g.input("value_target", &[batch_size, 1]);

    // Phase G v2: per-option direct-to-logits bias head. Each option
    // has its own learned `[action_dim]` bias vector; the bias is
    // selected per-lane by matmul with a one-hot option encoding.
    // This bypasses the shared trunk so the option-conditional signal
    // is not attenuated by Xavier-init trunk weights or crowded out
    // by the base-reward gradient through z.
    //
    // Phase G Tier-3: when `per_option_heads` is set, the shared fc2
    // is replaced by N per-option fc2 heads, giving each option its
    // own `[hidden_dim → action_dim]` matrix — the bias subsumes into
    // each head's own bias so we drop the separate option_bias layer.
    let raw_logits = if num_options > 1 && per_option_heads {
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let fc1 = nn::Linear::new(&mut g, "policy.fc1", latent_dim, hidden_dim);
        let h = fc1.forward(&mut g, z);
        let h = g.relu(h);
        per_option_fc2(
            &mut g,
            h,
            option_onehot,
            num_options,
            hidden_dim,
            action_dim,
        )
    } else if num_options > 1 {
        let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
        let trunk_logits = policy.forward(&mut g, z);
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let option_bias =
            nn::Linear::no_bias(&mut g, "policy.option_bias", num_options, action_dim);
        let bias_out = option_bias.forward(&mut g, option_onehot);
        g.add(trunk_logits, bias_out)
    } else {
        let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
        policy.forward(&mut g, z)
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
    let value_loss_raw = g.mse_loss(value, value_target);
    // Value-loss coefficient (PPO/A2C `vf_coef`). Under a shared
    // optimizer the value head tends to dominate the combined
    // gradient — its MSE target is reward-scale while the policy-CE
    // loss is O(log K) — so we weight value down explicitly. Default
    // 1.0 keeps the old behavior.
    let value_loss = if (value_loss_coef - 1.0).abs() < 1e-6 {
        value_loss_raw
    } else {
        let coef = g.scalar(value_loss_coef);
        g.mul(value_loss_raw, coef)
    };
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

/// Build the discrete PPO-style policy + value training graph.
///
/// Inputs:
/// - `"z"`: `[batch_size, latent_dim]`
/// - `"action"`: `[batch_size, action_dim]` — one-hot of the taken action
///   (MUST sum to 1 per row — plain one-hot, no advantage weighting)
/// - `"advantage"`: `[batch_size, 1]` — per-row scalar advantage Â_t
/// - `"old_prob_taken"`: `[batch_size, 1]` — π_old(a_t|s_t), the probability
///   of the taken action under the policy that collected the data (positive,
///   non-zero)
/// - `"value_target"`: `[batch_size, 1]` — TD target for the value head
///
/// Outputs: `[loss, logits, value]`.
///
/// Loss: the PPO clipped surrogate
/// ```text
/// r_t = π_new(a_t|s_t) / π_old(a_t|s_t)
/// L = -E[ min(r_t · Â_t, clip(r_t, 1-ε, 1+ε) · Â_t) ]
/// ```
/// Decomposed using only the ops meganeura has today: softmax (→prob),
/// per-row gather via matmul with a `[K,1]` ones column, elementwise
/// division, and `greater()`-gated min/max selectors. The `greater` op
/// has zero backward gradient, so the clip boundary correctly produces
/// zero gradient — the defining property that makes PPO a proper trust
/// region and stops committed policies from overshooting.
///
/// `entropy_beta` and `value_loss_coef` behave identically to the
/// non-PPO variant.
pub fn build_ppo_policy_graph(
    latent_dim: usize,
    action_dim: usize,
    hidden_dim: usize,
    batch_size: usize,
    clip_eps: f32,
    value_loss_coef: f32,
    entropy_beta: f32,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let action = g.input("action", &[batch_size, action_dim]);
    let advantage = g.input("advantage", &[batch_size, 1]);
    let old_prob_taken = g.input("old_prob_taken", &[batch_size, 1]);
    let value_target = g.input("value_target", &[batch_size, 1]);

    let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
    let logits = policy.forward(&mut g, z);

    // π_new(a | s) — probability of the taken action under the current
    // policy. Built from `softmax(logits) * one_hot` followed by a
    // per-row sum implemented as matmul with a `[K, 1]` ones vector.
    let sm_new = g.softmax(logits);
    let p_per_class = g.mul(sm_new, action);
    let ones_k1 = g.constant(vec![1.0; action_dim], &[action_dim, 1]);
    let new_prob_taken = g.matmul(p_per_class, ones_k1);

    // ratio r_t = π_new / π_old
    let ratio = g.div(new_prob_taken, old_prob_taken);

    // Clip ratio to [1-ε, 1+ε] via two `greater()`-gated selectors:
    // ratio_hi = min(ratio, 1+ε)
    // ratio_clip = max(ratio_hi, 1-ε)
    let one_plus_eps = g.constant(vec![1.0 + clip_eps; batch_size], &[batch_size, 1]);
    let one_minus_eps = g.constant(vec![1.0 - clip_eps; batch_size], &[batch_size, 1]);
    let ones_bx1 = g.constant(vec![1.0; batch_size], &[batch_size, 1]);

    // min(a, b) = (1 - gate) * a + gate * b  where gate = greater(a, b)
    let gate_over = g.greater(ratio, one_plus_eps);
    let neg_gate_over = g.neg(gate_over);
    let inv_gate_over = g.add(ones_bx1, neg_gate_over);
    let term1 = g.mul(ratio, inv_gate_over);
    let term2 = g.mul(one_plus_eps, gate_over);
    let ratio_hi = g.add(term1, term2);

    // max(a, b) = (1 - gate) * a + gate * b  where gate = greater(b, a)
    let gate_under = g.greater(one_minus_eps, ratio_hi);
    let neg_gate_under = g.neg(gate_under);
    let inv_gate_under = g.add(ones_bx1, neg_gate_under);
    let cterm1 = g.mul(ratio_hi, inv_gate_under);
    let cterm2 = g.mul(one_minus_eps, gate_under);
    let ratio_clip = g.add(cterm1, cterm2);

    // surr1 = r_t · Â_t,  surr2 = clip(r_t) · Â_t
    let surr1 = g.mul(ratio, advantage);
    let surr2 = g.mul(ratio_clip, advantage);

    // min(surr1, surr2) → pessimistic of the two (PPO's conservative choice)
    let gate_min = g.greater(surr1, surr2);
    let neg_gate_min = g.neg(gate_min);
    let inv_gate_min = g.add(ones_bx1, neg_gate_min);
    let smin1 = g.mul(surr1, inv_gate_min);
    let smin2 = g.mul(surr2, gate_min);
    let surr_min = g.add(smin1, smin2);

    // Policy loss = -mean(surr_min). We MAXIMIZE surr_min, so loss negates.
    let neg_surr_min = g.neg(surr_min);
    let policy_loss = g.mean_all(neg_surr_min);

    // Value head — same structure as the non-PPO path
    let value_head = ValueHead::new(&mut g, latent_dim, hidden_dim);
    let raw_value = value_head.forward(&mut g, z);
    let value = scaled_tanh(&mut g, raw_value, 200.0, batch_size, 1);
    let value_loss_raw = g.mse_loss(value, value_target);
    let value_loss = if (value_loss_coef - 1.0).abs() < 1e-6 {
        value_loss_raw
    } else {
        let coef = g.scalar(value_loss_coef);
        g.mul(value_loss_raw, coef)
    };
    let base_loss = g.add(policy_loss, value_loss);

    let total_loss = if entropy_beta == 0.0 {
        base_loss
    } else {
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
    per_option_heads: bool,
    value_loss_coef: f32,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let action = g.input("action", &[batch_size, action_dim]);
    let value_target = g.input("value_target", &[batch_size, 1]);

    // Phase G v2: per-option direct-to-mean bias head. See discrete
    // variant for the rationale; here the bias acts on the Gaussian
    // mean, so each option chooses a different centroid for the action
    // distribution while sharing the trunk's state-conditioned part.
    //
    // Phase G Tier-3: with `per_option_heads`, the shared fc2 is
    // replaced by N per-option heads outputting the Gaussian mean
    // directly — same gating mechanism as the discrete variant.
    let raw_mean = if num_options > 1 && per_option_heads {
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let fc1 = nn::Linear::new(&mut g, "policy.fc1", latent_dim, hidden_dim);
        let h = fc1.forward(&mut g, z);
        let h = g.relu(h);
        per_option_fc2(
            &mut g,
            h,
            option_onehot,
            num_options,
            hidden_dim,
            action_dim,
        )
    } else if num_options > 1 {
        let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
        let trunk_mean = policy.forward(&mut g, z);
        let option_onehot = g.input("option_onehot", &[batch_size, num_options]);
        let option_bias =
            nn::Linear::no_bias(&mut g, "policy.option_bias", num_options, action_dim);
        let bias_out = option_bias.forward(&mut g, option_onehot);
        g.add(trunk_mean, bias_out)
    } else {
        let policy = Policy::new(&mut g, latent_dim, action_dim, hidden_dim);
        policy.forward(&mut g, z)
    };
    // No soft-clamp on the Gaussian mean either — see the discrete
    // build_policy_graph for rationale.
    let mean = raw_mean;

    let value_head = ValueHead::new(&mut g, latent_dim, hidden_dim);
    let raw_value = value_head.forward(&mut g, z);
    let value = scaled_tanh(&mut g, raw_value, 200.0, batch_size, 1);

    // Policy loss: MSE(μ, taken_action) ≡ Gaussian NLL with σ² = 1
    let policy_loss = g.mse_loss(mean, action);
    let value_loss_raw = g.mse_loss(value, value_target);
    let value_loss = if (value_loss_coef - 1.0).abs() < 1e-6 {
        value_loss_raw
    } else {
        let coef = g.scalar(value_loss_coef);
        g.mul(value_loss_raw, coef)
    };
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
