//! Credit Assigner: attributes reward to past actions via causal attention.
//!
//! The credit assigner answers: *which past actions caused the reward I just
//! received?* It uses causal self-attention over a history window:
//!
//! ```text
//! CreditNet(r_t, [(z_{t-H}, a_{t-H}), ..., (z_t, a_t)]) -> alpha in R^H
//! credit_i = r_t * alpha_i    (alpha normalized via softmax)
//! ```
//!
//! Training: contrastive loss over pairs of similar states with divergent
//! rewards. The credit assigner should attribute the reward difference to
//! the diverging action sequences.
//!
//! Effective temporal scope diagnostic: `H_eff = sum_i (i * alpha_i)`

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// Causal attention credit assigner.
pub struct CreditAssigner {
    pub input_proj: nn::Linear,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub output_proj: nn::Linear,
    pub num_heads: u32,
    pub head_dim: u32,
    pub history_len: usize,
}

impl CreditAssigner {
    /// Per-timestep input dimension: `latent_dim + action_dim + 1`.
    pub fn input_dim(latent_dim: usize, action_dim: usize) -> usize {
        latent_dim + action_dim + 1
    }

    /// Build the credit assigner parameters.
    pub fn new(
        g: &mut Graph,
        latent_dim: usize,
        action_dim: usize,
        history_len: usize,
        hidden_dim: usize,
    ) -> Self {
        let input_dim = Self::input_dim(latent_dim, action_dim);
        let num_heads = 2u32;
        let head_dim = (hidden_dim / num_heads as usize) as u32;
        let attn_dim = (num_heads * head_dim) as usize;

        Self {
            input_proj: nn::Linear::new(g, "credit.input_proj", input_dim, attn_dim),
            q_proj: nn::Linear::no_bias(g, "credit.q_proj", attn_dim, attn_dim),
            k_proj: nn::Linear::no_bias(g, "credit.k_proj", attn_dim, attn_dim),
            v_proj: nn::Linear::no_bias(g, "credit.v_proj", attn_dim, attn_dim),
            output_proj: nn::Linear::new(g, "credit.output_proj", attn_dim, 1),
            num_heads,
            head_dim,
            history_len,
        }
    }

    /// Forward pass: `[history_len, input_dim] -> [history_len, 1]`.
    ///
    /// Returns per-timestep credit logits. Apply softmax to get
    /// normalized credit weights alpha_i.
    pub fn forward(&self, g: &mut Graph, history: NodeId) -> NodeId {
        let h = self.input_proj.forward(g, history);
        let h = g.relu(h);

        let q = self.q_proj.forward(g, h);
        let k = self.k_proj.forward(g, h);
        let v = self.v_proj.forward(g, h);

        let attn_out =
            g.causal_attention(q, k, v, self.num_heads, self.num_heads, self.head_dim);

        self.output_proj.forward(g, attn_out)
    }

    /// MSE loss against a contrastive credit target.
    ///
    /// `credit_pred`: `[history_len, 1]` — logits from forward().
    /// `credit_target`: `[history_len, 1]` — target distribution from
    /// contrastive pair sampling (action divergence, softmax-normalized).
    pub fn loss(g: &mut Graph, credit_pred: NodeId, credit_target: NodeId) -> NodeId {
        g.mse_loss(credit_pred, credit_target)
    }
}

/// Build a complete credit assigner training graph.
///
/// Returns the graph ready for `build_session()`. Inputs:
/// - `"history"`: `[history_len, input_dim]`
/// - `"credit_target"`: `[history_len, 1]`
///
/// Outputs:
/// - `[0]`: loss (scalar)
/// - `[1]`: credit logits `[history_len, 1]`
pub fn build_credit_graph(
    latent_dim: usize,
    action_dim: usize,
    history_len: usize,
    hidden_dim: usize,
) -> Graph {
    let input_dim = CreditAssigner::input_dim(latent_dim, action_dim);

    let mut g = Graph::new();
    let history = g.input("history", &[history_len, input_dim]);
    let credit_target = g.input("credit_target", &[history_len, 1]);

    let ca = CreditAssigner::new(&mut g, latent_dim, action_dim, history_len, hidden_dim);
    let credit_pred = ca.forward(&mut g, history);
    let loss = CreditAssigner::loss(&mut g, credit_pred, credit_target);

    g.set_outputs(vec![loss, credit_pred]);
    g
}

/// Compute the effective temporal scope from credit weights.
///
/// `H_eff = sum_i (i * alpha_i)` where alpha is softmax-normalized.
pub fn effective_scope(credit_weights: &[f32]) -> f32 {
    let max = credit_weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = credit_weights.iter().map(|&w| (w - max).exp()).sum();

    credit_weights
        .iter()
        .enumerate()
        .map(|(i, &w)| {
            let alpha = (w - max).exp() / exp_sum;
            i as f32 * alpha
        })
        .sum()
}

/// Softmax-normalize raw logits into a probability distribution.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_scope_uniform() {
        let weights = vec![0.0; 10];
        let h_eff = effective_scope(&weights);
        assert!((h_eff - 4.5).abs() < 1e-4);
    }

    #[test]
    fn effective_scope_recent() {
        let mut weights = vec![-100.0; 10];
        weights[9] = 0.0;
        let h_eff = effective_scope(&weights);
        assert!((h_eff - 9.0).abs() < 0.01);
    }

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // Should be monotonically increasing
        for i in 1..probs.len() {
            assert!(probs[i] > probs[i - 1]);
        }
    }
}
