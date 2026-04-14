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
//! Effective temporal scope diagnostic: `H_eff = sum_i (i * alpha_i)`
//! - Growing H_eff -> agent learns actions have longer consequences (healthy)
//! - Shrinking H_eff -> agent becoming myopic (investigate)

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
    /// Build the credit assigner parameters.
    ///
    /// `input_dim` = `latent_dim + action_dim + 1` (the +1 is for the reward scalar).
    pub fn new(
        g: &mut Graph,
        latent_dim: usize,
        action_dim: usize,
        history_len: usize,
        hidden_dim: usize,
    ) -> Self {
        let input_dim = latent_dim + action_dim + 1;
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
    /// Returns per-timestep attention weights (pre-softmax logits).
    /// The caller applies softmax to get normalized credit weights.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_scope_uniform() {
        // Uniform weights -> H_eff = mean index
        let weights = vec![0.0; 10];
        let h_eff = effective_scope(&weights);
        // Mean of 0..9 = 4.5
        assert!((h_eff - 4.5).abs() < 1e-4);
    }

    #[test]
    fn effective_scope_recent() {
        // All weight on the last timestep -> H_eff = 9
        let mut weights = vec![-100.0; 10];
        weights[9] = 0.0;
        let h_eff = effective_scope(&weights);
        assert!((h_eff - 9.0).abs() < 0.01);
    }
}
