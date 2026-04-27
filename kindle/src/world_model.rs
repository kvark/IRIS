//! World Model: forward dynamics predictor.
//!
//! Predicts the next latent state given the current latent and action:
//! `W(z_t, a_t) -> z_hat_{t+1}`
//!
//! Loss: `MSE(z_hat_{t+1}, stop_grad(z_{t+1}))`.
//! The target `z_{t+1}` is fed as a graph input (not parameter), which
//! prevents gradients from flowing through it — achieving stop-gradient.
//!
//! The prediction error also serves as the "surprise" component of the
//! reward circuit.

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// Forward dynamics world model.
///
/// The first layer is split into two projections (`z_proj` + `a_proj`)
/// whose outputs are summed. This is equivalent to `W @ [z; a]` but
/// avoids needing a concat op and — critically — keeps the encoder's
/// z_t on the loss path so it receives gradients.
pub struct WorldModel {
    pub z_proj: nn::Linear,
    pub a_proj: nn::Linear,
    pub fc2: nn::Linear,
    pub fc_out: nn::Linear,
}

impl WorldModel {
    /// Build the world model parameters.
    pub fn new(g: &mut Graph, latent_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            z_proj: nn::Linear::new(g, "world_model.z_proj", latent_dim, hidden_dim),
            a_proj: nn::Linear::no_bias(g, "world_model.a_proj", action_dim, hidden_dim),
            fc2: nn::Linear::new(g, "world_model.fc2", hidden_dim, hidden_dim),
            fc_out: nn::Linear::no_bias(g, "world_model.fc_out", hidden_dim, latent_dim),
        }
    }

    /// Forward pass: `(z_t, action) -> z_hat_{t+1}`.
    ///
    /// `z_t`: `[batch, latent_dim]` — encoder output (on the gradient path).
    /// `action`: `[batch, action_dim]` — one-hot or continuous action vector.
    pub fn forward(&self, g: &mut Graph, z_t: NodeId, action: NodeId) -> NodeId {
        self.forward_with_mode(g, z_t, action, false)
    }

    /// Forward pass with optional residual parameterization.
    ///
    /// When `residual` is true the head predicts `z + delta(z, a)`
    /// instead of `z_{t+1}` directly. Empirically necessary on envs
    /// where consecutive latents are nearly equal: without residual
    /// the head must re-derive z from scratch every step, and the
    /// no-op baseline (`z' = z`) beats the learned predictor.
    pub fn forward_with_mode(
        &self,
        g: &mut Graph,
        z_t: NodeId,
        action: NodeId,
        residual: bool,
    ) -> NodeId {
        let h_z = self.z_proj.forward(g, z_t);
        let h_a = self.a_proj.forward(g, action);
        let h = g.add(h_z, h_a);
        let h = g.relu(h);
        let h = self.fc2.forward(g, h);
        let h = g.relu(h);
        let delta_or_pred = self.fc_out.forward(g, h);
        if residual {
            g.add(z_t, delta_or_pred)
        } else {
            delta_or_pred
        }
    }

    /// Build the MSE loss against the target latent.
    ///
    /// `z_target` must be a graph input node (not derived from parameters)
    /// to achieve stop-gradient semantics.
    pub fn loss(g: &mut Graph, z_pred: NodeId, z_target: NodeId) -> NodeId {
        g.mse_loss(z_pred, z_target)
    }
}
