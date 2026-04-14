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
pub struct WorldModel {
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub fc_out: nn::Linear,
}

impl WorldModel {
    /// Build the world model parameters.
    ///
    /// Input dimension is `latent_dim + action_dim` (concatenated).
    pub fn new(
        g: &mut Graph,
        latent_dim: usize,
        action_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        let input_dim = latent_dim + action_dim;
        Self {
            fc1: nn::Linear::new(g, "world_model.fc1", input_dim, hidden_dim),
            fc2: nn::Linear::new(g, "world_model.fc2", hidden_dim, hidden_dim),
            fc_out: nn::Linear::no_bias(g, "world_model.fc_out", hidden_dim, latent_dim),
        }
    }

    /// Forward pass: `[batch, latent_dim + action_dim] -> [batch, latent_dim]`.
    ///
    /// `za` is the concatenation of `z_t` and `a_t` (one-hot encoded action).
    pub fn forward(&self, g: &mut Graph, za: NodeId) -> NodeId {
        let h = self.fc1.forward(g, za);
        let h = g.relu(h);
        let h = self.fc2.forward(g, h);
        let h = g.relu(h);
        self.fc_out.forward(g, h)
    }

    /// Build the MSE loss against the target latent.
    ///
    /// `z_target` must be a graph input node (not derived from parameters)
    /// to achieve stop-gradient semantics.
    pub fn loss(g: &mut Graph, z_pred: NodeId, z_target: NodeId) -> NodeId {
        g.mse_loss(z_pred, z_target)
    }
}
