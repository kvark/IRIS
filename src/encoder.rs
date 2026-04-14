//! Encoder: converts raw observations into a compact latent representation `z_t`.
//!
//! The encoder is the shared backbone. All other modules consume `z_t`,
//! not raw observations. Training signals flow back from the world model
//! (primary), policy gradient (secondary), and value head TD error (secondary).

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// MLP-based encoder for structured (feature vector) observations.
pub struct Encoder {
    pub fc1: nn::Linear,
    pub norm: nn::RmsNorm,
    pub fc2: nn::Linear,
}

impl Encoder {
    /// Build the encoder parameters in the graph.
    pub fn new(g: &mut Graph, obs_dim: usize, latent_dim: usize, hidden_dim: usize) -> Self {
        Self {
            fc1: nn::Linear::new(g, "encoder.fc1", obs_dim, hidden_dim),
            norm: nn::RmsNorm::new(g, "encoder.norm.weight", hidden_dim, 1e-5),
            fc2: nn::Linear::no_bias(g, "encoder.fc2", hidden_dim, latent_dim),
        }
    }

    /// Forward pass: `[batch, obs_dim] -> [batch, latent_dim]`.
    pub fn forward(&self, g: &mut Graph, obs: NodeId) -> NodeId {
        let h = self.fc1.forward(g, obs);
        let h = g.relu(h);
        let h = self.norm.forward(g, h);
        self.fc2.forward(g, h)
    }
}
