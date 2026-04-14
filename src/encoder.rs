//! Encoder: converts raw observations into a compact latent representation `z_t`.
//!
//! The encoder is the shared backbone. All other modules consume `z_t`,
//! not raw observations. Training signals flow back from the world model
//! (primary), policy gradient (secondary), and value head TD error (secondary).
//!
//! Two variants:
//! - **MLP** (default): for structured (feature vector) observations.
//! - **CNN** (feature `cnn`): for pixel observations (NCHW flat layout).

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

/// CNN-based encoder for pixel observations.
///
/// Architecture: conv(8 filters, 3x3, stride 2) → relu → conv(16, 3x3, stride 2)
/// → relu → global_avg_pool → linear → latent.
///
/// Input: flat NCHW tensor `[batch * channels * H * W]`.
/// Output: `[batch, latent_dim]`.
#[cfg(feature = "cnn")]
pub struct CnnEncoder {
    pub conv1: nn::Conv2d,
    pub conv2: nn::Conv2d,
    pub fc: nn::Linear,
    pub batch: u32,
    pub pool_channels: u32,
}

#[cfg(feature = "cnn")]
impl CnnEncoder {
    /// Build a CNN encoder for images of size `channels x height x width`.
    pub fn new(
        g: &mut Graph,
        channels: u32,
        height: u32,
        width: u32,
        latent_dim: usize,
        batch: u32,
    ) -> Self {
        let out_ch1 = 8u32;
        let out_ch2 = 16u32;
        let h1 = (height - 3 + 2) / 2 + 1; // stride-2 conv output
        let w1 = (width - 3 + 2) / 2 + 1;

        let conv1 = nn::Conv2d::new(g, "encoder.conv1", channels, out_ch1, 3, height, width, 2, 1);
        let conv2 = nn::Conv2d::new(g, "encoder.conv2", out_ch1, out_ch2, 3, h1, w1, 2, 1);

        let fc = nn::Linear::no_bias(g, "encoder.fc", out_ch2 as usize, latent_dim);

        Self {
            conv1,
            conv2,
            fc,
            batch,
            pool_channels: out_ch2,
        }
    }

    /// Forward pass: flat NCHW input → latent `[batch, latent_dim]`.
    pub fn forward(&self, g: &mut Graph, obs: NodeId) -> NodeId {
        let h = self.conv1.forward(g, obs, self.batch);
        let h = g.relu(h);
        let h = self.conv2.forward(g, h, self.batch);
        let h = g.relu(h);
        // global_avg_pool: [batch * channels * spatial] -> [batch * channels]
        let spatial = {
            let shape = &g.node(h).ty.shape;
            (shape[0] / (self.batch as usize * self.pool_channels as usize)) as u32
        };
        let h = g.global_avg_pool(h, self.batch, self.pool_channels, spatial);
        self.fc.forward(g, h)
    }
}
