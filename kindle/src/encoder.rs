//! Encoder: converts raw observations into a compact latent representation `z_t`.
//!
//! The encoder is the shared backbone. All other modules consume `z_t`,
//! not raw observations. Training signals flow back from the world model
//! (primary), policy gradient (secondary), and value head TD error (secondary).
//!
//! Two variants:
//! - **MLP**: for structured (feature vector) observations.
//! - **CNN**: for pixel observations (NCHW flat layout).

use meganeura::graph::{Graph, NodeId};
use meganeura::nn;

/// MLP-based encoder for structured (feature vector) observations.
///
/// Takes both the observation token and a per-env task embedding, summing
/// their projections at the hidden layer. This is mathematically
/// equivalent to concatenating `[obs, task]` and applying one wide linear
/// layer, but avoids needing a general concat op.
pub struct Encoder {
    pub obs_proj: nn::Linear,
    pub task_proj: nn::Linear,
    pub norm: nn::RmsNorm,
    pub fc2: nn::Linear,
}

impl Encoder {
    /// Build the encoder parameters in the graph.
    pub fn new(
        g: &mut Graph,
        obs_dim: usize,
        task_dim: usize,
        latent_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            obs_proj: nn::Linear::new(g, "encoder.obs_proj", obs_dim, hidden_dim),
            task_proj: nn::Linear::no_bias(g, "encoder.task_proj", task_dim, hidden_dim),
            norm: nn::RmsNorm::new(g, "encoder.norm.weight", hidden_dim, 1e-5),
            fc2: nn::Linear::no_bias(g, "encoder.fc2", hidden_dim, latent_dim),
        }
    }

    /// Forward pass: obs `[batch, obs_dim]` + task `[batch, task_dim]`
    /// → `[batch, latent_dim]`.
    pub fn forward(&self, g: &mut Graph, obs: NodeId, task: NodeId) -> NodeId {
        let h_obs = self.obs_proj.forward(g, obs);
        let h_task = self.task_proj.forward(g, task);
        let h = g.add(h_obs, h_task);
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
pub struct CnnEncoder {
    pub conv1: nn::Conv2d,
    pub conv2: nn::Conv2d,
    pub fc: nn::Linear,
    pub batch: u32,
    pub pool_channels: u32,
}

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

        let conv1 = nn::Conv2d::new(
            g,
            "encoder.conv1",
            channels,
            out_ch1,
            3,
            height,
            width,
            2,
            1,
        );
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

/// Nature-DQN-scale CNN encoder for visual RL (Mnih et al. 2015).
///
/// Architecture (designed for 84×84×4 frame-stacked Atari input):
/// ```text
/// input  : N × 4 × 84 × 84
/// conv1  : 32 filters, 8×8, stride 4  →  N × 32 × 20 × 20
/// relu
/// conv2  : 64 filters, 4×4, stride 2  →  N × 64 × 9 × 9
/// relu
/// conv3  : 64 filters, 3×3, stride 1  →  N × 64 × 7 × 7
/// relu
/// flatten:                              →  N × 3136
/// fc1    : 512 units                    →  N × 512
/// relu
/// fc2    : latent_dim                   →  N × latent_dim
/// ```
/// ~1.7M parameters. Preserves spatial structure (no global pool —
/// the small CnnEncoder pools to 16 dims and is unsuitable for any
/// task requiring spatial reasoning like Atari).
pub struct CnnEncoderDqn {
    pub conv1: nn::Conv2d,
    pub conv2: nn::Conv2d,
    pub conv3: nn::Conv2d,
    pub fc1: nn::Linear,
    pub fc2: nn::Linear,
    pub batch: u32,
    pub flat_dim: usize,
}

impl CnnEncoderDqn {
    pub fn new(
        g: &mut Graph,
        channels: u32,
        height: u32,
        width: u32,
        latent_dim: usize,
        batch: u32,
    ) -> Self {
        // Compute layer-by-layer output sizes (no padding, integer math).
        let h1 = (height - 8) / 4 + 1;
        let w1 = (width - 8) / 4 + 1;
        let h2 = (h1 - 4) / 2 + 1;
        let w2 = (w1 - 4) / 2 + 1;
        let h3 = h2 - 3 + 1;
        let w3 = w2 - 3 + 1;
        let out_ch3 = 64u32;
        let flat_dim = (out_ch3 as usize) * (h3 as usize) * (w3 as usize);

        let conv1 = nn::Conv2d::new(g, "encoder.conv1", channels, 32, 8, height, width, 4, 0);
        let conv2 = nn::Conv2d::new(g, "encoder.conv2", 32, 64, 4, h1, w1, 2, 0);
        let conv3 = nn::Conv2d::new(g, "encoder.conv3", 64, out_ch3, 3, h2, w2, 1, 0);
        let fc1 = nn::Linear::new(g, "encoder.fc1", flat_dim, 512);
        let fc2 = nn::Linear::no_bias(g, "encoder.fc2", 512, latent_dim);

        Self {
            conv1,
            conv2,
            conv3,
            fc1,
            fc2,
            batch,
            flat_dim,
        }
    }

    pub fn forward(&self, g: &mut Graph, obs: NodeId) -> NodeId {
        let h = self.conv1.forward(g, obs, self.batch);
        let h = g.relu(h);
        let h = self.conv2.forward(g, h, self.batch);
        let h = g.relu(h);
        let h = self.conv3.forward(g, h, self.batch);
        let h = g.relu(h);
        // The conv output is a flat [batch * out_ch3 * h3 * w3] tensor.
        // For the FC layer we need [batch, flat_dim] — same flat layout
        // works since meganeura's Linear treats the input as
        // [batch, in_features] when in_features = flat_dim. No reshape op
        // required.
        let h = self.fc1.forward(g, h);
        let h = g.relu(h);
        self.fc2.forward(g, h)
    }
}
