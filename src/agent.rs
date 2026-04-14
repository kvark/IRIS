//! Top-level Agent struct and training loop.
//!
//! The agent orchestrates the full pipeline:
//! observation -> encoder -> world model -> reward -> buffer -> train.
//!
//! Phase 0: only the encoder + world model are trained (via MSE loss).
//! The policy outputs random actions. Credit and value are scaffolded
//! but not yet integrated into the training graph.

use crate::buffer::{ExperienceBuffer, Transition};
use crate::credit;
use crate::encoder::Encoder;
use crate::env::{Action, Environment, Observation};
use crate::reward::{RewardCircuit, RewardWeights};
use crate::world_model::WorldModel;
use crate::OptLevel;
use meganeura::graph::Graph;
use meganeura::Session;
use rand::Rng;

/// Agent configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub obs_dim: usize,
    pub action_dim: usize,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub history_len: usize,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub reward_weights: RewardWeights,
    pub warmup_steps: usize,
    pub replay_ratio: f32,
    pub grid_resolution: f32,
    pub opt_level: OptLevel,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            obs_dim: 25,
            action_dim: 4,
            latent_dim: 16,
            hidden_dim: 32,
            history_len: 16,
            buffer_capacity: 10_000,
            batch_size: 1,
            learning_rate: 1e-3,
            reward_weights: RewardWeights::default(),
            warmup_steps: 100,
            replay_ratio: 0.2,
            grid_resolution: 0.5,
            opt_level: OptLevel::Full,
        }
    }
}

/// Diagnostics snapshot for observability.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Diagnostics {
    pub step: usize,
    pub loss_world_model: f32,
    pub reward_mean: f32,
    pub reward_surprise: f32,
    pub reward_novelty: f32,
    pub reward_homeo: f32,
    pub h_eff: f32,
    pub buffer_len: usize,
}

/// The IRIS agent.
pub struct Agent {
    pub config: AgentConfig,
    buffer: ExperienceBuffer,
    reward_circuit: RewardCircuit,
    session: Session,
    /// Latent dimension size (for output read-back).
    latent_size: usize,
    step_count: usize,
    // Running diagnostic accumulators
    last_loss: f32,
    last_surprise: f32,
    last_novelty: f32,
    last_homeo: f32,
    last_reward: f32,
}

impl Agent {
    /// Build the agent: construct graphs, compile to GPU sessions.
    ///
    /// The Phase 0 training graph contains:
    /// - Encoder: obs -> z_t
    /// - World Model: (z_t, action) -> z_hat, MSE(z_hat, z_target)
    ///
    /// The z_t latent is exposed as a second graph output so it can be
    /// read back and stored in the experience buffer.
    pub fn new(config: AgentConfig) -> Self {
        let mut g = Graph::new();

        // --- Graph inputs ---
        let obs_input = g.input("obs", &[config.batch_size, config.obs_dim]);
        let _action_input = g.input("action", &[config.batch_size, config.action_dim]);
        let z_target_input = g.input("z_target", &[config.batch_size, config.latent_dim]);

        // --- Encoder ---
        let encoder = Encoder::new(&mut g, config.obs_dim, config.latent_dim, config.hidden_dim);
        let z_t = encoder.forward(&mut g, obs_input);

        // --- World Model ---
        // Concatenate z_t and action into [batch, latent_dim + action_dim]
        // meganeura's concat is channel-wise for vision; for 2D tensors we
        // use matmul-free concatenation by building two halves.
        // Simple approach: use bias_add-like pattern or just declare a wider input.
        //
        // For Phase 0, we feed the concatenated [z_t; action] as a single
        // input by restructuring: the world model takes a pre-concatenated input.
        // We'll concatenate on CPU and feed as input instead.
        let za_input = g.input("za", &[config.batch_size, config.latent_dim + config.action_dim]);
        let world_model =
            WorldModel::new(&mut g, config.latent_dim, config.action_dim, config.hidden_dim);
        let z_hat = world_model.forward(&mut g, za_input);

        // --- World Model Loss ---
        let wm_loss = WorldModel::loss(&mut g, z_hat, z_target_input);

        // --- Graph outputs ---
        // Output 0: loss (for training)
        // Output 1: z_t (for buffer storage)
        // Output 2: z_hat (for surprise computation)
        g.set_outputs(vec![wm_loss, z_t, z_hat]);

        // --- Compile ---
        let session = match config.opt_level {
            OptLevel::Full => meganeura::build_session(&g),
            OptLevel::None => meganeura::build_session_unoptimized(&g),
        };

        // Note: Policy, ValueHead, and CreditAssigner are scaffolded as
        // standalone modules. They will be integrated into the training
        // graph in later phases when their training signals are ready.
        // For now, they exist as importable structs with working forward()
        // methods that can be demonstrated in separate test graphs.

        let latent_size = config.batch_size * config.latent_dim;

        Self {
            buffer: ExperienceBuffer::new(config.buffer_capacity, config.grid_resolution),
            reward_circuit: RewardCircuit::new(config.reward_weights.clone()),
            session,
            latent_size,
            step_count: 0,
            last_loss: 0.0,
            last_surprise: 0.0,
            last_novelty: 0.0,
            last_homeo: 0.0,
            last_reward: 0.0,
            config,
        }
    }

    /// Select an action given the current observation.
    ///
    /// Phase 0: random discrete action (policy is not yet trained).
    pub fn act<R: Rng>(&self, rng: &mut R) -> Action {
        Action::Discrete(rng.gen_range(0..self.config.action_dim))
    }

    /// Observe a transition and optionally train.
    ///
    /// 1. Encode the observation to get z_t
    /// 2. Compute reward from surprise, novelty, and homeostatic signals
    /// 3. Store transition in buffer
    /// 4. If enough data, run one training step
    pub fn observe(&mut self, obs: &Observation, action: &Action, env: &dyn Environment) {
        // --- Encode observation via the training graph's forward pass ---
        self.session.set_input("obs", &obs.data);

        // For the world model forward pass, we need z_t concatenated with action.
        // In Phase 0 we feed zeros since we're bootstrapping.
        let action_vec = action.to_one_hot(self.config.action_dim);
        let mut za = vec![0.0f32; self.config.latent_dim + self.config.action_dim];
        // z_t part is zeros on first step, filled from buffer on subsequent steps
        if let Some(prev) = self.buffer.last() {
            za[..self.config.latent_dim].copy_from_slice(&prev.latent);
        }
        za[self.config.latent_dim..].copy_from_slice(&action_vec);
        self.session.set_input("za", &za);

        // z_target: use previous latent or zeros
        let z_target = if let Some(prev) = self.buffer.last() {
            prev.latent.clone()
        } else {
            vec![0.0f32; self.config.latent_dim]
        };
        self.session.set_input("z_target", &z_target);

        // --- Run forward + backward ---
        self.session
            .set_learning_rate(self.config.learning_rate);
        self.session.step();
        self.session.wait();

        // --- Read outputs ---
        self.last_loss = self.session.read_loss();

        let mut z_t = vec![0.0f32; self.latent_size];
        self.session.read_output_by_index(1, &mut z_t);

        let mut z_hat = vec![0.0f32; self.latent_size];
        self.session.read_output_by_index(2, &mut z_hat);

        // --- Compute reward ---
        // Surprise: L2 distance between predicted and actual latent
        let pred_error: f32 = z_t
            .iter()
            .zip(z_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let surprise = RewardCircuit::surprise(pred_error);

        // Novelty
        let visit_count = self.buffer.visit_count(&z_t);
        let novelty = RewardCircuit::novelty(visit_count);

        // Homeostatic
        let homeo = RewardCircuit::homeostatic(env.homeostatic_variables());

        // Combined
        let reward = self.reward_circuit.compute(surprise, novelty, homeo);

        // --- Store transition ---
        self.buffer.push(Transition {
            observation: obs.data.clone(),
            latent: z_t,
            action: action_vec,
            reward,
            credit: 0.0, // placeholder — credit assigner not yet wired
            pred_error,
        });

        // --- Update diagnostics ---
        self.last_surprise = surprise;
        self.last_novelty = novelty;
        self.last_homeo = homeo;
        self.last_reward = reward;
        self.step_count += 1;
    }

    /// Current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Snapshot of current diagnostics.
    pub fn diagnostics(&self) -> Diagnostics {
        // Compute H_eff from recent credit weights
        let recent = self.buffer.recent_window(self.config.history_len);
        let credit_weights: Vec<f32> = recent.iter().map(|t| t.credit).collect();
        let h_eff = if credit_weights.is_empty() {
            0.0
        } else {
            credit::effective_scope(&credit_weights)
        };

        Diagnostics {
            step: self.step_count,
            loss_world_model: self.last_loss,
            reward_mean: self.last_reward,
            reward_surprise: self.last_surprise,
            reward_novelty: self.last_novelty,
            reward_homeo: self.last_homeo,
            h_eff,
            buffer_len: self.buffer.len(),
        }
    }
}
