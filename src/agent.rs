//! Top-level Agent struct and training loop.
//!
//! The agent orchestrates the full pipeline:
//! observation -> encoder -> world model -> reward -> credit -> buffer -> train.

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
    pub lr_credit: f32,
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
            lr_credit: 3e-4, // 0.3x base
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
    pub loss_credit: f32,
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
    wm_session: Session,
    credit_session: Session,
    latent_size: usize,
    step_count: usize,
    last_wm_loss: f32,
    last_credit_loss: f32,
    last_surprise: f32,
    last_novelty: f32,
    last_homeo: f32,
    last_reward: f32,
}

/// Xavier (Glorot uniform) initialization for a weight matrix.
fn xavier_init(fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let n = fan_in * fan_out;
    (0..n)
        .map(|i| {
            let x = ((i as f64 + seed as f64) * 0.618_033_988_749_895).fract() as f32;
            (x * PI * 2.0).sin() * scale
        })
        .collect()
}

impl Agent {
    /// Build the agent: construct graphs, compile to GPU sessions, init params.
    ///
    /// Two sessions:
    /// 1. **World model session**: encoder → world model → MSE loss.
    ///    Trains encoder + world model at `learning_rate`.
    /// 2. **Credit session**: causal attention credit assigner → MSE loss.
    ///    Trains credit assigner at `lr_credit` (0.3× base).
    pub fn new(config: AgentConfig) -> Self {
        // --- World model graph ---
        let wm_session = {
            let mut g = Graph::new();
            let obs_input = g.input("obs", &[config.batch_size, config.obs_dim]);
            let action_input = g.input("action", &[config.batch_size, config.action_dim]);
            let z_target_input = g.input("z_target", &[config.batch_size, config.latent_dim]);

            let encoder =
                Encoder::new(&mut g, config.obs_dim, config.latent_dim, config.hidden_dim);
            let z_t = encoder.forward(&mut g, obs_input);

            let world_model =
                WorldModel::new(&mut g, config.latent_dim, config.action_dim, config.hidden_dim);
            let z_hat = world_model.forward(&mut g, z_t, action_input);
            let wm_loss = WorldModel::loss(&mut g, z_hat, z_target_input);

            g.set_outputs(vec![wm_loss, z_t, z_hat]);

            let mut session = match config.opt_level {
                OptLevel::Full => meganeura::build_session(&g),
                OptLevel::None => meganeura::build_session_unoptimized(&g),
            };
            init_parameters(&mut session);
            session
        };

        // --- Credit assigner graph ---
        let credit_session = {
            let g = credit::build_credit_graph(
                config.latent_dim,
                config.action_dim,
                config.history_len,
                config.hidden_dim,
            );
            let mut session = match config.opt_level {
                OptLevel::Full => meganeura::build_session(&g),
                OptLevel::None => meganeura::build_session_unoptimized(&g),
            };
            init_parameters(&mut session);
            session
        };

        let latent_size = config.batch_size * config.latent_dim;

        Self {
            buffer: ExperienceBuffer::new(config.buffer_capacity, config.grid_resolution),
            reward_circuit: RewardCircuit::new(config.reward_weights.clone()),
            wm_session,
            credit_session,
            latent_size,
            step_count: 0,
            last_wm_loss: 0.0,
            last_credit_loss: 0.0,
            last_surprise: 0.0,
            last_novelty: 0.0,
            last_homeo: 0.0,
            last_reward: 0.0,
            config,
        }
    }

    /// Select an action. Phase 2: still random (policy not yet trained).
    pub fn act<R: Rng>(&self, rng: &mut R) -> Action {
        Action::Discrete(rng.gen_range(0..self.config.action_dim))
    }

    /// Observe a transition, train world model, and run credit assignment.
    pub fn observe<R: Rng>(
        &mut self,
        obs: &Observation,
        action: &Action,
        env: &dyn Environment,
        rng: &mut R,
    ) {
        // --- World model forward + backward ---
        self.wm_session.set_input("obs", &obs.data);

        let action_vec = action.to_one_hot(self.config.action_dim);
        self.wm_session.set_input("action", &action_vec);

        let z_target = if let Some(prev) = self.buffer.last() {
            prev.latent.clone()
        } else {
            vec![0.0f32; self.latent_size]
        };
        self.wm_session.set_input("z_target", &z_target);

        self.wm_session.set_learning_rate(self.config.learning_rate);
        self.wm_session.step();
        self.wm_session.wait();

        self.last_wm_loss = self.wm_session.read_loss();

        let mut z_t = vec![0.0f32; self.latent_size];
        self.wm_session.read_output_by_index(1, &mut z_t);

        let mut z_hat = vec![0.0f32; self.latent_size];
        self.wm_session.read_output_by_index(2, &mut z_hat);

        // --- Compute reward ---
        let pred_error: f32 = z_t
            .iter()
            .zip(z_hat.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let surprise = RewardCircuit::surprise(pred_error);
        let visit_count = self.buffer.visit_count(&z_t);
        let novelty = RewardCircuit::novelty(visit_count);
        let homeo = RewardCircuit::homeostatic(env.homeostatic_variables());
        let reward = self.reward_circuit.compute(surprise, novelty, homeo);

        // --- Store transition ---
        self.buffer.push(Transition {
            observation: obs.data.clone(),
            latent: z_t,
            action: action_vec,
            reward,
            credit: 0.0,
            pred_error,
        });

        // --- Credit assignment (once enough history) ---
        if self.buffer.len() >= self.config.history_len {
            self.credit_step(rng);
        }

        self.last_surprise = surprise;
        self.last_novelty = novelty;
        self.last_homeo = homeo;
        self.last_reward = reward;
        self.step_count += 1;
    }

    /// Run one credit assignment step: forward pass on recent history,
    /// contrastive training, and write credits back to buffer.
    fn credit_step<R: Rng>(&mut self, rng: &mut R) {
        let h = self.config.history_len;

        // --- Forward pass: get credit weights for recent history ---
        if let Some(history_flat) = self.buffer.flatten_history(h) {
            self.credit_session.set_input("history", &history_flat);

            // --- Contrastive training target ---
            let target = if let Some((hi, lo)) =
                self.buffer
                    .find_contrastive_pair(rng, h, self.config.latent_dim)
            {
                self.buffer.contrastive_target(hi, lo, h)
            } else {
                // No good pair found — use uniform target (no learning signal)
                vec![1.0 / h as f32; h]
            };
            self.credit_session.set_input("credit_target", &target);

            // --- Train ---
            self.credit_session.set_learning_rate(self.config.lr_credit);
            self.credit_session.step();
            self.credit_session.wait();

            self.last_credit_loss = self.credit_session.read_loss();

            // --- Read credit logits and write back to buffer ---
            let mut credit_logits = vec![0.0f32; h];
            self.credit_session.read_output_by_index(1, &mut credit_logits);

            let alpha = credit::softmax(&credit_logits);
            let r_t = self.last_reward;
            let credits: Vec<f32> = alpha.iter().map(|&a| r_t * a).collect();
            self.buffer.write_credits(&credits);
        }
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let recent = self.buffer.recent_window(self.config.history_len);
        let credit_weights: Vec<f32> = recent.iter().map(|t| t.credit).collect();
        let h_eff = if credit_weights.is_empty() {
            0.0
        } else {
            credit::effective_scope(&credit_weights)
        };

        Diagnostics {
            step: self.step_count,
            loss_world_model: self.last_wm_loss,
            loss_credit: self.last_credit_loss,
            reward_mean: self.last_reward,
            reward_surprise: self.last_surprise,
            reward_novelty: self.last_novelty,
            reward_homeo: self.last_homeo,
            h_eff,
            buffer_len: self.buffer.len(),
        }
    }
}

/// Initialize all parameters with Xavier (Glorot) initialization.
#[allow(clippy::pattern_type_mismatch)]
fn init_parameters(session: &mut Session) {
    let params: Vec<(String, usize)> = session
        .plan()
        .param_buffers
        .iter()
        .map(|(name, buf_ref)| {
            let size_bytes = session.plan().buffers[buf_ref.0 as usize];
            (name.clone(), size_bytes / 4)
        })
        .collect();

    for (i, (name, num_elements)) in params.iter().enumerate() {
        let num_elements = *num_elements;
        if name.ends_with(".bias") || name.ends_with(".weight") && num_elements <= 1 {
            let data = vec![0.0f32; num_elements];
            session.set_parameter(name, &data);
        } else if name.contains("norm") {
            let data = vec![1.0f32; num_elements];
            session.set_parameter(name, &data);
        } else {
            let fan = (num_elements as f32).sqrt() as usize;
            let data = xavier_init(fan, fan, i as u64 * 7919);
            session.set_parameter(name, &data);
        }
    }
}
