//! Top-level Agent struct and training loop.
//!
//! The agent orchestrates the full pipeline:
//! observation → encoder → world model → reward → credit → policy → action.
//!
//! Three GPU sessions with independent learning rates:
//! 1. World model (encoder + world model): base LR
//! 2. Credit assigner: 0.3× base
//! 3. Policy + value: 0.5× base, gated on warmup
//!
//! Phase 4 continual learning mechanisms:
//! - Replay mixing: each step additionally trains on a random past sample
//!   (with probability `replay_ratio`) to prevent catastrophic forgetting
//! - Representation drift monitor: encoder output on a fixed probe set is
//!   compared against a reference snapshot; large drift reduces encoder LR
//! - Entropy floor: policy updates are suppressed when entropy drops below
//!   a floor, preserving exploration

use crate::OptLevel;
use crate::buffer::{ExperienceBuffer, Transition};
use crate::credit;
use crate::encoder::Encoder;
use crate::env::{Action, ActionKind, Environment, Observation};
use crate::policy;
use crate::reward::{RewardCircuit, RewardWeights};
use crate::world_model::WorldModel;
use meganeura::Session;
use meganeura::graph::Graph;
use rand::Rng;

/// Agent configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub obs_dim: usize,
    pub action_dim: usize,
    /// Kind of action space (discrete categorical or continuous Gaussian).
    pub action_kind: ActionKind,
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub history_len: usize,
    pub buffer_capacity: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub lr_credit: f32,
    pub lr_policy: f32,
    pub reward_weights: RewardWeights,
    pub warmup_steps: usize,
    /// Probability of running an additional replay training step per observe().
    pub replay_ratio: f32,
    pub grid_resolution: f32,
    pub entropy_beta: f32,
    /// Floor for policy entropy — updates suppressed when entropy falls below this.
    pub entropy_floor: f32,
    /// Step interval between representation drift measurements.
    pub drift_interval: usize,
    /// Drift threshold beyond which encoder LR is reduced.
    pub drift_threshold: f32,
    pub opt_level: OptLevel,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            obs_dim: 25,
            action_dim: 4,
            action_kind: ActionKind::Discrete { n: 4 },
            latent_dim: 16,
            hidden_dim: 32,
            history_len: 16,
            buffer_capacity: 10_000,
            batch_size: 1,
            learning_rate: 1e-3,
            lr_credit: 3e-4, // 0.3× base
            lr_policy: 5e-4, // 0.5× base
            reward_weights: RewardWeights::default(),
            warmup_steps: 100,
            replay_ratio: 0.2,
            grid_resolution: 0.5,
            entropy_beta: 0.01,
            entropy_floor: 0.1, // minimum entropy before policy updates are gated
            drift_interval: 500,
            drift_threshold: 1.0,
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
    pub loss_policy: f32,
    pub loss_replay: f32,
    pub reward_mean: f32,
    pub reward_surprise: f32,
    pub reward_novelty: f32,
    pub reward_homeo: f32,
    pub reward_order: f32,
    pub h_eff: f32,
    pub policy_entropy: f32,
    pub repr_drift: f32,
    pub buffer_len: usize,
}

/// The IRIS agent.
pub struct Agent {
    pub config: AgentConfig,
    buffer: ExperienceBuffer,
    reward_circuit: RewardCircuit,
    wm_session: Session,
    credit_session: Session,
    policy_session: Session,
    latent_size: usize,
    step_count: usize,
    /// Fixed set of probe observations for drift monitoring (set after warmup).
    probe_obs: Option<Vec<Vec<f32>>>,
    /// Reference latents from the encoder at the moment the probe was fixed.
    probe_reference: Option<Vec<Vec<f32>>>,
    last_wm_loss: f32,
    last_credit_loss: f32,
    last_policy_loss: f32,
    last_replay_loss: f32,
    last_surprise: f32,
    last_novelty: f32,
    last_homeo: f32,
    last_order: f32,
    last_reward: f32,
    last_entropy: f32,
    last_drift: f32,
    /// Current encoder LR scale (reduced when drift is large).
    encoder_lr_scale: f32,
    /// Most recent logits from the policy network (for action sampling).
    last_logits: Vec<f32>,
    /// Most recent value estimate.
    last_value: f32,
}

/// Xavier (Glorot uniform) initialization.
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

fn build_session(g: &Graph, opt_level: OptLevel) -> Session {
    match opt_level {
        OptLevel::Full => meganeura::build_session(g),
        OptLevel::None => meganeura::build_session_unoptimized(g),
    }
}

impl Agent {
    /// Build the agent: three GPU sessions, Xavier init.
    pub fn new(config: AgentConfig) -> Self {
        // --- World model graph ---
        let wm_session = {
            let mut g = Graph::new();
            let obs = g.input("obs", &[config.batch_size, config.obs_dim]);
            let action = g.input("action", &[config.batch_size, config.action_dim]);
            let z_target = g.input("z_target", &[config.batch_size, config.latent_dim]);

            let enc = Encoder::new(&mut g, config.obs_dim, config.latent_dim, config.hidden_dim);
            let z_t = enc.forward(&mut g, obs);
            let wm = WorldModel::new(
                &mut g,
                config.latent_dim,
                config.action_dim,
                config.hidden_dim,
            );
            let z_hat = wm.forward(&mut g, z_t, action);
            let loss = WorldModel::loss(&mut g, z_hat, z_target);

            g.set_outputs(vec![loss, z_t, z_hat]);
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        // --- Credit assigner graph ---
        let credit_session = {
            let g = credit::build_credit_graph(
                config.latent_dim,
                config.action_dim,
                config.history_len,
                config.hidden_dim,
            );
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        // --- Policy + value graph (discrete or continuous) ---
        let policy_session = {
            let g = match config.action_kind {
                ActionKind::Discrete { .. } => policy::build_policy_graph(
                    config.latent_dim,
                    config.action_dim,
                    config.hidden_dim,
                ),
                ActionKind::Continuous { .. } => policy::build_continuous_policy_graph(
                    config.latent_dim,
                    config.action_dim,
                    config.hidden_dim,
                ),
            };
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        let latent_size = config.batch_size * config.latent_dim;

        Self {
            buffer: ExperienceBuffer::new(config.buffer_capacity, config.grid_resolution),
            reward_circuit: RewardCircuit::new(config.reward_weights.clone()),
            wm_session,
            credit_session,
            policy_session,
            latent_size,
            step_count: 0,
            probe_obs: None,
            probe_reference: None,
            last_wm_loss: 0.0,
            last_credit_loss: 0.0,
            last_policy_loss: 0.0,
            last_replay_loss: 0.0,
            last_surprise: 0.0,
            last_novelty: 0.0,
            last_homeo: 0.0,
            last_order: 0.0,
            last_reward: 0.0,
            last_entropy: 0.0,
            last_drift: 0.0,
            encoder_lr_scale: 1.0,
            last_logits: vec![0.0; config.action_dim],
            last_value: 0.0,
            config,
        }
    }

    /// Select an action using the policy network.
    pub fn act<R: Rng>(&mut self, _obs: &Observation, rng: &mut R) -> Action {
        let z_t = if let Some(prev) = self.buffer.last() {
            prev.latent.clone()
        } else {
            vec![0.0f32; self.latent_size]
        };

        self.policy_session.set_input("z", &z_t);
        self.policy_session
            .set_input("action", &vec![0.0f32; self.config.action_dim]);
        self.policy_session.set_input("value_target", &[0.0f32]);
        self.policy_session.set_learning_rate(0.0);
        self.policy_session.step();
        self.policy_session.wait();

        // Output 1 is logits (discrete) or mean (continuous), both shape [1, action_dim]
        let mut head_out = vec![0.0f32; self.config.action_dim];
        self.policy_session.read_output_by_index(1, &mut head_out);
        let mut value = [0.0f32; 1];
        self.policy_session.read_output_by_index(2, &mut value);

        self.last_logits = head_out.clone();
        self.last_value = value[0];

        match self.config.action_kind {
            ActionKind::Discrete { .. } => {
                self.last_entropy = policy::entropy(&head_out);
                Action::Discrete(policy::sample_action(&head_out, rng))
            }
            ActionKind::Continuous { scale, .. } => {
                // For continuous, use fixed exploration noise as "entropy" proxy
                self.last_entropy = scale;
                Action::Continuous(policy::sample_gaussian_action(&head_out, scale, rng))
            }
        }
    }

    /// Observe a transition, train all modules.
    pub fn observe<R: Rng>(
        &mut self,
        obs: &Observation,
        action: &Action,
        env: &dyn Environment,
        rng: &mut R,
    ) {
        // --- World model on current step ---
        // For discrete actions, one-hot encode. For continuous, use the
        // action vector directly.
        let action_vec = match self.config.action_kind {
            ActionKind::Discrete { .. } => action.to_one_hot(self.config.action_dim),
            ActionKind::Continuous { .. } => match *action {
                Action::Continuous(ref v) => v.clone(),
                Action::Discrete(_) => panic!("continuous action space got discrete action"),
            },
        };
        let z_target = if let Some(prev) = self.buffer.last() {
            prev.latent.clone()
        } else {
            vec![0.0f32; self.latent_size]
        };

        let (wm_loss, z_t, z_hat) = self.wm_forward_backward(
            &obs.data,
            &action_vec,
            &z_target,
            self.config.learning_rate * self.encoder_lr_scale,
        );
        self.last_wm_loss = wm_loss;

        // --- Reward ---
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
        let order = RewardCircuit::order(&obs.data);
        let reward = self.reward_circuit.compute(surprise, novelty, homeo, order);

        // --- Store transition ---
        self.buffer.push(Transition {
            observation: obs.data.clone(),
            latent: z_t.clone(),
            action: action_vec.clone(),
            reward,
            credit: 0.0,
            pred_error,
        });

        // --- Credit assignment ---
        if self.buffer.len() >= self.config.history_len {
            self.credit_step(rng);
        }

        // --- Policy + value training (gated on warmup + entropy floor) ---
        if self.step_count >= self.config.warmup_steps
            && self.last_entropy >= self.config.entropy_floor
        {
            self.policy_step(&z_t, &action_vec, reward);
        }

        // --- Replay mixing: occasionally train on a historical sample ---
        if rng.random_range(0.0..1.0) < self.config.replay_ratio {
            self.replay_step(rng);
        }

        // --- Representation drift monitor ---
        if self.step_count == self.config.warmup_steps && self.probe_obs.is_none() {
            self.capture_probe_reference();
        }
        if self.step_count > 0 && self.step_count.is_multiple_of(self.config.drift_interval) {
            self.measure_drift();
        }

        self.last_surprise = surprise;
        self.last_novelty = novelty;
        self.last_homeo = homeo;
        self.last_order = order;
        self.last_reward = reward;
        self.step_count += 1;
    }

    /// Run one world-model forward+backward pass.
    /// Returns (loss, z_t, z_hat).
    fn wm_forward_backward(
        &mut self,
        obs: &[f32],
        action: &[f32],
        z_target: &[f32],
        lr: f32,
    ) -> (f32, Vec<f32>, Vec<f32>) {
        self.wm_session.set_input("obs", obs);
        self.wm_session.set_input("action", action);
        self.wm_session.set_input("z_target", z_target);
        self.wm_session.set_learning_rate(lr);
        self.wm_session.step();
        self.wm_session.wait();

        let loss = self.wm_session.read_loss();
        let mut z_t = vec![0.0f32; self.latent_size];
        self.wm_session.read_output_by_index(1, &mut z_t);
        let mut z_hat = vec![0.0f32; self.latent_size];
        self.wm_session.read_output_by_index(2, &mut z_hat);
        (loss, z_t, z_hat)
    }

    /// Replay mixing: sample a past transition and run an extra world-model
    /// training step on it. This re-anchors the encoder + world model to
    /// historical experience, reducing catastrophic forgetting.
    fn replay_step<R: Rng>(&mut self, rng: &mut R) {
        if self.buffer.len() < 2 {
            return;
        }
        // Sample an older transition and its successor (for z_target)
        let n = self.buffer.len();
        let i = rng.random_range(0..n - 1);
        let ti = self.buffer.get(i);
        let tj = self.buffer.get(i + 1);
        let obs = ti.observation.clone();
        let action = ti.action.clone();
        let z_target = tj.latent.clone();

        let (loss, _, _) = self.wm_forward_backward(
            &obs,
            &action,
            &z_target,
            self.config.learning_rate * self.encoder_lr_scale * 0.5, // smaller step for replay
        );
        self.last_replay_loss = loss;
    }

    /// Capture a fixed probe set of observations and their reference latents.
    fn capture_probe_reference(&mut self) {
        let n_probe = 16.min(self.buffer.len());
        if n_probe == 0 {
            return;
        }
        let step = self.buffer.len() / n_probe.max(1);
        let mut observations = Vec::with_capacity(n_probe);
        let mut references = Vec::with_capacity(n_probe);
        for i in 0..n_probe {
            let idx = i * step;
            if idx < self.buffer.len() {
                let t = self.buffer.get(idx);
                observations.push(t.observation.clone());
                references.push(t.latent.clone());
            }
        }
        self.probe_obs = Some(observations);
        self.probe_reference = Some(references);
    }

    /// Measure how far the current encoder drifts from the reference.
    /// Reduces `encoder_lr_scale` if drift exceeds threshold.
    fn measure_drift(&mut self) {
        let (probes, references) = match (self.probe_obs.as_ref(), self.probe_reference.as_ref()) {
            (Some(p), Some(r)) => (p.clone(), r.clone()),
            _ => return,
        };

        let mut total = 0.0f32;
        let mut count = 0;

        let zero_action = vec![0.0f32; self.config.action_dim];
        let zero_target = vec![0.0f32; self.latent_size];

        for (obs, reference) in probes.iter().zip(references.iter()) {
            // Forward-only with lr=0 to get current encoder output
            self.wm_session.set_input("obs", obs);
            self.wm_session.set_input("action", &zero_action);
            self.wm_session.set_input("z_target", &zero_target);
            self.wm_session.set_learning_rate(0.0);
            self.wm_session.step();
            self.wm_session.wait();

            let mut current = vec![0.0f32; self.latent_size];
            self.wm_session.read_output_by_index(1, &mut current);

            let dist: f32 = current
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if dist.is_finite() {
                total += dist;
                count += 1;
            }
        }

        let drift = if count > 0 { total / count as f32 } else { 0.0 };
        self.last_drift = drift;

        // Reduce encoder LR if drift exceeds threshold, recover slowly otherwise
        if drift > self.config.drift_threshold {
            self.encoder_lr_scale = (self.encoder_lr_scale * 0.5).max(0.01);
        } else {
            self.encoder_lr_scale = (self.encoder_lr_scale * 1.1).min(1.0);
        }
    }

    /// Credit assignment step.
    fn credit_step<R: Rng>(&mut self, rng: &mut R) {
        let h = self.config.history_len;

        if let Some(history_flat) = self.buffer.flatten_history(h) {
            // Clamp all values to prevent NaN propagation
            let history_clean: Vec<f32> = history_flat
                .iter()
                .map(|v| {
                    if v.is_finite() {
                        v.clamp(-5.0, 5.0)
                    } else {
                        0.0
                    }
                })
                .collect();
            self.credit_session.set_input("history", &history_clean);

            let target = if let Some((hi, lo)) =
                self.buffer
                    .find_contrastive_pair(rng, h, self.config.latent_dim)
            {
                self.buffer.contrastive_target(hi, lo, h)
            } else {
                vec![1.0 / h as f32; h]
            };
            let target_clean: Vec<f32> = target
                .iter()
                .map(|v| if v.is_finite() { *v } else { 1.0 / h as f32 })
                .collect();
            self.credit_session
                .set_input("credit_target", &target_clean);

            self.credit_session.set_learning_rate(self.config.lr_credit);
            self.credit_session.step();
            self.credit_session.wait();

            self.last_credit_loss = self.credit_session.read_loss();

            let mut credit_logits = vec![0.0f32; h];
            self.credit_session
                .read_output_by_index(1, &mut credit_logits);

            let alpha = credit::softmax(&credit_logits);
            let r_t = self.last_reward;
            let credits: Vec<f32> = alpha.iter().map(|&a| r_t * a).collect();
            self.buffer.write_credits(&credits);
        }
    }

    /// Policy + value training step.
    fn policy_step(&mut self, z_t: &[f32], action_vec: &[f32], reward: f32) {
        self.policy_session.set_input("z", z_t);
        self.policy_session.set_input("action", action_vec);

        let value_target = reward;
        self.policy_session
            .set_input("value_target", &[value_target]);

        let advantage = reward - self.last_value;

        // Only reinforce when advantage > 0 (good actions). Negative-advantage
        // updates via negative LR are numerically unstable.
        if advantage > 0.0 {
            let scale = advantage.min(1.0);
            self.policy_session
                .set_learning_rate(self.config.lr_policy * scale);
        } else {
            // Small lr to keep the value head learning via MSE
            self.policy_session
                .set_learning_rate(self.config.lr_policy * 0.1);
        }
        self.policy_session.step();
        self.policy_session.wait();

        self.last_policy_loss = self.policy_session.read_loss();
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
            loss_policy: self.last_policy_loss,
            loss_replay: self.last_replay_loss,
            reward_mean: self.last_reward,
            reward_surprise: self.last_surprise,
            reward_novelty: self.last_novelty,
            reward_homeo: self.last_homeo,
            reward_order: self.last_order,
            h_eff,
            policy_entropy: self.last_entropy,
            repr_drift: self.last_drift,
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
