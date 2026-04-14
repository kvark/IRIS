//! Top-level Agent struct and training loop.
//!
//! The agent orchestrates the full pipeline:
//! observation → (adapter) → encoder → world model → reward → credit → policy → (adapter) → action.
//!
//! The agent's GPU graphs are built once with universal token sizes
//! (`OBS_TOKEN_DIM`, `MAX_ACTION_DIM`); a per-env `EnvAdapter` translates
//! between the env's native shapes and these token sizes. `switch_env`
//! swaps the active adapter without touching any compiled graph.
//!
//! Three GPU sessions with independent learning rates:
//! 1. World model (encoder + world model): base LR
//! 2. Credit assigner: 0.3× base
//! 3. Policy + value: 0.5× base, gated on warmup
//!
//! Phase 4 continual learning mechanisms:
//! - Replay mixing
//! - Representation drift monitor
//! - Entropy floor

use crate::OptLevel;
use crate::adapter::{EnvAdapter, MAX_ACTION_DIM, OBS_TOKEN_DIM, TASK_DIM};
use crate::buffer::{ExperienceBuffer, Transition};
use crate::credit;
use crate::encoder::Encoder;
use crate::env::{Action, Environment, Observation};
use crate::policy;
use crate::reward::{RewardCircuit, RewardWeights};
use crate::world_model::WorldModel;
use hashbrown::HashMap;
use meganeura::Session;
use meganeura::graph::Graph;
use rand::Rng;

/// Agent configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
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
            entropy_floor: 0.1,
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
    pub env_id: u32,
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
    adapter: Box<dyn EnvAdapter>,
    /// Per-env task embedding (key = env_id). Each env gets a fixed
    /// deterministic-random vector based on its id; we feed the active
    /// one into the encoder as a graph input each step. Not trained
    /// (the encoder learns to map (obs, env_embedding) into per-env
    /// latents).
    task_embeddings: HashMap<u32, Vec<f32>>,
    /// True for exactly one step after `switch_env`, so the next stored
    /// transition is tagged as an env boundary.
    pending_boundary: bool,
    buffer: ExperienceBuffer,
    reward_circuit: RewardCircuit,
    wm_session: Session,
    credit_session: Session,
    /// Policy graph always uses the continuous branch (MSE loss on Gaussian
    /// means). For discrete envs, the adapter softmax+samples over the
    /// first `n` head dims. This gives one universal policy graph.
    policy_session: Session,
    latent_size: usize,
    step_count: usize,
    probe_obs: Option<Vec<Vec<f32>>>,
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
    encoder_lr_scale: f32,
    last_value: f32,
    /// Scratch buffers to avoid per-step allocations.
    obs_token_scratch: Vec<f32>,
    action_token_scratch: Vec<f32>,
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
    /// Build the agent for a specific environment's adapter. The underlying
    /// graphs use universal token sizes and will never need rebuilding for
    /// subsequent env switches.
    pub fn new(config: AgentConfig, adapter: Box<dyn EnvAdapter>) -> Self {
        // --- World model graph (uses universal token sizes + task) ---
        //
        // The task embedding is fed as a graph **input** named "task".
        // Per-env values are deterministic-random and persisted CPU-side
        // in `task_embeddings`. The encoder learns to map (obs_token,
        // task) into per-env latents.
        let wm_session = {
            let mut g = Graph::new();
            let obs = g.input("obs", &[config.batch_size, OBS_TOKEN_DIM]);
            let action = g.input("action", &[config.batch_size, MAX_ACTION_DIM]);
            let z_target = g.input("z_target", &[config.batch_size, config.latent_dim]);
            // Task embedding is fed as a graph **input**, not a parameter.
            // The encoder sees per-env conditioning and can specialize its
            // representations, but we don't backprop into the embedding
            // itself (meganeura's autodiff over a parameter on this code
            // path is unstable). Each env's embedding is a fixed
            // deterministic-random vector keyed off the env_id; the
            // encoder learns to map (obs_token, env_embedding) into
            // env-specific latents.
            let task = g.input("task", &[config.batch_size, TASK_DIM]);

            let enc = Encoder::new(
                &mut g,
                OBS_TOKEN_DIM,
                TASK_DIM,
                config.latent_dim,
                config.hidden_dim,
            );
            let z_t = enc.forward(&mut g, obs, task);
            let wm = WorldModel::new(&mut g, config.latent_dim, MAX_ACTION_DIM, config.hidden_dim);
            let z_hat = wm.forward(&mut g, z_t, action);
            let loss = WorldModel::loss(&mut g, z_hat, z_target);

            // Note: we only expose `loss` and `z_t` as outputs.
            // `z_hat` is intentionally NOT an output — meganeura reuses
            // the intermediate buffer of any internal node that's also
            // exposed as an output for backward-pass scratch, which gives
            // back garbage on read. We only need z_t (for the buffer's
            // latent column); surprise is recovered from `sqrt(wm_loss
            // * latent_dim)` instead of `||z_t - z_hat||`.
            g.set_outputs(vec![loss, z_t]);
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        // --- Credit assigner graph ---
        let credit_session = {
            let g = credit::build_credit_graph(
                config.latent_dim,
                MAX_ACTION_DIM,
                config.history_len,
                config.hidden_dim,
            );
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        // --- Policy + value graph ---
        //
        // We always build the continuous-Gaussian graph (MSE loss against
        // the action token). For discrete envs the adapter interprets the
        // first `n` head dims as logits and samples categorically at act()
        // time; during training the taken action is encoded as a one-hot
        // in the action token, so MSE against one-hot has the same
        // gradient sign as cross-entropy: minimizing MSE with a one-hot
        // target pushes the mean toward the taken dim. It's not strictly
        // equivalent to cross-entropy but it's numerically stable and
        // unifies the discrete/continuous training path into a single
        // graph — a big simplification for env hopping.
        let policy_session = {
            let g = policy::build_continuous_policy_graph(
                config.latent_dim,
                MAX_ACTION_DIM,
                config.hidden_dim,
            );
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        let latent_size = config.batch_size * config.latent_dim;
        let task_size = config.batch_size * TASK_DIM;

        // Initialize this env's task embedding to its deterministic seed.
        let initial_task = embedding_for(adapter.id(), task_size);
        let mut task_embeddings = HashMap::new();
        task_embeddings.insert(adapter.id(), initial_task);

        Self {
            task_embeddings,
            adapter,
            pending_boundary: false,
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
            last_value: 0.0,
            obs_token_scratch: vec![0.0; config.batch_size * OBS_TOKEN_DIM],
            action_token_scratch: vec![0.0; config.batch_size * MAX_ACTION_DIM],
            config,
        }
    }

    /// Swap the active environment adapter. Preserves all learned
    /// parameters; the next transition stored is marked as an env
    /// boundary so the world model and credit assigner don't try to
    /// attribute dynamics or reward across the switch.
    ///
    /// Per-env state swap:
    /// 1. Read the current task embedding out of `wm_session` and save
    ///    it under the *outgoing* env's id.
    /// 2. Look up the incoming env's saved embedding (or initialize a
    ///    new one to the mean of all existing embeddings, so a fresh
    ///    env warm-starts from related learned ones).
    /// 3. Upload the new embedding to `wm_session`.
    pub fn switch_env(&mut self, adapter: Box<dyn EnvAdapter>) {
        let incoming_id = adapter.id();
        let task_size = self.config.batch_size * TASK_DIM;

        // Generate the new env's deterministic embedding if we haven't seen
        // it before. Re-entering an env will reuse the same vector (its
        // env_id determines it), which keeps the encoder's per-env
        // specialization consistent across hops.
        self.task_embeddings
            .entry(incoming_id)
            .or_insert_with(|| embedding_for(incoming_id, task_size));

        self.adapter = adapter;
        self.pending_boundary = true;
    }

    /// Currently active env id.
    pub fn env_id(&self) -> u32 {
        self.adapter.id()
    }

    /// Select an action using the policy network.
    pub fn act<R: Rng>(&mut self, _obs: &Observation, rng: &mut R) -> Action {
        let z_t = if let Some(prev) = self.buffer.last() {
            prev.latent.clone()
        } else {
            vec![0.0f32; self.latent_size]
        };

        self.policy_session.set_input("z", &z_t);
        self.action_token_scratch.fill(0.0);
        self.policy_session
            .set_input("action", &self.action_token_scratch);
        self.policy_session.set_input("value_target", &[0.0f32]);
        self.policy_session.set_learning_rate(0.0);
        self.policy_session.step();
        self.policy_session.wait();

        let mut head = vec![0.0f32; MAX_ACTION_DIM];
        self.policy_session.read_output_by_index(1, &mut head);
        let mut value = [0.0f32; 1];
        self.policy_session.read_output_by_index(2, &mut value);

        self.last_value = value[0];
        self.last_entropy = self.adapter.head_entropy(&head);

        self.adapter.sample_action(&head, rng)
    }

    /// Observe a transition, train all modules.
    pub fn observe<R: Rng>(
        &mut self,
        obs: &Observation,
        action: &Action,
        env: &dyn Environment,
        rng: &mut R,
    ) {
        // --- Translate through the adapter into universal tokens ---
        self.adapter.obs_to_token(obs, &mut self.obs_token_scratch);
        self.adapter
            .action_to_token(action, &mut self.action_token_scratch);

        // z_target: the previous latent, or zeros if bootstrapping / at boundary
        let z_target = if self.pending_boundary {
            vec![0.0f32; self.latent_size]
        } else if let Some(prev) = self.buffer.last() {
            prev.latent.clone()
        } else {
            vec![0.0f32; self.latent_size]
        };

        let obs_token = self.obs_token_scratch.clone();
        let action_token = self.action_token_scratch.clone();
        let (wm_loss, z_t) = self.wm_forward_backward(
            &obs_token,
            &action_token,
            &z_target,
            self.config.learning_rate * self.encoder_lr_scale,
        );
        self.last_wm_loss = wm_loss;

        // --- Reward ---
        // Surprise = sqrt(latent_dim · wm_loss) ≈ ‖z_hat − z_target‖.
        // We don't expose z_hat as an output (meganeura aliases buffers
        // for graph internals that are also outputs); reconstruct the
        // magnitude from the loss instead.
        let pred_error = (self.config.latent_dim as f32 * wm_loss.max(0.0)).sqrt();
        let surprise = RewardCircuit::surprise(pred_error);
        let visit_count = self.buffer.visit_count(&z_t);
        let novelty = RewardCircuit::novelty(visit_count);
        let homeo = RewardCircuit::homeostatic(env.homeostatic_variables());
        let order = RewardCircuit::order(&obs.data);
        let reward = self.reward_circuit.compute(surprise, novelty, homeo, order);

        // --- Store transition (tagged with current adapter's env id) ---
        let env_boundary = self.pending_boundary;
        self.pending_boundary = false;
        self.buffer.push(Transition {
            observation: obs_token,
            latent: z_t.clone(),
            action: action_token,
            reward,
            credit: 0.0,
            pred_error,
            env_id: self.adapter.id(),
            env_boundary,
        });

        // --- Credit assignment ---
        if self.buffer.len() >= self.config.history_len {
            self.credit_step(rng);
        }

        // --- Policy + value training (gated on warmup + entropy floor) ---
        if self.step_count >= self.config.warmup_steps
            && self.last_entropy >= self.config.entropy_floor
        {
            let action_token = self.action_token_scratch.clone();
            self.policy_step(&z_t, &action_token, reward);
        }

        // --- Replay mixing ---
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
    /// Returns (loss, z_t).
    fn wm_forward_backward(
        &mut self,
        obs: &[f32],
        action: &[f32],
        z_target: &[f32],
        lr: f32,
    ) -> (f32, Vec<f32>) {
        self.wm_session.set_input("obs", obs);
        self.wm_session.set_input("action", action);
        self.wm_session.set_input("z_target", z_target);
        let task = self
            .task_embeddings
            .get(&self.adapter.id())
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; self.config.batch_size * TASK_DIM]);
        self.wm_session.set_input("task", &task);
        self.wm_session.set_learning_rate(lr);
        self.wm_session.step();
        self.wm_session.wait();

        let loss = self.wm_session.read_loss();
        let mut z_t = vec![0.0f32; self.latent_size];
        self.wm_session.read_output_by_index(1, &mut z_t);
        (loss, z_t)
    }

    fn replay_step<R: Rng>(&mut self, rng: &mut R) {
        if self.buffer.len() < 2 {
            return;
        }
        let n = self.buffer.len();
        let i = rng.random_range(0..n - 1);
        let ti = self.buffer.get(i);
        let tj = self.buffer.get(i + 1);
        // Skip replay across env boundaries (latent → latent is
        // meaningless when env just switched)
        if tj.env_boundary || ti.env_id != tj.env_id {
            return;
        }
        let obs = ti.observation.clone();
        let action = ti.action.clone();
        let z_target = tj.latent.clone();

        let (loss, _) = self.wm_forward_backward(
            &obs,
            &action,
            &z_target,
            self.config.learning_rate * self.encoder_lr_scale * 0.5,
        );
        self.last_replay_loss = loss;
    }

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

    fn measure_drift(&mut self) {
        let (probes, references) = match (self.probe_obs.as_ref(), self.probe_reference.as_ref()) {
            (Some(p), Some(r)) => (p.clone(), r.clone()),
            _ => return,
        };

        let mut total = 0.0f32;
        let mut count = 0;

        let zero_action = vec![0.0f32; self.config.batch_size * MAX_ACTION_DIM];
        let zero_target = vec![0.0f32; self.latent_size];
        let task = self
            .task_embeddings
            .get(&self.adapter.id())
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; self.config.batch_size * TASK_DIM]);

        for (obs, reference) in probes.iter().zip(references.iter()) {
            self.wm_session.set_input("obs", obs);
            self.wm_session.set_input("action", &zero_action);
            self.wm_session.set_input("z_target", &zero_target);
            self.wm_session.set_input("task", &task);
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

        if drift > self.config.drift_threshold {
            self.encoder_lr_scale = (self.encoder_lr_scale * 0.5).max(0.01);
        } else {
            self.encoder_lr_scale = (self.encoder_lr_scale * 1.1).min(1.0);
        }
    }

    fn credit_step<R: Rng>(&mut self, rng: &mut R) {
        let h = self.config.history_len;

        if let Some(history_flat) = self.buffer.flatten_history(h) {
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

    fn policy_step(&mut self, z_t: &[f32], action_token: &[f32], reward: f32) {
        self.policy_session.set_input("z", z_t);
        self.policy_session.set_input("action", action_token);

        self.policy_session.set_input("value_target", &[reward]);

        let advantage = reward - self.last_value;

        if advantage > 0.0 {
            let scale = advantage.min(1.0);
            self.policy_session
                .set_learning_rate(self.config.lr_policy * scale);
        } else {
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
            env_id: self.adapter.id(),
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

/// Deterministic per-env task embedding. Same env_id always produces the
/// same vector, so swapping in and out across runs is consistent. Values
/// are spread in [-0.5, 0.5] via a golden-ratio hash.
fn embedding_for(env_id: u32, dim: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..dim)
        .map(|i| {
            let h =
                ((env_id as f64 + i as f64 * 17.0 + 1.0) * 0.618_033_988_749_895).fract() as f32;
            (h * PI * 2.0).sin() * 0.5
        })
        .collect()
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
            // xavier_init returns fan*fan elements; pad/truncate to exactly
            // num_elements so set_parameter overwrites the whole buffer
            // (otherwise tail bytes hold whatever was previously there).
            let fan = (num_elements as f32).sqrt().max(1.0) as usize;
            let mut data = xavier_init(fan, fan, i as u64 * 7919);
            data.resize(num_elements, 0.0);
            session.set_parameter(name, &data);
        }
    }
}
