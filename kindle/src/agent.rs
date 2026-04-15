//! Top-level Agent struct and training loop.
//!
//! The agent orchestrates the full pipeline:
//! observation → (adapter) → encoder → world model → reward → credit → policy → (adapter) → action.
//!
//! The agent's GPU graphs are built once with universal token sizes
//! (`OBS_TOKEN_DIM`, `MAX_ACTION_DIM`); a per-env `EnvAdapter` translates
//! between the env's native shapes and these token sizes. `switch_lane`
//! swaps one lane's adapter without touching any compiled graph.
//!
//! ## Batched lanes (Phase E)
//!
//! The agent is multi-lane: `N = config.batch_size` concurrent lanes share
//! the three compiled GPU sessions. Each lane owns its own adapter,
//! experience buffer, reward circuit and boundary flag; every `observe()`
//! call advances all N lanes in lockstep, stacking per-lane obs/action/
//! z_target/task rows into a single batched dispatch for the world model
//! and policy. Credit assignment is CPU-light per-lane (the credit graph
//! is sized for one lane's history and is called N times per step).
//!
//! For `N = 1` the runtime behaviour matches the pre-Phase-E single-lane
//! agent — construction takes a one-element `vec![adapter]` and every
//! step fed a one-element slice.
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
use crate::env::{Action, ActionKind, Environment, Observation};
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
    /// Action persistence: the sampled action is held for this many
    /// consecutive `act()` calls per lane before the policy is resampled.
    /// `1` (default) is the classic per-step reactive policy. `K > 1`
    /// stretches the effective credit-assigner horizon by K× with no
    /// graph change — a cheap precursor to the Phase G option layer.
    /// See `docs/phase-g-l1-options.md`.
    pub action_repeat: usize,
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
            action_repeat: 1,
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

/// Per-lane state. One per concurrent batch slot. Every slot owns its own
/// adapter, buffer, reward circuit and boundary flag; GPU graphs are shared
/// across lanes and feed the stacked per-lane inputs in a single dispatch.
/// One sampled transition from a lane's buffer, staged for a batched
/// replay forward+backward. We materialize this as a struct (rather than
/// a 4-tuple) so the replay-step code reads cleanly.
#[derive(Clone)]
struct ReplaySample {
    obs: Vec<f32>,
    action: Vec<f32>,
    z_target: Vec<f32>,
    env_id: u32,
}

struct Lane {
    adapter: Box<dyn EnvAdapter>,
    buffer: ExperienceBuffer,
    reward_circuit: RewardCircuit,
    pending_boundary: bool,

    // Action-persistence state (AgentConfig::action_repeat). When `repeats_left`
    // is > 0 we hand back `cached_action` instead of resampling, but the batched
    // policy forward still runs every step so the other lanes' fresh samples
    // come through in the same dispatch.
    cached_action: Option<Action>,
    repeats_left: usize,

    // Cached last-step values for diagnostics & policy advantage.
    last_value: f32,
    last_entropy: f32,
    last_surprise: f32,
    last_novelty: f32,
    last_homeo: f32,
    last_order: f32,
    last_reward: f32,
}

/// The kindle agent.
pub struct Agent {
    pub config: AgentConfig,
    /// N lanes, N = config.batch_size. Fixed at construction.
    lanes: Vec<Lane>,
    /// Per-env task embedding (key = env_id, value length = TASK_DIM). Each
    /// env gets a fixed deterministic-random vector based on its id; we
    /// tile the active per-lane embeddings row-wise into the encoder input
    /// each step. Not trained (the encoder learns to map (obs,
    /// env_embedding) into per-env latents).
    task_embeddings: HashMap<u32, Vec<f32>>,
    wm_session: Session,
    credit_session: Session,
    /// Policy graph always uses the continuous branch (MSE loss on Gaussian
    /// means). For discrete envs, the adapter softmax+samples over the
    /// first `n` head dims. This gives one universal policy graph.
    policy_session: Session,
    /// Per-lane latent dim (the WM graph is [N, latent_dim]).
    latent_dim: usize,
    step_count: usize,
    probe_obs: Option<Vec<Vec<f32>>>,
    probe_reference: Option<Vec<Vec<f32>>>,
    last_wm_loss: f32,
    last_credit_loss: f32,
    last_policy_loss: f32,
    last_replay_loss: f32,
    last_drift: f32,
    encoder_lr_scale: f32,
    /// Batch LR compensation: user's `learning_rate` is per-sample, but
    /// every WM/credit/policy loss is averaged over N rows, so per-sample
    /// gradient magnitude shrinks linearly with N. We multiply every
    /// learning rate by √N at the use sites so the effective per-sample
    /// update matches the N = 1 reference. √N (not N) is the standard
    /// large-batch rule of thumb — linear scaling tends to destabilize
    /// at larger N.
    batch_lr_scale: f32,
    /// Scratch buffers, sized [N × per-lane-dim], reused each step.
    obs_token_scratch: Vec<f32>,
    action_token_scratch: Vec<f32>,
    z_target_scratch: Vec<f32>,
    task_scratch: Vec<f32>,
    value_target_scratch: Vec<f32>,
    /// Per-row advantage-weighted action targets for the policy dispatch,
    /// sized `[N, MAX_ACTION_DIM]`. Computed as `advantage_i · one_hot_i`,
    /// which gives each lane its own signed gradient magnitude through
    /// either the cross-entropy or MSE loss path without needing a
    /// per-row loss weighting input on the graph side.
    policy_action_scratch: Vec<f32>,
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
    /// Build an N-lane agent. `adapters.len()` must equal `config.batch_size`;
    /// mismatching shapes panic at construction. For a single-lane agent,
    /// pass a single-element vec: `Agent::new(cfg, vec![adapter])`.
    ///
    /// The underlying graphs use universal token sizes and will never need
    /// rebuilding for subsequent lane-adapter swaps (`switch_lane`).
    pub fn new(config: AgentConfig, adapters: Vec<Box<dyn EnvAdapter>>) -> Self {
        assert!(
            !adapters.is_empty(),
            "Agent::new requires at least one adapter (one lane)"
        );
        assert_eq!(
            adapters.len(),
            config.batch_size,
            "adapters.len() ({}) must equal config.batch_size ({})",
            adapters.len(),
            config.batch_size
        );
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

            // Per-lane squared-error output. Exposing `(z_hat − z_target)²`
            // as `[N, latent_dim]` lets us compute per-lane prediction error
            // (and therefore per-lane surprise reward) on the CPU without
            // mutating the scalar `loss` used for backprop. This is the
            // Phase E.v2 "per-lane surprise" hook that the design doc
            // flagged as the thing to build when the shared mean-loss
            // surrogate starts hurting at large N.
            //
            // We use `z_target − z_hat` (i.e. `add(z_target, neg(z_hat))`)
            // instead of the reverse to keep `z_hat`'s primary consumer the
            // mse_loss node — meganeura's forward-optimize can still fuse
            // the loss path cleanly without fighting for the `z_hat`
            // buffer.
            let neg_zhat = g.neg(z_hat);
            let diff = g.add(z_target, neg_zhat);
            let sq = g.mul(diff, diff);
            g.set_outputs(vec![loss, z_t, sq]);
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
        // Pick the policy graph shape from the adapters' action kinds. All
        // adapters must share a kind (agent-wide single policy session), so
        // we check all of them and pick:
        //
        //   Discrete    → cross_entropy_loss(logits, one_hot_action).
        //                 The gradient w.r.t. logits is
        //                 `softmax(logits) − one_hot`, which gives every
        //                 logit a well-conditioned update every step and
        //                 lets the softmax actually narrow toward preferred
        //                 actions. This is what kindle should use for any
        //                 all-discrete setup.
        //
        //   Continuous  → mse_loss(mean, action). Gaussian-NLL with fixed
        //                 unit variance reduces to MSE up to a constant,
        //                 which is the right loss for continuous actions.
        //
        // Mixed-kind adapter sets aren't currently supported because one
        // compiled graph has one loss op. If that ever matters, we'd route
        // per-lane into one of two graphs (different shape) — out of scope
        // here.
        let first_kind = adapters[0].action_kind();
        for (i, a) in adapters.iter().enumerate().skip(1) {
            assert!(
                kinds_match(first_kind, a.action_kind()),
                "Agent::new: all adapters must share the same ActionKind \
                 variant (lane 0 is {:?}, lane {} is {:?}). Mixed discrete/\
                 continuous in one batched session is not supported.",
                first_kind,
                i,
                a.action_kind()
            );
        }
        let is_discrete = matches!(first_kind, ActionKind::Discrete { .. });

        let policy_session = {
            let g = if is_discrete {
                policy::build_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                )
            } else {
                policy::build_continuous_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                )
            };
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        let n = config.batch_size;

        // Initialize per-env task embeddings (one TASK_DIM vector per env_id)
        // for every env id present in the initial lane set.
        let mut task_embeddings: HashMap<u32, Vec<f32>> = HashMap::new();
        for adapter in &adapters {
            task_embeddings
                .entry(adapter.id())
                .or_insert_with(|| embedding_for(adapter.id(), TASK_DIM));
        }

        // Build per-lane state. Each lane gets a lane-index-seeded reward
        // circuit so per-lane order digests don't collide across lanes.
        let lanes: Vec<Lane> = adapters
            .into_iter()
            .enumerate()
            .map(|(i, adapter)| Lane {
                adapter,
                buffer: ExperienceBuffer::new(config.buffer_capacity, config.grid_resolution),
                reward_circuit: RewardCircuit::with_seed(
                    config.reward_weights.clone(),
                    0xA11CE ^ i as u64,
                ),
                pending_boundary: false,
                cached_action: None,
                repeats_left: 0,
                last_value: 0.0,
                last_entropy: 0.0,
                last_surprise: 0.0,
                last_novelty: 0.0,
                last_homeo: 0.0,
                last_order: 0.0,
                last_reward: 0.0,
            })
            .collect();

        Self {
            lanes,
            task_embeddings,
            wm_session,
            credit_session,
            policy_session,
            latent_dim: config.latent_dim,
            step_count: 0,
            probe_obs: None,
            probe_reference: None,
            last_wm_loss: 0.0,
            last_credit_loss: 0.0,
            last_policy_loss: 0.0,
            last_replay_loss: 0.0,
            last_drift: 0.0,
            encoder_lr_scale: 1.0,
            batch_lr_scale: (config.batch_size as f32).sqrt(),
            obs_token_scratch: vec![0.0; n * OBS_TOKEN_DIM],
            action_token_scratch: vec![0.0; n * MAX_ACTION_DIM],
            z_target_scratch: vec![0.0; n * config.latent_dim],
            task_scratch: vec![0.0; n * TASK_DIM],
            value_target_scratch: vec![0.0; n],
            policy_action_scratch: vec![0.0; n * MAX_ACTION_DIM],
            config,
        }
    }

    /// Number of lanes (`N = config.batch_size`).
    pub fn num_lanes(&self) -> usize {
        self.lanes.len()
    }

    /// Mark the next observed transition on `lane_idx` as the start of a
    /// new episode within the same env. The world model will zero its
    /// `z_target` row for that lane on the next step, and the stored
    /// transition will be tagged `env_boundary = true`, so the credit
    /// assigner and world model skip attribution across the reset.
    pub fn mark_boundary(&mut self, lane_idx: usize) {
        let lane = &mut self.lanes[lane_idx];
        lane.pending_boundary = true;
        // An episode reset ends any in-flight action repeat: the post-reset
        // state is drawn from a fresh env distribution, so the cached action
        // is semantically stale.
        lane.cached_action = None;
        lane.repeats_left = 0;
    }

    /// Swap the active adapter on one lane. Preserves all learned
    /// parameters; the next transition stored on that lane is marked as
    /// an env boundary so the world model and credit assigner don't try
    /// to attribute dynamics or reward across the switch. Other lanes
    /// are unaffected.
    ///
    /// A new env's task embedding is lazily initialized on first sight;
    /// returning to a previously-seen env reuses the same deterministic
    /// vector, preserving the encoder's per-env specialization.
    pub fn switch_lane(&mut self, lane_idx: usize, adapter: Box<dyn EnvAdapter>) {
        let incoming_id = adapter.id();
        self.task_embeddings
            .entry(incoming_id)
            .or_insert_with(|| embedding_for(incoming_id, TASK_DIM));

        let lane = &mut self.lanes[lane_idx];
        lane.adapter = adapter;
        lane.pending_boundary = true;
        lane.cached_action = None;
        lane.repeats_left = 0;
    }

    /// Env id of the adapter currently bound to `lane_idx`.
    pub fn env_id(&self, lane_idx: usize) -> u32 {
        self.lanes[lane_idx].adapter.id()
    }

    /// Select one action per lane. `observations.len()` must equal N.
    /// Returns a `Vec<Action>` of length N, in lane order.
    ///
    /// The obs argument is currently unused inside the policy graph (the
    /// policy conditions on the previous latent), but is kept in the
    /// signature to match the multi-lane contract and to make room for a
    /// future obs-conditioned exploration policy.
    pub fn act<R: Rng>(&mut self, observations: &[Observation], rng: &mut R) -> Vec<Action> {
        let n = self.lanes.len();
        assert_eq!(
            observations.len(),
            n,
            "act: observations.len() ({}) must equal num_lanes ({})",
            observations.len(),
            n
        );

        // Stack per-lane previous latents into the batched `z` input.
        let ld = self.latent_dim;
        let mut z_stack = vec![0.0f32; n * ld];
        for (i, lane) in self.lanes.iter().enumerate() {
            if let Some(prev) = lane.buffer.last() {
                z_stack[i * ld..(i + 1) * ld].copy_from_slice(&prev.latent);
            }
        }

        self.policy_session.set_input("z", &z_stack);
        self.action_token_scratch.fill(0.0);
        self.policy_session
            .set_input("action", &self.action_token_scratch);
        self.value_target_scratch.fill(0.0);
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        self.policy_session.set_learning_rate(0.0);
        self.policy_session.step();
        self.policy_session.wait();

        // Read stacked outputs: head is [N, MAX_ACTION_DIM], value is [N, 1].
        let mut head_stack = vec![0.0f32; n * MAX_ACTION_DIM];
        self.policy_session.read_output_by_index(1, &mut head_stack);
        let mut value_stack = vec![0.0f32; n];
        self.policy_session
            .read_output_by_index(2, &mut value_stack);

        // Per-lane sampling with optional action persistence. The batched
        // policy forward ran for every lane above (so `head`/`value` rows
        // are always fresh); a lane in mid-repeat just ignores its row
        // this step and re-uses the cached action.
        let action_repeat = self.config.action_repeat.max(1);
        let mut actions = Vec::with_capacity(n);
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            let head = &head_stack[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            lane.last_value = value_stack[i];
            lane.last_entropy = lane.adapter.head_entropy(head);

            let resample = lane.repeats_left == 0 || lane.cached_action.is_none();
            let action = if resample {
                let a = lane.adapter.sample_action(head, rng);
                lane.cached_action = Some(a.clone());
                lane.repeats_left = action_repeat - 1;
                a
            } else {
                lane.repeats_left -= 1;
                lane.cached_action
                    .clone()
                    .expect("cached_action is Some by branch condition")
            };
            actions.push(action);
        }
        actions
    }

    /// Observe one synchronous step across all lanes. All input slices must
    /// have length `N = config.batch_size`.
    pub fn observe<R: Rng>(
        &mut self,
        observations: &[Observation],
        actions: &[Action],
        envs: &[&dyn Environment],
        rng: &mut R,
    ) {
        let n = self.lanes.len();
        assert_eq!(observations.len(), n, "observations.len() must equal N");
        assert_eq!(actions.len(), n, "actions.len() must equal N");
        assert_eq!(envs.len(), n, "envs.len() must equal N");

        let ld = self.latent_dim;

        // --- Build stacked inputs: obs, action, z_target, task ---
        for (i, lane) in self.lanes.iter().enumerate() {
            let obs_row = &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
            lane.adapter.obs_to_token(&observations[i], obs_row);

            let act_row =
                &mut self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            lane.adapter.action_to_token(&actions[i], act_row);

            // z_target row: previous latent, zeros at boundary or bootstrap.
            let z_row = &mut self.z_target_scratch[i * ld..(i + 1) * ld];
            if lane.pending_boundary {
                z_row.fill(0.0);
            } else if let Some(prev) = lane.buffer.last() {
                z_row.copy_from_slice(&prev.latent);
            } else {
                z_row.fill(0.0);
            }

            // task row: lookup per-lane env's embedding.
            let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
            match self.task_embeddings.get(&lane.adapter.id()) {
                Some(emb) => task_row.copy_from_slice(emb),
                None => task_row.fill(0.0),
            }
        }

        // --- One batched WM forward+backward ---
        let wm_loss = self.wm_forward_backward_stacked(
            self.config.learning_rate * self.encoder_lr_scale * self.batch_lr_scale,
        );
        self.last_wm_loss = wm_loss;

        // Read stacked z_t output [N, latent_dim] and per-lane
        // squared-error output [N, latent_dim] from the WM graph.
        let mut z_stack = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(1, &mut z_stack);
        let mut sq_stack = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(2, &mut sq_stack);

        // --- Per-lane reward + transition push ---
        // Per-lane surprise: `pred_error_i = ||z_hat_i − z_target_i||` is
        // the L2 norm of the i-th row of `(z_hat − z_target)`, i.e. the
        // sqrt of the sum of that row's squared-errors. Replaces the old
        // Phase E.v1 mean-loss surrogate, which blurred rare-event signal
        // across all N lanes at large batch sizes.
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            let z_row = &z_stack[i * ld..(i + 1) * ld];
            let obs_row = &self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
            let act_row = &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let sq_row = &sq_stack[i * ld..(i + 1) * ld];

            let row_sum: f32 = sq_row
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .sum();
            let pred_error = row_sum.max(0.0).sqrt();
            let surprise = RewardCircuit::surprise(pred_error);

            let visit_count = lane.buffer.visit_count(z_row);
            let novelty = RewardCircuit::novelty(visit_count);
            let homeo = RewardCircuit::homeostatic(envs[i].homeostatic_variables());
            let order = lane.reward_circuit.observe_order(obs_row);
            let reward = lane.reward_circuit.compute(surprise, novelty, homeo, order);

            let env_boundary = lane.pending_boundary;
            lane.pending_boundary = false;
            lane.buffer.push(Transition {
                observation: obs_row.to_vec(),
                latent: z_row.to_vec(),
                action: act_row.to_vec(),
                reward,
                credit: 0.0,
                pred_error,
                env_id: lane.adapter.id(),
                env_boundary,
            });

            lane.last_surprise = surprise;
            lane.last_novelty = novelty;
            lane.last_homeo = homeo;
            lane.last_order = order;
            lane.last_reward = reward;
        }

        // --- Credit assignment (per-lane, sequential CPU-light dispatches) ---
        for i in 0..n {
            if self.lanes[i].buffer.len() >= self.config.history_len {
                self.credit_step(i, rng);
            }
        }

        // --- Policy + value training (one batched dispatch over all lanes) ---
        //
        // Gate is applied per-lane via the reward/advantage signal. Lanes
        // whose entropy is below the floor, or which are still in warmup,
        // contribute a zero gradient signal (LR scale 0) but share the
        // single graph dispatch.
        if self.step_count >= self.config.warmup_steps {
            self.policy_step_batched(&z_stack);
        }

        // --- Replay mixing: one batched replay per call, one transition
        // sampled per lane (no zero-row dilution).
        if rng.random_range(0.0..1.0) < self.config.replay_ratio {
            self.replay_step(rng);
        }

        // --- Representation drift monitor (shared probe set, WM session) ---
        if self.step_count == self.config.warmup_steps && self.probe_obs.is_none() {
            self.capture_probe_reference();
        }
        if self.step_count > 0 && self.step_count.is_multiple_of(self.config.drift_interval) {
            self.measure_drift();
        }

        self.step_count += 1;
    }

    /// Run one world-model forward+backward pass on the currently staged
    /// `obs_token_scratch` / `action_token_scratch` / `z_target_scratch` /
    /// `task_scratch` inputs. Returns the scalar batch-mean loss.
    fn wm_forward_backward_stacked(&mut self, lr: f32) -> f32 {
        self.wm_session.set_input("obs", &self.obs_token_scratch);
        self.wm_session
            .set_input("action", &self.action_token_scratch);
        self.wm_session
            .set_input("z_target", &self.z_target_scratch);
        self.wm_session.set_input("task", &self.task_scratch);
        self.wm_session.set_learning_rate(lr);
        self.wm_session.step();
        self.wm_session.wait();
        self.wm_session.read_loss()
    }

    /// Sample one replay transition per lane and run a single batched WM
    /// forward+backward over the stacked rows. Each lane retries up to 8
    /// random indices to find a non-boundary pair in its own buffer; if
    /// that fails (buffer < 2, or deeply fragmented by env switches), we
    /// fall back to the most recent valid donor lane's sample so every
    /// batch row carries signal instead of zeros.
    fn replay_step<R: Rng>(&mut self, rng: &mut R) {
        let ld = self.latent_dim;
        let n = self.lanes.len();

        // Per-lane sampled transitions; `None` for lanes that couldn't
        // find a valid non-boundary pair after 8 retries.
        let mut samples: Vec<Option<ReplaySample>> = Vec::with_capacity(n);
        for lane in &self.lanes {
            let buf_len = lane.buffer.len();
            if buf_len < 2 {
                samples.push(None);
                continue;
            }
            let mut found: Option<ReplaySample> = None;
            for _ in 0..8 {
                let idx = rng.random_range(0..buf_len - 1);
                let ti = lane.buffer.get(idx);
                let tj = lane.buffer.get(idx + 1);
                // Skip replay across env boundaries (latent → latent is
                // meaningless when env just switched).
                if tj.env_boundary || ti.env_id != tj.env_id {
                    continue;
                }
                found = Some(ReplaySample {
                    obs: ti.observation.clone(),
                    action: ti.action.clone(),
                    z_target: tj.latent.clone(),
                    env_id: ti.env_id,
                });
                break;
            }
            samples.push(found);
        }

        // Fall back every failed lane to the most recent valid donor so
        // the batch has no zero rows (which dilute the shared gradient).
        // If *no* lane produced a sample, bail — nothing to replay.
        let donor = samples.iter().rev().find_map(|s| s.clone());
        let Some(donor) = donor else {
            return;
        };
        for s in samples.iter_mut() {
            if s.is_none() {
                *s = Some(donor.clone());
            }
        }

        for (i, sample) in samples.iter().enumerate() {
            let sample = sample.as_ref().expect("filled above");
            let obs = &sample.obs;
            let act = &sample.action;
            let z_target = &sample.z_target;
            let env_id = &sample.env_id;
            let obs_row = &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
            obs_row.copy_from_slice(obs);
            let act_row =
                &mut self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            act_row.copy_from_slice(act);
            let z_row = &mut self.z_target_scratch[i * ld..(i + 1) * ld];
            z_row.copy_from_slice(z_target);

            let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
            if let Some(emb) = self.task_embeddings.get(env_id) {
                task_row.copy_from_slice(emb);
            } else {
                task_row.fill(0.0);
            }
        }

        let loss = self.wm_forward_backward_stacked(
            self.config.learning_rate * self.encoder_lr_scale * self.batch_lr_scale * 0.5,
        );
        self.last_replay_loss = loss;
    }

    /// Capture a shared probe reference set from lane 0's buffer (any lane
    /// would do — drift is a global representation-stability signal).
    fn capture_probe_reference(&mut self) {
        let lane0 = &self.lanes[0];
        let n_probe = 16.min(lane0.buffer.len());
        if n_probe == 0 {
            return;
        }
        let step = lane0.buffer.len() / n_probe.max(1);
        let mut observations = Vec::with_capacity(n_probe);
        let mut references = Vec::with_capacity(n_probe);
        for i in 0..n_probe {
            let idx = i * step;
            if idx < lane0.buffer.len() {
                let t = lane0.buffer.get(idx);
                observations.push(t.observation.clone());
                references.push(t.latent.clone());
            }
        }
        self.probe_obs = Some(observations);
        self.probe_reference = Some(references);
    }

    /// Measure representation drift by forwarding the probe set through the
    /// batched WM graph. The probe set isn't necessarily a multiple of N;
    /// we pad the remaining rows with zeros and ignore their outputs.
    fn measure_drift(&mut self) {
        let (probes, references) = match (self.probe_obs.as_ref(), self.probe_reference.as_ref()) {
            (Some(p), Some(r)) => (p.clone(), r.clone()),
            _ => return,
        };

        let n = self.lanes.len();
        let ld = self.latent_dim;

        let mut total = 0.0f32;
        let mut count = 0;

        // Use lane 0's env's task embedding for all probe rows — drift is a
        // global representation signal and its absolute scale depends on
        // task conditioning anyway. Unused (padded) rows see the same
        // embedding but their outputs are discarded.
        let task_emb = self
            .task_embeddings
            .get(&self.lanes[0].adapter.id())
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; TASK_DIM]);

        for chunk_start in (0..probes.len()).step_by(n) {
            // Stage batch inputs.
            self.obs_token_scratch.fill(0.0);
            self.action_token_scratch.fill(0.0);
            self.z_target_scratch.fill(0.0);
            for i in 0..n {
                let task_row = &mut self.task_scratch[i * TASK_DIM..(i + 1) * TASK_DIM];
                task_row.copy_from_slice(&task_emb);
            }
            let chunk_len = (probes.len() - chunk_start).min(n);
            for i in 0..chunk_len {
                let probe = &probes[chunk_start + i];
                let obs_row =
                    &mut self.obs_token_scratch[i * OBS_TOKEN_DIM..(i + 1) * OBS_TOKEN_DIM];
                let copy_len = probe.len().min(OBS_TOKEN_DIM);
                obs_row[..copy_len].copy_from_slice(&probe[..copy_len]);
            }

            self.wm_session.set_input("obs", &self.obs_token_scratch);
            self.wm_session
                .set_input("action", &self.action_token_scratch);
            self.wm_session
                .set_input("z_target", &self.z_target_scratch);
            self.wm_session.set_input("task", &self.task_scratch);
            self.wm_session.set_learning_rate(0.0);
            self.wm_session.step();
            self.wm_session.wait();

            let mut z_stack = vec![0.0f32; n * ld];
            self.wm_session.read_output_by_index(1, &mut z_stack);

            for i in 0..chunk_len {
                let current = &z_stack[i * ld..(i + 1) * ld];
                let reference = &references[chunk_start + i];
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
        }

        let drift = if count > 0 { total / count as f32 } else { 0.0 };
        self.last_drift = drift;

        if drift > self.config.drift_threshold {
            self.encoder_lr_scale = (self.encoder_lr_scale * 0.5).max(0.01);
        } else {
            self.encoder_lr_scale = (self.encoder_lr_scale * 1.1).min(1.0);
        }
    }

    /// Run one credit-assigner pass on a single lane's history. The credit
    /// graph is sized `[history_len, input_dim]` (one-lane by design — see
    /// the design doc); we call it N times per step, once per lane.
    fn credit_step<R: Rng>(&mut self, lane_idx: usize, rng: &mut R) {
        let h = self.config.history_len;
        let latent_dim = self.config.latent_dim;
        // Credit runs per-lane (serialized), not batched — so it doesn't
        // get the loss-averaging dilution. But the shared credit weights
        // absorb N updates per synchronous step instead of 1, and we want
        // each of those updates scaled consistently with the WM/policy
        // updates on the same step. √N keeps the magnitudes aligned.
        let lr_credit = self.config.lr_credit * self.batch_lr_scale;

        // Build the history + contrastive target on the immutable borrow of
        // the lane, then drop it before touching `self.credit_session`.
        let (history_clean, target_clean, r_t) = {
            let lane = &self.lanes[lane_idx];
            let Some(history_flat) = lane.buffer.flatten_history(h) else {
                return;
            };
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
            let target =
                if let Some((hi, lo)) = lane.buffer.find_contrastive_pair(rng, h, latent_dim) {
                    lane.buffer.contrastive_target(hi, lo, h)
                } else {
                    vec![1.0 / h as f32; h]
                };
            let target_clean: Vec<f32> = target
                .iter()
                .map(|v| if v.is_finite() { *v } else { 1.0 / h as f32 })
                .collect();
            (history_clean, target_clean, lane.last_reward)
        };

        self.credit_session.set_input("history", &history_clean);
        self.credit_session
            .set_input("credit_target", &target_clean);
        self.credit_session.set_learning_rate(lr_credit);
        self.credit_session.step();
        self.credit_session.wait();

        self.last_credit_loss = self.credit_session.read_loss();

        let mut credit_logits = vec![0.0f32; h];
        self.credit_session
            .read_output_by_index(1, &mut credit_logits);

        let alpha = credit::softmax(&credit_logits);
        let credits: Vec<f32> = alpha.iter().map(|&a| r_t * a).collect();
        self.lanes[lane_idx].buffer.write_credits(&credits);
    }

    /// Batched policy + value step over all lanes. The graph input `"z"` is
    /// the stacked per-lane latents produced this step; `"action"` is the
    /// stacked action tokens already in `action_token_scratch`;
    /// `"value_target"` is the stacked per-lane rewards. `"action"` is
    /// fed as a per-row advantage-weighted scaled one-hot — see below.
    ///
    /// Per-row advantage weighting, without a graph change: for either
    /// policy loss variant the gradient w.r.t. logits is
    /// `target_weight · (pred − one_hot)`. So if we feed the action
    /// input as `advantage_i · one_hot_i` (signed, clamped) rather than
    /// just `one_hot_i`, each lane's gradient magnitude and sign come
    /// from its own advantage. Lanes with positive advantage push the
    /// policy toward the taken action, lanes with negative advantage
    /// push it away, and lanes with ~zero advantage contribute ~nothing.
    /// The shared LR is a fixed `lr_policy · batch_lr_scale`.
    ///
    /// This is the zero-graph-change analog of a proper per-row loss
    /// weighting input. It works for the discrete cross-entropy graph
    /// and the continuous MSE graph uniformly (both use `"action"` as
    /// their target).
    fn policy_step_batched(&mut self, z_stack: &[f32]) {
        // Build stacked value targets (the rewards computed this step).
        for (i, lane) in self.lanes.iter().enumerate() {
            self.value_target_scratch[i] = lane.last_reward;
        }

        // Build per-row advantage-weighted action targets. Entropy floor
        // still gates per-lane; a gated-out lane contributes a zero row
        // (no gradient). Clamp advantages to ±1 so a single outlier lane
        // can't dominate the shared-weight update.
        self.policy_action_scratch.fill(0.0);
        let mut any_active = false;
        for (i, lane) in self.lanes.iter().enumerate() {
            if lane.last_entropy < self.config.entropy_floor {
                continue;
            }
            let advantage = (lane.last_reward - lane.last_value).clamp(-1.0, 1.0);
            if advantage == 0.0 {
                continue;
            }
            any_active = true;
            let act_src = &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let act_dst =
                &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            for (dst, &src) in act_dst.iter_mut().zip(act_src.iter()) {
                *dst = advantage * src;
            }
        }
        if !any_active {
            return;
        }

        self.policy_session.set_input("z", z_stack);
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        self.policy_session
            .set_learning_rate(self.config.lr_policy * self.batch_lr_scale);
        self.policy_session.step();
        self.policy_session.wait();

        self.last_policy_loss = self.policy_session.read_loss();
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Per-lane diagnostics, one entry per lane in lane order.
    ///
    /// Global (batch-shared) signals — `loss_world_model`, `loss_credit`,
    /// `loss_policy`, `loss_replay`, `repr_drift` — are broadcast to every
    /// lane's row. Lane-specific fields (`env_id`, `reward_*`,
    /// `policy_entropy`, `buffer_len`, `h_eff`) vary per row.
    pub fn diagnostics(&self) -> Vec<Diagnostics> {
        self.lanes
            .iter()
            .map(|lane| {
                let recent = lane.buffer.recent_window(self.config.history_len);
                let credit_weights: Vec<f32> = recent.iter().map(|t| t.credit).collect();
                let h_eff = if credit_weights.is_empty() {
                    0.0
                } else {
                    credit::effective_scope(&credit_weights)
                };

                Diagnostics {
                    step: self.step_count,
                    env_id: lane.adapter.id(),
                    loss_world_model: self.last_wm_loss,
                    loss_credit: self.last_credit_loss,
                    loss_policy: self.last_policy_loss,
                    loss_replay: self.last_replay_loss,
                    reward_mean: lane.last_reward,
                    reward_surprise: lane.last_surprise,
                    reward_novelty: lane.last_novelty,
                    reward_homeo: lane.last_homeo,
                    reward_order: lane.last_order,
                    h_eff,
                    policy_entropy: lane.last_entropy,
                    repr_drift: self.last_drift,
                    buffer_len: lane.buffer.len(),
                }
            })
            .collect()
    }
}

/// Deterministic per-env task embedding. Same env_id always produces the
/// Variant-only equality on `ActionKind`. The enum isn't `Eq` because it
/// carries `f32` in the continuous branch; we just want to check the two
/// adapters agree on which branch of the sum type they are.
fn kinds_match(a: ActionKind, b: ActionKind) -> bool {
    matches!(
        (a, b),
        (ActionKind::Discrete { .. }, ActionKind::Discrete { .. })
            | (ActionKind::Continuous { .. }, ActionKind::Continuous { .. })
    )
}

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
