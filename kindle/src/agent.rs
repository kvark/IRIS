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
use crate::option;
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
    /// Label-smoothing coefficient for the per-row advantage-weighted
    /// action target sent to the policy graph. At `ε > 0`, the pure
    /// one-hot target `[0, 0, 1, 0]` becomes `[(1−ε)·one_hot + ε/K]`
    /// before advantage scaling, preventing the softmax from collapsing
    /// to deterministic (gradient is nonzero on every logit even at a
    /// one-hot softmax, so the policy can always recover from bad local
    /// minima). At `ε = 0` (default), the target is exact one-hot —
    /// backward compatible. `ε = 0.1` is a standard starting point.
    pub label_smoothing: f32,
    /// Number of discrete options for the L1 option-policy (Phase G).
    /// `1` (default) skips L1 entirely — no option_session compiled, no
    /// goal conditioning, byte-parity with the pre-Phase-G agent.
    /// `≥ 2` activates the full L1 path: option_session, per-lane goal
    /// conditioning on L0's z input, option-return training.
    pub num_options: usize,
    /// Goal-latent width decoded per option. Defaults to `latent_dim`.
    pub option_dim: usize,
    /// Fixed number of env steps per option (v1: no learned termination).
    pub option_horizon: usize,
    /// L1 option-policy learning rate.
    pub lr_option: f32,
    /// Phase G v5: number of recent options held in the L1 credit
    /// assigner's history window. A value < 2 disables the L1 credit
    /// assigner (its session is not compiled); the option policy then
    /// trains on per-option advantage only, no cross-option credit.
    /// Default `8` mirrors the design-doc recommendation.
    pub option_history_len: usize,
    /// L1 credit assigner learning rate (analogous to `lr_credit` for
    /// L0). Applied with the same `√N` batch scale as other L1 LRs.
    pub lr_option_credit: f32,
    /// Goal-achievement bonus coefficient. When L1 is active, each step's
    /// L0 reward is augmented with `−α · ‖z_t − goal‖`, giving L0 a
    /// self-supervised signal to drive the latent toward the option's goal
    /// regardless of the frozen reward circuit's output. `α = 0` disables.
    pub goal_bonus_alpha: f32,
    /// Phase G v4: enable the learned-termination head. When `true`, the
    /// agent forwards the option session every step, samples a Bernoulli
    /// from the predicted `β(z_t)`, and terminates the current option if
    /// either the sample or the fixed-horizon cap fires. The termination
    /// head is trained via BCE against a target derived from whether
    /// switching options would have been beneficial at that state.
    /// When `false`, termination is purely horizon-based (v1/v2/v3
    /// behaviour) and the head receives no gradient; the option-session
    /// graph shape is identical either way to keep parameter layouts
    /// stable across config changes.
    pub learned_termination: bool,
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
            label_smoothing: 0.0,
            num_options: 1,
            option_dim: 0, // 0 = use latent_dim
            option_horizon: 10,
            lr_option: 2.5e-4,
            option_history_len: 8,
            lr_option_credit: 7.5e-5,
            goal_bonus_alpha: 0.1,
            learned_termination: false,
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
    /// L1: which option this lane is currently executing.
    pub current_option: u32,
    /// L1: accumulated return for the current option so far.
    pub option_return: f32,
    /// L1: ‖z_t − goal‖ — how close the lane's latent is to its goal.
    pub goal_distance: f32,
    /// L1 effective credit scope: `Σ_i (i · α_i)` over the last
    /// `option_history_len` options. Zero when the L1 credit assigner
    /// is disabled (`option_history_len < 2`). Increasing over training
    /// means option credit is being attributed to older options — the
    /// agent is learning longer-horizon option structure.
    pub h_eff_l1: f32,
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

/// One completed-option entry for the L1 credit assigner's history.
#[derive(Clone)]
struct OptionEntry {
    /// Encoder latent at option start — the "state" input row for the
    /// L1 credit graph.
    z_start: Vec<f32>,
    /// Option index (one-hot-encoded into the history row).
    option_idx: u32,
    /// Sum of `last_reward` over this option's window.
    option_return: f32,
    /// Number of env steps the option ran before termination.
    option_length: u32,
}

/// Fixed-capacity ring buffer of recently-terminated options per lane.
struct OptionHistory {
    entries: std::collections::VecDeque<OptionEntry>,
    capacity: usize,
}

impl OptionHistory {
    fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
        }
    }

    fn push(&mut self, entry: OptionEntry) {
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, i: usize) -> &OptionEntry {
        &self.entries[i]
    }

    /// Flatten the last `history_len` entries row-major into a single
    /// `[history_len × input_dim]` vector for the credit graph. The
    /// per-row layout is `[z_start | option_onehot | option_return | length]`.
    /// Panics if `self.len() < history_len`.
    fn flatten(&self, history_len: usize, num_options: usize, latent_dim: usize) -> Vec<f32> {
        debug_assert!(self.len() >= history_len);
        let input_dim = latent_dim + num_options + 2;
        let mut out = Vec::with_capacity(history_len * input_dim);
        let start = self.len() - history_len;
        for e in self.entries.iter().skip(start) {
            out.extend_from_slice(&e.z_start);
            for k in 0..num_options {
                out.push(if k == e.option_idx as usize { 1.0 } else { 0.0 });
            }
            out.push(e.option_return);
            out.push(e.option_length as f32);
        }
        out
    }

    /// Locate a contrastive pair in the last `history_len` entries: two
    /// options with similar `z_start` but divergent realized returns.
    /// Returns `(high_return_idx_within_window, low_return_idx_within_window)`
    /// in `[0, history_len)`. None if not enough data or no useful pair.
    fn find_contrastive_pair<R: Rng>(
        &self,
        rng: &mut R,
        history_len: usize,
    ) -> Option<(usize, usize)> {
        if self.len() < history_len {
            return None;
        }
        let start = self.len() - history_len;
        let mut best: Option<(usize, usize)> = None;
        let mut best_score = 0.0f32;
        // Brute-force over all (i, j) pairs in the window; history_len
        // is O(8) so this is 56 comparisons.
        for a in 0..history_len {
            for b in (a + 1)..history_len {
                let ei = self.get(start + a);
                let ej = self.get(start + b);
                let z_dist: f32 = ei
                    .z_start
                    .iter()
                    .zip(ej.z_start.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt();
                let ret_diff = (ei.option_return - ej.option_return).abs();
                let score = ret_diff / (z_dist + 0.1);
                if score > best_score {
                    best_score = score;
                    best = if ei.option_return >= ej.option_return {
                        Some((a, b))
                    } else {
                        Some((b, a))
                    };
                }
            }
        }
        // Optional RNG usage so the seed remains referenced; swaps the
        // returned pair with a small probability for sample diversity.
        if let Some((hi, lo)) = best {
            if rng.random_range(0.0..1.0) < 0.1 {
                return Some((lo, hi));
            }
            return Some((hi, lo));
        }
        None
    }

    /// Option-divergence contrastive target for the credit assigner.
    /// Produces a softmax-normalized `[history_len]` vector whose peak
    /// is at indices where the hi/lo pair took different options.
    fn contrastive_target(&self, hi: usize, lo: usize, history_len: usize) -> Vec<f32> {
        let start = self.len() - history_len;
        let mut divergence = vec![0.0f32; history_len];
        // We don't align two windows here (unlike L0 which compares
        // parallel histories at two end-points). Instead we measure
        // divergence as "this step's option differs from the
        // high-return option at hi" — a cheap heuristic that scores
        // steps where the agent took an option distinct from the
        // locally best one.
        let hi_idx = self.get(start + hi).option_idx;
        for (i, div) in divergence.iter_mut().enumerate() {
            let ent = self.get(start + i);
            *div = if ent.option_idx == hi_idx { 0.0 } else { 1.0 };
        }
        // Softmax normalize.
        let max = divergence.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = divergence.iter().map(|d| (d - max).exp()).sum();
        if exp_sum > 0.0 {
            for d in divergence.iter_mut() {
                *d = (*d - max).exp() / exp_sum;
            }
        } else {
            for d in divergence.iter_mut() {
                *d = 1.0 / history_len as f32;
            }
        }
        let _ = lo;
        divergence
    }
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

    // --- L1 option state (Phase G) ---
    current_option: u32,
    option_goal: Vec<f32>,
    option_steps_left: usize,
    /// Number of env steps since the current option was last (re)sampled.
    option_elapsed: u32,
    option_return: f32,
    /// Value prediction cached at option-start for advantage computation.
    option_start_value: f32,
    /// Encoder latent captured at option-start, used as the per-option
    /// `z_start` row fed to the L1 credit assigner.
    option_start_z: Vec<f32>,
    /// Option-level history ring buffer for the L1 credit assigner.
    /// Each entry is one completed option window.
    option_history: OptionHistory,

    // Cached last-step values for diagnostics & policy advantage.
    last_value: f32,
    last_entropy: f32,
    last_surprise: f32,
    last_novelty: f32,
    last_homeo: f32,
    last_order: f32,
    last_reward: f32,
    /// Reward excluding the L1 goal-alignment bonus. Used as the value
    /// head's TD target so that the value baseline does NOT absorb the
    /// option-conditioned bonus; otherwise advantage = reward − value
    /// would cancel the bonus out and the policy-gradient signal that
    /// distinguishes options would collapse. When L1 is inactive this
    /// equals `last_reward`.
    last_base_reward: f32,
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
    /// L1 option-policy session. `None` when `num_options <= 1` (L0-only).
    option_session: Option<Session>,
    /// L1 credit-assigner session. `None` when L1 is off or when
    /// `option_history_len < 2`. Runs contrastively per-lane like the
    /// L0 credit graph.
    option_credit_session: Option<Session>,
    /// Per-lane latent dim (the WM graph is [N, latent_dim]).
    latent_dim: usize,
    step_count: usize,
    probe_obs: Option<Vec<Vec<f32>>>,
    probe_reference: Option<Vec<Vec<f32>>>,
    last_wm_loss: f32,
    last_credit_loss: f32,
    last_policy_loss: f32,
    last_replay_loss: f32,
    last_option_credit_loss: f32,
    last_h_eff_l1: f32,
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
    /// Effective option_dim (resolved from config: 0 → latent_dim). The
    /// goal vector width used by the goal-alignment reward bonus.
    option_dim: usize,
    /// Stacked per-lane one-hot option encodings fed to the policy
    /// graph's `option_onehot` input when L1 is active. Each row's
    /// active-option slot is `1.0`, others `0.0`. Empty (`n * 0`) when
    /// `num_options = 1`.
    option_onehot_scratch: Vec<f32>,
    /// L1 scratch buffers.
    option_taken_scratch: Vec<f32>,
    option_return_scratch: Vec<f32>,
    /// Termination BCE target, `[N, 1]`. Zero unless the agent decided
    /// (this step) that the current option should have ended; then 1.
    termination_target_scratch: Vec<f32>,
    /// Fixed goal lookup table [num_options × option_dim]. Each option
    /// maps to a pre-set orthogonal direction in latent space. L1 learns
    /// which option to pick; the goal vectors themselves are constants.
    goal_table: Vec<f32>,
}

/// Cosine similarity between two equal-length vectors. Returns 0 when
/// either side has zero norm (no direction defined).
fn unit_cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let a_norm: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let b_norm: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if a_norm < 1e-6 || b_norm < 1e-6 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (a_norm * b_norm)
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

        // Resolve option_dim: 0 in config means "use latent_dim". This
        // is the dimensionality of the `option_goal` vector used by the
        // goal-alignment reward bonus; it is no longer mixed into the
        // policy's `z` input (Phase G v2 uses a per-option bias head
        // instead, which is plumbed through `option_onehot`).
        let option_dim = if config.option_dim == 0 {
            config.latent_dim
        } else {
            config.option_dim
        };
        let l1_active = config.num_options >= 2;

        // L0 policy graph — `z` is just the encoder latent. When L1 is
        // active, the graph also takes a one-hot `option_onehot` input
        // and routes it through a direct-to-logits bias head (see
        // `policy::build_policy_graph`).
        let policy_session = {
            let g = if is_discrete {
                policy::build_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                    config.entropy_beta,
                    config.num_options,
                )
            } else {
                policy::build_continuous_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                    config.num_options,
                )
            };
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            s
        };

        // L1 credit-assigner session — only built when L1 is active
        // and the user asked for a non-degenerate history window.
        let option_credit_session = if l1_active && config.option_history_len >= 2 {
            let g = credit::build_option_credit_graph(
                config.latent_dim,
                config.num_options,
                config.option_history_len,
                config.hidden_dim,
            );
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
        };

        // L1 option-policy session — only built when num_options >= 2.
        let option_session = if l1_active {
            let g = option::build_option_graph(
                config.latent_dim,
                config.num_options,
                config.hidden_dim,
                config.batch_size,
            );
            let mut s = build_session(&g, config.opt_level);
            init_parameters(&mut s);
            Some(s)
        } else {
            None
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
                current_option: 0,
                option_goal: vec![0.0; option_dim],
                option_steps_left: 0, // triggers initial option sample
                option_elapsed: 0,
                option_return: 0.0,
                option_start_value: 0.0,
                option_start_z: vec![0.0; config.latent_dim],
                option_history: OptionHistory::new(config.option_history_len.max(1)),
                last_value: 0.0,
                last_entropy: 0.0,
                last_surprise: 0.0,
                last_novelty: 0.0,
                last_homeo: 0.0,
                last_order: 0.0,
                last_reward: 0.0,
                last_base_reward: 0.0,
            })
            .collect();

        Self {
            lanes,
            task_embeddings,
            wm_session,
            credit_session,
            policy_session,
            option_session,
            option_credit_session,
            latent_dim: config.latent_dim,
            step_count: 0,
            probe_obs: None,
            probe_reference: None,
            last_wm_loss: 0.0,
            last_credit_loss: 0.0,
            last_policy_loss: 0.0,
            last_replay_loss: 0.0,
            last_option_credit_loss: 0.0,
            last_h_eff_l1: 0.0,
            last_drift: 0.0,
            encoder_lr_scale: 1.0,
            batch_lr_scale: (config.batch_size as f32).sqrt(),
            obs_token_scratch: vec![0.0; n * OBS_TOKEN_DIM],
            action_token_scratch: vec![0.0; n * MAX_ACTION_DIM],
            z_target_scratch: vec![0.0; n * config.latent_dim],
            task_scratch: vec![0.0; n * TASK_DIM],
            value_target_scratch: vec![0.0; n],
            policy_action_scratch: vec![0.0; n * MAX_ACTION_DIM],
            option_dim,
            option_onehot_scratch: vec![0.0; n * config.num_options.max(1)],
            option_taken_scratch: vec![0.0; n * config.num_options],
            option_return_scratch: vec![0.0; n],
            termination_target_scratch: vec![0.0; n],
            goal_table: option::build_goal_table(config.num_options, option_dim),
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
        // Episode reset terminates the current option early — force a
        // resample next act().
        lane.option_steps_left = 0;
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
        lane.option_steps_left = 0;
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

        // Stack per-lane previous latents.
        let ld = self.latent_dim;
        let od = self.option_dim;
        let mut z_stack = vec![0.0f32; n * ld];
        for (i, lane) in self.lanes.iter().enumerate() {
            if let Some(prev) = lane.buffer.last() {
                z_stack[i * ld..(i + 1) * ld].copy_from_slice(&prev.latent);
            }
        }

        // --- L1 option management (Phase G) ---
        if let Some(ref mut opt_sess) = self.option_session {
            let num_options = self.config.num_options;
            let horizon = self.config.option_horizon;
            let learned_term = self.config.learned_termination;
            let warmup_done = self.step_count >= self.config.warmup_steps;

            // Forward option session every step to read current option
            // logits, state value, and termination β(z_t). LR = 0 so this
            // is forward-only; the training pass below optionally runs at
            // the real LR on just the terminated-lane rows.
            opt_sess.set_input("z", &z_stack);
            self.option_taken_scratch.fill(0.0);
            opt_sess.set_input("option_taken", &self.option_taken_scratch);
            self.option_return_scratch.fill(0.0);
            opt_sess.set_input("option_return", &self.option_return_scratch);
            self.termination_target_scratch.fill(0.0);
            opt_sess.set_input("termination_target", &self.termination_target_scratch);
            opt_sess.set_learning_rate(0.0);
            opt_sess.step();
            opt_sess.wait();

            let mut logits = vec![0.0f32; n * num_options];
            opt_sess.read_output_by_index(1, &mut logits);
            let mut values = vec![0.0f32; n];
            opt_sess.read_output_by_index(2, &mut values);
            let mut term_probs = vec![0.0f32; n];
            opt_sess.read_output_by_index(3, &mut term_probs);

            // Per-lane termination decision: horizon cap (always) or
            // Bernoulli(β) sample (only when learned_termination is on
            // and past warmup).
            let mut lanes_to_terminate: Vec<usize> = Vec::with_capacity(n);
            for (i, lane) in self.lanes.iter().enumerate() {
                let horizon_expired = lane.option_steps_left == 0;
                let learned_fire = learned_term
                    && warmup_done
                    && rng.random_range(0.0..1.0) < term_probs[i];
                if horizon_expired || learned_fire {
                    lanes_to_terminate.push(i);
                }
            }

            if !lanes_to_terminate.is_empty() {
                // --- L1 backward: train at the lanes that just
                // terminated. Other lanes contribute zero rows and are
                // effectively excluded from this step's gradient.
                self.option_taken_scratch.fill(0.0);
                self.option_return_scratch.fill(0.0);
                self.termination_target_scratch.fill(0.0);
                let mut any_train = false;
                for &i in &lanes_to_terminate {
                    let lane = &self.lanes[i];
                    let advantage =
                        (lane.option_return - lane.option_start_value).clamp(-1.0, 1.0);
                    if advantage.abs() < 1e-8 {
                        continue;
                    }
                    any_train = true;
                    let row = &mut self.option_taken_scratch
                        [i * num_options..(i + 1) * num_options];
                    row[lane.current_option as usize] = advantage;
                    self.option_return_scratch[i] = lane.option_return;
                    // Termination target at the step where the option
                    // ended, with a deadband: only train β when the
                    // option's realized advantage is clearly signed.
                    //
                    //   |adv| < 0.3 → target = 0 (keep β low; noisy
                    //                  signals shouldn't raise β).
                    //   adv < −0.3   → target = 1 (raise β here).
                    //   adv > +0.3   → target = 0 (correct to have
                    //                  continued).
                    //
                    // Paired with the −3 logit bias in `option.rs`,
                    // this keeps β strongly low by default; β only
                    // rises when states with consistently-negative
                    // option returns accumulate training signal.
                    let deadband = 0.3f32;
                    self.termination_target_scratch[i] = if advantage < -deadband {
                        1.0
                    } else {
                        0.0
                    };
                }

                if any_train && warmup_done {
                    opt_sess.set_input("z", &z_stack);
                    opt_sess.set_input("option_taken", &self.option_taken_scratch);
                    opt_sess.set_input("option_return", &self.option_return_scratch);
                    opt_sess
                        .set_input("termination_target", &self.termination_target_scratch);
                    opt_sess.set_learning_rate(self.config.lr_option * self.batch_lr_scale);
                    opt_sess.step();
                    opt_sess.wait();
                }

                // --- Record each terminated option into its lane's
                // history and, if L1 credit is enabled, run one credit
                // forward+backward pass on the resulting window. The
                // option's `z_start` was captured at its last
                // resample; `option_elapsed` tracks the realized
                // length even when learned-termination fires early.
                for &i in &lanes_to_terminate {
                    let lane = &mut self.lanes[i];
                    if lane.option_elapsed > 0 {
                        let entry = OptionEntry {
                            z_start: lane.option_start_z.clone(),
                            option_idx: lane.current_option,
                            option_return: lane.option_return,
                            option_length: lane.option_elapsed,
                        };
                        lane.option_history.push(entry);
                    }
                }
                self.option_credit_step(rng, &lanes_to_terminate);

                // --- Sample new option for each terminated lane ---
                for &i in &lanes_to_terminate {
                    let lane = &mut self.lanes[i];
                    let row = &logits[i * num_options..(i + 1) * num_options];
                    let opt_idx = crate::adapter::sample_discrete_from_logits(row, rng);
                    lane.current_option = opt_idx as u32;
                    lane.option_start_value = values[i];
                    lane.option_steps_left = horizon;
                    lane.option_elapsed = 0;
                    lane.option_return = 0.0;
                    // Capture `z_start` for this new option from the
                    // current z_stack row — this is the latent we just
                    // computed above and will begin this option from.
                    let z_row = &z_stack[i * ld..(i + 1) * ld];
                    lane.option_start_z.clear();
                    lane.option_start_z.extend_from_slice(z_row);
                    let base = opt_idx * od;
                    lane.option_goal
                        .copy_from_slice(&self.goal_table[base..base + od]);
                }
            }

            // Build per-lane one-hot option encodings and feed both z
            // (pure latent) and option_onehot to the policy graph.
            // The option identity signals directly into the per-option
            // bias head inside the policy graph.
            self.option_onehot_scratch.fill(0.0);
            for (i, lane) in self.lanes.iter().enumerate() {
                let row = &mut self.option_onehot_scratch[i * num_options..(i + 1) * num_options];
                row[lane.current_option as usize] = 1.0;
            }
            self.policy_session.set_input("z", &z_stack);
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        } else {
            // L0-only path — feed z directly.
            self.policy_session.set_input("z", &z_stack);
        }

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
        // Stability guard: sanitize any NaN and clamp extreme logits so
        // softmax(head) stays finite. Combined with the graph-internal
        // soft-tanh-clamp (see `policy.rs::scaled_tanh`) this forms a
        // defense-in-depth against long-run numerical drift. ±60 is
        // just beyond the graph clamp (±50) so a well-behaved graph
        // output is never touched here; only truly bad values are
        // sanitized.
        for v in head_stack.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            } else {
                *v = v.clamp(-60.0, 60.0);
            }
        }
        for v in value_stack.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            } else {
                *v = v.clamp(-1e6, 1e6);
            }
        }

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
        if wm_loss.is_finite() {
            self.last_wm_loss = wm_loss;
        } else {
            log::warn!(
                "WM loss went non-finite at step {}, re-initialized WM params",
                self.step_count
            );
            init_parameters(&mut self.wm_session);
            self.last_wm_loss = 0.0;
        }

        // Read stacked z_t output [N, latent_dim] and per-lane
        // squared-error output [N, latent_dim] from the WM graph.
        let mut z_stack = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(1, &mut z_stack);
        let mut sq_stack = vec![0.0f32; n * ld];
        self.wm_session.read_output_by_index(2, &mut sq_stack);
        // Stability guard: clamp latent components to a finite range
        // before anything downstream (surprise computation, policy
        // input, reward bonus, buffer writes). The encoder's final
        // `fc2` is unbounded — under long training on simple envs it
        // can drive `|z|` arbitrarily large, and the resulting policy
        // logits blow up to NaN within a few tens of thousands of
        // steps. Clamping to ±10 preserves all normal-range
        // representations (Xavier init produces values in ~[-2, 2])
        // while bounding logit magnitudes through the policy's
        // `Linear(latent_dim, hidden)` — exp(10·‖w‖) stays well inside
        // f32 before any softmax.
        for v in z_stack.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            } else {
                *v = v.clamp(-10.0, 10.0);
            }
        }

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
            let base_reward = lane.reward_circuit.compute(surprise, novelty, homeo, order);

            // Goal-alignment bonus (Phase G): when L1 is active, add
            // `α · cos(z_t, goal)` — scale-invariant, bounded in
            // `[-α, +α]`. Kept in `last_reward` (advantage input) but
            // NOT in `last_base_reward` (value TD target), so the
            // value baseline can't absorb the option signal.
            let mut bonus = 0.0f32;
            if self.option_session.is_some() && self.config.goal_bonus_alpha > 0.0 {
                let cos = unit_cosine(z_row, &lane.option_goal);
                bonus = self.config.goal_bonus_alpha * cos;
            }
            let reward = base_reward + bonus;

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
            lane.last_base_reward = base_reward;

            // L1: accumulate reward into the current option's return and
            // count down. The next act() call will detect steps_left == 0
            // and handle training + resampling.
            lane.option_return += reward;
            lane.option_steps_left = lane.option_steps_left.saturating_sub(1);
            lane.option_elapsed = lane.option_elapsed.saturating_add(1);
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
        if loss.is_finite() {
            self.last_replay_loss = loss;
        } else {
            log::warn!(
                "replay loss went non-finite at step {}, re-initialized WM params",
                self.step_count
            );
            init_parameters(&mut self.wm_session);
            self.last_replay_loss = 0.0;
        }
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

        let loss = self.credit_session.read_loss();
        if loss.is_finite() {
            self.last_credit_loss = loss;
        } else {
            log::warn!(
                "credit loss went non-finite at step {}, re-initialized credit params",
                self.step_count
            );
            init_parameters(&mut self.credit_session);
            self.last_credit_loss = 0.0;
        }

        let mut credit_logits = vec![0.0f32; h];
        self.credit_session
            .read_output_by_index(1, &mut credit_logits);

        let alpha = credit::softmax(&credit_logits);
        let credits: Vec<f32> = alpha.iter().map(|&a| r_t * a).collect();
        self.lanes[lane_idx].buffer.write_credits(&credits);
    }

    /// Run the L1 credit assigner for each lane that just terminated an
    /// option (and thus has a fresh row in `option_history`). Trains the
    /// option-credit graph contrastively on the lane's own option
    /// history; reads back credit logits, softmaxes them, and updates
    /// the per-lane `h_eff_l1` diagnostic. Agent-wide loss reported as
    /// the last lane's loss (sufficient for diagnostic tracking).
    ///
    /// Called from `act()` after new history entries are pushed and
    /// before the next options are sampled.
    fn option_credit_step<R: Rng>(&mut self, rng: &mut R, lanes_to_update: &[usize]) {
        let Some(ref mut opt_credit_sess) = self.option_credit_session else {
            return;
        };
        let history_len = self.config.option_history_len;
        let num_options = self.config.num_options;
        let latent_dim = self.latent_dim;
        let lr = self.config.lr_option_credit * self.batch_lr_scale;
        let warmup_done = self.step_count >= self.config.warmup_steps;

        for &i in lanes_to_update {
            // Stage the per-lane history + contrastive target, then
            // drop the borrow before touching the session mutably.
            let (history_flat, target_flat) = {
                let lane = &self.lanes[i];
                if lane.option_history.len() < history_len {
                    continue;
                }
                let hist = lane
                    .option_history
                    .flatten(history_len, num_options, latent_dim);
                let tgt = if let Some((hi, lo)) =
                    lane.option_history.find_contrastive_pair(rng, history_len)
                {
                    lane.option_history.contrastive_target(hi, lo, history_len)
                } else {
                    vec![1.0 / history_len as f32; history_len]
                };
                (hist, tgt)
            };

            opt_credit_sess.set_input("history", &history_flat);
            opt_credit_sess.set_input("credit_target", &target_flat);
            opt_credit_sess.set_learning_rate(if warmup_done { lr } else { 0.0 });
            opt_credit_sess.step();
            opt_credit_sess.wait();

            let loss = opt_credit_sess.read_loss();
            if loss.is_finite() {
                self.last_option_credit_loss = loss;
            } else {
                log::warn!(
                    "option-credit loss went non-finite at step {}, re-initialized option-credit params",
                    self.step_count
                );
                init_parameters(opt_credit_sess);
                self.last_option_credit_loss = 0.0;
            }

            let mut credit_logits = vec![0.0f32; history_len];
            opt_credit_sess.read_output_by_index(1, &mut credit_logits);
            // Sanitize any NaN/Inf before softmax: the credit graph
            // occasionally produces non-finite values in early training
            // on small-variance envs (Pendulum). Fall back to 0 so the
            // softmax becomes uniform and `h_eff_l1` stays finite.
            for v in credit_logits.iter_mut() {
                if !v.is_finite() {
                    *v = 0.0;
                }
            }
            self.last_h_eff_l1 = credit::effective_scope(&credit_logits);
        }
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
        // Build stacked value targets: use the *base* reward (no L1
        // goal-alignment bonus). The bonus is retained in `last_reward`
        // for the advantage computation below — so the value baseline
        // doesn't learn to cancel the option-discriminating signal.
        // When L1 is inactive the two are identical. Clamp the target
        // to a sane range so a runaway reward stream (e.g. a very
        // negative homeo deviation) can't drive the value MSE into
        // gradient magnitudes that explode the value-head weights.
        for (i, lane) in self.lanes.iter().enumerate() {
            let v = lane.last_base_reward;
            self.value_target_scratch[i] = if v.is_finite() {
                v.clamp(-100.0, 100.0)
            } else {
                0.0
            };
        }

        // Build per-row advantage-weighted action targets.
        //
        // Raw clamped advantages (not normalized). Normalization was
        // tested and made L0 worse at N=64: compressing all advantages
        // to similar magnitudes removes the signal diversity that lets
        // the cross-entropy gradient find a good local optimum. The
        // 462-landing best (commit 18d422f) used raw clamped advantages;
        // re-introducing normalization later caused L0-42 to degrade
        // from −131 to −849.
        //
        // Phase G v3: `entropy_floor` no longer suppresses updates
        // below the floor (which locked deterministic collapse once
        // entropy hit zero — with batch_size=1 the policy then never
        // recovered). Instead, `entropy_deficit = (floor - entropy) /
        // floor`, clamped to `[0, 1]`, drives two self-correcting
        // changes to the update target for that lane:
        //
        //   1. Label-smoothing `eps` is amplified toward `1.0` —
        //      the softmax target shifts from the taken action's
        //      one-hot toward the uniform distribution. The
        //      cross-entropy gradient then pulls the dominant logit
        //      down and the rest up, restoring entropy.
        //   2. When the real reward advantage is near zero (value
        //      tracks reward well, so the policy gradient would
        //      otherwise carry no signal), `entropy_deficit` is used
        //      as the effective advantage magnitude — a positive
        //      recovery signal applied to the now-uniform target.
        //
        // Both reduce to the old behaviour at entropy ≥ floor
        // (deficit = 0 → eps = eps_base, effective_adv = advantage).
        self.policy_action_scratch.fill(0.0);
        let mut any_active = false;
        let eps_base = self.config.label_smoothing;
        let floor = self.config.entropy_floor;
        let k = MAX_ACTION_DIM as f32;
        for (i, lane) in self.lanes.iter().enumerate() {
            let advantage = (lane.last_reward - lane.last_value).clamp(-1.0, 1.0);
            let entropy_deficit = if floor > 0.0 {
                ((floor - lane.last_entropy) / floor).clamp(0.0, 1.0)
            } else {
                0.0
            };

            // Nothing to do only when there's no reward signal AND
            // entropy is comfortably above the floor.
            if advantage.abs() < 1e-8 && entropy_deficit < 1e-6 {
                continue;
            }
            any_active = true;

            // Amplify label-smoothing toward uniform when entropy is low.
            let eps = (eps_base + (1.0 - eps_base) * entropy_deficit).min(1.0);
            // Synthesize a recovery advantage only when the real
            // reward advantage has gone silent — otherwise use the
            // signal advantage unchanged. The soft-tanh-clamp on the
            // policy logits (see `policy.rs`) prevents the NaN-from-
            // unbounded-logits failure mode on its own, so we don't
            // need to amplify recovery to dominate the signal.
            let effective_adv = if advantage.abs() < 1e-3 && entropy_deficit > 0.0 {
                entropy_deficit
            } else {
                advantage
            };
            let act_src = &self.action_token_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            let act_dst =
                &mut self.policy_action_scratch[i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
            for (dst, &src) in act_dst.iter_mut().zip(act_src.iter()) {
                let smoothed = (1.0 - eps) * src + eps / k;
                *dst = effective_adv * smoothed;
            }
        }
        if !any_active {
            return;
        }

        // Feed the pure latent `z` plus, when L1 is active, the per-lane
        // one-hot `option_onehot` so the policy graph's option bias head
        // receives the current option identity for training.
        self.policy_session.set_input("z", z_stack);
        if self.option_session.is_some() {
            let num_options = self.config.num_options;
            self.option_onehot_scratch.fill(0.0);
            for (i, lane) in self.lanes.iter().enumerate() {
                let row = &mut self.option_onehot_scratch[i * num_options..(i + 1) * num_options];
                row[lane.current_option as usize] = 1.0;
            }
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        }
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        self.policy_session
            .set_learning_rate(self.config.lr_policy * self.batch_lr_scale);
        self.policy_session.step();
        self.policy_session.wait();

        let loss = self.policy_session.read_loss();
        // Watchdog: reset on non-finite OR absolute magnitude > 1000.
        // The latter catches the "finite but runaway" regime observed
        // on LunarLander after a brief performance peak — the
        // cross-entropy loss can plunge into the thousands of
        // magnitude when `log_softmax` produces values of order -1000
        // from extreme logit spreads, even though every individual
        // number is technically finite. Reset restores uniform-ish
        // softmax and lets the agent re-climb.
        if !loss.is_finite() || loss.abs() > 1000.0 {
            init_parameters(&mut self.policy_session);
            log::warn!(
                "policy loss {:.1} unstable at step {}, re-initialized policy params",
                loss,
                self.step_count
            );
            self.last_policy_loss = 0.0;
        } else {
            self.last_policy_loss = loss;
        }
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

                // L1 goal distance: ‖last_latent − goal‖.
                let goal_distance = if self.option_session.is_some() {
                    if let Some(prev) = lane.buffer.last() {
                        prev.latent
                            .iter()
                            .zip(lane.option_goal.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>()
                            .sqrt()
                    } else {
                        0.0
                    }
                } else {
                    0.0
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
                    current_option: lane.current_option,
                    option_return: lane.option_return,
                    goal_distance,
                    h_eff_l1: self.last_h_eff_l1,
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
