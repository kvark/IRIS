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
use crate::approach;
use crate::buffer::{ExperienceBuffer, Transition};
use crate::coord;
use crate::credit;
use crate::delta_goals;
use crate::encoder::{CnnEncoder, Encoder};
use crate::env::{Action, ActionKind, Environment, Observation};
use crate::option;
use crate::outcome;
use crate::planner;
use crate::policy;
use crate::reward::{RewardCircuit, RewardWeights};
use crate::rnd;
use crate::world_model::WorldModel;
use crate::xeps_memory;
use hashbrown::HashMap;
use meganeura::Session;
use meganeura::graph::Graph;
use meganeura::nn;
use rand::Rng;

/// What encoder kindle builds as the WM graph's backbone.
///
/// `Mlp` (default) is kindle's original dense encoder — flat
/// obs-token vector → hidden → latent. Suits structured
/// low-dim obs (CartPole, LunarLander, Taxi, etc.).
///
/// `Cnn { channels, height, width }` builds a small conv-net
/// encoder on raw NCHW pixel/grid input. Intended for visual
/// tasks (ARC-AGI-3's 64×64 colour grid, Atari-like frames).
/// When this variant is selected, `Agent::set_visual_obs` must
/// be called every step before `observe` with a flat
/// `[batch_size · channels · height · width]` input; the obs
/// token path still flows to the reward circuit in parallel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    Mlp,
    Cnn {
        channels: u32,
        height: u32,
        width: u32,
    },
}

impl EncoderKind {
    /// Flat element count for visual obs: `0` for Mlp,
    /// `channels · height · width` for Cnn.
    pub fn visual_dim(&self) -> usize {
        match *self {
            EncoderKind::Mlp => 0,
            EncoderKind::Cnn {
                channels,
                height,
                width,
            } => (channels as usize) * (height as usize) * (width as usize),
        }
    }
}

/// Criterion for ranking terminal entries into the M7 prototype
/// buffer's top-P% fraction. See `AgentConfig::approach_rank_by`.
///
/// The M7 confidence-weighting run (commit ac27d5c) showed that
/// ranking by `Return` converges on the wrong prototype on
/// LunarLander, because the highest-return episodes under v3
/// homeo are the shortest crashes. `Novelty` ranks by terminal
/// rarity instead, which on LunarLander promotes the rare
/// soft-landing terminal basin. Trade-off: on envs where success
/// is a *common* terminal state (e.g. CartPole timeouts), `Novelty`
/// promotes rare-crash terminals and is counter-productive;
/// `Return` is the right choice there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApproachRankBy {
    /// Cumulative intrinsic episode return (default; M7 v1 behaviour).
    Return,
    /// `1 / sqrt(visit_count(z_end))` — standard kindle novelty at
    /// the terminal latent, computed on the lane's grid-discretized
    /// experience buffer. Rare terminals rank high.
    Novelty,
}

/// What target the M6 outcome-value head trains against at
/// episode completion. See `AgentConfig::outcome_target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeTarget {
    /// Train `R̂` against the centered *sum* of per-step `r_base`
    /// over the just-ended episode. Every step of the episode
    /// shares one target. The original design; M6 v2 showed this
    /// converges on an episode-wide mean (intra-episode-flat `R̂`).
    EpisodeSum,
    /// Train `R̂` against the *last step's* `r_base` before episode
    /// boundary. Same shared-target-per-episode shape as
    /// `EpisodeSum` but aimed at the terminal state's homeo
    /// profile.
    TerminalReward,
    /// Train `R̂` against a *per-step* reward-to-go target:
    /// `target(t) = Σ_{k=t}^{T} r_base_k`. Each step in a
    /// completed episode now carries its own supervision signal,
    /// which defeats the M6 v2 "intra-episode-flat" failure mode —
    /// early windows of a soft-landing episode get a higher
    /// expected-future-reward target than early windows of a crash
    /// episode, so `R̂` learns per-step differentiation instead of
    /// a single per-episode bias.
    RewardToGo,
}

/// How M6 injects its output into the agent-facing reward.
///
/// `Raw` is the default — simply add `α · clamp(R̂)` to the step
/// reward. `PotentialDelta` adds `α · (R̂_t − R̂_{t-1})` instead,
/// i.e. the per-step *change* in the state-value estimate rather
/// than its absolute value. Classical potential-based shaping (Ng
/// et al. 1999) — guaranteed policy-invariant up to a constant,
/// converts a state-value head into a per-step signal
/// automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeBonus {
    Raw,
    PotentialDelta,
}

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
    /// Phase G Tier-3: replace the shared L0 `fc2` head with N
    /// per-option `[hidden_dim → action_dim]` heads gated by
    /// `option_onehot`. Each option's head only receives gradient
    /// when that option is active, silo'ing per-option capacity so
    /// different options can commit to genuinely different action
    /// distributions without fighting in one shared parameter space.
    /// Only effective when `num_options >= 2`; ignored otherwise.
    /// Default `true` — the additive-bias-only behaviour (Phase G v2
    /// through Tier-1) is kept as the `false` path for A/B.
    pub per_option_heads: bool,
    /// Sequence-level credit: discount factor for the n-step return
    /// used as the policy advantage baseline target. `0.0 ≤ γ < 1`.
    /// Only consulted when `n_step >= 2`. Default `0.95`.
    pub gamma: f32,
    /// Sequence-level credit: lookahead horizon for the policy
    /// advantage. `1` (default) preserves the pre-GAE single-step
    /// advantage (`r_t − V(s_t)`). `n ≥ 2` delays the policy update
    /// by `n−1` steps so it can fold in the next `n` rewards as a
    /// Monte-Carlo return:
    ///   R_t = Σ_{k=0}^{n-1} γ^k · r_{t+k}   (truncated at
    ///         `env_boundary`), normalized by Σ γ^k so the magnitude
    ///         stays comparable to a single-step reward.
    ///   advantage_t = clamp(R_t − V(s_t), -1, 1).
    /// The value head still trains on the single-step reward target
    /// (`lane.last_base_reward`) to avoid the bootstrap-instability
    /// failure mode that killed the earlier GAE attempt (commit
    /// 8f291e5). 4–8 is a reasonable range for LunarLander-scale
    /// episodes (100–300 env steps).
    pub n_step: usize,
    /// M6 learnable reward (see `docs/phase-m6-learnable-reward.md`):
    /// weight α on the outcome-value head's prediction when it's added
    /// as a fifth reward primitive — `r_t = r_base + α · R̂(z_t)`.
    /// `0.0` (default) disables M6; behaviour is byte-identical to the
    /// pre-M6 agent.
    pub outcome_reward_alpha: f32,
    /// M6: learning rate for the outcome-value head. `None` (default)
    /// resolves to `learning_rate × 0.3` — same scale as the credit head.
    pub lr_outcome: Option<f32>,
    /// M6: EMA rate for the running baseline used to center episode-
    /// return targets fed to the outcome head. Lower = smoother
    /// baseline, higher variance reduction but slower drift. Default
    /// `0.05`.
    pub outcome_baseline_ema: f32,
    /// M6: what target the outcome head trains against on episode
    /// completion. The M6 mechanism check (2026-04-19) showed that
    /// `EpisodeSum` — the original design — inherits the silence of
    /// kindle's four primitives on event-ordering: soft and crash
    /// LunarLander episodes produce nearly-identical cumulative
    /// returns, so the head learns an episode-length correlate
    /// instead of a quality signal. `TerminalReward` trains R̂ on
    /// just the *last* step's `r_base` instead, a different self-
    /// observable that can (and does, for v5 shaping) differ
    /// strongly between soft and crash if the reward circuit's
    /// terminal homeo profile already differs.
    pub outcome_target: OutcomeTarget,
    /// M6: how the head's output enters the per-step reward. See
    /// `OutcomeBonus`. `Raw` is the default; `PotentialDelta` is
    /// the Ng-et-al potential-based-shaping variant.
    pub outcome_bonus: OutcomeBonus,
    /// Symmetric clamp applied to the per-row policy advantage
    /// `(reward − value)` before it scales the action target. The
    /// historical default `1.0` bounds the cross-entropy gradient
    /// but also *destroys* advantage magnitude on steps where the
    /// reward actually differentiates actions (typical LunarLander
    /// mid-flight decisions, where r−V can be ±5 but gets clipped
    /// to ±1). Raise this when diagnosing a policy that fails to
    /// commit under default-regularized kindle.
    pub advantage_clamp: f32,
    /// Magnitude threshold above which the policy-loss watchdog
    /// re-initializes policy parameters (alongside the non-finite
    /// check). Default `1000.0` catches the "finite but runaway"
    /// regime that can follow a brief performance peak. Raise this
    /// to a very large value to effectively disable the watchdog
    /// when ablating whether its resets are preventing convergence.
    pub policy_loss_watchdog_threshold: f32,
    /// M7 approach-reward weight. When `> 0`, kindle maintains a
    /// single prototype centroid in latent space (the mean of the
    /// top-`approach_top_frac` fraction of recent terminal latents
    /// by episode return) and adds `−α · ‖z_t − centroid‖` to the
    /// per-step reward. Gives the reward a continuous
    /// approach-shaping signal that the homeo primitives cannot
    /// express. Zero (default) disables M7; byte-identical pre-M7.
    /// See `docs/phase-m7-approach-reward.md`.
    pub approach_reward_alpha: f32,
    /// M7 rolling buffer of (terminal_latent, episode_return) pairs.
    pub approach_buffer_size: usize,
    /// M7 fraction of highest-return terminals to average into the
    /// prototype. 0.2 = the top-20%.
    pub approach_top_frac: f32,
    /// M7 episodes between centroid recomputes. Lower = prototype
    /// tracks recent experience faster but jitters more; higher =
    /// stable but stale.
    pub approach_update_interval: usize,
    /// M7 warmup: no approach reward until this many completed
    /// episodes have been observed (so the prototype is built from
    /// a non-trivial distribution).
    pub approach_warmup_episodes: usize,
    /// M7 symmetric distance clamp (pre-α multiplication) to bound
    /// worst-case per-step bonus magnitude.
    pub approach_distance_clamp: f32,
    /// M7 confidence saturation: number of completed episodes
    /// required for the prototype to reach full confidence
    /// (`c = 1`). Confidence ramps linearly from `0` at
    /// `approach_warmup_episodes` up to `1` at
    /// `warmup + saturation` episodes. When ramping is active, the
    /// per-step approach bonus is scaled by `c` and the homeo
    /// reward is tapered by `homeo_confidence_taper · c`. `0`
    /// (default) disables ramping — `c = 1` always, behaviour
    /// identical to the non-confidence-aware M7 v1.
    pub approach_confidence_saturation: usize,
    /// M7↔homeo integration: fraction of the homeo reward to
    /// remove once `c = 1`. `homeo_effective = homeo_raw · (1 − τ · c)`.
    /// `0` (default) keeps homeo at full weight even when M7 is
    /// fully confident (M7 is purely additive). `0.5` halves the
    /// homeo contribution at full confidence so M7 can lead
    /// without homeo's misaligned basin dominating. `1.0` turns
    /// homeo off entirely at full confidence — M7 becomes the sole
    /// reward signal.
    pub homeo_confidence_taper: f32,
    /// M7 prototype-selection criterion. `Return` (default) ranks
    /// terminal entries by cumulative kindle intrinsic reward —
    /// fine when the env's high-return terminals are also task-
    /// success terminals (the 6 kindle-friendly envs, to varying
    /// strengths). `Novelty` ranks by `1 / sqrt(visit_count)` at
    /// the terminal latent, promoting rare terminals over common
    /// ones. On LunarLander this is expected to help (soft
    /// landings are rare, crashes are common); on CartPole etc.
    /// it would hurt (timeout-success is common, so novelty
    /// demotes it).
    pub approach_rank_by: ApproachRankBy,
    /// RND curiosity weight. When `> 0`, kindle builds a
    /// Random-Network-Distillation pair (frozen random target +
    /// trainable predictor, both MLPs on the agent's latent
    /// `z`) and adds `α · mse(predictor(z), target(z))` to the
    /// per-step reward. Zero (default) disables RND and skips
    /// both net construction and per-step training overhead.
    /// Unlike the surprise primitive, RND's prediction error
    /// doesn't decay as the world model converges, so it remains
    /// a live exploration signal throughout training — designed
    /// for the curiosity-death failure mode the ARC-AGI-3 sweep
    /// exposed (see commit 780c9a2).
    pub rnd_reward_alpha: f32,
    /// RND feature dim (target + predictor output size). Default 16.
    pub rnd_feature_dim: usize,
    /// RND hidden-layer width. Default 64.
    pub rnd_hidden_dim: usize,
    /// RND predictor learning rate. `None` → `learning_rate × 0.3`,
    /// matching the credit / outcome-head scale.
    pub rnd_lr: Option<f32>,
    /// Continuous coordinate-action head. `0.0` (default)
    /// disables it — kindle doesn't build the head and
    /// `sample_coords()` returns zeros. `α > 0` constructs a
    /// CPU MLP `z → (μ_x, μ_y) ∈ [−1, 1]` that the harness can
    /// sample from to pick spatial-click coordinates on envs
    /// that support them (ARC-AGI-3 complex actions). The head
    /// trains via REINFORCE on a per-step advantage supplied by
    /// the harness; `α` scales the reinforcement magnitude.
    pub coord_action_alpha: f32,
    /// Hidden width for the coord head. Default 32.
    pub coord_hidden_dim: usize,
    /// Gaussian exploration noise stddev in the `[−1, 1]` action
    /// space. Default 0.3 — enough diversity to cover the coord
    /// space, narrow enough to leave signal in the mean.
    pub coord_sigma: f32,
    /// LR for the coord head's REINFORCE update. `None` →
    /// `learning_rate × 0.3`.
    pub coord_lr: Option<f32>,
    /// M8 delta-goal reward. When `> 0`, kindle maintains a bank of
    /// latent positions where a significant state change was just
    /// observed (`‖z_cur − z_prev‖ ≥ delta_goal_threshold`) and
    /// rewards the policy with `−α · min_i ‖z − g_i‖` (clamped by
    /// `delta_goal_distance_clamp`). Self-supervised: no task labels.
    /// Goal-bank stays diverse because candidates within
    /// `delta_goal_merge_radius` of an existing goal are dropped.
    /// Zero (default) disables M8 and skips construction.
    pub delta_goal_alpha: f32,
    /// M8 minimum per-step latent-delta to trigger a new goal entry.
    /// Below this, a step's `z_cur` is not recorded even if it lands
    /// in a novel region. Calibrate alongside `latent_dim`.
    pub delta_goal_threshold: f32,
    /// M8 merge radius: a candidate goal within this L2 distance of
    /// any existing bank entry is considered a duplicate and dropped.
    pub delta_goal_merge_radius: f32,
    /// M8 maximum bank size. Oldest entries are evicted first.
    pub delta_goal_bank_size: usize,
    /// M8 symmetric distance clamp (pre-α) to bound worst-case
    /// per-step reward magnitude.
    pub delta_goal_distance_clamp: f32,
    /// M8 v2: minimum world-model prediction error required to
    /// consider recording a new goal. Only transitions where
    /// `pred_error >= delta_goal_surprise_threshold` AND
    /// `‖Δobs‖ >= delta_goal_threshold` enter the bank. Banking on
    /// raw obs-deltas alone (v1) collects the agent's routine
    /// trajectory — no gradient toward new behaviour. Gating on
    /// WM surprise biases the bank toward transitions the WM
    /// didn't predict, i.e. genuinely unexpected events. `0.0`
    /// (treat as "no surprise gate") preserves v1 semantics.
    pub delta_goal_surprise_threshold: f32,
    /// Cross-episode state-action novelty weight. When `> 0`,
    /// kindle maintains a persistent `(quantized_state, action)`
    /// visit counter (shared across lanes, spanning episode
    /// boundaries) and emits `α / sqrt(1 + count)` as an intrinsic
    /// per-step reward keyed on the PREVIOUS step's (state, action)
    /// pair. Unlike the existing state-only novelty primitive,
    /// this discriminates between actions at the same state —
    /// targeting the "reach L1 every episode and retry the same
    /// actions there" failure mode observed on ARC-AGI-3. Zero
    /// (default) disables the memory and skips construction.
    pub xeps_reward_alpha: f32,
    /// Grid resolution for the cross-episode memory's state key.
    /// `None` → reuse `grid_resolution` (the state-novelty bucket
    /// size), keeping the two quantization schemes consistent.
    pub xeps_grid_resolution: Option<f32>,
    /// Extrinsic-reward weight. When `> 0`, the harness can supply
    /// a per-lane scalar via `Agent::set_extrinsic_reward(&[f32])`
    /// before each `observe()` call; kindle adds
    /// `α · extrinsic[i]` to the per-step reward used by credit
    /// and policy training. Unlike the homeo primitive — which
    /// always subtracts a positive deviation from a target and
    /// thus produces a one-sided signal — extrinsic reward is
    /// signed and passes through unchanged, so sparse ±1 signals
    /// (Atari, Gym classic control reward) reach policy gradient
    /// with the correct sign. Kindle's value head absorbs the DC
    /// offset automatically; the advantage sees the per-step
    /// variance. Zero (default) preserves the pre-primitive
    /// behaviour byte-for-byte.
    ///
    /// Note: using the extrinsic primitive couples kindle's policy
    /// to a task-specific reward signal and thus violates the
    /// cold-start self-training thesis. Prefer expressing goals
    /// via homeostatic variables when possible; this channel is
    /// primarily diagnostic (validating kindle's policy-gradient
    /// machinery on envs where the reward signal is known).
    pub extrinsic_reward_alpha: f32,
    /// Global advantage-norm clip applied across the batch before
    /// the policy update is built. When `> 0`, the L2 norm of the
    /// per-lane advantage vector is bounded: if
    /// `‖adv‖_2 > policy_adv_global_clip`, every lane's advantage
    /// is rescaled by `clip / ‖adv‖_2`. This is the policy-gradient
    /// analogue of global-grad-norm clipping — since each lane's
    /// contribution to the policy gradient is linear in its
    /// advantage, clipping advantages bounds the gradient norm.
    /// Complements per-lane `advantage_clamp` (which is an L∞
    /// bound). Zero (default) disables.
    pub policy_adv_global_clip: f32,
    /// Adaptive LR target for policy updates. When `> 0`, kindle
    /// maintains an EMA of `|pi_loss|` (smoothed by
    /// `policy_lr_adaptive_ema`) and scales the per-step policy
    /// learning rate by `target / max(ema, target)`. When the
    /// policy is taking huge update steps (loss magnitude far
    /// above the target), effective LR drops proportionally;
    /// when the policy has settled into a range below the target,
    /// full LR is used. Classic adaptive-step-size damping — the
    /// simpler cousin of KL-constrained TRPO. Zero (default)
    /// disables.
    pub policy_lr_adaptive_target: f32,
    /// EMA rate for the `|pi_loss|` tracking in adaptive LR.
    /// Default `0.05` (20-step effective window). Higher = more
    /// responsive but noisier.
    pub policy_lr_adaptive_ema: f32,
    /// TD-bootstrap the value-head target (only affects
    /// `policy_step_n_step`, i.e. when `n_step >= 2`).
    ///
    /// Default `false` — the value head trains on single-step
    /// reward at the ripe state, which on sparse-reward envs
    /// leaves V≈0 everywhere and the advantage signal is
    /// sparse-but-correctly-signed.
    ///
    /// When `true`: value target becomes the bootstrapped n-step
    /// return
    /// ```text
    /// V_target(s_ripe) = Σ_{k=0}^{n-1} γ^k r_{ripe+k}
    ///                  + γ^n · V(s_{ripe+n})
    /// ```
    /// (with bootstrap suppressed when the episode terminates
    /// inside the window or at the bootstrap point). This is the
    /// classical n-step TD target — rewards propagate backward
    /// through V via Bellman recursion, so every state with any
    /// causal connection to a future reward gets a dense
    /// non-zero V estimate and the advantage `ret - V(s_ripe)`
    /// carries gradient at every step, not just at reward events.
    ///
    /// Requires `buf_len >= n_step + 1` (one extra transition for
    /// the bootstrap). Lanes with shorter buffers fall back to
    /// bootstrap=0 (equivalent to treating them as episode-
    /// terminating, which is the safe conservative default).
    ///
    /// The bootstrap uses the STORED `τ_{ripe+n_step}.value` — a
    /// value prediction from the policy session at the time that
    /// transition was observed. This is a "target-network" style
    /// stale estimate, which is stable by construction (the
    /// bootstrap target doesn't change under the current update,
    /// avoiding the DQN-style divergence mode).
    pub value_bootstrap: bool,
    /// GAE (Generalized Advantage Estimation) λ parameter.
    /// `0.0` = disabled (advantage = `R_n − V(s_ripe)` as before).
    /// `(0, 1]` = enable:
    /// ```text
    /// Â_t = Σ_{k=0}^{n-1} (γλ)^k · δ_{t+k}
    /// δ_{t+k} = r_{t+k} + γ·V(s_{t+k+1}) - V(s_{t+k})
    /// ```
    /// Exponentially-weighted average over all n-step TD targets —
    /// λ=0 → pure 1-step TD (low variance, high bias),
    /// λ=1 → Monte-Carlo return − V (high variance, unbiased),
    /// λ≈0.95 is the PPO/A2C default.
    ///
    /// Why this matters over plain `value_bootstrap`: GAE decouples
    /// the *advantage* estimator from the *value target*. With only
    /// `value_bootstrap`, advantage = `R_n − V(s_ripe)` and V is
    /// trained to fit `R_n` — so as V gets accurate, advantage → 0
    /// and the policy-gradient signal dies. GAE advantages are
    /// TD-error-based, so they stay non-trivial (E[Â_t]=0 but
    /// Var[Â_t] > 0) even when V has converged.
    ///
    /// Enabling GAE also enables the bootstrap headroom (needs
    /// `V(s_{t+1})` for each per-step δ). The value target itself
    /// stays on the `value_bootstrap` path — set `value_bootstrap =
    /// true` alongside `gae_lambda > 0` for the standard A2C setup.
    pub gae_lambda: f32,
    /// Coefficient on the value-head MSE before it's summed with
    /// the policy loss. Standard PPO/A2C use `0.5`; defaults to
    /// `1.0` here for backward compatibility.
    ///
    /// With a shared optimizer, the combined-loss gradient is
    /// dominated by whichever head produces the larger loss
    /// magnitude. On dense-reward envs the value MSE is on the
    /// reward scale (potentially tens or hundreds), while the
    /// policy cross-entropy is on a log-scale (O(log K) ≈ 0.7 for
    /// CartPole). Leaving the value at 1.0 makes the policy
    /// effectively learn ~50× slower than the value. Setting this
    /// to 0.1–0.5 rebalances without separate LRs or sessions.
    pub value_loss_coef: f32,
    /// Update the policy only every N env-steps, then do N
    /// gradient steps in a row on the accumulated rollout.
    /// Default `1` = per-env-step update (the existing behavior).
    ///
    /// Why this matters: kindle's per-step update means by the
    /// time a transition becomes "ripe" (n_step in the past), the
    /// policy has already taken n_step + 1 gradient steps since
    /// that transition was collected. The data is effectively
    /// off-policy from the very first update, which biases the
    /// policy-gradient estimator and produces the commit/uncommit
    /// oscillation seen on CartPole. Setting `policy_update_interval
    /// = n_step + 1` keeps the rollout fully on-policy with respect
    /// to the policy that collected it — the standard A2C/PPO
    /// setup. On the fire step, the agent does `interval` sequential
    /// gradient steps, each trained on the ripe transition at a
    /// different offset back through the rollout buffer.
    pub policy_update_interval: usize,
    /// Normalize advantages per-batch to zero mean / unit std before
    /// feeding them into the policy update. Standard PPO/A2C trick.
    /// Default `false` (backward compat).
    ///
    /// On many envs V lags the reward early on, so advantages are all
    /// same-sign (e.g. +1/+2 every step when reward is +1). Same-sign
    /// advantages push the policy toward whichever action a majority of
    /// lanes happened to take, regardless of whether that action is
    /// actually better — the advantage signal carries the *bias* from
    /// "V doesn't predict reward yet" rather than the *differential*
    /// information we actually need. Mean-centering strips the bias;
    /// variance-normalization fixes the gradient-magnitude scale.
    pub advantage_normalize: bool,
    /// Enable the PPO clipped-surrogate policy loss. When `true`, the
    /// agent uses `build_ppo_policy_graph` instead of the plain
    /// advantage-weighted CE path — see that function's docstring for
    /// the formula. The clipped ratio provides a mathematical trust
    /// region: once the policy has moved ε away from the data-
    /// collection policy on a given transition, the gradient through
    /// that transition drops to zero. This is what stabilizes committed
    /// policies and closes the commit/uncommit oscillation that
    /// advantage-normalized on-policy PG cannot escape on its own.
    /// Default `false`. Requires `policy_update_interval > 1` (at least
    /// some rollout window) for the ratio to be meaningfully ≠ 1 across
    /// inner-loop steps.
    pub use_ppo: bool,
    /// PPO clip radius ε. Standard value `0.2`. Ratio is clipped to
    /// `[1 − ε, 1 + ε]`.
    pub ppo_clip_eps: f32,
    /// Number of epochs to replay each rollout through the PPO
    /// update. On epoch 1 the ratio is ≈ 1 everywhere (policy
    /// hasn't moved since collection), so the clip does nothing —
    /// the update is identical to plain advantage-weighted PG. On
    /// epochs 2+, the policy has drifted from π_old, ratios diverge
    /// from 1, and the clip actually caps updates that would move
    /// the policy past the trust region. Standard PPO uses 3–10.
    /// Only has effect when `use_ppo = true`. Default 1 (equivalent
    /// to the no-clip baseline for single-epoch runs).
    pub ppo_n_epochs: usize,
    /// Model-based planner. When `> 0`, kindle maintains a CPU
    /// copy of the world-model weights and can simulate candidate
    /// action sequences via `plan_and_queue()` — sampling
    /// `planner_samples` random sequences of length `planner_horizon`,
    /// rolling each out through the frozen WM, scoring by sum of
    /// `1/sqrt(1+visit_count)` over the predicted latents, and
    /// queueing the best sequence for subsequent `act()` calls to
    /// consume. Disabled (0) by default. Applies to discrete-action
    /// envs only (continuous actions are skipped; the queue stays
    /// empty).
    pub planner_horizon: usize,
    /// Number of random action sequences sampled per plan call.
    /// Cost scales O(samples × horizon × hidden²). Default 32.
    pub planner_samples: usize,
    /// Steps between WM-weight refreshes into the planner's CPU
    /// cache. Too frequent wastes cycles; too infrequent uses
    /// stale weights. Default 200.
    pub planner_refresh_interval: usize,
    /// WM encoder backbone. `Mlp` (default) = kindle's original
    /// obs-token encoder; `Cnn { channels, height, width }` =
    /// conv-net encoder for visual/grid inputs (ARC-AGI-3 etc.).
    /// See `EncoderKind` doc for the protocol.
    pub encoder_kind: EncoderKind,
    /// M6 v2: window size for the outcome head's input. `1`
    /// (default) reduces to single-frame `R̂(z_t)` — the back-compat
    /// M6 v1 path. `k ≥ 2` concatenates the last `k` encoder
    /// latents `[z_{t-k+1}, ..., z_t]` so the head can read
    /// trajectory momentum, not just present-state. The M6 v1
    /// mechanism check (2026-04-19) showed that single-frame inputs
    /// can't discriminate soft from crash mid-flight because two
    /// identical mid-flight z's can precede different outcomes — a
    /// windowed input gets a richer condition.
    pub outcome_window: usize,
    /// M6: symmetric clamp applied to the raw outcome-head output
    /// before multiplication by `α`. Caps the worst-case per-step
    /// bonus at `α · outcome_clamp`. Default `5.0`; raise alongside
    /// `α` when probing whether the learned signal is correct-but-
    /// too-quiet relative to the ~3–15/step homeostatic penalty.
    pub outcome_clamp: f32,
    /// M6: hard cap on the number of trajectory latents kept per
    /// episode for the batched end-of-episode backward pass. Must be
    /// large enough to contain a typical episode for the target env;
    /// episodes longer than this get their tail truncated (a warn is
    /// emitted). Default `256` — covers LunarLander episodes (100–300
    /// steps). Also used as the outcome-head's compiled batch size, so
    /// changing this recompiles the graph.
    pub outcome_max_episode_len: usize,
    /// Phase G Tier-3: EMA rate for the continuous goal-latent
    /// update. At every option termination, the terminated option's
    /// goal vector is pulled toward the observed end-state latent:
    ///   `goal[o] ← (1 − β) · goal[o] + β · z_end`.
    /// This turns the fixed-table orthogonal anchors (from
    /// `option::build_goal_table`) into learned prototypes that
    /// track where L0 actually ends up under each option, making the
    /// goal-alignment bonus a self-consistency signal rather than an
    /// arbitrary pull toward a latent axis. `0.0` (the default here
    /// is `0.02`) disables the update, giving back the pre-Tier-3
    /// fixed-table behaviour.
    pub goal_ema_rate: f32,
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
            per_option_heads: true,
            gamma: 0.95,
            n_step: 1,
            outcome_reward_alpha: 0.0,
            lr_outcome: None,
            outcome_baseline_ema: 0.05,
            outcome_target: OutcomeTarget::EpisodeSum,
            outcome_bonus: OutcomeBonus::Raw,
            advantage_clamp: 1.0,
            policy_loss_watchdog_threshold: 1000.0,
            approach_reward_alpha: 0.0,
            approach_buffer_size: 100,
            approach_top_frac: 0.2,
            approach_update_interval: 10,
            approach_warmup_episodes: 20,
            approach_distance_clamp: 10.0,
            approach_confidence_saturation: 0,
            homeo_confidence_taper: 0.0,
            approach_rank_by: ApproachRankBy::Return,
            rnd_reward_alpha: 0.0,
            rnd_feature_dim: 16,
            rnd_hidden_dim: 64,
            rnd_lr: None,
            coord_action_alpha: 0.0,
            coord_hidden_dim: 32,
            coord_sigma: 0.3,
            coord_lr: None,
            delta_goal_alpha: 0.0,
            delta_goal_threshold: 0.5,
            delta_goal_merge_radius: 0.1,
            delta_goal_bank_size: 64,
            delta_goal_distance_clamp: 5.0,
            delta_goal_surprise_threshold: 0.5,
            xeps_reward_alpha: 0.0,
            xeps_grid_resolution: None,
            extrinsic_reward_alpha: 0.0,
            policy_adv_global_clip: 0.0,
            policy_lr_adaptive_target: 0.0,
            policy_lr_adaptive_ema: 0.05,
            value_bootstrap: false,
            gae_lambda: 0.0,
            value_loss_coef: 1.0,
            policy_update_interval: 1,
            advantage_normalize: false,
            use_ppo: false,
            ppo_clip_eps: 0.2,
            ppo_n_epochs: 1,
            planner_horizon: 0,
            planner_samples: 32,
            planner_refresh_interval: 200,
            encoder_kind: EncoderKind::Mlp,
            outcome_window: 1,
            outcome_clamp: 5.0,
            outcome_max_episode_len: 256,
            goal_ema_rate: 0.02,
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
    /// L1 continuous goal prototypes: mean pairwise Euclidean distance
    /// across the `num_options` goal vectors. Zero until the table
    /// diverges from init; drops toward zero if prototypes collapse to
    /// a single point (mode collapse — a sign the EMA rate is too
    /// aggressive relative to L0's per-option differentiation).
    pub goal_diversity: f32,
    /// M6 outcome-value head prediction at this lane's latest latent.
    /// Signed (centered by the lane's baseline EMA), clamped to
    /// `[-5, +5]`. Zero when `outcome_reward_alpha == 0`. Rising magnitude
    /// over training means the head is discriminating between
    /// high-return and low-return trajectories.
    pub r_hat: f32,
    /// M6 running EMA baseline — mean episode return across all
    /// completed episodes observed so far (per-lane baselines; this
    /// diagnostic reports the last updated value across all lanes).
    /// Drifts toward the agent's asymptotic per-episode return.
    pub outcome_baseline: f32,
    /// M6 most recent training loss on a completed episode. Zero
    /// until the first episode boundary fires.
    pub outcome_loss: f32,
    /// M7 L2 distance from this lane's current latent to the
    /// prototype centroid. Zero when M7 is disabled or the
    /// prototype hasn't seeded yet.
    pub approach_distance: f32,
    /// M7 number of completed episodes in the prototype buffer.
    /// Rises until it caps at `approach_buffer_size`.
    pub approach_buffer_fill: usize,
    /// M7 centroid drift at the last recompute — L2 distance from
    /// the previous centroid. High = prototype unstable; zero =
    /// not yet recomputed twice.
    pub approach_centroid_drift: f32,
    /// M7 centroid age in episodes since last recompute. Caps at
    /// `approach_update_interval`.
    pub approach_centroid_age: usize,
    /// M7 current confidence `c ∈ [0, 1]`. Zero before warmup;
    /// ramps linearly from 0 → 1 over
    /// `approach_confidence_saturation` episodes once warmup is
    /// met. Used to scale the approach bonus and (optionally)
    /// taper the homeo reward. Always 0 when M7 is disabled.
    pub approach_confidence: f32,
    /// RND predictor MSE averaged across lanes on the most recent
    /// step. Proportional to the per-step curiosity reward
    /// (ignoring the α weight). Tracks how unfamiliar the current
    /// latent cluster is to the predictor; should be positive when
    /// curiosity is driving exploration, and drift toward zero on
    /// over-visited state regions. Always 0 when RND is disabled.
    pub rnd_mse: f32,
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
    /// π_old(a | s) for the action just sampled in `act()`. Stored
    /// into `Transition.prob_taken` in `observe()` so the PPO path can
    /// compute importance ratios. In (0, 1]; default 1.0 before first
    /// `act()`.
    last_prob_taken: f32,
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

    // --- M6 learnable reward (outcome-value head) ---
    /// Sequence of encoder latents since this lane's last episode
    /// boundary. Capped at `outcome_max_episode_len`; the tail is
    /// truncated if the episode overflows (warn-logged). Cleared on
    /// episode reset.
    outcome_ep_trajectory: Vec<Vec<f32>>,
    /// Running sum of `r_base` over the current episode.
    outcome_ep_return: f32,
    /// Single-step `r_base` of the just-finished step. Becomes the
    /// previous episode's terminal reward when the *next* step
    /// carries `env_boundary=true`. Used by
    /// `OutcomeTarget::TerminalReward`.
    outcome_last_step_reward: f32,
    /// Per-step `r_base` history within the current episode — the
    /// raw material for `OutcomeTarget::RewardToGo`, which
    /// back-accumulates these into per-step targets at episode end.
    /// Cleared alongside `outcome_ep_trajectory` on boundary. Same
    /// cap so we match it row-by-row.
    outcome_ep_step_rewards: Vec<f32>,
    /// Previous step's `R̂` value (post-clamp). Used by
    /// `OutcomeBonus::PotentialDelta` to emit `α · (R̂_t − R̂_{t-1})`
    /// as the per-step bonus instead of `α · R̂_t`. Reset to 0 at
    /// `env_boundary`.
    prev_r_hat: f32,
    /// EMA of completed-episode returns for variance reduction.
    outcome_baseline: f32,
    /// Last `R̂(z_t)` forward read; used for the per-step reward bonus
    /// and diagnostics.
    last_r_hat: f32,
    /// True once this lane has seen at least one completed episode,
    /// so the baseline has meaningful value (before that, `baseline = 0`
    /// and we treat the first episode's return as the baseline seed).
    outcome_baseline_seeded: bool,
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
    /// EMA of `|last_policy_loss|`. Updated inside the policy
    /// training paths after each `step()`. Drives
    /// `policy_lr_adaptive_target` scaling when enabled. Zero
    /// at init — the first few updates run at full LR until the
    /// EMA warms up.
    policy_loss_ema: f32,
    /// Env-step counter modulo `policy_update_interval`. When it
    /// reaches the configured interval, the policy fires
    /// `interval` consecutive gradient steps on a sliding window
    /// of ripe transitions and resets to 0.
    policy_update_ticks: usize,
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
    /// Cached expected byte-size of the WM session's `visual_obs`
    /// input slot. Non-zero only when `encoder_kind` is `Cnn`.
    /// Used to sanity-check caller-supplied slices. The actual
    /// data lives in the meganeura-owned, device-local, host-
    /// visible graph buffer — accessed via
    /// `wm_session.input_host_ptr("visual_obs")` — so kindle keeps
    /// no CPU-side scratch for the CNN input.
    visual_obs_size_bytes: usize,
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
    /// PPO mode: per-row advantage `[N, 1]` (separate input, not
    /// baked into `policy_action_scratch` like the plain path).
    ppo_advantage_scratch: Vec<f32>,
    /// PPO mode: per-row `π_old(a | s)` for the taken action
    /// `[N, 1]`. Positive, non-zero.
    ppo_old_prob_scratch: Vec<f32>,
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

    /// M6 outcome-value head (CPU). `None` when
    /// `outcome_reward_alpha == 0.0` (default) — no compute cost.
    outcome_head: Option<outcome::OutcomeHead>,
    /// M6: last episode return observed across any lane, smoothed by
    /// the baseline EMA. Diagnostic only — each lane keeps its own
    /// baseline for the actual centering.
    last_outcome_baseline: f32,

    /// M7 approach-reward state. `None` when
    /// `approach_reward_alpha == 0.0` (default) — zero CPU cost.
    approach_state: Option<approach::ApproachState>,
    /// RND curiosity state. `None` when `rnd_reward_alpha == 0.0`
    /// (default) — zero compute cost.
    rnd_state: Option<rnd::RndState>,
    /// Most recent RND per-step MSE across lanes, averaged.
    /// Diagnostic only.
    last_rnd_mse: f32,
    /// Continuous coord-action head. `None` when
    /// `coord_action_alpha == 0.0`.
    coord_head: Option<coord::CoordHead>,
    /// Per-lane last reward, cached so the coord head's
    /// REINFORCE update at the NEXT step can use the advantage
    /// `reward − running_baseline`. We recompute on observe.
    coord_last_reward: Vec<f32>,
    /// EMA baseline of per-step reward for coord-head advantage
    /// centering.
    coord_reward_baseline: f32,
    /// M7 per-lane episode-return accumulator (in kindle's
    /// intrinsic reward). Used so the prototype-updater sees the
    /// same `r_ep` that trained the policy, not a separate
    /// quantity.
    approach_ep_returns: Vec<f32>,
    /// M7 latest approach-distance per lane, for diagnostics.
    approach_distances: Vec<f32>,
    /// M8 delta-goal bank (shared across lanes). `None` when
    /// `delta_goal_alpha == 0.0`.
    delta_goal_bank: Option<delta_goals::DeltaGoalBank>,
    /// M8 per-lane previous latent, cleared at episode boundaries
    /// so the cross-episode jump never triggers a spurious goal.
    delta_goal_prev_latent: Vec<Option<Vec<f32>>>,
    /// M8 number of goal-events recorded during the most recent
    /// `observe()` call, summed across lanes. Diagnostic only.
    last_delta_goal_events: usize,
    /// Cross-episode state-action memory (shared across lanes).
    /// `None` when `xeps_reward_alpha == 0.0`.
    xeps_memory: Option<xeps_memory::StateActionMemory>,
    /// Last action sampled for each lane (discrete id), cached so
    /// the next `observe()` can credit the preceding (state,
    /// action) pair. `None` before any action has been sampled on
    /// that lane, or after an episode boundary reset.
    xeps_prev_action: Vec<Option<u32>>,
    /// Track 3 model-based planner. `None` when
    /// `planner_horizon == 0`. Weights are refreshed every
    /// `planner_refresh_interval` calls to `plan_and_queue()`.
    planner: Option<planner::WmRollout>,
    /// Per-lane action queue populated by the planner. `act()`
    /// pops from the front of this queue before policy sampling,
    /// so a queued sequence commits the next-K actions.
    planner_queue: Vec<std::collections::VecDeque<u32>>,
    /// Number of `plan_and_queue` calls since the last WM-weight
    /// refresh. Triggers a refresh when it hits
    /// `planner_refresh_interval`.
    planner_calls_since_refresh: usize,
    /// Per-lane extrinsic reward for the NEXT `observe()` call.
    /// Populated by the harness via `set_extrinsic_reward`. Zero-
    /// initialized; cleared back to 0 after each `observe()` so a
    /// missed `set_extrinsic_reward` doesn't silently repeat the
    /// previous step's value. Only consumed when
    /// `extrinsic_reward_alpha > 0`.
    extrinsic_reward: Vec<f32>,
}

/// Cosine similarity between two equal-length vectors. Returns 0 when
/// either side has zero norm (no direction defined).
/// Compute the n-step discounted return starting at `ripe_idx` in
/// `buffer`, with optional TD-bootstrap from the stored value at
/// `ripe_idx + n_step`.
///
/// Returns `(ret, gk_at_end, terminated)`:
/// - `ret`: the accumulated `Σ γ^k r_{ripe+k}` (0..end) plus the
///   bootstrap `γ^n · V(s_{ripe+n_step})` when `bootstrap=true` and
///   the trajectory didn't terminate within the window.
/// - `gk_at_end`: `γ^k_end` — the discount factor at the point where
///   accumulation stopped. Equals `γ^n_step` on normal completion.
/// - `terminated`: true iff the window hit an `env_boundary` (i.e.
///   the episode ended strictly inside the window). A bootstrap is
///   NOT added in that case — terminal states have zero future value
///   by definition.
///
/// `bootstrap_value_clamp` is the symmetric L∞ bound applied to the
/// stored `V(s_{ripe+n_step})` before discounting — a safeguard
/// against a drifted value head poisoning the TD target.
///
/// Caller guarantees `ripe_idx + n_step ≤ buffer.len()` when
/// `bootstrap=true`; `ripe_idx + n_step - 1 ≤ buffer.len() - 1` when
/// `bootstrap=false` (i.e. n_step rewards must all be in-buffer).
fn compute_td_n_step_return(
    buffer: &crate::buffer::ExperienceBuffer,
    ripe_idx: usize,
    n_step: usize,
    gamma: f32,
    bootstrap: bool,
    bootstrap_value_clamp: f32,
) -> (f32, f32, bool) {
    let mut ret = 0.0f32;
    let mut gk = 1.0f32;
    let mut terminated = false;
    for k in 0..n_step {
        let idx = ripe_idx + k;
        let tr = buffer.get(idx);
        if k > 0 && tr.env_boundary {
            terminated = true;
            break;
        }
        ret += gk * tr.reward;
        gk *= gamma;
    }
    if bootstrap && !terminated {
        let boot_idx = ripe_idx + n_step;
        let boot_tr = buffer.get(boot_idx);
        if !boot_tr.env_boundary && boot_tr.value.is_finite() {
            let boot_v = boot_tr
                .value
                .clamp(-bootstrap_value_clamp, bootstrap_value_clamp);
            ret += gk * boot_v;
        }
    }
    (ret, gk, terminated)
}

/// Compute the GAE (Generalized Advantage Estimation) advantage at
/// `ripe_idx` using `n_step` one-step TD errors folded back with
/// factor `γλ`.
///
/// Formula:
/// ```text
/// δ_t       = r_t + γ·V(s_{t+1}) - V(s_t)         per-step TD error
/// Â_t       = δ_t + γλ·Â_{t+1}·(1 - done_{t+1})   GAE recursion
/// ```
/// where `done_{t+1}` is true if the episode terminated between t
/// and t+1 (i.e. `tr_{t+1}.env_boundary` is set — a fresh episode
/// started, so V(s_{t+1}) is semantically zero for the old episode).
///
/// Requires `ripe_idx + n_step ≤ buffer.len() - 1` (one extra slot
/// past the n-step window to read `V(s_{ripe+n_step})` for the last
/// δ's bootstrap). The caller enforces this via
/// `bootstrap_headroom = 1` when `gae_lambda > 0`.
///
/// `bootstrap_value_clamp` is applied symmetrically to both
/// `V(s_t)` and `V(s_{t+1})` readings before computing δ — the
/// same safeguard as `compute_td_n_step_return`.
fn compute_gae_advantage(
    buffer: &crate::buffer::ExperienceBuffer,
    ripe_idx: usize,
    n_step: usize,
    gamma: f32,
    lambda: f32,
    bootstrap_value_clamp: f32,
) -> f32 {
    // First pass: accumulate δ's forward, stopping if the episode
    // ended strictly inside the window (after k=0).
    let mut deltas: Vec<f32> = Vec::with_capacity(n_step);
    for k in 0..n_step {
        let t = ripe_idx + k;
        let tr = buffer.get(t);
        if k > 0 && tr.env_boundary {
            break;
        }
        let next = buffer.get(t + 1);
        let next_v = if next.env_boundary || !next.value.is_finite() {
            0.0
        } else {
            next.value.clamp(-bootstrap_value_clamp, bootstrap_value_clamp)
        };
        let v_t = if tr.value.is_finite() {
            tr.value.clamp(-bootstrap_value_clamp, bootstrap_value_clamp)
        } else {
            0.0
        };
        let r = if tr.reward.is_finite() { tr.reward } else { 0.0 };
        let delta = r + gamma * next_v - v_t;
        deltas.push(delta);
    }
    // Second pass: fold back with γλ discount.
    // Â_t = δ_t + γλ · Â_{t+1}. done_{t+1} handling is baked in
    // via the loop cutoff above — once we stopped at a boundary,
    // all later δ's are implicitly zero (no accumulation past
    // termination).
    let mut adv = 0.0f32;
    for &delta in deltas.iter().rev() {
        adv = delta + gamma * lambda * adv;
    }
    adv
}

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

            // Encoder backbone: default MLP on an obs-token vector, or
            // a small CNN on raw visual input when configured.
            let z_t = match config.encoder_kind {
                EncoderKind::Mlp => {
                    let obs = g.input("obs", &[config.batch_size, OBS_TOKEN_DIM]);
                    let enc = Encoder::new(
                        &mut g,
                        OBS_TOKEN_DIM,
                        TASK_DIM,
                        config.latent_dim,
                        config.hidden_dim,
                    );
                    enc.forward(&mut g, obs, task)
                }
                EncoderKind::Cnn {
                    channels,
                    height,
                    width,
                } => {
                    // Flat NCHW visual input.
                    let flat_dim = (channels as usize)
                        * (height as usize)
                        * (width as usize)
                        * config.batch_size;
                    let visual = g.input("visual_obs", &[flat_dim]);
                    let cnn = CnnEncoder::new(
                        &mut g,
                        channels,
                        height,
                        width,
                        config.latent_dim,
                        config.batch_size as u32,
                    );
                    let z_cnn = cnn.forward(&mut g, visual);
                    // Fold the task embedding in post-CNN via a tiny
                    // projection so the visual encoder still gets
                    // per-env conditioning without rebuilding the whole
                    // graph around task concat.
                    let task_proj = nn::Linear::no_bias(
                        &mut g,
                        "encoder.task_proj_cnn",
                        TASK_DIM,
                        config.latent_dim,
                    );
                    let task_h = task_proj.forward(&mut g, task);
                    g.add(z_cnn, task_h)
                }
            };
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
            let g = if is_discrete && config.use_ppo {
                // PPO mode does not support L1 options yet — options are
                // an orthogonal feature; the PPO graph assumes a flat
                // policy without per-option bias heads.
                assert!(
                    config.num_options <= 1,
                    "use_ppo is not compatible with num_options > 1 (L1 options)"
                );
                policy::build_ppo_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                    config.ppo_clip_eps,
                    config.value_loss_coef,
                    config.entropy_beta,
                )
            } else if is_discrete {
                policy::build_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                    config.entropy_beta,
                    config.num_options,
                    config.per_option_heads,
                    config.value_loss_coef,
                )
            } else {
                policy::build_continuous_policy_graph(
                    config.latent_dim,
                    MAX_ACTION_DIM,
                    config.hidden_dim,
                    config.batch_size,
                    config.num_options,
                    config.per_option_heads,
                    config.value_loss_coef,
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

        // M7 approach-reward state. Constructed only when
        // `approach_reward_alpha > 0`. Cheap CPU data structure.
        // Coord-action head. Constructed only when
        // `coord_action_alpha > 0`. Cheap CPU MLP.
        let coord_head = if config.coord_action_alpha > 0.0 {
            let lr = config.coord_lr.unwrap_or(config.learning_rate * 0.3);
            Some(coord::CoordHead::new(
                config.latent_dim,
                config.coord_hidden_dim,
                config.batch_size,
                lr,
                config.coord_sigma,
                0xC001_0D25_DECA_FBADu64 ^ (config.batch_size as u64).wrapping_mul(0x9E37_79B9),
            ))
        } else {
            None
        };

        // RND curiosity state. Constructed only when
        // `rnd_reward_alpha > 0` so the default agent pays no
        // per-step overhead. RND operates on the obs TOKEN
        // (64-dim, pre-encoder) — see the docstring on
        // `rnd_reward_alpha` for the rationale.
        let rnd_state = if config.rnd_reward_alpha > 0.0 {
            let lr = config.rnd_lr.unwrap_or(config.learning_rate * 0.3);
            Some(rnd::RndState::new(
                OBS_TOKEN_DIM,
                config.rnd_feature_dim,
                config.rnd_hidden_dim,
                lr,
                0x42_BEEF_D15E_A53Eu64 ^ (config.batch_size as u64).wrapping_mul(0x9E37_79B9),
            ))
        } else {
            None
        };

        let approach_state = if config.approach_reward_alpha > 0.0 {
            Some(approach::ApproachState::new(
                config.latent_dim,
                config.approach_buffer_size,
                config.approach_top_frac,
                config.approach_update_interval,
                config.approach_warmup_episodes,
            ))
        } else {
            None
        };

        // M8 delta-goal bank. Shared across lanes so all of them
        // benefit from discoveries by any one lane. Feature vector
        // is the obs TOKEN (OBS_TOKEN_DIM), NOT the post-encoder
        // latent: a well-trained encoder compresses state into a
        // narrow region where per-step latent-deltas fall below
        // any useful threshold (the same saturation RND hit, see
        // `rnd_reward_alpha` doc). The obs token carries the raw
        // per-frame variation that "something just changed in the
        // world" should key off.
        let delta_goal_bank = if config.delta_goal_alpha > 0.0 {
            Some(delta_goals::DeltaGoalBank::new(
                OBS_TOKEN_DIM,
                config.delta_goal_bank_size,
                config.delta_goal_threshold,
                config.delta_goal_merge_radius,
            ))
        } else {
            None
        };

        // Cross-episode state-action memory. Keyed on the encoder
        // latent (post-encoder), quantized by
        // `xeps_grid_resolution` (or `grid_resolution` when
        // unspecified). Shared across lanes — lanes pool
        // exploration credit.
        let xeps_memory = if config.xeps_reward_alpha > 0.0 {
            let res = config
                .xeps_grid_resolution
                .unwrap_or(config.grid_resolution);
            Some(xeps_memory::StateActionMemory::new(res))
        } else {
            None
        };

        // Track 3 model-based planner. Allocated when horizon > 0.
        // Weights are zero-initialized; the first `plan_and_queue`
        // call triggers a refresh that pulls the current WM state.
        let planner = if config.planner_horizon > 0 {
            Some(planner::WmRollout::new(
                config.latent_dim,
                MAX_ACTION_DIM,
                config.hidden_dim,
            ))
        } else {
            None
        };

        // M6 outcome-value head (CPU MLP). Constructed only when the
        // user has asked for a non-zero bonus weight. Its LR derives
        // from `lr_outcome` or falls back to `learning_rate × 0.3`.
        let outcome_head = if config.outcome_reward_alpha > 0.0 {
            let lr = config.lr_outcome.unwrap_or(config.learning_rate * 0.3);
            Some(outcome::OutcomeHead::new(
                config.latent_dim,
                config.outcome_window.max(1),
                config.hidden_dim,
                lr,
                0xA11CE ^ 0xD0C_EFF,
            ))
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
                last_prob_taken: 1.0,
                last_entropy: 0.0,
                last_surprise: 0.0,
                last_novelty: 0.0,
                last_homeo: 0.0,
                last_order: 0.0,
                last_reward: 0.0,
                last_base_reward: 0.0,
                outcome_ep_trajectory: Vec::new(),
                outcome_ep_return: 0.0,
                outcome_last_step_reward: 0.0,
                outcome_ep_step_rewards: Vec::new(),
                prev_r_hat: 0.0,
                outcome_baseline: 0.0,
                last_r_hat: 0.0,
                outcome_baseline_seeded: false,
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
            policy_update_ticks: 0,
            policy_loss_ema: 0.0,
            last_replay_loss: 0.0,
            last_option_credit_loss: 0.0,
            last_h_eff_l1: 0.0,
            last_drift: 0.0,
            encoder_lr_scale: 1.0,
            batch_lr_scale: (config.batch_size as f32).sqrt(),
            obs_token_scratch: vec![0.0; n * OBS_TOKEN_DIM],
            visual_obs_size_bytes: n
                * config.encoder_kind.visual_dim()
                * std::mem::size_of::<f32>(),
            action_token_scratch: vec![0.0; n * MAX_ACTION_DIM],
            z_target_scratch: vec![0.0; n * config.latent_dim],
            task_scratch: vec![0.0; n * TASK_DIM],
            value_target_scratch: vec![0.0; n],
            policy_action_scratch: vec![0.0; n * MAX_ACTION_DIM],
            ppo_advantage_scratch: vec![0.0; n],
            ppo_old_prob_scratch: vec![1.0; n],
            option_dim,
            option_onehot_scratch: vec![0.0; n * config.num_options.max(1)],
            option_taken_scratch: vec![0.0; n * config.num_options],
            option_return_scratch: vec![0.0; n],
            termination_target_scratch: vec![0.0; n],
            goal_table: option::build_goal_table(config.num_options, option_dim),
            outcome_head,
            last_outcome_baseline: 0.0,
            approach_state,
            approach_ep_returns: vec![0.0; n],
            approach_distances: vec![0.0; n],
            rnd_state,
            last_rnd_mse: 0.0,
            coord_head,
            coord_last_reward: vec![0.0; n],
            coord_reward_baseline: 0.0,
            delta_goal_bank,
            delta_goal_prev_latent: (0..n).map(|_| None).collect(),
            last_delta_goal_events: 0,
            xeps_memory,
            xeps_prev_action: vec![None; n],
            planner,
            planner_queue: (0..n).map(|_| std::collections::VecDeque::new()).collect(),
            planner_calls_since_refresh: 0,
            extrinsic_reward: vec![0.0; n],
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
                let learned_fire =
                    learned_term && warmup_done && rng.random_range(0.0..1.0) < term_probs[i];
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
                    let advantage = (lane.option_return - lane.option_start_value).clamp(-1.0, 1.0);
                    if advantage.abs() < 1e-8 {
                        continue;
                    }
                    any_train = true;
                    let row =
                        &mut self.option_taken_scratch[i * num_options..(i + 1) * num_options];
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
                    self.termination_target_scratch[i] =
                        if advantage < -deadband { 1.0 } else { 0.0 };
                }

                if any_train && warmup_done {
                    opt_sess.set_input("z", &z_stack);
                    opt_sess.set_input("option_taken", &self.option_taken_scratch);
                    opt_sess.set_input("option_return", &self.option_return_scratch);
                    opt_sess.set_input("termination_target", &self.termination_target_scratch);
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
                let ema_rate = self.config.goal_ema_rate;
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

                        // Tier-3 continuous goal prototype: pull the
                        // just-terminated option's goal toward the
                        // observed end-state latent. Uses the first
                        // `copy_dim` dims of z (the goal vector lives
                        // in R^{option_dim}, which may be ≤ latent_dim).
                        if ema_rate > 0.0 {
                            let old_opt = lane.current_option as usize;
                            let z_end = &z_stack[i * ld..(i + 1) * ld];
                            let base = old_opt * od;
                            let copy_dim = od.min(ld);
                            let goal = &mut self.goal_table[base..base + od];
                            for k in 0..copy_dim {
                                goal[k] += ema_rate * (z_end[k] - goal[k]);
                            }
                        }
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

            // Model-based planner: if the queue has a pre-planned
            // action for this lane, play it instead of sampling from
            // the policy head. Skips action_repeat accounting — a
            // planned sequence is semantically one committed plan,
            // not independent samples to repeat.
            let queued = self.planner_queue[i].pop_front();
            let action = if let Some(a) = queued {
                let act = Action::Discrete(a as usize);
                lane.cached_action = Some(act.clone());
                lane.repeats_left = 0;
                act
            } else {
                let resample = lane.repeats_left == 0 || lane.cached_action.is_none();
                if resample {
                    let a = lane.adapter.sample_action(head, rng);
                    lane.cached_action = Some(a.clone());
                    lane.repeats_left = action_repeat - 1;
                    a
                } else {
                    lane.repeats_left -= 1;
                    lane.cached_action
                        .clone()
                        .expect("cached_action is Some by branch condition")
                }
            };
            // Cache π_old(a | s) for the PPO path — head is logits for
            // discrete adapters; re-softmax it here rather than threading
            // an extra array out of the batched forward. Continuous
            // actions use a Gaussian with fixed scale, we approximate
            // their π_old as 1.0 (the MSE surrogate is scale-invariant).
            lane.last_prob_taken = match &action {
                Action::Discrete(a) => {
                    let probs = crate::policy::softmax_probs(head);
                    let idx = (*a).min(probs.len().saturating_sub(1));
                    probs[idx].max(1e-8)
                }
                Action::Continuous(_) => 1.0,
            };
            // Cache the discrete action id for cross-episode memory.
            // Continuous actions don't key the xeps memory (no natural
            // discrete index), so they leave `xeps_prev_action` as None.
            self.xeps_prev_action[i] = match action {
                Action::Discrete(a) => Some(a as u32),
                Action::Continuous(_) => None,
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
        //
        // M6: capture the outcome-head knobs + handle in one split so the
        // inner loop can both train (episode boundary) and forward (every
        // step) without fighting the borrow checker against `self.lanes`.
        let m6_alpha = self.config.outcome_reward_alpha;
        let m6_ema = self.config.outcome_baseline_ema;
        let m6_clamp = self.config.outcome_clamp.max(0.0);
        let m6_max_ep = self.config.outcome_max_episode_len;
        let m6_target = self.config.outcome_target;
        let m6_bonus_mode = self.config.outcome_bonus;
        let m7_alpha = self.config.approach_reward_alpha;
        let m7_rank_by = self.config.approach_rank_by;
        let m7_clamp = self.config.approach_distance_clamp;
        let m7_saturation = self.config.approach_confidence_saturation;
        let rnd_alpha = self.config.rnd_reward_alpha;
        let mut rnd_state = self.rnd_state.take();
        let mut rnd_mse_sum = 0.0f32;
        let mut rnd_mse_count = 0usize;
        let dg_alpha = self.config.delta_goal_alpha;
        let dg_clamp = self.config.delta_goal_distance_clamp;
        let dg_surprise_gate = self.config.delta_goal_surprise_threshold;
        let mut dg_bank = self.delta_goal_bank.take();
        let mut dg_events_this_step: usize = 0;
        let xeps_alpha = self.config.xeps_reward_alpha;
        let mut xeps_memory = self.xeps_memory.take();
        let ext_alpha = self.config.extrinsic_reward_alpha;
        let m7_warmup = self.config.approach_warmup_episodes;
        let m7_homeo_taper = self.config.homeo_confidence_taper.clamp(0.0, 1.0);
        let mut m7_state = self.approach_state.take();
        // Confidence for this step. Zero before warmup; linear ramp
        // from 0 → 1 over `saturation` episodes once warmup is met;
        // clamped to [0, 1]. `saturation = 0` disables ramping
        // (c = 1 once warmup is satisfied, matching M7 v1 byte-parity
        // for an M7-enabled config).
        let m7_confidence = match m7_state.as_ref() {
            None => 0.0,
            Some(state) => {
                if state.episodes_seen < m7_warmup {
                    0.0
                } else if m7_saturation == 0 {
                    1.0
                } else {
                    let past_warmup = (state.episodes_seen - m7_warmup) as f32;
                    (past_warmup / m7_saturation as f32).clamp(0.0, 1.0)
                }
            }
        };
        let mut m6_head = self.outcome_head.take();
        let mut m6_baseline_diag = self.last_outcome_baseline;
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
            let homeo_raw = RewardCircuit::homeostatic(envs[i].homeostatic_variables());
            // M7↔homeo confidence taper: once M7 has confidence,
            // reduce the homeo contribution so M7's approach signal
            // isn't dominated by homeo's (potentially misaligned)
            // basin. `τ = 0` (default) preserves pre-confidence M7
            // behaviour.
            let homeo = homeo_raw * (1.0 - m7_homeo_taper * m7_confidence);
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
            // `reward_pre_m6` is what trains the outcome head; what the
            // rest of the agent sees also folds in the M6 bonus below.
            let reward_pre_m6 = base_reward + bonus;

            let env_boundary = lane.pending_boundary;
            lane.pending_boundary = false;

            // M7 episode-boundary update: when this step is the first
            // of a new episode, the previous episode's terminal
            // `z_end` is still at `lane.buffer.last()` (current step
            // hasn't been pushed yet) and `approach_ep_returns[i]`
            // holds the previous episode's intrinsic-reward total.
            // Push that pair so the prototype-updater can re-compute.
            if env_boundary {
                if let Some(state) = m7_state.as_mut() {
                    if let Some(prev) = lane.buffer.last() {
                        let rank_score = match m7_rank_by {
                            ApproachRankBy::Return => self.approach_ep_returns[i],
                            ApproachRankBy::Novelty => {
                                // Terminal-specific rarity — reads
                                // from the M7 state's own
                                // `terminal_visit_counts`, NOT the
                                // full buffer's visit_count (which
                                // would conflate "terminated here"
                                // with "passed through here"). Note:
                                // this lookup happens BEFORE the
                                // subsequent `push_terminal` call
                                // increments the count, so the rank
                                // reflects historical rarity, not
                                // including self.
                                state.terminal_novelty(&prev.latent)
                            }
                        };
                        state.push_terminal(&prev.latent, rank_score);
                    }
                }
                self.approach_ep_returns[i] = 0.0;
            }

            // M6 episode-boundary training: when this step is the first
            // after a reset, the PREVIOUS episode's trajectory and
            // summary stats are complete and ready to train the outcome
            // head. We do this BEFORE the per-step M6 forward so the
            // bonus uses the just-updated head (marginally better
            // signal).
            if env_boundary {
                if let Some(head) = m6_head.as_mut() {
                    if !lane.outcome_ep_trajectory.is_empty() {
                        // Build one windowed input per step.
                        let mut windows: Vec<Vec<f32>> =
                            Vec::with_capacity(lane.outcome_ep_trajectory.len());
                        for i in 0..lane.outcome_ep_trajectory.len() {
                            if let Some(w) = head.build_window(&lane.outcome_ep_trajectory, i) {
                                windows.push(w);
                            }
                        }
                        match m6_target {
                            OutcomeTarget::EpisodeSum | OutcomeTarget::TerminalReward => {
                                let target_raw = match m6_target {
                                    OutcomeTarget::EpisodeSum => lane.outcome_ep_return,
                                    OutcomeTarget::TerminalReward => lane.outcome_last_step_reward,
                                    OutcomeTarget::RewardToGo => unreachable!(),
                                };
                                if !lane.outcome_baseline_seeded {
                                    lane.outcome_baseline = target_raw;
                                    lane.outcome_baseline_seeded = true;
                                }
                                let centered = target_raw - lane.outcome_baseline;
                                head.train_batch(&windows, centered);
                                lane.outcome_baseline =
                                    (1.0 - m6_ema) * lane.outcome_baseline + m6_ema * target_raw;
                                m6_baseline_diag = lane.outcome_baseline;
                            }
                            OutcomeTarget::RewardToGo => {
                                // Back-accumulate per-step RTG targets.
                                let l = lane.outcome_ep_step_rewards.len();
                                let mut targets = vec![0.0f32; l];
                                let mut running = 0.0f32;
                                for i in (0..l).rev() {
                                    running += lane.outcome_ep_step_rewards[i];
                                    targets[i] = running;
                                }
                                // Baseline = EMA of RTG[0] = full
                                // episode return, so early-step targets
                                // are centered like EpisodeSum would
                                // be. Late-step targets retain their
                                // raw magnitude (smaller by
                                // construction) because they reference
                                // the same baseline, giving per-step
                                // differentiation.
                                let ep_ret = targets.first().copied().unwrap_or(0.0);
                                if !lane.outcome_baseline_seeded {
                                    lane.outcome_baseline = ep_ret;
                                    lane.outcome_baseline_seeded = true;
                                }
                                for t in targets.iter_mut() {
                                    *t -= lane.outcome_baseline;
                                }
                                // Align lengths in case of a truncated
                                // trajectory cap: train on the shorter
                                // of the two.
                                let n_train = windows.len().min(targets.len());
                                head.train_batch_variable(&windows[..n_train], &targets[..n_train]);
                                lane.outcome_baseline =
                                    (1.0 - m6_ema) * lane.outcome_baseline + m6_ema * ep_ret;
                                m6_baseline_diag = lane.outcome_baseline;
                            }
                        }
                    }
                }
                lane.outcome_ep_trajectory.clear();
                lane.outcome_ep_return = 0.0;
                lane.outcome_ep_step_rewards.clear();
                lane.prev_r_hat = 0.0;
            }

            // Push z into the current episode's trajectory (cap at
            // `outcome_max_episode_len` — overflow truncates the tail).
            // Done *before* the per-step forward so the window ending
            // at `z_t` is available, including for the first step of a
            // new episode.
            if m6_head.is_some() && lane.outcome_ep_trajectory.len() < m6_max_ep {
                lane.outcome_ep_trajectory.push(z_row.to_vec());
            }

            // M6 per-step forward: read `R̂(window_ending_at_z_t)` for
            // the reward bonus.
            let r_hat = if let Some(head) = m6_head.as_ref() {
                let end = lane.outcome_ep_trajectory.len().saturating_sub(1);
                if let Some(win) = head.build_window(&lane.outcome_ep_trajectory, end) {
                    let raw = head.forward(&win);
                    if raw.is_finite() {
                        raw.clamp(-m6_clamp, m6_clamp)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let m6_bonus = match m6_bonus_mode {
                OutcomeBonus::Raw => m6_alpha * r_hat,
                OutcomeBonus::PotentialDelta => {
                    // α · (R̂_t − R̂_{t-1}). At episode boundary
                    // `prev_r_hat` is reset to 0, so the bonus on the
                    // first step of a new episode is `α · r_hat`
                    // (no previous reference). Ng-et-al shaping is
                    // well-defined here: treating the pre-episode
                    // potential as 0 is equivalent to re-setting the
                    // shaping at each episode boundary.
                    m6_alpha * (r_hat - lane.prev_r_hat)
                }
            };
            lane.prev_r_hat = r_hat;
            lane.last_r_hat = r_hat;

            // M7 approach bonus: `-α · ‖z_t − centroid‖` (clamped).
            // Zero until the prototype has been seeded (warmup
            // episodes across all lanes satisfied, see
            // `approach::ApproachState::reward`).
            // Scale the M7 bonus by confidence — starts at 0 during
            // warmup, ramps to 1 over `approach_confidence_saturation`
            // episodes. When saturation is 0, confidence is 1 once
            // warmup is satisfied (v1 behaviour).
            let m7_reward = if let Some(state) = m7_state.as_ref() {
                m7_confidence * state.reward(z_row, m7_alpha, m7_clamp)
            } else {
                0.0
            };
            // Cache raw distance for diagnostics.
            self.approach_distances[i] = if let Some(state) = m7_state.as_ref() {
                state.distance(z_row)
            } else {
                0.0
            };

            // Accumulate pre-M6 reward into the episode return — the
            // outcome head trains on this (or on the single-step value
            // when `outcome_target == TerminalReward`, read from
            // `outcome_last_step_reward` below) so it can't chase its
            // own output (stability guarantee from the design doc).
            lane.outcome_ep_return += reward_pre_m6;
            lane.outcome_last_step_reward = reward_pre_m6;
            if m6_head.is_some() && lane.outcome_ep_step_rewards.len() < m6_max_ep {
                lane.outcome_ep_step_rewards.push(reward_pre_m6);
            }

            // M7 episode-return accumulator: uses `reward_pre_m6`,
            // NOT the post-bonus reward. Prevents the prototype
            // from being built out of the M7 bonus's own echo.
            self.approach_ep_returns[i] += reward_pre_m6;

            // RND curiosity: train predictor on current z, emit
            // MSE as intrinsic reward. Unlike kindle's surprise
            // (which decays as the WM converges), RND's target is
            // a frozen random net so the curiosity signal stays
            // alive as long as there are states the predictor
            // hasn't been fit against.
            // RND reads the obs TOKEN (pre-encoder, 64-dim) rather
            // than the post-encoder latent. An encoder that has
            // converged tightly clusters latents into a narrow
            // region where the predictor can match the target
            // quickly on any z — killing RND's signal. The obs
            // token carries more raw variation across frames, so
            // MSE stays informative longer.
            let rnd_reward = if let Some(state) = rnd_state.as_mut() {
                let mse = state.step(obs_row);
                rnd_mse_sum += mse;
                rnd_mse_count += 1;
                rnd_alpha * mse
            } else {
                0.0
            };

            // M8 delta-goal reward. First, consider recording a new
            // goal from the (prev, cur) OBS pair — gated by
            // `pred_error >= surprise_threshold` so only
            // WM-surprising transitions populate the bank. Then
            // score the current obs against the (possibly
            // just-updated) bank. `prev_obs` is cleared on episode
            // boundaries so cross-episode jumps never register.
            let dg_reward = if let Some(bank) = dg_bank.as_mut() {
                if env_boundary {
                    self.delta_goal_prev_latent[i] = None;
                }
                if pred_error >= dg_surprise_gate {
                    let prev = self.delta_goal_prev_latent[i].as_deref();
                    if bank.observe_delta(prev, obs_row) {
                        dg_events_this_step += 1;
                    }
                }
                // Always update prev_obs, even when the surprise
                // gate blocks recording, so the next step's delta
                // is measured against this step rather than against
                // a stale pre-gate observation.
                self.delta_goal_prev_latent[i] = Some(obs_row.to_vec());
                bank.reward(obs_row, dg_alpha, dg_clamp)
            } else {
                0.0
            };

            // Cross-episode state-action novelty. Keyed on the
            // PREVIOUS step's (obs_token, action) — that's the
            // transition we're rewarding. The obs token is used
            // (not the post-encoder latent) because a trained
            // encoder clusters latents tight enough that the
            // quantized key collapses to 1-2 cells — verified
            // empirically on cd82 (xeps=6 at default grid, xeps=12
            // at fine grid) before switching to obs. The obs
            // token carries the raw per-frame variation that
            // distinguishes states the agent should care about.
            let xeps_reward = if let Some(memory) = xeps_memory.as_mut() {
                if let (Some(prev), Some(prev_a)) = (lane.buffer.last(), self.xeps_prev_action[i]) {
                    let bonus = if env_boundary {
                        0.0
                    } else {
                        memory.reward(&prev.observation, prev_a, xeps_alpha)
                    };
                    memory.observe(&prev.observation, prev_a);
                    bonus
                } else {
                    0.0
                }
            } else {
                0.0
            };
            // Also clear prev_action on the boundary step itself, so
            // the NEXT step (first full step of the new episode)
            // sees an empty prev — no spurious credit before a
            // fresh action has been sampled.
            if env_boundary {
                self.xeps_prev_action[i] = None;
            }

            // Extrinsic reward (signed, passed through from the env).
            // Kindle's value head absorbs the mean; the per-step
            // variance stays in the advantage. This is the channel
            // stock RL consumes — scaled by alpha it sits alongside
            // kindle's intrinsic primitives. Harness sets
            // `self.extrinsic_reward[i]` via `set_extrinsic_reward`
            // before calling observe; we consume and zero it here.
            let ext_reward = ext_alpha * self.extrinsic_reward[i];
            self.extrinsic_reward[i] = 0.0;

            let reward = reward_pre_m6
                + m6_bonus
                + m7_reward
                + rnd_reward
                + dg_reward
                + xeps_reward
                + ext_reward;

            // Cache per-lane reward for the coord head's next
            // REINFORCE update; the head uses this step's reward
            // as the advantage signal for the coordinates it
            // sampled PRE-step. The head's `sample` call caches μ
            // and sample; `train_coord_head` (called by the
            // harness after this `observe` returns) runs
            // train_step using that cached state + this reward.
            self.coord_last_reward[i] = reward;

            lane.buffer.push(Transition {
                observation: obs_row.to_vec(),
                latent: z_row.to_vec(),
                action: act_row.to_vec(),
                reward,
                credit: 0.0,
                pred_error,
                value: lane.last_value,
                prob_taken: lane.last_prob_taken,
                option_idx: lane.current_option,
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
        // Restore the outcome head after the loop finishes.
        self.outcome_head = m6_head;
        self.last_outcome_baseline = m6_baseline_diag;
        // Restore the M7 state.
        self.approach_state = m7_state;
        self.rnd_state = rnd_state;
        self.last_rnd_mse = if rnd_mse_count > 0 {
            rnd_mse_sum / rnd_mse_count as f32
        } else {
            0.0
        };
        self.delta_goal_bank = dg_bank;
        self.last_delta_goal_events = dg_events_this_step;
        self.xeps_memory = xeps_memory;

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
        // Encoder-dependent: MLP uses the obs-token buffer; CNN uses
        // the raw visual_obs buffer (populated by `set_visual_obs`
        // before this method runs).
        match self.config.encoder_kind {
            EncoderKind::Mlp => {
                self.wm_session.set_input("obs", &self.obs_token_scratch);
            }
            EncoderKind::Cnn { .. } => {
                // The `visual_obs` graph buffer is allocated by
                // meganeura as `Memory::Shared` (device-local +
                // host-visible + host-coherent). The harness writes
                // preprocessed frames directly into the mapped
                // pointer via `set_visual_obs` or
                // `visual_obs_host_ptr`, so there's no CPU-side
                // scratch to upload here — the data is already in
                // the GPU buffer. Queue submit's implicit host-
                // domain barrier makes the writes visible.
            }
        }
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

    /// Populate the visual-obs buffer for the next `observe()`.
    ///
    /// Per-lane extrinsic (env) reward for the NEXT `observe()` call.
    /// `rewards.len()` must equal `batch_size`. No-op when
    /// `extrinsic_reward_alpha == 0.0`.
    ///
    /// Call between env step and `observe` — the harness is the only
    /// party that knows the env's native reward. Kindle zeroes the
    /// per-lane value inside `observe` after consuming it, so a
    /// missed call doesn't silently replay the previous step.
    pub fn set_extrinsic_reward(&mut self, rewards: &[f32]) {
        if self.config.extrinsic_reward_alpha == 0.0 {
            return;
        }
        assert_eq!(
            rewards.len(),
            self.extrinsic_reward.len(),
            "set_extrinsic_reward: expected {} rewards, got {}",
            self.extrinsic_reward.len(),
            rewards.len()
        );
        self.extrinsic_reward.copy_from_slice(rewards);
    }

    /// Must be called by the harness when `encoder_kind = Cnn` and
    /// the input shape is flat NCHW of size `batch · channels · h · w`.
    /// No-op when the encoder is `Mlp`.
    ///
    /// This writes directly into meganeura's device-local,
    /// host-visible graph buffer via `Session::input_host_ptr` —
    /// the legacy scratch copy + `set_input` upload is gone, so
    /// one memcpy lands the frames in the GPU-side memory with
    /// no intermediary.
    pub fn set_visual_obs(&mut self, visual_obs: &[f32]) {
        if self.visual_obs_size_bytes == 0 {
            return;
        }
        let want = self.visual_obs_size_bytes / std::mem::size_of::<f32>();
        assert_eq!(
            visual_obs.len(),
            want,
            "set_visual_obs: expected {} floats (batch · channels · h · w), got {}",
            want,
            visual_obs.len()
        );
        let (dst, size) = self
            .wm_session
            .input_host_ptr("visual_obs")
            .expect("visual_obs input slot present under CNN encoder");
        let need = std::mem::size_of_val(visual_obs);
        debug_assert!(need <= size, "visual_obs write {need} > slot {size}");
        // Safety: `dst` points at a host-visible, host-coherent
        // buffer owned by wm_session for the lifetime of this
        // Agent. We write exactly `need` bytes which fits within
        // the slot size. `observe()` calls `wm_session.wait()`
        // before returning, so the previous step's GPU read has
        // completed — no race on the next host write.
        unsafe {
            std::ptr::copy_nonoverlapping(visual_obs.as_ptr() as *const u8, dst, need);
        }
    }

    /// Raw host pointer + byte size of the WM session's
    /// `visual_obs` input buffer. `None` when the encoder is
    /// `Mlp` (no visual input slot).
    ///
    /// The buffer is allocated as `Memory::Shared` (device-local,
    /// host-visible, host-coherent). Writes through the pointer
    /// are picked up by the next `observe()` without any upload,
    /// staging, or external-memory import — a single memcpy lands
    /// the frame in GPU-visible memory.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid for the lifetime of this
    /// Agent. The caller must write at most `size_bytes` and must
    /// not initiate a write before the previous `observe()`
    /// returns (wm_session's `wait()` inside `observe` synchronizes
    /// with the GPU's read of the buffer).
    pub fn visual_obs_host_ptr(&self) -> Option<(*mut u8, usize)> {
        self.wm_session.input_host_ptr("visual_obs")
    }

    /// Byte size of the `visual_obs` slot, or 0 when the encoder
    /// is `Mlp`. Diagnostic.
    pub fn visual_obs_host_size(&self) -> usize {
        self.visual_obs_size_bytes
    }

    /// Sample one replay transition per lane and run a single batched WM
    /// forward+backward over the stacked rows. Each lane retries up to 8
    /// random indices to find a non-boundary pair in its own buffer; if
    /// that fails (buffer < 2, or deeply fragmented by env switches), we
    /// fall back to the most recent valid donor lane's sample so every
    /// batch row carries signal instead of zeros.
    fn replay_step<R: Rng>(&mut self, rng: &mut R) {
        // Replay in CNN-encoder mode would need visual frames
        // stored per-transition, which they aren't (buffer holds
        // the 64-dim token). Skip replay in that case; the online
        // WM gradient still flows every step.
        if matches!(self.config.encoder_kind, EncoderKind::Cnn { .. }) {
            return;
        }
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
        // Drift probes are 64-dim obs tokens captured at warmup;
        // they're not visual frames, so a CNN encoder can't consume
        // them. Leave `last_drift = 0` for CNN-mode agents.
        if matches!(self.config.encoder_kind, EncoderKind::Cnn { .. }) {
            return;
        }
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
        let n_step = self.config.n_step.max(1);
        // When the user opts into n-step lookahead, defer to the
        // dedicated path that trains on old (z, action, value) with
        // an n-step Monte-Carlo return as the advantage.
        if n_step >= 2 {
            // Rollout-gated dispatch. With `policy_update_interval = 1`
            // (default) this fires every env-step on the single ripe
            // transition at `buf_len - n_step - bootstrap_headroom` —
            // identical to the old per-step behavior. With
            // `policy_update_interval > 1` we accumulate transitions
            // and, when the tick hits the interval, fire `interval`
            // sequential gradient steps on the last `interval` ripe
            // transitions (ripe_back_offset = 0, 1, …, interval-1).
            // Each step's advantage is still computed from the stored
            // V's, so `interval = n_step + 1` keeps the whole rollout
            // on-policy w.r.t. the collector.
            let interval = self.config.policy_update_interval.max(1);
            self.policy_update_ticks += 1;
            if self.policy_update_ticks < interval {
                return;
            }
            self.policy_update_ticks = 0;
            let n_epochs = if self.config.use_ppo {
                self.config.ppo_n_epochs.max(1)
            } else {
                1
            };
            // Epoch loop: replay the same `interval`-step rollout
            // `n_epochs` times. With PPO, epoch 1's ratio ≈ 1 (π_new
            // ≈ π_old since collection), so it runs as plain PG.
            // Epochs 2+ see the policy drifted from π_old (because
            // epoch 1 updated weights), so ratios diverge from 1 and
            // the PPO clip starts actually capping updates that would
            // push past the trust region. This is the standard PPO
            // training schedule (e.g. SB3 defaults to n_epochs=10).
            //
            // Non-PPO path keeps n_epochs=1 — replaying the same
            // rollout under plain PG would just multiply the update
            // magnitude (every epoch is unclipped, so each repeat is
            // basically an LR boost).
            for _ in 0..n_epochs {
                // Oldest ripe first so the gradient order matches
                // the temporal order of the collected rollout —
                // Adam's moment estimates then update in the same
                // order the data was produced, which matches how
                // A2C trains on a rollout.
                for k in (0..interval).rev() {
                    self.policy_step_n_step_at(n_step, k);
                }
            }
            return;
        }
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
            let adv_clamp = self.config.advantage_clamp.max(0.0);
            let advantage = (lane.last_reward - lane.last_value).clamp(-adv_clamp, adv_clamp);
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

        // Global advantage-norm clip — see `policy_step_n_step` for
        // rationale.
        let clip = self.config.policy_adv_global_clip;
        if clip > 0.0 {
            let sum_sq: f32 = self.policy_action_scratch.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            if norm > clip {
                let scale = clip / norm;
                for v in self.policy_action_scratch.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Adaptive learning-rate scaling — see `policy_step_n_step`.
        let target = self.config.policy_lr_adaptive_target;
        let lr_scale = if target > 0.0 && self.policy_loss_ema > target {
            target / self.policy_loss_ema
        } else {
            1.0
        };

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
            .set_learning_rate(self.config.lr_policy * self.batch_lr_scale * lr_scale);
        self.policy_session.step();
        self.policy_session.wait();

        let loss = self.policy_session.read_loss();
        // Watchdog: reset on non-finite OR absolute magnitude above
        // `policy_loss_watchdog_threshold`. The latter catches the
        // "finite but runaway" regime observed on LunarLander after
        // a brief performance peak. Setting the threshold very high
        // effectively disables the magnitude branch (NaN branch
        // remains active).
        let wd = self.config.policy_loss_watchdog_threshold;
        if !loss.is_finite() || loss.abs() > wd {
            init_parameters(&mut self.policy_session);
            log::warn!(
                "policy loss {:.1} unstable at step {}, re-initialized policy params",
                loss,
                self.step_count
            );
            self.last_policy_loss = 0.0;
        } else {
            self.last_policy_loss = loss;
            let rate = self.config.policy_lr_adaptive_ema.clamp(0.001, 1.0);
            self.policy_loss_ema = (1.0 - rate) * self.policy_loss_ema + rate * loss.abs();
        }
    }

    /// N-step advantage variant of `policy_step_batched`.
    ///
    /// Trains the policy on the transition at buffer offset
    /// `len - n_step` — the step at which the lane took an action
    /// whose `n_step`-lookahead reward window is now fully observed.
    /// The advantage is `R_t − V(s_t)` where `R_t` is the γ-discounted
    /// and effective-horizon-normalized sum over the next `n_step`
    /// rewards, truncated at any `env_boundary` flag (so rewards from
    /// after an episode reset don't bleed into the prior option's
    /// credit).
    ///
    /// Value head still trains on the single-step base reward at the
    /// old state, preserving the value-head stability that the
    /// previous discounted-return attempt (commit 8f291e5) lost.
    /// Train the policy on the ripe transition at
    /// `ripe_idx = buf_len - n_step - headroom - ripe_back_offset`.
    /// `ripe_back_offset = 0` → most-recent ripe (the per-step default).
    /// Larger offsets walk back through the rollout for
    /// `policy_update_interval > 1`.
    fn policy_step_n_step_at(&mut self, n_step: usize, ripe_back_offset: usize) {
        let n = self.lanes.len();
        let gamma = self.config.gamma;
        let eps_base = self.config.label_smoothing;
        let floor = self.config.entropy_floor;
        let k_actions = MAX_ACTION_DIM as f32;

        // The n-step return `Σ γ^k r_{t+k}` is compared against a
        // value-head baseline that tracks the *single-step* reward —
        // the previous implementation normalized ret by
        // `Σ γ^k ≈ n_step` to keep the two scales comparable. That
        // works for dense reward streams but is catastrophic on
        // sparse ones: a single ±1 event in a 16-step window divides
        // down to ±0.06, which trips the `advantage.abs() < 1e-3`
        // skip check in the label-smoothing block below and produces
        // a near-zero policy gradient. Empirically (Pong 400k
        // env-steps, all ±1 events averaged away to silence).
        //
        // Correct handling: treat ret as the Monte-Carlo return
        // directly. `ripe.value` is an EMA of single-step reward at
        // that state — a correctly-signed (if slightly biased)
        // baseline. Advantage = ret - value in the reward-scale
        // units. advantage_clamp still bounds the per-step update
        // magnitude.

        self.policy_action_scratch.fill(0.0);
        self.value_target_scratch.fill(0.0);

        // Build old-state z stack and per-row targets. Lanes with a
        // too-short buffer contribute zeros (no gradient).
        let ld = self.latent_dim;
        let mut old_z_stack = vec![0.0f32; n * ld];
        let num_options = self.config.num_options;
        let has_options = self.option_session.is_some();
        if has_options {
            self.option_onehot_scratch.fill(0.0);
        }

        let mut any_active = false;

        // Under `value_bootstrap` OR `gae_lambda > 0`, shift ripe
        // one step earlier so we reserve the last buffer slot as
        // the bootstrap state for `V(s_{ripe+n_step})`. Without it,
        // ripe_idx + n_step would be out of range (== buf_len) —
        // both the value-bootstrap and GAE paths read that slot.
        let use_gae = self.config.gae_lambda > 0.0;
        let needs_bootstrap_slot = self.config.value_bootstrap || use_gae;
        let bootstrap_headroom = needs_bootstrap_slot as usize;
        let value_target_bootstrap = self.config.value_bootstrap || use_gae;

        // First pass: compute raw advantage + value target for each
        // active lane. We collect them all before labeling so that
        // `advantage_normalize` can zero-mean / unit-std the batch
        // before it feeds into the CE labels.
        let mut raw_advantages = vec![0.0f32; n];
        let mut value_targets = vec![0.0f32; n];
        let mut lane_active = vec![false; n];
        let adv_clamp = self.config.advantage_clamp.max(0.0);
        for (i, lane) in self.lanes.iter().enumerate() {
            let buf_len = lane.buffer.len();
            if buf_len < n_step + bootstrap_headroom + ripe_back_offset {
                continue;
            }
            let ripe_idx = buf_len - n_step - bootstrap_headroom - ripe_back_offset;
            let ripe = lane.buffer.get(ripe_idx);
            let (ret, _gk_end, _terminated) = compute_td_n_step_return(
                &lane.buffer,
                ripe_idx,
                n_step,
                gamma,
                value_target_bootstrap,
                100.0,
            );
            let adv_raw = if use_gae {
                compute_gae_advantage(
                    &lane.buffer,
                    ripe_idx,
                    n_step,
                    gamma,
                    self.config.gae_lambda,
                    100.0,
                )
            } else {
                ret - ripe.value
            };
            raw_advantages[i] = adv_raw;
            value_targets[i] = if value_target_bootstrap {
                ret.clamp(-100.0, 100.0)
            } else if ripe.reward.is_finite() {
                ripe.reward.clamp(-100.0, 100.0)
            } else {
                0.0
            };
            lane_active[i] = true;
        }

        // Optional zero-mean / unit-std normalization across the active
        // lanes. Strips the "V lags reward" bias and rescales gradient
        // magnitudes to O(1) regardless of reward scale. Computed over
        // active lanes only; inactive lanes contribute nothing.
        if self.config.advantage_normalize {
            let active_count = lane_active.iter().filter(|&&a| a).count();
            if active_count >= 2 {
                let mut sum = 0.0f32;
                for (i, &a) in raw_advantages.iter().enumerate() {
                    if lane_active[i] {
                        sum += a;
                    }
                }
                let mean = sum / active_count as f32;
                let mut sq_sum = 0.0f32;
                for (i, &a) in raw_advantages.iter().enumerate() {
                    if lane_active[i] {
                        let d = a - mean;
                        sq_sum += d * d;
                    }
                }
                let std = (sq_sum / active_count as f32).sqrt().max(1e-3);
                for (i, a) in raw_advantages.iter_mut().enumerate() {
                    if lane_active[i] {
                        *a = (*a - mean) / std;
                    }
                }
            }
        }

        let use_ppo = self.config.use_ppo;
        if use_ppo {
            self.ppo_advantage_scratch.fill(0.0);
            self.ppo_old_prob_scratch.fill(1.0);
        }

        for (i, lane) in self.lanes.iter().enumerate() {
            if !lane_active[i] {
                continue;
            }
            let buf_len = lane.buffer.len();
            let ripe_idx = buf_len - n_step - bootstrap_headroom - ripe_back_offset;
            let ripe = lane.buffer.get(ripe_idx);
            let advantage = raw_advantages[i].clamp(-adv_clamp, adv_clamp);
            self.value_target_scratch[i] = value_targets[i];

            let entropy_deficit = if floor > 0.0 {
                ((floor - lane.last_entropy) / floor).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if advantage.abs() < 1e-8 && entropy_deficit < 1e-6 {
                continue;
            }
            any_active = true;

            old_z_stack[i * ld..(i + 1) * ld].copy_from_slice(&ripe.latent);

            if use_ppo {
                // PPO feeds: plain one-hot `action` (sum=1), scalar
                // `advantage`, scalar `old_prob_taken`. The clip and
                // the advantage-weighting live inside the graph.
                let act_dst = &mut self.policy_action_scratch
                    [i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                for (dst, &src) in act_dst.iter_mut().zip(ripe.action.iter()) {
                    *dst = src;
                }
                self.ppo_advantage_scratch[i] = advantage;
                self.ppo_old_prob_scratch[i] = ripe.prob_taken.max(1e-8);
            } else {
                let eps = (eps_base + (1.0 - eps_base) * entropy_deficit).min(1.0);
                let effective_adv = if advantage.abs() < 1e-3 && entropy_deficit > 0.0 {
                    entropy_deficit
                } else {
                    advantage
                };
                let act_src = &ripe.action;
                let act_dst = &mut self.policy_action_scratch
                    [i * MAX_ACTION_DIM..(i + 1) * MAX_ACTION_DIM];
                for (dst, &src) in act_dst.iter_mut().zip(act_src.iter()) {
                    let smoothed = (1.0 - eps) * src + eps / k_actions;
                    *dst = effective_adv * smoothed;
                }
            }

            if has_options {
                let row = &mut self.option_onehot_scratch[i * num_options..(i + 1) * num_options];
                let oi = (ripe.option_idx as usize).min(num_options.saturating_sub(1));
                row[oi] = 1.0;
            }
        }

        if !any_active {
            return;
        }

        // Global advantage-norm clip (policy-gradient-analogue of
        // grad-norm clipping). The per-lane `advantage_clamp` is an
        // L∞ bound; this is the L2 bound across the batch. Since
        // each lane contributes linearly to the policy gradient,
        // bounding the batch L2 norm bounds the update magnitude.
        let clip = self.config.policy_adv_global_clip;
        if clip > 0.0 {
            let sum_sq: f32 = self.policy_action_scratch.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            if norm > clip {
                let scale = clip / norm;
                for v in self.policy_action_scratch.iter_mut() {
                    *v *= scale;
                }
            }
        }

        // Adaptive learning-rate scaling based on EMA(|pi_loss|).
        // When the recent policy updates have been unusually loud
        // (EMA loss magnitude far above the target), scale LR down
        // so the next update is smaller. Damps the commit-recover
        // oscillation we observed on CartPole.
        let target = self.config.policy_lr_adaptive_target;
        let lr_scale = if target > 0.0 && self.policy_loss_ema > target {
            target / self.policy_loss_ema
        } else {
            1.0
        };

        self.policy_session.set_input("z", &old_z_stack);
        if has_options {
            self.policy_session
                .set_input("option_onehot", &self.option_onehot_scratch);
        }
        self.policy_session
            .set_input("action", &self.policy_action_scratch);
        self.policy_session
            .set_input("value_target", &self.value_target_scratch);
        if use_ppo {
            self.policy_session
                .set_input("advantage", &self.ppo_advantage_scratch);
            self.policy_session
                .set_input("old_prob_taken", &self.ppo_old_prob_scratch);
        }
        self.policy_session
            .set_learning_rate(self.config.lr_policy * self.batch_lr_scale * lr_scale);
        self.policy_session.step();
        self.policy_session.wait();

        let loss = self.policy_session.read_loss();
        let wd = self.config.policy_loss_watchdog_threshold;
        if !loss.is_finite() || loss.abs() > wd {
            init_parameters(&mut self.policy_session);
            log::warn!(
                "policy loss {:.1} unstable at step {} (n-step), re-initialized policy params",
                loss,
                self.step_count
            );
            self.last_policy_loss = 0.0;
        } else {
            self.last_policy_loss = loss;
            let rate = self.config.policy_lr_adaptive_ema.clamp(0.001, 1.0);
            self.policy_loss_ema = (1.0 - rate) * self.policy_loss_ema + rate * loss.abs();
        }
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Cheap per-lane read of `last_r_hat` without building the full
    /// Diagnostics (used by M6 instrumentation harnesses that log this
    /// every step).
    pub fn r_hats(&self) -> Vec<f32> {
        self.lanes.iter().map(|l| l.last_r_hat).collect()
    }

    /// Current M7 confidence ∈ [0, 1]. Zero until warmup is met;
    /// ramps linearly to 1 over `approach_confidence_saturation`
    /// episodes. 1 once warmup is met if saturation = 0. Always 0
    /// when M7 is disabled.
    /// Sample per-lane `(x, y)` coordinates from the coord head's
    /// current policy. Returns a `[(x, y); N]` vector in `[−1, 1]`
    /// space; callers rescale to the target env's coord range.
    /// Returns zeros when the coord head is disabled.
    ///
    /// After the next `observe()` completes, call
    /// `train_coord_head()` to update the head via REINFORCE on
    /// whatever advantage signal the harness chooses (typically
    /// the last per-step reward minus a baseline).
    pub fn sample_coords<R: Rng>(&mut self, rng: &mut R) -> Vec<(f32, f32)> {
        let n = self.lanes.len();
        let Some(head) = self.coord_head.as_mut() else {
            return vec![(0.0, 0.0); n];
        };
        let mut out = Vec::with_capacity(n);
        for (i, lane) in self.lanes.iter().enumerate() {
            let z = match lane.buffer.last() {
                Some(t) => t.latent.clone(),
                None => vec![0.0; self.latent_dim],
            };
            let [sx, sy] = head.sample(i, &z, || {
                // Box-Muller from a pair of uniforms.
                let u1: f32 = rng.random_range(1e-7..1.0);
                let u2: f32 = rng.random_range(0.0..1.0);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f32::consts::PI * u2;
                (r * theta.cos(), r * theta.sin())
            });
            out.push((sx, sy));
        }
        out
    }

    /// Train the coord head on the most recent step's rewards via
    /// REINFORCE. `advantage = per_lane_reward − running_baseline`;
    /// the running baseline is an EMA updated inside the agent.
    /// No-op when the coord head is disabled.
    pub fn train_coord_head(&mut self) {
        let Some(head) = self.coord_head.as_mut() else {
            return;
        };
        let alpha = self.config.coord_action_alpha;
        // Per-step reward baseline (shared across lanes — simple
        // rolling average is enough).
        let mean_r: f32 = self.coord_last_reward.iter().copied().sum::<f32>()
            / self.coord_last_reward.len().max(1) as f32;
        let ema = 0.02f32;
        self.coord_reward_baseline = (1.0 - ema) * self.coord_reward_baseline + ema * mean_r;
        for (i, lane) in self.lanes.iter().enumerate() {
            let z = match lane.buffer.last() {
                Some(t) => t.latent.clone(),
                None => continue,
            };
            let adv = (self.coord_last_reward[i] - self.coord_reward_baseline) * alpha;
            head.train_step(i, &z, adv);
        }
    }

    /// Re-initialize the RND predictor (keeps target frozen). Used
    /// by the harness to re-activate curiosity when the agent
    /// enters a qualitatively new state distribution (e.g. on
    /// level-up in ARC-AGI-3). No-op when RND is disabled.
    pub fn reset_rnd_predictor(&mut self) {
        if let Some(state) = self.rnd_state.as_mut() {
            state.reset_predictor();
        }
    }

    /// Current size of the M8 delta-goal bank, or 0 when M8 is off.
    pub fn delta_goal_bank_size(&self) -> usize {
        self.delta_goal_bank.as_ref().map_or(0, |b| b.len())
    }

    /// Number of M8 goal-events recorded in the most recent
    /// `observe()` call, summed across lanes. Zero when M8 is off.
    pub fn last_delta_goal_events(&self) -> usize {
        self.last_delta_goal_events
    }

    /// Distinct `(quantized_state, action)` pairs in the
    /// cross-episode memory. 0 when disabled. Diagnostic.
    pub fn xeps_distinct_pairs(&self) -> usize {
        self.xeps_memory.as_ref().map_or(0, |m| m.distinct_pairs())
    }

    /// Run the Track 3 model-based planner for every lane whose
    /// planner queue is currently empty. For each such lane:
    /// - Sample `planner_samples` random discrete action sequences
    ///   of length `planner_horizon` over `num_actions` (passed by
    ///   the caller — typically from the env's action space).
    /// - Roll each sequence out through the frozen WM starting
    ///   from the lane's most recent latent.
    /// - Score each by the sum of `1/sqrt(1+visit_count)` across
    ///   predicted latents — pulling toward under-visited regions
    ///   of the state space.
    /// - Queue the highest-scoring sequence for consumption by
    ///   subsequent `act()` calls.
    ///
    /// No-op when `planner_horizon == 0` or when a lane has no
    /// prior observation (first step of the agent's life).
    /// `num_actions` is clamped to the WM's compiled action width.
    pub fn plan_and_queue<R: Rng>(&mut self, num_actions: usize, rng: &mut R) {
        let Some(planner) = self.planner.as_mut() else {
            return;
        };
        let k = self.config.planner_horizon;
        let m = self.config.planner_samples;
        if k == 0 || m == 0 {
            return;
        }
        // Refresh WM weights into the CPU cache at the configured
        // cadence. First call always triggers a refresh.
        if self.planner_calls_since_refresh == 0
            || self.planner_calls_since_refresh >= self.config.planner_refresh_interval
        {
            planner.refresh_from_session(&self.wm_session);
            self.planner_calls_since_refresh = 0;
        }
        self.planner_calls_since_refresh += 1;

        let num_actions_eff = num_actions.min(planner.action_dim).max(1);
        let mut one_hot = vec![0.0f32; planner.action_dim];
        let mut traj = vec![0.0f32; k * planner.latent_dim];

        for (lane_idx, lane) in self.lanes.iter().enumerate() {
            if !self.planner_queue[lane_idx].is_empty() {
                continue;
            }
            let z0 = match lane.buffer.last() {
                Some(t) => t.latent.clone(),
                None => continue,
            };

            let mut best_score = f32::NEG_INFINITY;
            let mut best_actions: Vec<u32> = Vec::new();
            for _ in 0..m {
                let actions: Vec<u32> = (0..k)
                    .map(|_| rng.random_range(0..num_actions_eff) as u32)
                    .collect();
                planner.rollout(&z0, &actions, &mut one_hot, &mut traj);

                let mut score = 0.0f32;
                for step in 0..k {
                    let z_step = &traj[step * planner.latent_dim..(step + 1) * planner.latent_dim];
                    let c = lane.buffer.visit_count(z_step);
                    score += 1.0 / ((c as f32 + 1.0).sqrt());
                }
                if score > best_score {
                    best_score = score;
                    best_actions = actions;
                }
            }
            for a in best_actions {
                self.planner_queue[lane_idx].push_back(a);
            }
        }
    }

    /// Total number of actions currently queued by the planner
    /// across all lanes (diagnostic).
    pub fn planner_queue_len(&self) -> usize {
        self.planner_queue.iter().map(|q| q.len()).sum()
    }

    pub fn approach_confidence(&self) -> f32 {
        let Some(state) = self.approach_state.as_ref() else {
            return 0.0;
        };
        let warmup = self.config.approach_warmup_episodes;
        if state.episodes_seen < warmup {
            return 0.0;
        }
        let saturation = self.config.approach_confidence_saturation;
        if saturation == 0 {
            return 1.0;
        }
        let past_warmup = (state.episodes_seen - warmup) as f32;
        (past_warmup / saturation as f32).clamp(0.0, 1.0)
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
            .enumerate()
            .map(|(lane_idx, lane)| {
                let recent = lane.buffer.recent_window(self.config.history_len);
                let credit_weights: Vec<f32> = recent.iter().map(|t| t.credit).collect();
                let h_eff = if credit_weights.is_empty() {
                    0.0
                } else {
                    credit::effective_scope(&credit_weights)
                };

                let goal_diversity =
                    if self.option_session.is_some() && self.config.num_options >= 2 {
                        let n_opt = self.config.num_options;
                        let od = self.goal_table.len() / n_opt.max(1);
                        let mut sum = 0.0f32;
                        let mut pairs = 0usize;
                        for a in 0..n_opt {
                            for b in (a + 1)..n_opt {
                                let ga = &self.goal_table[a * od..(a + 1) * od];
                                let gb = &self.goal_table[b * od..(b + 1) * od];
                                let d2: f32 =
                                    ga.iter().zip(gb.iter()).map(|(x, y)| (x - y).powi(2)).sum();
                                sum += d2.sqrt();
                                pairs += 1;
                            }
                        }
                        if pairs == 0 { 0.0 } else { sum / pairs as f32 }
                    } else {
                        0.0
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
                    goal_diversity,
                    r_hat: lane.last_r_hat,
                    outcome_baseline: lane.outcome_baseline,
                    outcome_loss: self
                        .outcome_head
                        .as_ref()
                        .map(|h| h.last_loss)
                        .unwrap_or(0.0),
                    approach_distance: self
                        .approach_distances
                        .get(lane_idx)
                        .copied()
                        .unwrap_or(0.0),
                    approach_buffer_fill: self
                        .approach_state
                        .as_ref()
                        .map(|s| s.buffer.len())
                        .unwrap_or(0),
                    approach_centroid_drift: self
                        .approach_state
                        .as_ref()
                        .map(|s| s.last_centroid_drift)
                        .unwrap_or(0.0),
                    approach_centroid_age: self
                        .approach_state
                        .as_ref()
                        .map(|s| s.centroid_age)
                        .unwrap_or(0),
                    approach_confidence: self.approach_confidence(),
                    rnd_mse: self.last_rnd_mse,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{ExperienceBuffer, Transition};

    /// Build an ExperienceBuffer with one transition per tuple
    /// `(reward, value, env_boundary)`. Latent and other fields are
    /// filled with zeros; they don't matter for the TD-target math.
    fn mk_buffer(spec: &[(f32, f32, bool)]) -> ExperienceBuffer {
        let mut buf = ExperienceBuffer::new(64, 0.5);
        for &(reward, value, env_boundary) in spec {
            buf.push(Transition {
                observation: vec![0.0; 4],
                latent: vec![0.0; 4],
                action: vec![0.0; MAX_ACTION_DIM],
                reward,
                credit: 0.0,
                pred_error: 0.0,
                value,
                prob_taken: 1.0,
                option_idx: 0,
                env_id: 0,
                env_boundary,
            });
        }
        buf
    }

    #[test]
    fn td_n_step_return_no_bootstrap_matches_plain_sum() {
        // Three rewards, no bootstrap. Expect ret = Σ γ^k r_k.
        let buf = mk_buffer(&[(1.0, 0.0, false), (2.0, 0.0, false), (4.0, 0.0, false)]);
        let gamma = 0.9;
        let (ret, gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, false, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
        assert!((gk - 0.9f32.powi(3)).abs() < 1e-5, "gk: {}", gk);
        assert!(!term, "should not be terminated");
    }

    #[test]
    fn td_n_step_return_bootstrap_adds_gamma_n_v() {
        // Same three rewards + a 4th transition carrying value=10.
        // With bootstrap, expect ret = Σ γ^k r_k + γ^3 · 10.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (4.0, 0.0, false),
            (99.0, 10.0, false), // reward ignored; only its .value is read for bootstrap
        ]);
        let gamma = 0.9;
        let (ret, _gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0 + 0.729 * 10.0;
        assert!(
            (ret - expected).abs() < 1e-4,
            "ret: {} vs {}",
            ret,
            expected
        );
        assert!(!term);
    }

    #[test]
    fn td_n_step_return_termination_inside_window_suppresses_bootstrap() {
        // Transition at k=2 is env_boundary → window terminates at k=1,
        // bootstrap suppressed even though the 4th transition has value.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (999.0, 0.0, true), // boundary
            (99.0, 10.0, false),
        ]);
        let gamma = 0.9;
        let (ret, _gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        // Only k=0 and k=1 rewards included; no bootstrap.
        let expected = 1.0 + 0.9 * 2.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
        assert!(term, "should be terminated");
    }

    #[test]
    fn td_n_step_return_bootstrap_state_across_boundary_suppresses_bootstrap() {
        // Window fills cleanly (k=0..2 no boundary), but the bootstrap
        // transition at index 3 IS a boundary — means it's the first
        // transition of a new episode, not a continuation. Bootstrap
        // should be dropped.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (4.0, 0.0, false),
            (99.0, 10.0, true), // boundary at bootstrap point
        ]);
        let gamma = 0.9;
        let (ret, _gk, term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
        // The WINDOW itself didn't terminate — we just have no bootstrap.
        assert!(!term);
    }

    #[test]
    fn td_n_step_return_bootstrap_clamp_bounds_runaway_value() {
        // Value head diverged to +1e6; clamp should keep target sane.
        let buf = mk_buffer(&[
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (0.0, 0.0, false),
            (0.0, 1.0e6, false),
        ]);
        let gamma = 0.9;
        let (ret, _gk, _term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        // With clamp=100 on the bootstrap, ret = γ^3 · 100 = 72.9.
        let expected = 0.729 * 100.0;
        assert!(
            (ret - expected).abs() < 1e-3,
            "ret: {} vs {}",
            ret,
            expected
        );
    }

    #[test]
    fn td_n_step_return_bootstrap_ignores_nonfinite_value() {
        // NaN value at bootstrap position — should be skipped.
        let buf = mk_buffer(&[
            (1.0, 0.0, false),
            (2.0, 0.0, false),
            (3.0, 0.0, false),
            (99.0, f32::NAN, false),
        ]);
        let gamma = 0.9;
        let (ret, _gk, _term) = compute_td_n_step_return(&buf, 0, 3, gamma, true, 100.0);
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 3.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
    }

    #[test]
    fn td_n_step_return_single_reward_bootstrap_is_td0() {
        // n_step=1 with bootstrap = classical TD(0) target.
        let buf = mk_buffer(&[(2.0, 0.0, false), (0.0, 5.0, false)]);
        let gamma = 0.95;
        let (ret, _gk, _term) = compute_td_n_step_return(&buf, 0, 1, gamma, true, 100.0);
        let expected = 2.0 + 0.95 * 5.0;
        assert!(
            (ret - expected).abs() < 1e-5,
            "ret: {} vs {}",
            ret,
            expected
        );
    }

    #[test]
    fn td_n_step_return_gamma_zero_myopic() {
        // With γ=0, only the ripe reward counts — discount annihilates everything else.
        let buf = mk_buffer(&[
            (7.0, 0.0, false),
            (999.0, 0.0, false),
            (999.0, 0.0, false),
            (99.0, 999.0, false),
        ]);
        let (ret, gk, _term) = compute_td_n_step_return(&buf, 0, 3, 0.0, true, 100.0);
        // γ^0 · 7 + 0 · 999 + 0 · 999 + 0 · bootstrap = 7
        assert!((ret - 7.0).abs() < 1e-5, "ret: {}", ret);
        assert_eq!(gk, 0.0);
    }

    // ---- GAE advantage tests ----

    #[test]
    fn gae_lambda_zero_is_pure_td0() {
        // λ=0 → Â_t = δ_0 = r_0 + γ·V(s_1) − V(s_0).
        // ripe has reward=2, value=1; next has value=5.
        let buf = mk_buffer(&[
            (2.0, 1.0, false),
            (0.0, 5.0, false),
            (0.0, 0.0, false), // padding so buf_len > ripe+n_step
        ]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 1, gamma, 0.0, 100.0);
        let expected = 2.0 + 0.9 * 5.0 - 1.0;
        assert!(
            (adv - expected).abs() < 1e-5,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_lambda_one_equals_mc_minus_value() {
        // λ=1 → Â_t = Σ γ^k · δ_{t+k} = (Σ γ^k · r_{t+k}) + γ^n · V(s_{t+n}) − V(s_t).
        // Verify via telescoping.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),  // ripe: r=1, V=0.5
            (2.0, 0.3, false),  // t+1: r=2, V=0.3
            (4.0, 0.2, false),  // t+2: r=4, V=0.2
            (99.0, 10.0, false), // bootstrap slot: V=10
        ]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 3, gamma, 1.0, 100.0);
        // MC return + γ^3·V_boot − V(s_ripe)
        let expected = 1.0 + 0.9 * 2.0 + 0.81 * 4.0 + 0.729 * 10.0 - 0.5;
        assert!(
            (adv - expected).abs() < 1e-4,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_recursive_identity() {
        // Â_t = δ_t + γλ · Â_{t+1}. Compute twice — once for the
        // full window at ripe_idx=0, once for the shorter window at
        // ripe_idx=1 — and verify the identity numerically.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),
            (2.0, 0.3, false),
            (4.0, 0.2, false),
            (0.0, 6.0, false),
        ]);
        let gamma = 0.9;
        let lambda = 0.95;
        let adv_full = compute_gae_advantage(&buf, 0, 3, gamma, lambda, 100.0);
        let adv_tail = compute_gae_advantage(&buf, 1, 2, gamma, lambda, 100.0);
        // δ_0 = 1 + 0.9·0.3 − 0.5 = 0.77
        let delta0 = 1.0 + 0.9 * 0.3 - 0.5;
        let expected = delta0 + gamma * lambda * adv_tail;
        assert!(
            (adv_full - expected).abs() < 1e-5,
            "adv_full: {} vs δ_0 + γλ·adv_tail = {}",
            adv_full,
            expected
        );
    }

    #[test]
    fn gae_termination_inside_window_stops_accumulation() {
        // Boundary at k=2 (index 2 is the start of a new episode)
        // → only δ's for k=0 and k=1 accumulate.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),
            (2.0, 0.3, false),
            (999.0, 777.0, true), // boundary → don't accumulate this δ,
            //                     and its V is NOT read as V(s_{t+1}) for k=1
            (0.0, 0.0, false),
        ]);
        let gamma = 0.9;
        let lambda = 0.95;
        let adv = compute_gae_advantage(&buf, 0, 3, gamma, lambda, 100.0);
        // δ_0: r=1, V(s_1)=0.3 (next has no boundary) → 1 + 0.9·0.3 − 0.5 = 0.77
        // δ_1: r=2, V(s_2) — but index 2 has env_boundary=true → V(s_2) treated as 0
        //      → 2 + 0.9·0 − 0.3 = 1.7
        // Stop (index 2 boundary also cuts accumulation for k≥2).
        // Â_0 = δ_0 + γλ · δ_1 = 0.77 + 0.9·0.95·1.7 = 0.77 + 1.4535 = 2.2235
        let d0 = 1.0 + 0.9 * 0.3 - 0.5;
        let d1 = 2.0 + 0.9 * 0.0 - 0.3;
        let expected = d0 + gamma * lambda * d1;
        assert!(
            (adv - expected).abs() < 1e-4,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_boundary_at_bootstrap_zeros_last_next_v() {
        // No boundary inside the window, but the n+1-th slot IS a
        // boundary → V(s_{ripe+n_step}) treated as 0 for the last δ.
        let buf = mk_buffer(&[
            (1.0, 0.5, false),
            (2.0, 0.3, false),
            (4.0, 0.2, false),
            (99.0, 777.0, true), // boundary → V at bootstrap slot → 0
        ]);
        let gamma = 0.9;
        let lambda = 0.95;
        let adv = compute_gae_advantage(&buf, 0, 3, gamma, lambda, 100.0);
        let d0 = 1.0 + 0.9 * 0.3 - 0.5;
        let d1 = 2.0 + 0.9 * 0.2 - 0.3;
        // δ_2: V(s_3) treated as 0 (boundary) → 4 + 0.9·0 − 0.2 = 3.8
        let d2 = 4.0 + 0.9 * 0.0 - 0.2;
        let expected = d0 + gamma * lambda * (d1 + gamma * lambda * d2);
        assert!(
            (adv - expected).abs() < 1e-4,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_clamp_bounds_runaway_value() {
        // V=1e6 at ripe; clamp to ±100 before computing δ.
        let buf = mk_buffer(&[
            (0.0, 1e6, false),
            (0.0, 0.0, false),
            (0.0, 0.0, false),
        ]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 1, gamma, 0.5, 100.0);
        // δ_0 = 0 + 0.9·0 − 100 (V clamped) = −100. λ=0.5 has only 1 step, so Â=δ_0.
        assert!(
            (adv - (-100.0)).abs() < 1e-3,
            "adv: {} (expected −100)",
            adv
        );
    }

    #[test]
    fn gae_nonfinite_value_treated_as_zero() {
        // NaN at ripe's V — treated as 0.
        let buf = mk_buffer(&[
            (3.0, f32::NAN, false),
            (0.0, 2.0, false),
            (0.0, 0.0, false),
        ]);
        let gamma = 0.9;
        let adv = compute_gae_advantage(&buf, 0, 1, gamma, 0.0, 100.0);
        // δ_0 = 3 + 0.9·2 − 0 = 4.8
        let expected = 3.0 + 0.9 * 2.0;
        assert!(
            (adv - expected).abs() < 1e-5,
            "adv: {} vs {}",
            adv,
            expected
        );
    }

    #[test]
    fn gae_gamma_zero_is_single_step_residual() {
        // γ=0 → all later δ's have γλ=0 weight, and within each δ
        // the γ·V(s_{t+1}) term is also zero. Â_0 = r_0 − V(s_0).
        let buf = mk_buffer(&[
            (5.0, 1.0, false),
            (999.0, 999.0, false),
            (999.0, 999.0, false),
        ]);
        let adv = compute_gae_advantage(&buf, 0, 2, 0.0, 0.95, 100.0);
        assert!((adv - 4.0).abs() < 1e-5, "adv: {}", adv);
    }
}
