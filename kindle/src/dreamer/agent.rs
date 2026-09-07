//! Online acting, replay learning, and latent imagination.

mod vector;
pub use vector::VectorDreamerAgent;

use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use meganeura::{Mode, Session};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use super::behavior;
use super::checkpoint::{self, TensorFingerprints};
use super::checkpoint::{
    BEHAVIOR as CHECKPOINT_BEHAVIOR, SLOW_VALUE as CHECKPOINT_SLOW_VALUE, WORLD as CHECKPOINT_WORLD,
};
use super::config::DreamerConfig;
use super::cpu;
use super::distributions::{
    PercentileNormalizer, TwoHotBins, continuation_weights, lambda_returns, sample_probabilities,
    sample_probabilities_at, softmax_unimix,
};
use super::intrinsic::{VisitationBonus, VisitationState};
use super::readback::{Readback, ReadbackTiming};
use super::replay::{FrameFlags, ReplayFrame, Reward, SequenceBatch, SequenceReplay};
use super::runtime::{
    build_session, configure_d3_optimizer, ema_matching, initialize_d3, sync_matching,
};
use super::world;
use super::{BLADE_REV, DREAMERV3_UPSTREAM_REV, MEGANEURA_REV};
use crate::env::{RgbFrame, Transition};
use crate::vision::{
    OBSERVATION_CHANNELS, OBSERVATION_GRID, Observation, PROJECTION_SEED, Perception,
    PerceptionIdentity, PerceptionKind,
};

const CHECKPOINT_FORMAT: u32 = 3;
const CHECKPOINT_ARCHITECTURE: &str = "dreamerv3-visual-features";
const CHECKPOINT_METADATA: &str = "metadata.json";
const RNG_POLICY: u64 = 0x706f_6c69_6379_0001;
const RNG_LIVE_POSTERIOR: u64 = 0x6c69_7665_706f_7374;
const RNG_REPLAY: u64 = 0x7265_706c_6179_0001;
const RNG_TRAIN_POSTERIOR: u64 = 0x7472_6169_6e70_6f73;
const RNG_IMAGINATION: u64 = 0x696d_6167_696e_6501;
const RNG_DIAGNOSTIC: u64 = 0x6469_6167_6e6f_7374;
const RNG_RESUME: u64 = 0x7265_7375_6d65_0001;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct CheckpointMetadata {
    format: u32,
    architecture: String,
    dreamerv3_revision: String,
    meganeura_revision: String,
    blade_revision: String,
    projection_seed: u64,
    #[serde(default)]
    perception: Option<PerceptionIdentity>,
    observation_grid: usize,
    observation_channels: usize,
    config: DreamerConfig,
    learner_step: u64,
    environment_step: u64,
    #[serde(default = "single_stream")]
    collection_streams: usize,
    return_low: f32,
    return_high: f32,
    #[serde(default)]
    visitation: Option<VisitationState>,
    #[serde(default)]
    future_head_revision: Option<String>,
    #[serde(default)]
    tensor_sha256: Option<TensorFingerprints>,
}

fn single_stream() -> usize {
    1
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ActionMode {
    #[default]
    Sample,
    Greedy,
}

/// Executable graph and reward identities, separate from run hyperparameters.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ModelProvenance {
    pub dreamerv3_revision: &'static str,
    pub meganeura_revision: &'static str,
    pub blade_revision: &'static str,
    pub future_head_revision: Option<&'static str>,
    pub visitation_hash_version: Option<u32>,
    /// Bound by the pixel agent; absent for a core fed precomputed features.
    pub perception: Option<PerceptionIdentity>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct WorldMetrics {
    pub total_loss: f32,
    pub reconstruction_loss: f32,
    pub future_prediction_loss: f32,
    /// Unclipped posterior-to-prior KL before either free-nat floor.
    pub raw_kl: f32,
    pub dynamics_kl: f32,
    pub representation_kl: f32,
    pub reward_loss: f32,
    pub continuation_loss: f32,
    /// Replay-value loss as seen by the frozen critic inside the world graph.
    /// Its gradient updates RSSM/representation parameters, not critic weights.
    pub replay_value_loss: f32,
    pub replay_reward_prediction_mean: f32,
    pub replay_reward_target_mean: f32,
    pub replay_reward_mae: f32,
    /// Mean prediction for nonzero reward targets. Retained for compatibility
    /// with sparse tasks whose only nonzero reward is positive.
    pub rewarded_prediction_mean: f32,
    /// Mean prediction for zero reward targets.
    pub unrewarded_prediction_mean: f32,
    pub rewarded_count: usize,
    pub positive_reward_prediction_mean: f32,
    pub zero_reward_prediction_mean: f32,
    pub negative_reward_prediction_mean: f32,
    pub positive_reward_count: usize,
    pub zero_reward_count: usize,
    pub negative_reward_count: usize,
    pub replay_continuation_prediction_mean: f32,
    pub replay_continuation_target_mean: f32,
    pub replay_continuation_mae: f32,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct BehaviorMetrics {
    pub total_loss: f32,
    pub policy_loss: f32,
    /// One when the policy objective contributes gradients, otherwise zero.
    pub actor_update_scale: f32,
    pub value_loss: f32,
    pub replay_value_loss: f32,
    /// Mean categorical entropy before trajectory weighting.
    pub policy_entropy: f32,
    /// Entropy after the same continuation weighting used by the actor loss.
    pub weighted_policy_entropy: f32,
    pub imagined_reward_mean: f32,
    pub imagined_continuation_mean: f32,
    pub return_mean: f32,
    pub return_scale: f32,
    pub advantage_abs_mean: f32,
    pub weighted_advantage_abs_mean: f32,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LearnReport {
    pub learner_step: u64,
    pub replay_len: usize,
    pub world: WorldMetrics,
    pub behavior: BehaviorMetrics,
    pub timing: LearnTiming,
}

/// Host wall time and submitted payload in `Session::set_input`, not bus traffic.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct InputWriteTiming {
    pub calls: usize,
    pub bytes: usize,
    pub seconds: f64,
}

impl InputWriteTiming {
    fn set_input(&mut self, session: &mut Session, name: &str, values: &[f32]) {
        let started = Instant::now();
        session.set_input(name, values);
        self.seconds += started.elapsed().as_secs_f64();
        self.calls += 1;
        self.bytes += std::mem::size_of_val(values);
    }
}

/// Wall time includes the GPU waits and host transfers performed by each stage.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct LearnTiming {
    pub replay_seconds: f64,
    pub posterior_seconds: f64,
    pub imagination_seconds: f64,
    /// Subsets of the corresponding stage totals, not additional elapsed time.
    pub posterior_readback: ReadbackTiming,
    pub imagination_readback: ReadbackTiming,
    pub posterior_inputs: InputWriteTiming,
    pub imagination_inputs: InputWriteTiming,
    pub world_train_seconds: f64,
    pub replay_refresh_seconds: f64,
    pub world_sync_seconds: f64,
    pub behavior_train_seconds: f64,
    pub behavior_sync_seconds: f64,
    pub total_seconds: f64,
}

struct PosteriorBatch {
    deter: Vec<Vec<f32>>,
    stoch: Vec<Vec<f32>>,
}

struct BehaviorTrainingBatch {
    imagined_feature: Vec<f32>,
    action_target: Vec<f32>,
    imagined_weight: Vec<f32>,
    imagined_value_target: Vec<f32>,
    imagined_slow_target: Vec<f32>,
    replay_feature: Vec<f32>,
    replay_weight: Vec<f32>,
    replay_value_target: Vec<f32>,
    replay_slow_target: Vec<f32>,
    imagined_reward_mean: f32,
    imagined_continuation_mean: f32,
    return_mean: f32,
    return_scale: f32,
    advantage_abs_mean: f32,
    weighted_advantage_abs_mean: f32,
    replay_reward_prediction_mean: f32,
    replay_reward_target_mean: f32,
    replay_reward_mae: f32,
    rewarded_prediction_mean: f32,
    unrewarded_prediction_mean: f32,
    rewarded_count: usize,
    positive_reward_prediction_mean: f32,
    zero_reward_prediction_mean: f32,
    negative_reward_prediction_mean: f32,
    positive_reward_count: usize,
    zero_reward_count: usize,
    negative_reward_count: usize,
    replay_continuation_prediction_mean: f32,
    replay_continuation_target_mean: f32,
    replay_continuation_mae: f32,
}

#[derive(Default)]
struct PriorRollout {
    rewards: Vec<f32>,
    continuations: Vec<f32>,
    values: Vec<f32>,
    observations: Vec<Vec<f32>>,
}

/// D3's runner does not call its ratio scheduler before replay is ready. The
/// first eligible call yields one update; only later environment steps accrue
/// fractional train-ratio credit.
#[derive(Default)]
struct D3TrainScheduler {
    started: bool,
    credit: f32,
}

struct DreamerRngs {
    policy: StdRng,
    live_posterior: StdRng,
    replay: StdRng,
    train_posterior: StdRng,
    imagination: StdRng,
}

impl DreamerRngs {
    fn new(seed: u64) -> Self {
        Self {
            policy: StdRng::seed_from_u64(seed ^ RNG_POLICY),
            live_posterior: StdRng::seed_from_u64(seed ^ RNG_LIVE_POSTERIOR),
            replay: StdRng::seed_from_u64(seed ^ RNG_REPLAY),
            train_posterior: StdRng::seed_from_u64(seed ^ RNG_TRAIN_POSTERIOR),
            imagination: StdRng::seed_from_u64(seed ^ RNG_IMAGINATION),
        }
    }

    fn resumed(seed: u64, learner_step: u64, environment_step: u64) -> Self {
        Self::new(
            seed ^ RNG_RESUME ^ learner_step.rotate_left(17) ^ environment_step.rotate_left(41),
        )
    }

    fn diagnostic(seed: u64, learner_step: u64, environment_step: u64) -> StdRng {
        StdRng::seed_from_u64(
            seed ^ RNG_DIAGNOSTIC ^ learner_step.rotate_left(23) ^ environment_step.rotate_left(47),
        )
    }
}

impl D3TrainScheduler {
    fn observe(&mut self, credit: f32) {
        if self.started {
            self.credit += credit;
        }
    }

    fn update_due(&mut self, replay_ready: bool) -> bool {
        if !replay_ready {
            return false;
        }
        if !self.started {
            self.started = true;
            self.credit = 1.0;
        }
        self.credit >= 1.0
    }

    fn consume_update(&mut self) {
        assert!(self.credit >= 1.0);
        self.credit -= 1.0;
    }
}

/// DreamerV3 learner over precomputed frozen visual observations.
///
/// Acting never trains implicitly. Call [`Self::learn`] explicitly, or use
/// [`Self::learn_scheduled`] to honor D3's replay-samples-per-environment-step
/// ratio without putting learner work on the control deadline.
pub struct DreamerCore {
    gpu: Arc<blade_graphics::Context>,
    readback: Readback,
    config: DreamerConfig,
    perception_identity: Option<PerceptionIdentity>,
    bins: TwoHotBins,
    replay: SequenceReplay,
    visitation: Option<VisitationBonus>,
    rngs: DreamerRngs,
    return_normalizer: PercentileNormalizer,
    world_train: Session,
    world_observe_batch: Session,
    world_observe_live: Session,
    world_transition: Session,
    world_transition_live: Session,
    world_heads: Session,
    world_heads_live: Session,
    world_prediction_live: Option<Session>,
    behavior_train: Session,
    behavior_online: Session,
    behavior_slow: Session,
    policy_live: Session,
    behavior_value_live: Option<Session>,
    deter: Vec<f32>,
    stoch: Vec<f32>,
    observation: Vec<f32>,
    encoded_observation: Vec<f32>,
    feature: Vec<f32>,
    pending_action: Option<usize>,
    active: bool,
    needs_reset: bool,
    learner_step: u64,
    environment_step: u64,
    train_scheduler: D3TrainScheduler,
    collection_streams: usize,
}

impl DreamerCore {
    pub fn new(config: DreamerConfig) -> Result<Self, blade_graphics::NotSupportedError> {
        let gpu = Arc::new(crate::init_gpu_context()?);
        Ok(Self::with_gpu(config, gpu))
    }

    /// Restore a model-sized training checkpoint into a fresh runtime.
    ///
    /// Replay and the in-flight episode are deliberately not checkpointed.
    /// A restored agent starts between episodes with empty replay, while both
    /// optimizer moments, the slow critic, counters, and return normalization
    /// continue exactly from the saved learner state.
    pub fn restore(checkpoint: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata = read_checkpoint_metadata(checkpoint.as_ref())?;
        let gpu = Arc::new(crate::init_gpu_context()?);
        Self::restore_with_gpu(checkpoint.as_ref(), gpu, metadata)
    }

    fn restore_with_gpu(
        checkpoint: &Path,
        gpu: Arc<blade_graphics::Context>,
        metadata: CheckpointMetadata,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut core = Self::with_gpu(metadata.config.clone(), gpu);
        core.perception_identity = metadata.perception;
        checkpoint::load_session(&mut core.world_train, &checkpoint.join(CHECKPOINT_WORLD))?;
        checkpoint::load_session(
            &mut core.behavior_train,
            &checkpoint.join(CHECKPOINT_BEHAVIOR),
        )?;
        checkpoint::load_session(
            &mut core.behavior_slow,
            &checkpoint.join(CHECKPOINT_SLOW_VALUE),
        )?;
        core.learner_step = metadata.learner_step;
        core.environment_step = metadata.environment_step;
        core.return_normalizer
            .restore(metadata.return_low, metadata.return_high);
        if let (Some(bonus), Some(state)) = (&mut core.visitation, metadata.visitation) {
            bonus.restore(state);
        }
        core.rngs =
            DreamerRngs::resumed(core.config.seed, core.learner_step, core.environment_step);
        core.sync_world_inference();
        sync_matching(&core.behavior_train, &mut core.behavior_online, "behavior.");
        sync_matching(
            &core.behavior_train,
            &mut core.policy_live,
            "behavior.actor.",
        );
        sync_matching(
            &core.behavior_train,
            &mut core.world_train,
            "behavior.value.",
        );
        Ok(core)
    }

    pub fn with_gpu(config: DreamerConfig, gpu: Arc<blade_graphics::Context>) -> Self {
        config.validate();
        let starts = config.batch_size * config.batch_length;
        let imagined_rows = starts * config.imagination_length;
        let replay_rows = config.batch_size * (config.batch_length - 1);

        let world_train_config = world_training_config(&config);
        let world_train_graph =
            world::build_training_graph(&world_train_config, config.world_backprop_length);
        let world_observe_batch_graph = world::build_observe_graph(&config, config.batch_size);
        let world_observe_live_graph = world::build_observe_graph(&config, 1);
        let world_transition_graph = world::build_transition_graph(&config, starts);
        let world_transition_live_graph = world::build_transition_graph(&config, 1);
        let world_head_graph = world::build_head_graph(&config, starts);
        let world_head_live_graph = world::build_head_graph(&config, 1);
        let behavior_train_graph =
            behavior::build_training_graph(&config, imagined_rows, replay_rows);
        let behavior_online_graph = behavior::build_inference_graph(&config, starts);
        let behavior_slow_graph = behavior::build_value_inference_graph(&config, starts);
        let policy_live_graph = behavior::build_actor_inference_graph(&config, 1);

        let mut world_train = build_session(
            &world_train_graph,
            &gpu,
            Mode::Training,
            config.skip_full_optimize,
        );
        let mut world_observe_batch =
            build_session(&world_observe_batch_graph, &gpu, Mode::Inference, false);
        let mut world_observe_live =
            build_session(&world_observe_live_graph, &gpu, Mode::Inference, false);
        let mut world_transition =
            build_session(&world_transition_graph, &gpu, Mode::Inference, false);
        let mut world_transition_live =
            build_session(&world_transition_live_graph, &gpu, Mode::Inference, false);
        let mut world_heads = build_session(&world_head_graph, &gpu, Mode::Inference, false);
        let mut world_heads_live =
            build_session(&world_head_live_graph, &gpu, Mode::Inference, false);
        let mut behavior_train = build_session(
            &behavior_train_graph,
            &gpu,
            Mode::Training,
            config.skip_full_optimize,
        );
        let mut behavior_online =
            build_session(&behavior_online_graph, &gpu, Mode::Inference, false);
        let mut behavior_slow = build_session(&behavior_slow_graph, &gpu, Mode::Inference, false);
        let mut policy_live = build_session(&policy_live_graph, &gpu, Mode::Inference, false);

        initialize_d3(&mut world_train, &world_train_graph, config.seed);
        initialize_d3(
            &mut behavior_train,
            &behavior_train_graph,
            config.seed ^ 0x5eed_0000_0000_0001,
        );
        for target in [
            &mut world_observe_batch,
            &mut world_observe_live,
            &mut world_transition,
            &mut world_transition_live,
            &mut world_heads,
            &mut world_heads_live,
        ] {
            sync_matching(&world_train, target, "world.");
        }
        sync_matching(&behavior_train, &mut behavior_online, "behavior.");
        sync_matching(&behavior_train, &mut behavior_slow, "behavior.value.");
        sync_matching(&behavior_train, &mut policy_live, "behavior.actor.");
        sync_matching(&behavior_train, &mut world_train, "behavior.value.");

        let size = config.network();
        Self {
            readback: Readback::new(Arc::clone(&gpu)),
            gpu,
            bins: TwoHotBins::new(config.value_bins),
            replay: SequenceReplay::new(config.replay_capacity),
            visitation: config.visitation_bonus.then(VisitationBonus::new),
            rngs: DreamerRngs::new(config.seed),
            return_normalizer: PercentileNormalizer::new(config.return_norm_rate, 1.0),
            deter: vec![0.0; size.deter],
            stoch: vec![0.0; size.stoch * size.classes],
            observation: vec![0.0; Observation::LEN],
            encoded_observation: vec![0.0; OBSERVATION_GRID * OBSERVATION_GRID * size.vision_depth],
            feature: vec![0.0; config.feature_dim()],
            pending_action: None,
            active: false,
            needs_reset: false,
            learner_step: 0,
            environment_step: 0,
            train_scheduler: D3TrainScheduler::default(),
            collection_streams: 1,
            config,
            perception_identity: None,
            world_train,
            world_observe_batch,
            world_observe_live,
            world_transition,
            world_transition_live,
            world_heads,
            world_heads_live,
            world_prediction_live: None,
            behavior_train,
            behavior_online,
            behavior_slow,
            policy_live,
            behavior_value_live: None,
        }
    }

    pub fn config(&self) -> &DreamerConfig {
        &self.config
    }

    pub fn provenance(&self) -> ModelProvenance {
        ModelProvenance {
            dreamerv3_revision: DREAMERV3_UPSTREAM_REV,
            meganeura_revision: MEGANEURA_REV,
            blade_revision: BLADE_REV,
            future_head_revision: (self.config.loss_scales.future_prediction > 0.0)
                .then_some(world::FUTURE_HEAD_REVISION),
            visitation_hash_version: self
                .config
                .visitation_bonus
                .then_some(super::intrinsic::VERSION),
            perception: self.perception_identity.clone(),
        }
    }

    pub fn replay_len(&self) -> usize {
        self.replay.len()
    }

    pub fn learner_step(&self) -> u64 {
        self.learner_step
    }

    pub fn environment_step(&self) -> u64 {
        self.environment_step
    }

    pub fn cpu_worker_threads(&self) -> usize {
        cpu::threads()
    }

    /// Trainable world and behavior parameter counts, excluding frozen perception,
    /// inference-session copies, and the EMA value model.
    pub fn trainable_parameter_counts(&self) -> (usize, usize) {
        (
            trainable_parameter_count(&self.world_train),
            trainable_parameter_count(&self.behavior_train),
        )
    }

    /// Adapter and driver used by all perception and Dreamer sessions.
    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        crate::gpu_device_info(self.gpu.device_information())
    }

    /// Profile sessions on inputs left by a completed learner update. Requires
    /// a context created with `MEGANEURA_GPU_TIMING=1`. Training captures only
    /// forward/backward work on the final row microbatch, without optimizer,
    /// clipping or accumulation passes. Parameters, optimizer moments, replay,
    /// counters and RNG streams are unchanged. The next learner update installs
    /// its ordinary optimizer settings and overwrites scratch gradients.
    pub fn profile_sessions(
        &mut self,
        directory: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use meganeura::profiler::{CaptureOptions, ProfileError, capture_session_profile};

        assert!(self.learner_step > 0, "learn once before profiling");
        fs::create_dir_all(directory.as_ref())?;
        for session in [&mut self.world_train, &mut self.behavior_train] {
            session.clear_optimizer();
            session.clear_grad_accumulate();
            session.disable_grad_clip();
        }
        for (name, session) in [
            ("posterior", &mut self.world_observe_batch),
            ("actor_value", &mut self.behavior_online),
            ("slow_value", &mut self.behavior_slow),
            ("world_heads", &mut self.world_heads),
            ("transition", &mut self.world_transition),
            ("world_gradient", &mut self.world_train),
            ("behavior_gradient", &mut self.behavior_train),
        ] {
            let mut durations = Vec::new();
            let mut gpu_durations = Vec::new();
            for iteration in 0..13 {
                let started = Instant::now();
                session.step();
                session.wait();
                let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
                // Advance the command-buffer ring before retaining timestamps
                // from these fixed-input, optimizer-free normal executions.
                if iteration >= 2 {
                    durations.push(wall_ms);
                    let timings = session.gpu_timings();
                    if !timings.is_empty() {
                        gpu_durations.push(
                            timings
                                .iter()
                                .map(|(_, duration)| duration.as_secs_f64())
                                .sum::<f64>()
                                * 1_000.0,
                        );
                    }
                }
            }
            durations.sort_by(f64::total_cmp);
            gpu_durations.sort_by(f64::total_cmp);
            let unprofiled_median_ms = durations[durations.len() / 2];
            let profile = capture_session_profile(
                session,
                |_| {},
                CaptureOptions {
                    unprofiled_median_ms: Some(unprofiled_median_ms),
                    ..CaptureOptions::default()
                },
            );
            let path = directory.as_ref().join(format!("{name}.json"));
            let mut data = match profile {
                Ok(profile) => serde_json::to_value(profile)?,
                Err(ProfileError::TooManyDispatches { count, limit }) => {
                    // Keep the wall measurement and the reason no GPU trace
                    // exists. A truncated timestamp stream is not a profile.
                    serde_json::json!({
                        "unprofiled_median_ms": unprofiled_median_ms,
                        "dispatch_count": count,
                        "timestamp_pass_limit": limit,
                        "per_dispatch_timing_unavailable": "dispatch count exceeds timestamp capacity",
                    })
                }
                Err(error) => return Err(error.into()),
            };
            data["ordinary_gpu_median_ms"] =
                serde_json::json!(gpu_durations.get(gpu_durations.len() / 2));
            data["ordinary_gpu_samples_ms"] = serde_json::json!(gpu_durations);
            data["ordinary_wall_samples_ms"] = serde_json::json!(durations);
            fs::write(path, serde_json::to_vec_pretty(&data)?)?;
        }
        Ok(())
    }

    /// Current posterior feature (`deter` followed by flattened categoricals).
    ///
    /// This read-only view is useful for representation probes. It must not be
    /// persisted as agent state; replay owns the recurrent context it needs.
    pub fn latent_feature(&self) -> &[f32] {
        &self.feature
    }

    /// Current output of the trainable adapter between perception and the RSSM.
    pub fn encoded_observation(&self) -> &[f32] {
        &self.encoded_observation
    }

    /// Current frozen visual observation before the trainable adapter.
    pub fn visual_observation(&self) -> &[f32] {
        &self.observation
    }

    /// Forecast made before this observation when future prediction is enabled,
    /// otherwise posterior reconstruction. Reset observations have no forecast
    /// target and should be excluded from diagnostic error summaries.
    pub fn observation_prediction(&mut self) -> Vec<f32> {
        self.ensure_world_prediction_live();
        let decoder = self
            .world_prediction_live
            .as_mut()
            .expect("diagnostic decoder initialized");
        decoder.set_input("deter", &self.deter);
        decoder.set_input("stoch", &self.stoch);
        decoder.step();
        decoder.wait();
        let mut observation = vec![0.0; Observation::LEN];
        decoder.read_output_by_index(0, &mut observation);
        observation
    }

    /// Reward predicted from the current posterior state.
    ///
    /// This runs the synchronized inference head without changing recurrent
    /// state and is intended for frozen evaluation and representation probes.
    pub fn posterior_reward_prediction(&mut self) -> f32 {
        self.world_heads_live.set_input("deter", &self.deter);
        self.world_heads_live.set_input("stoch", &self.stoch);
        self.world_heads_live.step();
        self.world_heads_live.wait();
        let mut reward_logits = vec![0.0; self.config.value_bins];
        self.world_heads_live
            .read_output_by_index(0, &mut reward_logits);
        decode_rows(&reward_logits, 1, &self.bins)[0]
    }

    /// Value predicted from the current posterior state.
    ///
    /// The diagnostic value session is initialized lazily and synchronized
    /// with the behavior critic without changing recurrent or optimizer state.
    pub fn posterior_value_prediction(&mut self) -> f32 {
        self.ensure_behavior_value_live();
        let value = self
            .behavior_value_live
            .as_mut()
            .expect("diagnostic value session initialized");
        value.set_input("feature", &self.feature);
        value.step();
        value.wait();
        let mut logits = vec![0.0; self.config.value_bins];
        value.read_output_by_index(0, &mut logits);
        decode_rows(&logits, 1, &self.bins)[0]
    }

    /// Reward predicted after one prior transition from the current state.
    ///
    /// This read-only diagnostic uses an independent categorical RNG, so
    /// probing does not perturb subsequent acting or posterior sampling.
    pub fn prior_reward_prediction(&mut self, action: usize) -> f32 {
        self.prior_reward_rollout(&[action])[0]
    }

    /// Open-loop prior reward predictions for a proposed action sequence.
    /// The live posterior state and RNG are left unchanged.
    pub fn prior_reward_rollout(&mut self, actions: &[usize]) -> Vec<f32> {
        self.prior_rollout(actions, false, false).rewards
    }

    /// Open-loop prior rewards and decoded visual observations.
    /// This diagnostic leaves the live posterior and all RNG streams intact.
    pub fn prior_diagnostic_rollout(&mut self, actions: &[usize]) -> (Vec<f32>, Vec<Vec<f32>>) {
        let rollout = self.prior_rollout(actions, true, false);
        (rollout.rewards, rollout.observations)
    }

    /// Open-loop reward, continuation, and value predictions.
    ///
    /// Calling this independently for candidate actions reuses identical
    /// categorical draws, making one-step counterfactual comparisons paired.
    pub fn prior_behavior_rollout(&mut self, actions: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let rollout = self.prior_rollout(actions, false, true);
        (rollout.rewards, rollout.continuations, rollout.values)
    }

    fn prior_rollout(
        &mut self,
        actions: &[usize],
        decode_observations: bool,
        decode_behavior: bool,
    ) -> PriorRollout {
        assert!(self.active, "call begin_episode before probing the prior");
        assert!(!actions.is_empty());
        assert!(
            actions
                .iter()
                .all(|action| *action < self.config.action_count)
        );
        let size = self.config.network();
        let mut deter = self.deter.clone();
        let mut stoch = self.stoch.clone();
        let mut rng =
            DreamerRngs::diagnostic(self.config.seed, self.learner_step, self.environment_step);
        let mut rollout = PriorRollout {
            rewards: Vec::with_capacity(actions.len()),
            continuations: Vec::with_capacity(if decode_behavior { actions.len() } else { 0 }),
            values: Vec::with_capacity(if decode_behavior { actions.len() } else { 0 }),
            observations: Vec::with_capacity(if decode_observations {
                actions.len()
            } else {
                0
            }),
        };
        if decode_observations {
            self.ensure_world_prediction_live();
        }
        if decode_behavior {
            self.ensure_behavior_value_live();
        }
        for &action in actions {
            let mut action_one_hot = vec![0.0; self.config.action_count];
            action_one_hot[action] = 1.0;
            self.world_transition_live.set_input("deter", &deter);
            self.world_transition_live.set_input("stoch", &stoch);
            self.world_transition_live
                .set_input("action", &action_one_hot);
            self.world_transition_live.step();
            self.world_transition_live.wait();
            let mut prior_logits = vec![0.0; size.stoch * size.classes];
            self.world_transition_live
                .read_output_by_index(0, &mut deter);
            self.world_transition_live
                .read_output_by_index(1, &mut prior_logits);
            stoch = sample_latents(
                &prior_logits,
                1,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut rng,
            );
            self.world_heads_live.set_input("deter", &deter);
            self.world_heads_live.set_input("stoch", &stoch);
            self.world_heads_live.step();
            self.world_heads_live.wait();
            let mut reward_logits = vec![0.0; self.config.value_bins];
            self.world_heads_live
                .read_output_by_index(0, &mut reward_logits);
            rollout
                .rewards
                .push(decode_rows(&reward_logits, 1, &self.bins)[0]);
            if decode_behavior {
                let mut continuation = [0.0];
                self.world_heads_live
                    .read_output_by_index(1, &mut continuation);
                rollout.continuations.push(continuation[0]);
                let feature = join_features(&deter, &stoch, 1, &self.config);
                let value = self
                    .behavior_value_live
                    .as_mut()
                    .expect("diagnostic value session initialized");
                value.set_input("feature", &feature);
                value.step();
                value.wait();
                let mut value_logits = vec![0.0; self.config.value_bins];
                value.read_output_by_index(0, &mut value_logits);
                rollout
                    .values
                    .push(decode_rows(&value_logits, 1, &self.bins)[0]);
            }
            if decode_observations {
                let decoder = self
                    .world_prediction_live
                    .as_mut()
                    .expect("diagnostic decoder initialized");
                decoder.set_input("deter", &deter);
                decoder.set_input("stoch", &stoch);
                decoder.step();
                decoder.wait();
                let mut observation = vec![0.0; Observation::LEN];
                decoder.read_output_by_index(0, &mut observation);
                rollout.observations.push(observation);
            }
        }
        rollout
    }

    fn ensure_world_prediction_live(&mut self) {
        if self.world_prediction_live.is_some() {
            return;
        }
        let graph = world::build_observation_prediction_graph(&self.config, 1);
        let mut decoder = build_session(&graph, &self.gpu, Mode::Inference, false);
        sync_matching(&self.world_train, &mut decoder, "world.");
        self.world_prediction_live = Some(decoder);
    }

    fn ensure_behavior_value_live(&mut self) {
        if self.behavior_value_live.is_some() {
            return;
        }
        let graph = behavior::build_value_inference_graph(&self.config, 1);
        let mut value = build_session(&graph, &self.gpu, Mode::Inference, false);
        sync_matching(&self.behavior_train, &mut value, "behavior.value.");
        self.behavior_value_live = Some(value);
    }

    /// Save model parameters, optimizer state, the EMA critic, and scalar learner
    /// state. Replay is intentionally excluded and can be refilled online.
    pub fn save_checkpoint(&mut self, checkpoint: impl AsRef<Path>) -> io::Result<()> {
        let checkpoint = checkpoint.as_ref();
        fs::create_dir_all(checkpoint)?;
        let world_temporary = checkpoint.join(format!("{CHECKPOINT_WORLD}.tmp"));
        let behavior_temporary = checkpoint.join(format!("{CHECKPOINT_BEHAVIOR}.tmp"));
        let slow_temporary = checkpoint.join(format!("{CHECKPOINT_SLOW_VALUE}.tmp"));
        self.world_train.save_checkpoint(&world_temporary)?;
        self.behavior_train.save_checkpoint(&behavior_temporary)?;
        self.behavior_slow.save_checkpoint(&slow_temporary)?;
        fs::rename(&world_temporary, checkpoint.join(CHECKPOINT_WORLD))?;
        fs::rename(&behavior_temporary, checkpoint.join(CHECKPOINT_BEHAVIOR))?;
        fs::rename(&slow_temporary, checkpoint.join(CHECKPOINT_SLOW_VALUE))?;

        let (return_low, return_high) = self.return_normalizer.state();
        let metadata = CheckpointMetadata {
            format: CHECKPOINT_FORMAT,
            architecture: CHECKPOINT_ARCHITECTURE.to_owned(),
            dreamerv3_revision: DREAMERV3_UPSTREAM_REV.to_owned(),
            meganeura_revision: MEGANEURA_REV.to_owned(),
            blade_revision: BLADE_REV.to_owned(),
            projection_seed: PROJECTION_SEED,
            perception: self.perception_identity.clone(),
            observation_grid: OBSERVATION_GRID,
            observation_channels: OBSERVATION_CHANNELS,
            config: self.config.clone(),
            learner_step: self.learner_step,
            environment_step: self.environment_step,
            collection_streams: self.collection_streams,
            return_low,
            return_high,
            visitation: self.visitation.as_ref().map(VisitationBonus::state),
            future_head_revision: self.provenance().future_head_revision.map(str::to_owned),
            tensor_sha256: Some(TensorFingerprints::read(checkpoint)?),
        };
        let encoded = serde_json::to_vec_pretty(&metadata).map_err(io::Error::other)?;
        let metadata_temporary = checkpoint.join(format!("{CHECKPOINT_METADATA}.tmp"));
        fs::write(&metadata_temporary, encoded)?;
        fs::rename(metadata_temporary, checkpoint.join(CHECKPOINT_METADATA))?;
        Ok(())
    }

    /// Begin the first episode or the next episode after `is_last`.
    /// Replay and all optimizer/model state survive this recurrent reset.
    ///
    /// The reset observation enters replay but does not advance
    /// `environment_step` or earn scheduler credit. Kindle's environment
    /// budget counts executed actions; upstream D3's driver counter also
    /// counts these action-free reset records, a small disclosed accounting
    /// difference in episode-based environments.
    pub fn begin_episode(&mut self, observation: Observation) {
        self.check_episode_boundary();
        self.deter.fill(0.0);
        self.stoch.fill(0.0);
        if let Some(bonus) = &mut self.visitation {
            bonus.observe(&observation);
        }
        self.posterior_live(observation.as_slice(), None, true);
        self.replay.push(
            ReplayFrame {
                observation,
                previous_action: None,
                reward: Reward::default(),
                flags: FrameFlags {
                    is_first: true,
                    is_last: false,
                    is_terminal: false,
                },
                deter: self.deter.clone().into_boxed_slice(),
                stoch: self.stoch.clone().into_boxed_slice(),
            },
            &self.config,
        );
        self.active = true;
        self.needs_reset = false;
    }

    fn check_episode_boundary(&self) {
        assert!(
            self.pending_action.is_none(),
            "cannot reset with a pending action"
        );
        assert!(
            !self.active || self.needs_reset,
            "begin_episode is only legal initially or after an is_last frame"
        );
    }

    pub fn act(&mut self, mode: ActionMode, action_mask: Option<&[bool]>) -> usize {
        assert!(self.active, "call begin_episode before act");
        assert!(
            !self.needs_reset,
            "call begin_episode after an is_last frame"
        );
        assert!(
            self.pending_action.is_none(),
            "observe the pending action first"
        );
        let probabilities = self.posterior_action_probabilities(action_mask);
        let action = match mode {
            ActionMode::Sample => sample_probabilities(&probabilities, &mut self.rngs.policy),
            ActionMode::Greedy => argmax(&probabilities),
        };
        self.pending_action = Some(action);
        action
    }

    /// Current posterior policy after optional action masking and actor
    /// unimix. This read-only diagnostic does not sample or alter live state.
    pub fn posterior_action_probabilities(&mut self, action_mask: Option<&[bool]>) -> Vec<f32> {
        assert!(self.active, "call begin_episode before probing the policy");
        if let Some(mask) = action_mask {
            assert_eq!(mask.len(), self.config.action_count);
            assert!(mask.iter().any(|valid| *valid));
        }
        self.policy_live.set_input("feature", &self.feature);
        self.policy_live.step();
        self.policy_live.wait();
        let mut logits = vec![0.0; self.config.action_count];
        self.policy_live.read_output_by_index(0, &mut logits);
        if let Some(mask) = action_mask {
            for (logit, valid) in logits.iter_mut().zip(mask) {
                if !valid {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
        let mut probabilities = vec![0.0; self.config.action_count];
        if let Some(mask) = action_mask {
            let valid_count = mask.iter().filter(|valid| **valid).count();
            softmax_unimix(&logits, 0.0, &mut probabilities);
            for (probability, valid) in probabilities.iter_mut().zip(mask) {
                *probability = if *valid {
                    (1.0 - self.config.actor_unimix) * *probability
                        + self.config.actor_unimix / valid_count as f32
                } else {
                    0.0
                };
            }
        } else {
            softmax_unimix(&logits, self.config.actor_unimix, &mut probabilities);
        }
        probabilities
    }

    /// Ingest an arrival and return both reward channels as stored in replay,
    /// including optional visitation novelty. Scales apply only during learning.
    pub fn observe(
        &mut self,
        observation: Observation,
        mut reward: Reward,
        flags: FrameFlags,
    ) -> Reward {
        assert!(!flags.is_first, "use begin_episode for an is_first frame");
        if flags.is_terminal {
            assert!(flags.is_last);
        }
        let action = self
            .pending_action
            .take()
            .expect("act must precede observe");
        if let Some(bonus) = &mut self.visitation {
            reward.intrinsic += bonus.observe(&observation);
        }
        self.posterior_live(observation.as_slice(), Some(action), false);
        self.replay.push(
            ReplayFrame {
                observation,
                previous_action: Some(action),
                reward,
                flags,
                deter: self.deter.clone().into_boxed_slice(),
                stoch: self.stoch.clone().into_boxed_slice(),
            },
            &self.config,
        );
        self.environment_step += 1;
        self.train_scheduler.observe(
            self.config.train_ratio / (self.config.batch_size * self.config.batch_length) as f32,
        );
        self.needs_reset = flags.is_last;
        reward
    }

    /// Execute at most `maximum_updates` updates due under D3's numeric train
    /// ratio, clocked by Kindle's executed-action environment steps.
    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        let mut reports = Vec::new();
        if self.config.train_ratio == 0.0 {
            return reports;
        }
        let replay_ready =
            self.replay.valid_sequence_count(&self.config) >= self.config.replay_warmup_sequences();
        while reports.len() < maximum_updates && self.train_scheduler.update_due(replay_ready) {
            let Some(report) = self.learn() else {
                break;
            };
            self.train_scheduler.consume_update();
            reports.push(report);
        }
        reports
    }

    /// Perform one D3 learner update, independent of scheduler credit.
    pub fn learn(&mut self) -> Option<LearnReport> {
        let started = Instant::now();
        let stage = Instant::now();
        let batch = self.replay.sample(&self.config, &mut self.rngs.replay)?;
        let mut timing = LearnTiming {
            replay_seconds: stage.elapsed().as_secs_f64(),
            ..LearnTiming::default()
        };
        self.readback.take_timing();
        let stage = Instant::now();
        let posterior = self.sample_posterior_batch(&batch, &mut timing.posterior_inputs);
        timing.posterior_seconds = stage.elapsed().as_secs_f64();
        timing.posterior_readback = self.readback.take_timing();
        // D3 forms all targets from the same pre-update parameters. Keeping
        // this ordering also gives the world graph replay-value targets while
        // its frozen critic routes that auxiliary gradient into the RSSM.
        let stage = Instant::now();
        let behavior_batch =
            self.imagine_and_target(&batch, &posterior, &mut timing.imagination_inputs);
        timing.imagination_seconds = stage.elapsed().as_secs_f64();
        timing.imagination_readback = self.readback.take_timing();
        let stage = Instant::now();
        let world = self.train_world(&batch, &posterior, &behavior_batch);
        timing.world_train_seconds = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        self.replay
            .update_context(&batch, &posterior.deter, &posterior.stoch, &self.config);
        timing.replay_refresh_seconds = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        self.sync_world_inference();
        timing.world_sync_seconds = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let behavior = self.train_behavior(&behavior_batch);
        timing.behavior_train_seconds = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        self.sync_behavior_inference();
        timing.behavior_sync_seconds = stage.elapsed().as_secs_f64();
        self.learner_step += 1;
        timing.total_seconds = started.elapsed().as_secs_f64();
        Some(LearnReport {
            learner_step: self.learner_step,
            replay_len: self.replay.len(),
            world,
            behavior,
            timing,
        })
    }

    fn posterior_live(&mut self, observation: &[f32], action: Option<usize>, reset: bool) {
        self.observation.copy_from_slice(observation);
        let size = self.config.network();
        let mut action_one_hot = vec![0.0; self.config.action_count];
        if let Some(action) = action {
            action_one_hot[action] = 1.0;
        }
        let keep = if reset { 0.0 } else { 1.0 };
        self.world_observe_live
            .set_input("previous_deter", &self.deter);
        self.world_observe_live
            .set_input("previous_stoch", &self.stoch);
        self.world_observe_live
            .set_input("previous_action", &action_one_hot);
        self.world_observe_live
            .set_input("observation", observation);
        self.world_observe_live
            .set_input("keep_deter", &vec![keep; size.deter]);
        self.world_observe_live
            .set_input("keep_stoch", &vec![keep; size.stoch * size.classes]);
        self.world_observe_live
            .set_input("keep_action", &vec![keep; self.config.action_count]);
        self.world_observe_live.step();
        let mut logits = vec![0.0; size.stoch * size.classes];
        self.readback.read(
            &self.world_observe_live,
            &mut [
                (0, &mut self.deter),
                (1, &mut logits),
                (3, &mut self.encoded_observation),
            ],
        );
        self.stoch = sample_latents(
            &logits,
            1,
            size.stoch,
            size.classes,
            self.config.unimix,
            &mut self.rngs.live_posterior,
        );
        self.feature = join_features(&self.deter, &self.stoch, 1, &self.config);
        assert!(self.feature.iter().all(|value| value.is_finite()));
    }

    fn sample_posterior_batch(
        &mut self,
        batch: &SequenceBatch,
        inputs: &mut InputWriteTiming,
    ) -> PosteriorBatch {
        let size = self.config.network();
        let rows = self.config.batch_size;
        let mut previous_deter = batch.initial_deter.clone();
        let mut previous_stoch = batch.initial_stoch.clone();
        let mut deters = Vec::with_capacity(self.config.batch_length);
        let mut stochs = Vec::with_capacity(self.config.batch_length);
        for time in 0..self.config.batch_length {
            let (keep_deter, keep_stoch, keep_action) = keep_masks(batch, time, &self.config);
            inputs.set_input(
                &mut self.world_observe_batch,
                "previous_deter",
                &previous_deter,
            );
            inputs.set_input(
                &mut self.world_observe_batch,
                "previous_stoch",
                &previous_stoch,
            );
            inputs.set_input(
                &mut self.world_observe_batch,
                "previous_action",
                &batch.previous_actions[time],
            );
            inputs.set_input(
                &mut self.world_observe_batch,
                "observation",
                &batch.observations[time],
            );
            inputs.set_input(&mut self.world_observe_batch, "keep_deter", &keep_deter);
            inputs.set_input(&mut self.world_observe_batch, "keep_stoch", &keep_stoch);
            inputs.set_input(&mut self.world_observe_batch, "keep_action", &keep_action);
            self.world_observe_batch.step();
            let mut deter = vec![0.0; rows * size.deter];
            let mut logits = vec![0.0; rows * size.stoch * size.classes];
            self.readback.read(
                &self.world_observe_batch,
                &mut [(0, &mut deter), (1, &mut logits)],
            );
            let stoch = sample_latents(
                &logits,
                rows,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut self.rngs.train_posterior,
            );
            previous_deter = deter.clone();
            previous_stoch = stoch.clone();
            deters.push(deter);
            stochs.push(stoch);
        }
        PosteriorBatch {
            deter: deters,
            stoch: stochs,
        }
    }

    fn train_world(
        &mut self,
        batch: &SequenceBatch,
        posterior: &PosteriorBatch,
        behavior: &BehaviorTrainingBatch,
    ) -> WorldMetrics {
        let rows = self.config.batch_size;
        let network = self.config.network();
        let stochastic_width = network.stoch * network.classes;
        let observation_width = self.config.observation_dim();
        let microbatch_rows = self.config.world_microbatch_size();
        let microbatch_count = rows / microbatch_rows;
        let chunk_length = self.config.world_backprop_length;
        let chunk_count = self.config.batch_length / chunk_length;
        let pass_count = chunk_count * microbatch_count;
        self.world_train
            .set_grad_accumulate(pass_count.try_into().unwrap());
        self.world_train.zero_grad();
        let mut metrics = WorldMetrics::default();

        for chunk in 0..chunk_count {
            let start = chunk * chunk_length;
            for microbatch in 0..microbatch_count {
                let first_row = microbatch * microbatch_rows;
                let initial_deter = if start == 0 {
                    &batch.initial_deter
                } else {
                    &posterior.deter[start - 1]
                };
                let initial_stoch = if start == 0 {
                    &batch.initial_stoch
                } else {
                    &posterior.stoch[start - 1]
                };
                self.world_train.set_input(
                    "initial_deter",
                    row_slice(initial_deter, first_row, microbatch_rows, network.deter),
                );
                self.world_train.set_input(
                    "initial_stoch",
                    row_slice(initial_stoch, first_row, microbatch_rows, stochastic_width),
                );

                for local_time in 0..chunk_length {
                    let time = start + local_time;
                    let (keep_deter, keep_stoch, keep_action) =
                        keep_masks_range(batch, time, first_row, microbatch_rows, &self.config);
                    self.world_train.set_input(
                        &format!("observation_{local_time}"),
                        row_slice(
                            &batch.observations[time],
                            first_row,
                            microbatch_rows,
                            observation_width,
                        ),
                    );
                    self.world_train.set_input(
                        &format!("previous_action_{local_time}"),
                        row_slice(
                            &batch.previous_actions[time],
                            first_row,
                            microbatch_rows,
                            self.config.action_count,
                        ),
                    );
                    self.world_train
                        .set_input(&format!("keep_deter_{local_time}"), &keep_deter);
                    self.world_train
                        .set_input(&format!("keep_stoch_{local_time}"), &keep_stoch);
                    self.world_train
                        .set_input(&format!("keep_action_{local_time}"), &keep_action);
                    self.world_train.set_input(
                        &format!("posterior_sample_{local_time}"),
                        row_slice(
                            &posterior.stoch[time],
                            first_row,
                            microbatch_rows,
                            stochastic_width,
                        ),
                    );
                    let mut reward_target = vec![0.0; microbatch_rows * self.config.value_bins];
                    for local_row in 0..microbatch_rows {
                        self.bins.encode(
                            batch.rewards[time][first_row + local_row],
                            &mut reward_target[local_row * self.config.value_bins
                                ..(local_row + 1) * self.config.value_bins],
                        );
                    }
                    self.world_train
                        .set_input(&format!("reward_target_{local_time}"), &reward_target);
                    let continuation_target = batch.flags[time]
                        [first_row..first_row + microbatch_rows]
                        .iter()
                        .map(|flags| {
                            if flags.is_terminal {
                                0.0
                            } else {
                                self.config.continuation_discount()
                            }
                        })
                        .collect::<Vec<_>>();
                    self.world_train.set_input(
                        &format!("continuation_target_{local_time}"),
                        &continuation_target,
                    );
                    let mut replay_value_target =
                        vec![0.0; microbatch_rows * self.config.value_bins];
                    let mut replay_slow_target =
                        vec![0.0; microbatch_rows * self.config.value_bins];
                    let mut replay_value_weight = vec![0.0; microbatch_rows];
                    if time + 1 < self.config.batch_length {
                        let replay_row = time * rows + first_row;
                        let value_start = replay_row * self.config.value_bins;
                        let value_end = value_start + microbatch_rows * self.config.value_bins;
                        replay_value_target
                            .copy_from_slice(&behavior.replay_value_target[value_start..value_end]);
                        replay_slow_target
                            .copy_from_slice(&behavior.replay_slow_target[value_start..value_end]);
                        // This graph averages across all T states, whereas D3's
                        // replay-value loss contains T-1 targets. Correct that
                        // denominator while keeping the final state's weight 0.
                        let denominator_correction =
                            self.config.batch_length as f32 / (self.config.batch_length - 1) as f32;
                        for (target, source) in replay_value_weight
                            .iter_mut()
                            .zip(&behavior.replay_weight[replay_row..replay_row + microbatch_rows])
                        {
                            *target = *source * denominator_correction;
                        }
                    }
                    self.world_train.set_input(
                        &format!("replay_value_target_{local_time}"),
                        &replay_value_target,
                    );
                    self.world_train.set_input(
                        &format!("replay_slow_target_{local_time}"),
                        &replay_slow_target,
                    );
                    self.world_train.set_input(
                        &format!("replay_value_weight_{local_time}"),
                        &replay_value_weight,
                    );
                }

                let pass = chunk * microbatch_count + microbatch;
                if pass + 1 == pass_count {
                    configure_d3_optimizer(
                        &mut self.world_train,
                        &self.config,
                        self.learner_step,
                        self.config.learning_rate,
                    );
                } else {
                    self.world_train.clear_optimizer();
                }
                self.world_train.step();
                self.world_train.wait();
                metrics.total_loss += read_scalar(&self.world_train, world::LOSS_TOTAL);
                metrics.reconstruction_loss +=
                    read_scalar(&self.world_train, world::LOSS_RECONSTRUCTION);
                metrics.future_prediction_loss +=
                    read_scalar(&self.world_train, world::LOSS_FUTURE_PREDICTION);
                metrics.raw_kl += read_scalar(&self.world_train, world::RAW_KL);
                metrics.dynamics_kl += read_scalar(&self.world_train, world::LOSS_DYNAMICS);
                metrics.representation_kl +=
                    read_scalar(&self.world_train, world::LOSS_REPRESENTATION);
                metrics.reward_loss += read_scalar(&self.world_train, world::LOSS_REWARD);
                metrics.continuation_loss +=
                    read_scalar(&self.world_train, world::LOSS_CONTINUATION);
                metrics.replay_value_loss +=
                    read_scalar(&self.world_train, world::LOSS_REPLAY_VALUE);
            }
        }
        self.world_train.clear_grad_accumulate();
        let scale = 1.0 / pass_count as f32;
        metrics.total_loss *= scale;
        metrics.reconstruction_loss *= scale;
        metrics.future_prediction_loss *= scale;
        metrics.raw_kl *= scale;
        metrics.dynamics_kl *= scale;
        metrics.representation_kl *= scale;
        metrics.reward_loss *= scale;
        metrics.continuation_loss *= scale;
        metrics.replay_value_loss *= scale;
        metrics.replay_reward_prediction_mean = behavior.replay_reward_prediction_mean;
        metrics.replay_reward_target_mean = behavior.replay_reward_target_mean;
        metrics.replay_reward_mae = behavior.replay_reward_mae;
        metrics.rewarded_prediction_mean = behavior.rewarded_prediction_mean;
        metrics.unrewarded_prediction_mean = behavior.unrewarded_prediction_mean;
        metrics.rewarded_count = behavior.rewarded_count;
        metrics.positive_reward_prediction_mean = behavior.positive_reward_prediction_mean;
        metrics.zero_reward_prediction_mean = behavior.zero_reward_prediction_mean;
        metrics.negative_reward_prediction_mean = behavior.negative_reward_prediction_mean;
        metrics.positive_reward_count = behavior.positive_reward_count;
        metrics.zero_reward_count = behavior.zero_reward_count;
        metrics.negative_reward_count = behavior.negative_reward_count;
        metrics.replay_continuation_prediction_mean = behavior.replay_continuation_prediction_mean;
        metrics.replay_continuation_target_mean = behavior.replay_continuation_target_mean;
        metrics.replay_continuation_mae = behavior.replay_continuation_mae;
        assert!(all_finite(&[
            metrics.total_loss,
            metrics.reconstruction_loss,
            metrics.future_prediction_loss,
            metrics.raw_kl,
            metrics.dynamics_kl,
            metrics.representation_kl,
            metrics.reward_loss,
            metrics.continuation_loss,
            metrics.replay_value_loss,
            metrics.replay_reward_prediction_mean,
            metrics.replay_reward_target_mean,
            metrics.replay_reward_mae,
            metrics.rewarded_prediction_mean,
            metrics.unrewarded_prediction_mean,
            metrics.positive_reward_prediction_mean,
            metrics.zero_reward_prediction_mean,
            metrics.negative_reward_prediction_mean,
            metrics.replay_continuation_prediction_mean,
            metrics.replay_continuation_target_mean,
            metrics.replay_continuation_mae,
        ]));
        metrics
    }

    fn sync_world_inference(&mut self) {
        for target in [
            &mut self.world_observe_batch,
            &mut self.world_observe_live,
            &mut self.world_transition,
            &mut self.world_transition_live,
            &mut self.world_heads,
            &mut self.world_heads_live,
        ] {
            sync_matching(&self.world_train, target, "world.");
        }
        if let Some(decoder) = &mut self.world_prediction_live {
            sync_matching(&self.world_train, decoder, "world.");
        }
    }

    fn imagine_and_target(
        &mut self,
        batch: &SequenceBatch,
        posterior: &PosteriorBatch,
        inputs: &mut InputWriteTiming,
    ) -> BehaviorTrainingBatch {
        let size = self.config.network();
        let starts = self.config.batch_size * self.config.batch_length;
        let horizon = self.config.imagination_length;
        let mut deter = flatten_time(&posterior.deter);
        let mut stoch = flatten_time(&posterior.stoch);
        let mut features = Vec::with_capacity(horizon + 1);
        let mut actions = Vec::with_capacity(horizon);
        let mut rewards = Vec::with_capacity(horizon + 1);
        let mut continuations = Vec::with_capacity(horizon + 1);
        let mut values = Vec::with_capacity(horizon + 1);
        let mut slow_values = Vec::with_capacity(horizon + 1);

        for time in 0..=horizon {
            let state_feature = join_features(&deter, &stoch, starts, &self.config);
            inputs.set_input(&mut self.behavior_online, "feature", &state_feature);
            self.behavior_online.step();
            let mut actor_logits = vec![0.0; starts * self.config.action_count];
            let mut value_logits = vec![0.0; starts * self.config.value_bins];

            inputs.set_input(&mut self.behavior_slow, "feature", &state_feature);
            self.behavior_slow.step();
            let mut slow_logits = vec![0.0; starts * self.config.value_bins];

            inputs.set_input(&mut self.world_heads, "deter", &deter);
            inputs.set_input(&mut self.world_heads, "stoch", &stoch);
            self.world_heads.step();
            let mut reward_logits = vec![0.0; starts * self.config.value_bins];
            let mut continuation = vec![0.0; starts];
            self.readback.read_many(&mut [
                (&self.behavior_online, 0, &mut actor_logits),
                (&self.behavior_online, 1, &mut value_logits),
                (&self.behavior_slow, 0, &mut slow_logits),
                (&self.world_heads, 0, &mut reward_logits),
                (&self.world_heads, 1, &mut continuation),
            ]);

            let decoded_reward = decode_rows(&reward_logits, starts, &self.bins);
            let decoded_value = decode_rows(&value_logits, starts, &self.bins);
            let decoded_slow = decode_rows(&slow_logits, starts, &self.bins);
            features.push(state_feature);
            rewards.push(decoded_reward);
            continuations.push(continuation);
            values.push(decoded_value);
            slow_values.push(decoded_slow);

            if time == horizon {
                break;
            }
            let mut action_indices = vec![0; starts];
            let mut action_one_hot = vec![0.0; starts * self.config.action_count];
            let mut probabilities = vec![0.0; self.config.action_count];
            for row in 0..starts {
                let logits = &actor_logits
                    [row * self.config.action_count..(row + 1) * self.config.action_count];
                softmax_unimix(logits, self.config.actor_unimix, &mut probabilities);
                let action = sample_probabilities(&probabilities, &mut self.rngs.imagination);
                action_indices[row] = action;
                action_one_hot[row * self.config.action_count + action] = 1.0;
            }
            actions.push(action_indices);
            inputs.set_input(&mut self.world_transition, "deter", &deter);
            inputs.set_input(&mut self.world_transition, "stoch", &stoch);
            inputs.set_input(&mut self.world_transition, "action", &action_one_hot);
            self.world_transition.step();
            let mut next_deter = vec![0.0; starts * size.deter];
            let mut prior_logits = vec![0.0; starts * size.stoch * size.classes];
            self.readback.read(
                &self.world_transition,
                &mut [(0, &mut next_deter), (1, &mut prior_logits)],
            );
            let next_stoch = sample_latents(
                &prior_logits,
                starts,
                size.stoch,
                size.classes,
                self.config.unimix,
                &mut self.rngs.imagination,
            );
            deter = next_deter;
            stoch = next_stoch;
        }

        let mut returns = (0..horizon).map(|_| vec![0.0; starts]).collect::<Vec<_>>();
        let mut weights = (0..horizon).map(|_| vec![0.0; starts]).collect::<Vec<_>>();
        for start in 0..starts {
            let reward = (0..=horizon)
                .map(|time| rewards[time][start])
                .collect::<Vec<_>>();
            let continuation = (0..=horizon)
                .map(|time| continuations[time][start].clamp(0.0, 1.0))
                .collect::<Vec<_>>();
            let terminal = continuation
                .iter()
                .map(|continuation| 1.0 - continuation)
                .collect::<Vec<_>>();
            let value = (0..=horizon)
                .map(|time| values[time][start])
                .collect::<Vec<_>>();
            let trajectory = lambda_returns(
                &vec![0.0; horizon + 1],
                &terminal,
                &reward,
                &value,
                &value,
                1.0,
                self.config.lambda,
            );
            let trajectory_weights = continuation_weights(&continuation[..horizon]);
            for time in 0..horizon {
                returns[time][start] = trajectory[time];
                weights[time][start] = trajectory_weights[time];
            }
        }
        let all_returns = flatten_time(&returns);
        self.return_normalizer.update(&all_returns);
        let return_scale = self.return_normalizer.scale();

        let imagined_rows = starts * horizon;
        let mut imagined_feature = Vec::with_capacity(imagined_rows * self.config.feature_dim());
        let mut action_target = vec![0.0; imagined_rows * self.config.action_count];
        let mut imagined_weight = vec![0.0; imagined_rows];
        let mut imagined_value_target = vec![0.0; imagined_rows * self.config.value_bins];
        let mut imagined_slow_target = vec![0.0; imagined_rows * self.config.value_bins];
        let mut advantage_abs_sum = 0.0;
        let mut weighted_advantage_abs_sum = 0.0;
        for time in 0..horizon {
            imagined_feature.extend_from_slice(&features[time]);
            for start in 0..starts {
                let row = time * starts + start;
                let advantage = (returns[time][start] - values[time][start]) / return_scale;
                advantage_abs_sum += advantage.abs();
                weighted_advantage_abs_sum += (weights[time][start] * advantage).abs();
                action_target[row * self.config.action_count + actions[time][start]] =
                    weights[time][start] * advantage;
                imagined_weight[row] = weights[time][start];
                self.bins.encode(
                    returns[time][start],
                    &mut imagined_value_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
                self.bins.encode(
                    slow_values[time][start],
                    &mut imagined_slow_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
            }
        }

        let replay_reward_target = flatten_time(&batch.rewards);
        let replay_reward_prediction = &rewards[0];
        let replay_reward_prediction_mean = mean(replay_reward_prediction);
        let replay_reward_target_mean = mean(&replay_reward_target);
        let replay_reward_mae =
            mean_absolute_error(replay_reward_prediction, &replay_reward_target);
        let reward_summary =
            summarize_reward_predictions(replay_reward_prediction, &replay_reward_target);
        let replay_continuation_prediction = &continuations[0];
        let replay_continuation_target = batch
            .flags
            .iter()
            .flatten()
            .map(|flags| {
                if flags.is_terminal {
                    0.0
                } else {
                    self.config.continuation_discount()
                }
            })
            .collect::<Vec<_>>();

        let replay_rows = self.config.batch_size * (self.config.batch_length - 1);
        let mut replay_feature = vec![0.0; replay_rows * self.config.feature_dim()];
        let mut replay_weight = vec![0.0; replay_rows];
        let mut replay_value_target = vec![0.0; replay_rows * self.config.value_bins];
        let mut replay_slow_target = vec![0.0; replay_rows * self.config.value_bins];
        for batch_row in 0..self.config.batch_size {
            let last = (0..self.config.batch_length)
                .map(|time| f32::from(batch.flags[time][batch_row].is_last))
                .collect::<Vec<_>>();
            let terminal = (0..self.config.batch_length)
                .map(|time| f32::from(batch.flags[time][batch_row].is_terminal))
                .collect::<Vec<_>>();
            let reward = (0..self.config.batch_length)
                .map(|time| batch.rewards[time][batch_row])
                .collect::<Vec<_>>();
            let value = (0..self.config.batch_length)
                .map(|time| values[0][time * self.config.batch_size + batch_row])
                .collect::<Vec<_>>();
            let bootstrap = (0..self.config.batch_length)
                .map(|time| returns[0][time * self.config.batch_size + batch_row])
                .collect::<Vec<_>>();
            let replay_return = lambda_returns(
                &last,
                &terminal,
                &reward,
                &value,
                &bootstrap,
                self.config.continuation_discount(),
                self.config.lambda,
            );
            for time in 0..self.config.batch_length - 1 {
                let start = time * self.config.batch_size + batch_row;
                let row = time * self.config.batch_size + batch_row;
                replay_feature
                    [row * self.config.feature_dim()..(row + 1) * self.config.feature_dim()]
                    .copy_from_slice(
                        &features[0][start * self.config.feature_dim()
                            ..(start + 1) * self.config.feature_dim()],
                    );
                replay_weight[row] = 1.0 - last[time];
                self.bins.encode(
                    replay_return[time],
                    &mut replay_value_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
                self.bins.encode(
                    slow_values[0][start],
                    &mut replay_slow_target
                        [row * self.config.value_bins..(row + 1) * self.config.value_bins],
                );
            }
        }

        BehaviorTrainingBatch {
            imagined_feature,
            action_target,
            imagined_weight,
            imagined_value_target,
            imagined_slow_target,
            replay_feature,
            replay_weight,
            replay_value_target,
            replay_slow_target,
            imagined_reward_mean: mean_nested(&rewards[1..]),
            imagined_continuation_mean: mean_nested(&continuations),
            return_mean: mean_nested(&returns),
            return_scale,
            advantage_abs_mean: advantage_abs_sum / imagined_rows as f32,
            weighted_advantage_abs_mean: weighted_advantage_abs_sum / imagined_rows as f32,
            replay_reward_prediction_mean,
            replay_reward_target_mean,
            replay_reward_mae,
            rewarded_prediction_mean: reward_summary.nonzero_prediction_mean,
            unrewarded_prediction_mean: reward_summary.zero_prediction_mean,
            rewarded_count: reward_summary.nonzero_count,
            positive_reward_prediction_mean: reward_summary.positive_prediction_mean,
            zero_reward_prediction_mean: reward_summary.zero_prediction_mean,
            negative_reward_prediction_mean: reward_summary.negative_prediction_mean,
            positive_reward_count: reward_summary.positive_count,
            zero_reward_count: reward_summary.zero_count,
            negative_reward_count: reward_summary.negative_count,
            replay_continuation_prediction_mean: mean(replay_continuation_prediction),
            replay_continuation_target_mean: mean(&replay_continuation_target),
            replay_continuation_mae: mean_absolute_error(
                replay_continuation_prediction,
                &replay_continuation_target,
            ),
        }
    }

    fn train_behavior(&mut self, batch: &BehaviorTrainingBatch) -> BehaviorMetrics {
        let actor_update_scale = self.config.actor_update_scale(self.learner_step);
        self.behavior_train
            .set_input("imagined_feature", &batch.imagined_feature);
        self.behavior_train
            .set_input("action_target", &batch.action_target);
        self.behavior_train
            .set_input("imagined_weight", &batch.imagined_weight);
        self.behavior_train
            .set_input("imagined_value_target", &batch.imagined_value_target);
        self.behavior_train
            .set_input("imagined_slow_target", &batch.imagined_slow_target);
        self.behavior_train
            .set_input("replay_feature", &batch.replay_feature);
        self.behavior_train
            .set_input("replay_weight", &batch.replay_weight);
        self.behavior_train
            .set_input("replay_value_target", &batch.replay_value_target);
        self.behavior_train
            .set_input("replay_slow_target", &batch.replay_slow_target);
        // Freeze actor parameters, not their gradients. Letting LaProp track
        // moments while the actor LR is zero avoids turning the first enabled
        // update into a large bias-correction transient after an old, empty
        // shared actor/critic optimizer clock.
        self.behavior_train
            .set_lr_multiplier("behavior.actor.", actor_update_scale);
        configure_d3_optimizer(
            &mut self.behavior_train,
            &self.config,
            self.learner_step,
            self.config.behavior_learning_rate(),
        );
        self.behavior_train.step();
        self.behavior_train.wait();
        let metrics = BehaviorMetrics {
            total_loss: read_scalar(&self.behavior_train, behavior::LOSS_TOTAL),
            policy_loss: read_scalar(&self.behavior_train, behavior::LOSS_POLICY),
            actor_update_scale,
            value_loss: read_scalar(&self.behavior_train, behavior::LOSS_VALUE),
            replay_value_loss: read_scalar(&self.behavior_train, behavior::LOSS_REPLAY_VALUE),
            policy_entropy: read_scalar(&self.behavior_train, behavior::POLICY_ENTROPY),
            weighted_policy_entropy: read_scalar(
                &self.behavior_train,
                behavior::WEIGHTED_POLICY_ENTROPY,
            ),
            imagined_reward_mean: batch.imagined_reward_mean,
            imagined_continuation_mean: batch.imagined_continuation_mean,
            return_mean: batch.return_mean,
            return_scale: batch.return_scale,
            advantage_abs_mean: batch.advantage_abs_mean,
            weighted_advantage_abs_mean: batch.weighted_advantage_abs_mean,
        };
        assert!(all_finite(&[
            metrics.total_loss,
            metrics.policy_loss,
            metrics.actor_update_scale,
            metrics.value_loss,
            metrics.replay_value_loss,
            metrics.policy_entropy,
            metrics.weighted_policy_entropy,
            metrics.imagined_reward_mean,
            metrics.imagined_continuation_mean,
            metrics.return_mean,
            metrics.return_scale,
            metrics.advantage_abs_mean,
            metrics.weighted_advantage_abs_mean,
        ]));
        metrics
    }

    fn sync_behavior_inference(&mut self) {
        sync_matching(&self.behavior_train, &mut self.behavior_online, "behavior.");
        if let Some(value) = &mut self.behavior_value_live {
            sync_matching(&self.behavior_train, value, "behavior.value.");
        }
        ema_matching(
            &self.behavior_train,
            &mut self.behavior_slow,
            "behavior.value.",
            self.config.slow_value_rate,
        );
        sync_matching(
            &self.behavior_train,
            &mut self.policy_live,
            "behavior.actor.",
        );
        // Keep the world graph's fixed critic copy ready for the next replay
        // update and consistent in checkpoints.
        sync_matching(
            &self.behavior_train,
            &mut self.world_train,
            "behavior.value.",
        );
    }
}

/// Pixel-facing agent combining frozen visual perception and the Dreamer core.
pub struct DreamerAgent {
    perception: Perception,
    core: DreamerCore,
}

impl DreamerAgent {
    pub fn new(
        config: DreamerConfig,
        dino_checkpoint: impl AsRef<Path>,
        dino_plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_perception(
            config,
            PerceptionKind::DinoV3,
            dino_checkpoint,
            dino_plan_cache,
        )
    }

    pub fn with_perception(
        config: DreamerConfig,
        kind: PerceptionKind,
        encoder_checkpoint: impl AsRef<Path>,
        encoder_plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let fingerprint = crate::vision::checkpoint_sha256(encoder_checkpoint.as_ref())?;
        let identity = kind.identity(fingerprint);
        let gpu = Arc::new(crate::init_gpu_context()?);
        let perception = Perception::load(
            kind,
            encoder_checkpoint.as_ref(),
            Arc::clone(&gpu),
            encoder_plan_cache,
        )?;
        let mut core = DreamerCore::with_gpu(config, gpu);
        core.perception_identity = Some(identity);
        Ok(Self { perception, core })
    }

    pub fn restore(
        dreamer_checkpoint: impl AsRef<Path>,
        encoder_checkpoint: impl AsRef<Path>,
        encoder_plan_cache: Option<&Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let metadata = read_checkpoint_metadata(dreamer_checkpoint.as_ref())?;
        let identity = metadata
            .perception
            .as_ref()
            .ok_or("pixel restore requires a recorded perception identity")?;
        identity.verify_file(encoder_checkpoint.as_ref())?;
        let kind = identity.kind;
        let gpu = Arc::new(crate::init_gpu_context()?);
        let core =
            DreamerCore::restore_with_gpu(dreamer_checkpoint.as_ref(), Arc::clone(&gpu), metadata)?;
        let perception =
            Perception::load(kind, encoder_checkpoint.as_ref(), gpu, encoder_plan_cache)?;
        Ok(Self { perception, core })
    }

    pub fn core(&self) -> &DreamerCore {
        &self.core
    }

    pub fn core_mut(&mut self) -> &mut DreamerCore {
        &mut self.core
    }

    /// Current posterior feature for representation diagnostics.
    pub fn latent_feature(&self) -> &[f32] {
        self.core.latent_feature()
    }

    /// Current output of the trainable adapter between perception and the RSSM.
    pub fn encoded_observation(&self) -> &[f32] {
        self.core.encoded_observation()
    }

    /// Current frozen visual observation before the trainable adapter.
    pub fn visual_observation(&self) -> &[f32] {
        self.core.visual_observation()
    }

    /// Current configured observation-head output; see [`DreamerCore::observation_prediction`].
    pub fn observation_prediction(&mut self) -> Vec<f32> {
        self.core.observation_prediction()
    }

    /// Reward predicted from the current posterior state.
    pub fn posterior_reward_prediction(&mut self) -> f32 {
        self.core.posterior_reward_prediction()
    }

    /// Value predicted from the current posterior state.
    pub fn posterior_value_prediction(&mut self) -> f32 {
        self.core.posterior_value_prediction()
    }

    /// One-step prior reward prediction for a proposed categorical action.
    pub fn prior_reward_prediction(&mut self, action: usize) -> f32 {
        self.core.prior_reward_prediction(action)
    }

    /// Open-loop prior reward predictions for a proposed action sequence.
    pub fn prior_reward_rollout(&mut self, actions: &[usize]) -> Vec<f32> {
        self.core.prior_reward_rollout(actions)
    }

    /// Open-loop prior rewards and decoded visual observations.
    pub fn prior_diagnostic_rollout(&mut self, actions: &[usize]) -> (Vec<f32>, Vec<Vec<f32>>) {
        self.core.prior_diagnostic_rollout(actions)
    }

    /// Open-loop reward, continuation, and value predictions.
    pub fn prior_behavior_rollout(&mut self, actions: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.core.prior_behavior_rollout(actions)
    }

    /// Current posterior policy after optional action masking and actor
    /// unimix. This does not sample or alter the live recurrent state.
    pub fn posterior_action_probabilities(&mut self, action_mask: Option<&[bool]>) -> Vec<f32> {
        self.core.posterior_action_probabilities(action_mask)
    }

    pub fn begin_episode(&mut self, frame: &RgbFrame) {
        self.core.check_episode_boundary();
        self.perception.reset();
        let observation =
            self.perception
                .encode_frame_rgb8(frame.pixels(), frame.width(), frame.height());
        self.core.begin_episode(observation);
    }

    pub fn act(&mut self, mode: ActionMode, action_mask: Option<&[bool]>) -> usize {
        self.core.act(mode, action_mask)
    }

    /// Return the unscaled reward channels, including configured novelty.
    pub fn observe(&mut self, transition: &Transition) -> Reward {
        assert!(
            self.core.pending_action.is_some(),
            "act must precede observe"
        );
        let observation = self.perception.encode_frame_rgb8(
            transition.frame.pixels(),
            transition.frame.width(),
            transition.frame.height(),
        );
        self.core
            .observe(observation, transition.reward, transition.flags())
    }

    pub fn learn(&mut self) -> Option<LearnReport> {
        self.core.learn()
    }

    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        self.core.learn_scheduled(maximum_updates)
    }

    pub fn save_checkpoint(&mut self, checkpoint: impl AsRef<Path>) -> io::Result<()> {
        self.core.save_checkpoint(checkpoint)
    }
}

fn read_checkpoint_metadata(checkpoint: &Path) -> io::Result<CheckpointMetadata> {
    let encoded = fs::read(checkpoint.join(CHECKPOINT_METADATA))?;
    let metadata = serde_json::from_slice(&encoded)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    validate_checkpoint_metadata(&metadata)?;
    if let Some(fingerprints) = &metadata.tensor_sha256 {
        fingerprints.verify(checkpoint)?;
    }
    Ok(metadata)
}

fn validate_checkpoint_metadata(metadata: &CheckpointMetadata) -> io::Result<()> {
    if metadata.collection_streams == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint collection stream count must be positive",
        ));
    }
    if let Some(perception) = &metadata.perception {
        perception.validate()?;
    }
    let expected = [
        (
            "architecture",
            CHECKPOINT_ARCHITECTURE,
            metadata.architecture.as_str(),
        ),
        (
            "DreamerV3 revision",
            DREAMERV3_UPSTREAM_REV,
            metadata.dreamerv3_revision.as_str(),
        ),
        (
            "Meganeura revision",
            MEGANEURA_REV,
            metadata.meganeura_revision.as_str(),
        ),
        (
            "Blade revision",
            BLADE_REV,
            metadata.blade_revision.as_str(),
        ),
    ];
    if metadata.format != CHECKPOINT_FORMAT {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported Kindle checkpoint format {}, expected {}",
                metadata.format, CHECKPOINT_FORMAT
            ),
        ));
    }
    for (name, wanted, actual) in expected {
        if actual != wanted {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint {name} is {actual:?}, expected {wanted:?}"),
            ));
        }
    }
    let expected_vision = [
        ("projection seed", PROJECTION_SEED, metadata.projection_seed),
        (
            "observation grid",
            OBSERVATION_GRID as u64,
            metadata.observation_grid as u64,
        ),
        (
            "observation channels",
            OBSERVATION_CHANNELS as u64,
            metadata.observation_channels as u64,
        ),
    ];
    for (name, wanted, actual) in expected_vision {
        if actual != wanted {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint {name} is {actual}, expected {wanted}"),
            ));
        }
    }
    let expected_future_head = (metadata.config.loss_scales.future_prediction > 0.0)
        .then_some(world::FUTURE_HEAD_REVISION);
    if metadata.future_head_revision.as_deref() != expected_future_head {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incompatible future-prediction head; use the original source revision",
        ));
    }
    if metadata.config.visitation_bonus != metadata.visitation.is_some()
        || metadata
            .visitation
            .as_ref()
            .is_some_and(|state| !state.is_valid())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incompatible or missing visitation state",
        ));
    }
    metadata
        .config
        .check()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn keep_masks(
    batch: &SequenceBatch,
    time: usize,
    config: &DreamerConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    keep_masks_range(batch, time, 0, config.batch_size, config)
}

fn keep_masks_range(
    batch: &SequenceBatch,
    time: usize,
    first_row: usize,
    row_count: usize,
    config: &DreamerConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let end_row = first_row
        .checked_add(row_count)
        .expect("keep-mask row range overflowed");
    assert!(end_row <= config.batch_size);
    let size = config.network();
    let mut deter = vec![0.0; row_count * size.deter];
    let mut stoch = vec![0.0; row_count * size.stoch * size.classes];
    let mut action = vec![0.0; row_count * config.action_count];
    for local_row in 0..row_count {
        let keep = batch.keep(time, first_row + local_row);
        deter[local_row * size.deter..(local_row + 1) * size.deter].fill(keep);
        let width = size.stoch * size.classes;
        stoch[local_row * width..(local_row + 1) * width].fill(keep);
        action[local_row * config.action_count..(local_row + 1) * config.action_count].fill(keep);
    }
    (deter, stoch, action)
}

fn row_slice(values: &[f32], first_row: usize, row_count: usize, row_width: usize) -> &[f32] {
    assert!(row_width > 0);
    let start = first_row
        .checked_mul(row_width)
        .expect("row slice start overflowed");
    let end_row = first_row
        .checked_add(row_count)
        .expect("row slice range overflowed");
    let end = end_row
        .checked_mul(row_width)
        .expect("row slice end overflowed");
    &values[start..end]
}

fn world_training_config(config: &DreamerConfig) -> DreamerConfig {
    let mut world = config.clone();
    world.batch_size = config.world_microbatch_size();
    world
}

fn sample_latents(
    logits: &[f32],
    batch: usize,
    stoch: usize,
    classes: usize,
    unimix: f32,
    rng: &mut StdRng,
) -> Vec<f32> {
    assert_eq!(logits.len(), batch * stoch * classes);
    let mut output = vec![0.0; logits.len()];
    if batch * stoch >= 4096 {
        // Draw in the original flat order before distributing independent rows.
        // Neither thread scheduling nor worker count changes the RNG stream.
        let draws: Vec<f32> = (0..batch * stoch).map(|_| rng.random()).collect();
        cpu::parallel(|| {
            output
                .par_chunks_exact_mut(classes)
                .zip(logits.par_chunks_exact(classes))
                .zip(draws.par_iter())
                .for_each_init(
                    || vec![0.0; classes],
                    |probabilities, ((output, logits), &draw)| {
                        softmax_unimix(logits, unimix, probabilities);
                        output[sample_probabilities_at(probabilities, draw)] = 1.0;
                    },
                );
        });
        return output;
    }
    let mut probabilities = vec![0.0; classes];
    for row in 0..batch * stoch {
        let start = row * classes;
        softmax_unimix(&logits[start..start + classes], unimix, &mut probabilities);
        let class = sample_probabilities(&probabilities, rng);
        output[start + class] = 1.0;
    }
    output
}

fn join_features(deter: &[f32], stoch: &[f32], batch: usize, config: &DreamerConfig) -> Vec<f32> {
    let size = config.network();
    assert_eq!(deter.len(), batch * size.deter);
    let stoch_width = size.stoch * size.classes;
    assert_eq!(stoch.len(), batch * stoch_width);
    let mut output = vec![0.0; batch * config.feature_dim()];
    for row in 0..batch {
        let target = &mut output[row * config.feature_dim()..(row + 1) * config.feature_dim()];
        target[..size.deter].copy_from_slice(&deter[row * size.deter..(row + 1) * size.deter]);
        target[size.deter..].copy_from_slice(&stoch[row * stoch_width..(row + 1) * stoch_width]);
    }
    output
}

fn flatten_time(values: &[Vec<f32>]) -> Vec<f32> {
    let capacity = values.iter().map(Vec::len).sum();
    let mut output = Vec::with_capacity(capacity);
    for value in values {
        output.extend_from_slice(value);
    }
    output
}

fn decode_rows(logits: &[f32], rows: usize, bins: &TwoHotBins) -> Vec<f32> {
    assert_eq!(logits.len(), rows * bins.len());
    if rows >= 256 {
        return cpu::parallel(|| {
            logits
                .par_chunks_exact(bins.len())
                .map_init(
                    || vec![0.0; bins.len()],
                    |scratch, row| bins.decode_logits_with_scratch(row, scratch),
                )
                .collect()
        });
    }
    let mut scratch = vec![0.0; bins.len()];
    (0..rows)
        .map(|row| {
            bins.decode_logits_with_scratch(
                &logits[row * bins.len()..(row + 1) * bins.len()],
                &mut scratch,
            )
        })
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    assert!(!values.is_empty());
    let mut best = 0;
    for index in 1..values.len() {
        if values[index].total_cmp(&values[best]).is_gt() {
            best = index;
        }
    }
    best
}

fn read_scalar(session: &Session, index: usize) -> f32 {
    let mut output = [0.0];
    session.read_output_by_index(index, &mut output);
    output[0]
}

fn trainable_parameter_count(session: &Session) -> usize {
    session
        .param_names()
        .into_iter()
        .filter(|name| session.has_param_grad(name))
        .map(|name| session.param_size(name).expect("known parameter size"))
        .sum()
}

fn mean_nested(values: &[Vec<f32>]) -> f32 {
    let count = values.iter().map(Vec::len).sum::<usize>();
    if count == 0 {
        0.0
    } else {
        values.iter().flatten().sum::<f32>() / count as f32
    }
}

fn mean(values: &[f32]) -> f32 {
    assert!(!values.is_empty());
    values.iter().sum::<f32>() / values.len() as f32
}

fn mean_absolute_error(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len());
    assert!(!left.is_empty());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / left.len() as f32
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RewardPredictionSummary {
    nonzero_prediction_mean: f32,
    zero_prediction_mean: f32,
    positive_prediction_mean: f32,
    negative_prediction_mean: f32,
    nonzero_count: usize,
    zero_count: usize,
    positive_count: usize,
    negative_count: usize,
}

fn summarize_reward_predictions(predictions: &[f32], targets: &[f32]) -> RewardPredictionSummary {
    assert_eq!(predictions.len(), targets.len());
    let mut nonzero_sum = 0.0;
    let mut zero_sum = 0.0;
    let mut positive_sum = 0.0;
    let mut negative_sum = 0.0;
    let mut summary = RewardPredictionSummary::default();
    for (&prediction, &target) in predictions.iter().zip(targets) {
        if target > 0.0 {
            positive_sum += prediction;
            summary.positive_count += 1;
            nonzero_sum += prediction;
            summary.nonzero_count += 1;
        } else if target < 0.0 {
            negative_sum += prediction;
            summary.negative_count += 1;
            nonzero_sum += prediction;
            summary.nonzero_count += 1;
        } else {
            zero_sum += prediction;
            summary.zero_count += 1;
        }
    }
    summary.nonzero_prediction_mean = nonzero_sum / summary.nonzero_count.max(1) as f32;
    summary.zero_prediction_mean = zero_sum / summary.zero_count.max(1) as f32;
    summary.positive_prediction_mean = positive_sum / summary.positive_count.max(1) as f32;
    summary.negative_prediction_mean = negative_sum / summary.negative_count.max(1) as f32;
    summary
}

fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn valid_checkpoint_metadata() -> CheckpointMetadata {
        CheckpointMetadata {
            format: CHECKPOINT_FORMAT,
            architecture: CHECKPOINT_ARCHITECTURE.to_owned(),
            dreamerv3_revision: DREAMERV3_UPSTREAM_REV.to_owned(),
            meganeura_revision: MEGANEURA_REV.to_owned(),
            blade_revision: BLADE_REV.to_owned(),
            projection_seed: PROJECTION_SEED,
            observation_grid: OBSERVATION_GRID,
            observation_channels: OBSERVATION_CHANNELS,
            config: DreamerConfig::tiny(3),
            learner_step: 1,
            environment_step: 8,
            collection_streams: 1,
            return_low: 0.0,
            return_high: 1.0,
            visitation: None,
            future_head_revision: None,
            perception: Some(
                PerceptionKind::DinoV3.identity(crate::vision::VITS16_CHECKPOINT_SHA256.to_owned()),
            ),
            tensor_sha256: None,
        }
    }

    #[test]
    fn checkpoint_metadata_rejects_backend_revision_mismatch() {
        validate_checkpoint_metadata(&valid_checkpoint_metadata()).unwrap();

        let mut old_meganeura = valid_checkpoint_metadata();
        old_meganeura.meganeura_revision = "d904e12e52af6910b041873cd203a5d5e5fd3b3c".to_owned();
        let error = validate_checkpoint_metadata(&old_meganeura).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checkpoint Meganeura revision"));

        let mut wrong_blade = valid_checkpoint_metadata();
        wrong_blade.blade_revision = "wrong-revision".to_owned();
        let error = validate_checkpoint_metadata(&wrong_blade).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checkpoint Blade revision"));
    }

    #[test]
    fn perception_identity_rejects_changed_kind_semantics_and_malformed_hashes() {
        let mut metadata = valid_checkpoint_metadata();
        for malformed in ["", "xyz", &"A".repeat(64)] {
            metadata.perception.as_mut().unwrap().checkpoint_sha256 = malformed.to_owned();
            assert!(validate_checkpoint_metadata(&metadata).is_err());
        }
        metadata.perception = Some(
            PerceptionKind::LeVJepa.identity(crate::vision::levjepa::CHECKPOINT_SHA256.to_owned()),
        );
        validate_checkpoint_metadata(&metadata).unwrap();
        metadata.perception.as_mut().unwrap().kind = PerceptionKind::DinoV3;
        assert!(validate_checkpoint_metadata(&metadata).is_err());
        metadata.perception = Some(
            PerceptionKind::LeVJepa.identity(crate::vision::levjepa::CHECKPOINT_SHA256.to_owned()),
        );
        metadata.perception.as_mut().unwrap().encoding_revision = "full-attention".to_owned();
        assert!(validate_checkpoint_metadata(&metadata).is_err());
        metadata.perception = None;
        validate_checkpoint_metadata(&metadata).unwrap(); // feature-only core
        metadata.format = 2;
        assert!(validate_checkpoint_metadata(&metadata).is_err());
    }

    #[test]
    fn pixel_restore_rejects_wrong_weights_before_gpu_construction() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "kindle-fingerprint-{}-{unique}",
            std::process::id(),
        ));
        fs::create_dir(&directory).unwrap();
        fs::write(
            directory.join(CHECKPOINT_METADATA),
            serde_json::to_vec(&valid_checkpoint_metadata()).unwrap(),
        )
        .unwrap();
        let weights = directory.join("wrong-weights.safetensors");
        fs::write(&weights, b"not the pinned encoder").unwrap();
        let error = DreamerAgent::restore(&directory, weights, None)
            .err()
            .expect("different encoder must be rejected");
        fs::remove_dir_all(directory).unwrap();
        assert!(error.to_string().contains("perception checkpoint SHA-256"));
    }

    #[test]
    fn checkpoint_restore_rejects_mixed_tensor_files_before_gpu_construction() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "kindle-checkpoint-integrity-{}-{unique}",
            std::process::id(),
        ));
        fs::create_dir(&directory).unwrap();
        let files = [CHECKPOINT_WORLD, CHECKPOINT_BEHAVIOR, CHECKPOINT_SLOW_VALUE];
        for name in files {
            fs::write(directory.join(name), b"original tensor file").unwrap();
        }
        let mut metadata = valid_checkpoint_metadata();
        metadata.tensor_sha256 = Some(TensorFingerprints::read(&directory).unwrap());
        fs::write(
            directory.join(CHECKPOINT_METADATA),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        read_checkpoint_metadata(&directory).unwrap();
        for name in files {
            fs::write(directory.join(name), b"different learner generation").unwrap();
            let error = DreamerCore::restore(&directory)
                .err()
                .expect("a mixed save must be rejected");
            assert!(error.to_string().contains(name));
            assert!(error.to_string().contains("SHA-256 mismatch"));
            fs::write(directory.join(name), b"original tensor file").unwrap();
        }
        fs::remove_file(directory.join(CHECKPOINT_WORLD)).unwrap();
        assert!(read_checkpoint_metadata(&directory).is_err());
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn historical_metadata_may_lack_tensor_fingerprints() {
        let mut encoded = serde_json::to_value(valid_checkpoint_metadata()).unwrap();
        encoded.as_object_mut().unwrap().remove("tensor_sha256");
        let metadata: CheckpointMetadata = serde_json::from_value(encoded).unwrap();
        assert!(metadata.tensor_sha256.is_none());
        validate_checkpoint_metadata(&metadata).unwrap();
    }

    #[test]
    fn checkpoint_metadata_requires_matching_visitation_state() {
        let mut metadata = valid_checkpoint_metadata();
        metadata.config.visitation_bonus = true;
        assert!(validate_checkpoint_metadata(&metadata).is_err());
        metadata.visitation = Some(VisitationBonus::new().state());
        validate_checkpoint_metadata(&metadata).unwrap();
        metadata.config.visitation_bonus = false;
        assert!(validate_checkpoint_metadata(&metadata).is_err());
    }

    #[test]
    fn checkpoint_metadata_rejects_the_retired_global_future_head() {
        let mut metadata = valid_checkpoint_metadata();
        metadata.config.loss_scales.future_prediction = 0.25;
        assert!(validate_checkpoint_metadata(&metadata).is_err());
        metadata.future_head_revision = Some(world::FUTURE_HEAD_REVISION.to_owned());
        validate_checkpoint_metadata(&metadata).unwrap();
        metadata.config.loss_scales.future_prediction = 0.0;
        assert!(validate_checkpoint_metadata(&metadata).is_err());
    }

    #[test]
    fn categorical_latents_are_one_hot_per_variable() {
        let mut rng = StdRng::seed_from_u64(3);
        let sampled = sample_latents(&[0.0; 2 * 3 * 4], 2, 3, 4, 0.01, &mut rng);
        for row in sampled.as_chunks::<4>().0 {
            assert_eq!(row.iter().sum::<f32>(), 1.0);
            assert_eq!(row.iter().filter(|value| **value == 1.0).count(), 1);
        }
    }

    #[test]
    fn parallel_categorical_rows_preserve_samples_and_rng_position() {
        let (batch, stoch, classes) = (256, 32, 32);
        let logits: Vec<_> = (0..batch * stoch * classes)
            .map(|i| ((i * 137) % 1019) as f32 / 50.0 - 10.0)
            .collect();
        let mut serial_rng = StdRng::seed_from_u64(141);
        let mut parallel_rng = serial_rng.clone();
        let mut expected = vec![0.0; logits.len()];
        let mut probabilities = vec![0.0; classes];
        for (row, output) in logits
            .chunks_exact(classes)
            .zip(expected.chunks_exact_mut(classes))
        {
            softmax_unimix(row, 0.01, &mut probabilities);
            output[sample_probabilities(&probabilities, &mut serial_rng)] = 1.0;
        }
        let actual = sample_latents(&logits, batch, stoch, classes, 0.01, &mut parallel_rng);
        assert_eq!(actual, expected);
        assert_eq!(serial_rng.random::<u64>(), parallel_rng.random::<u64>());
    }

    #[test]
    fn parallel_scalar_decoding_preserves_row_order_and_bits() {
        let bins = TwoHotBins::new(255);
        let rows = 512;
        let logits: Vec<_> = (0..rows * bins.len())
            .map(|i| ((i * 137) % 1019) as f32 / 50.0 - 10.0)
            .collect();
        let mut scratch = vec![0.0; bins.len()];
        let expected: Vec<_> = logits
            .chunks_exact(bins.len())
            .map(|row| bins.decode_logits_with_scratch(row, &mut scratch))
            .collect();
        assert_eq!(decode_rows(&logits, rows, &bins), expected);
    }

    #[test]
    fn reward_prediction_summary_separates_reward_signs() {
        let summary =
            summarize_reward_predictions(&[-1.0, 0.0, 1.0, 3.0, 7.0], &[-1.0, 0.0, 1.0, 2.0, 0.0]);
        assert_eq!(summary.nonzero_prediction_mean, 1.0);
        assert_eq!(summary.zero_prediction_mean, 3.5);
        assert_eq!(summary.positive_prediction_mean, 2.0);
        assert_eq!(summary.negative_prediction_mean, -1.0);
        assert_eq!(summary.nonzero_count, 3);
        assert_eq!(summary.zero_count, 2);
        assert_eq!(summary.positive_count, 2);
        assert_eq!(summary.negative_count, 1);
    }

    #[test]
    fn d3_scheduler_discards_prefill_credit() {
        let mut scheduler = D3TrainScheduler::default();
        for _ in 0..100 {
            scheduler.observe(0.25);
        }
        assert!(!scheduler.update_due(false));
        assert!(scheduler.update_due(true));
        scheduler.consume_update();
        assert!(!scheduler.update_due(true));

        for _ in 0..3 {
            scheduler.observe(0.25);
            assert!(!scheduler.update_due(true));
        }
        scheduler.observe(0.25);
        assert!(scheduler.update_due(true));
    }

    #[test]
    fn dreamer_rng_streams_advance_independently() {
        let mut advanced = DreamerRngs::new(7);
        let mut baseline = DreamerRngs::new(7);
        for _ in 0..100 {
            let _: u64 = advanced.imagination.random();
            let _: u64 = advanced.replay.random();
            let _: u64 = advanced.train_posterior.random();
        }
        assert_eq!(
            advanced.policy.random::<u64>(),
            baseline.policy.random::<u64>()
        );
        assert_eq!(
            advanced.live_posterior.random::<u64>(),
            baseline.live_posterior.random::<u64>()
        );
        assert_ne!(
            advanced.imagination.random::<u64>(),
            baseline.imagination.random::<u64>()
        );

        let mut left = DreamerRngs::diagnostic(7, 11, 13);
        let mut right = DreamerRngs::diagnostic(7, 11, 13);
        let mut next_step = DreamerRngs::diagnostic(7, 11, 14);
        assert_eq!(left.random::<u64>(), right.random::<u64>());
        assert_ne!(left.random::<u64>(), next_step.random::<u64>());
    }

    #[test]
    fn dreamer_rng_streams_change_with_experiment_seed() {
        let mut seed_zero = DreamerRngs::new(0);
        let mut seed_one = DreamerRngs::new(1);
        assert_ne!(
            seed_zero.policy.random::<u64>(),
            seed_one.policy.random::<u64>()
        );
        assert_ne!(
            seed_zero.live_posterior.random::<u64>(),
            seed_one.live_posterior.random::<u64>()
        );
        assert_ne!(
            seed_zero.replay.random::<u64>(),
            seed_one.replay.random::<u64>()
        );
        assert_ne!(
            seed_zero.train_posterior.random::<u64>(),
            seed_one.train_posterior.random::<u64>()
        );
        assert_ne!(
            seed_zero.imagination.random::<u64>(),
            seed_one.imagination.random::<u64>()
        );
    }

    #[test]
    fn state_features_concatenate_deter_and_flat_categoricals() {
        let config = DreamerConfig::tiny(3);
        let size = config.network();
        let deter = vec![2.0; 2 * size.deter];
        let stoch = vec![1.0; 2 * size.stoch * size.classes];
        let feature = join_features(&deter, &stoch, 2, &config);
        assert_eq!(feature.len(), 2 * config.feature_dim());
        assert!(feature[..size.deter].iter().all(|value| *value == 2.0));
        assert!(
            feature[size.deter..config.feature_dim()]
                .iter()
                .all(|value| *value == 1.0)
        );
    }

    #[test]
    fn world_microbatch_changes_only_the_training_graph_batch() {
        let mut config = DreamerConfig::tiny(3);
        config.world_microbatch_size = Some(1);
        let world = world_training_config(&config);
        assert_eq!(config.batch_size, 2);
        assert_eq!(world.batch_size, 1);
        assert_eq!(world.batch_length, config.batch_length);
        assert_eq!(world.world_backprop_length, config.world_backprop_length);
        assert_eq!(world.network(), config.network());
    }

    #[test]
    fn row_slice_selects_complete_contiguous_rows() {
        let values = (0..15).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(row_slice(&values, 1, 3, 3), &values[3..12]);
        assert!(row_slice(&values, 5, 0, 3).is_empty());
    }

    #[test]
    #[ignore = "builds and runs all Dreamer GPU sessions"]
    fn tiny_world_microbatch_matches_the_effective_batch_update() {
        check_world_microbatch(0.0);
    }

    #[test]
    #[ignore = "builds and runs all Dreamer GPU sessions"]
    fn tiny_predictive_microbatch_matches_the_effective_batch_update() {
        check_world_microbatch(0.25);
    }

    fn check_world_microbatch(future_prediction_scale: f32) {
        struct ParameterSnapshot {
            name: String,
            initial: Vec<f32>,
            updated: Vec<f32>,
            second_moment: Vec<f32>,
        }

        #[derive(Debug, Default)]
        struct VectorComparison {
            difference_square_sum: f64,
            left_square_sum: f64,
            right_square_sum: f64,
            dot_product: f64,
        }

        impl VectorComparison {
            fn add(&mut self, left: f32, right: f32) {
                self.difference_square_sum += f64::from(left - right).powi(2);
                self.left_square_sum += f64::from(left).powi(2);
                self.right_square_sum += f64::from(right).powi(2);
                self.dot_product += f64::from(left) * f64::from(right);
            }

            fn relative_difference(&self) -> f64 {
                self.difference_square_sum.sqrt()
                    / self
                        .left_square_sum
                        .sqrt()
                        .max(self.right_square_sum.sqrt())
            }

            fn cosine(&self) -> f64 {
                self.dot_product / (self.left_square_sum * self.right_square_sum).sqrt()
            }
        }

        fn run(
            gpu: Arc<blade_graphics::Context>,
            world_microbatch_size: Option<usize>,
            future_prediction_scale: f32,
        ) -> (WorldMetrics, Vec<ParameterSnapshot>) {
            let mut config = DreamerConfig::tiny(3);
            config.batch_size = 4;
            config.batch_length = 4;
            config.world_backprop_length = 4;
            config.world_microbatch_size = world_microbatch_size;
            config.loss_scales.future_prediction = future_prediction_scale;
            config.learning_rate_warmup = 0;
            config.skip_full_optimize = true;
            let mut agent = DreamerCore::with_gpu(config, gpu);
            let names = agent
                .world_train
                .param_names()
                .into_iter()
                .filter(|name| name.starts_with("world.") && agent.world_train.has_param_grad(name))
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let initial = names
                .iter()
                .map(|name| {
                    let mut values = vec![0.0; agent.world_train.param_size(name).unwrap()];
                    agent.world_train.read_param(name, &mut values);
                    values
                })
                .collect::<Vec<_>>();
            let observation =
                |step: usize| Observation::from_vec(vec![step as f32 / 10.0; Observation::LEN]);

            agent.begin_episode(observation(0));
            for step in 1..=16 {
                let action = agent.act(ActionMode::Greedy, None);
                assert!(action < 3);
                agent.observe(
                    observation(step),
                    Reward {
                        extrinsic: if step.is_multiple_of(3) { 1.0 } else { 0.0 },
                        intrinsic: 0.0,
                    },
                    FrameFlags::default(),
                );
            }
            let report = agent
                .learn()
                .expect("seventeen frames fill four tiny sequences");
            let parameters = names
                .into_iter()
                .zip(initial)
                .map(|(name, initial)| {
                    let size = agent.world_train.param_size(&name).unwrap();
                    let mut updated = vec![0.0; size];
                    let mut second_moment = vec![0.0; size];
                    agent.world_train.read_param(&name, &mut updated);
                    agent.world_train.read_adam_v(&name, &mut second_moment);
                    ParameterSnapshot {
                        name,
                        initial,
                        updated,
                        second_moment,
                    }
                })
                .collect();
            (report.world, parameters)
        }

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let (full_metrics, full_parameters) = run(Arc::clone(&gpu), None, future_prediction_scale);
        let (micro_metrics, micro_parameters) = run(gpu, Some(1), future_prediction_scale);
        let metric_pairs = [
            (full_metrics.total_loss, micro_metrics.total_loss),
            (
                full_metrics.future_prediction_loss,
                micro_metrics.future_prediction_loss,
            ),
            (
                full_metrics.reconstruction_loss,
                micro_metrics.reconstruction_loss,
            ),
            (full_metrics.raw_kl, micro_metrics.raw_kl),
            (full_metrics.reward_loss, micro_metrics.reward_loss),
            (
                full_metrics.continuation_loss,
                micro_metrics.continuation_loss,
            ),
            (
                full_metrics.replay_value_loss,
                micro_metrics.replay_value_loss,
            ),
        ];
        for (full, micro) in metric_pairs {
            let tolerance = 1e-4 * full.abs().max(1.0);
            assert!(
                (full - micro).abs() <= tolerance,
                "full-batch metric {full} != microbatch metric {micro}"
            );
        }
        assert!(!full_parameters.is_empty());
        assert_eq!(full_parameters.len(), micro_parameters.len());
        let mut maximum_parameter_difference = (0.0f32, String::new(), 0usize, 0.0f32, 0.0f32);
        let mut update_comparison = VectorComparison::default();
        let mut second_moment_comparison = VectorComparison::default();
        for (full, micro) in full_parameters.iter().zip(&micro_parameters) {
            assert_eq!(full.name, micro.name);
            assert_eq!(full.initial, micro.initial);
            assert_eq!(full.updated.len(), micro.updated.len());
            assert_eq!(full.second_moment.len(), micro.second_moment.len());
            for (index, (&full_value, &micro_value)) in
                full.updated.iter().zip(&micro.updated).enumerate()
            {
                let difference = (full_value - micro_value).abs();
                if difference > maximum_parameter_difference.0 {
                    maximum_parameter_difference = (
                        difference,
                        full.name.clone(),
                        index,
                        full_value,
                        micro_value,
                    );
                }
                update_comparison.add(
                    full_value - full.initial[index],
                    micro_value - micro.initial[index],
                );
            }
            for (&full_value, &micro_value) in full.second_moment.iter().zip(&micro.second_moment) {
                second_moment_comparison.add(full_value, micro_value);
            }
        }

        // Equal row partitions reorder f32 reductions. At coordinates where
        // gradients nearly cancel, first-step LaProp can therefore turn an
        // infinitesimal sign change into opposite one-learning-rate updates.
        // Bound that pointwise effect while requiring the complete update and
        // RMS state to remain aligned.
        let learning_rate = f64::from(DreamerConfig::tiny(3).learning_rate);
        assert!(
            f64::from(maximum_parameter_difference.0) <= 2.1 * learning_rate,
            "microbatch update exceeded the reduction-order bound: {maximum_parameter_difference:?}"
        );
        assert!(
            update_comparison.relative_difference() < 0.025 && update_comparison.cosine() > 0.999,
            "microbatch changed the aggregate world update: {update_comparison:?}"
        );
        assert!(
            second_moment_comparison.relative_difference() < 0.001
                && second_moment_comparison.cosine() > 0.99999,
            "microbatch changed LaProp's RMS state: {second_moment_comparison:?}"
        );
    }

    #[test]
    fn input_write_timing_serializes_only_numeric_fields() {
        let value = serde_json::to_value(InputWriteTiming::default()).unwrap();
        let fields = value.as_object().unwrap();
        assert_eq!(fields.len(), 3);
        assert!(fields.values().all(|value| value.as_f64() == Some(0.0)));
    }

    fn assert_learner_transfer_ledger(report: &LearnReport, config: &DreamerConfig) {
        let size = config.network();
        let starts = config.batch_size * config.batch_length;
        let state_width = size.deter + size.stoch * size.classes;
        let head_width = config.action_count + 3 * config.value_bins + 1;
        let horizon = config.imagination_length;
        assert_eq!(
            report.timing.posterior_inputs.calls,
            7 * config.batch_length
        );
        assert_eq!(
            report.timing.posterior_inputs.bytes,
            starts
                * (2 * (state_width + config.action_count) + config.observation_dim())
                * std::mem::size_of::<f32>()
        );
        assert_eq!(
            report.timing.imagination_inputs.calls,
            4 * (horizon + 1) + 3 * horizon
        );
        assert_eq!(
            report.timing.imagination_inputs.bytes,
            starts
                * (3 * (horizon + 1) * state_width + horizon * (state_width + config.action_count))
                * std::mem::size_of::<f32>()
        );
        assert_eq!(report.timing.posterior_readback.calls, config.batch_length);
        assert_eq!(
            report.timing.posterior_readback.bytes,
            starts * state_width * std::mem::size_of::<f32>()
        );
        assert_eq!(report.timing.imagination_readback.calls, 2 * horizon + 1);
        assert_eq!(
            report.timing.imagination_readback.bytes,
            starts
                * ((horizon + 1) * head_width + horizon * state_width)
                * std::mem::size_of::<f32>()
        );
        for (readback, inputs, parent) in [
            (
                &report.timing.posterior_readback,
                &report.timing.posterior_inputs,
                report.timing.posterior_seconds,
            ),
            (
                &report.timing.imagination_readback,
                &report.timing.imagination_inputs,
                report.timing.imagination_seconds,
            ),
        ] {
            let phases = [
                readback.prepare_seconds,
                readback.submit_seconds,
                readback.wait_seconds,
                readback.copy_seconds,
                inputs.seconds,
            ];
            assert!(
                phases
                    .iter()
                    .all(|value| value.is_finite() && *value >= 0.0)
            );
            assert!(phases.iter().sum::<f64>() <= parent);
        }
    }

    #[test]
    #[ignore = "builds and runs all Dreamer GPU sessions"]
    fn tiny_agent_completes_an_act_and_learn_cycle() {
        let mut config = DreamerConfig::tiny(3);
        config.batch_length = 8;
        config.world_backprop_length = 4;
        config.dynamics_free_nats = Some(0.0);
        config.actor_learning_starts = 1;
        config.visitation_bonus = true;
        config.train_ratio = 0.0;
        config.skip_full_optimize = true;
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut agent = DreamerCore::with_gpu(config, Arc::clone(&gpu));
        let (world_parameters, behavior_parameters) = agent.trainable_parameter_counts();
        assert!(world_parameters > 0 && behavior_parameters > 0);
        assert!(
            !agent
                .world_train
                .has_param_grad("behavior.value.layer0.weight"),
            "the world optimizer must not own critic parameters"
        );
        assert!(
            agent
                .world_train
                .has_param_grad("world.dynamics.core.dynin0.weight"),
            "replay value must retain a gradient path into the world model"
        );
        let observation = || Observation::from_vec(vec![0.0; Observation::LEN]);

        agent.begin_episode(observation());
        assert_eq!(agent.visual_observation(), vec![0.0; Observation::LEN]);
        assert!(agent.posterior_reward_prediction().is_finite());
        assert!(agent.posterior_value_prediction().is_finite());
        let posterior_observation = agent.observation_prediction();
        assert_eq!(posterior_observation.len(), Observation::LEN);
        assert!(posterior_observation.iter().all(|value| value.is_finite()));
        let (prior_rewards, prior_observations) = agent.prior_diagnostic_rollout(&[0, 1]);
        assert_eq!(prior_rewards.len(), 2);
        assert_eq!(prior_observations.len(), 2);
        assert!(prior_rewards.iter().all(|value| value.is_finite()));
        assert!(prior_observations.iter().all(|observation| {
            observation.len() == Observation::LEN
                && observation.iter().all(|value| value.is_finite())
        }));
        let (rewards, continuations, values) = agent.prior_behavior_rollout(&[0, 1]);
        assert_eq!(
            (rewards.len(), continuations.len(), values.len()),
            (2, 2, 2)
        );
        assert!(
            rewards
                .iter()
                .chain(&continuations)
                .chain(&values)
                .all(|value| value.is_finite())
        );
        let probabilities = agent.posterior_action_probabilities(None);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let masked = agent.posterior_action_probabilities(Some(&[true, false, true]));
        assert_eq!(masked[1], 0.0);
        assert!((masked.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for step in 0..8 {
            let action = agent.act(ActionMode::Greedy, None);
            assert!(action < 3);
            let reward = agent.observe(
                observation(),
                Reward {
                    extrinsic: step as f32,
                    intrinsic: 0.0,
                },
                FrameFlags {
                    is_last: step == 7,
                    is_terminal: step == 7,
                    ..FrameFlags::default()
                },
            );
            assert_eq!(reward.extrinsic, step as f32);
            assert!((reward.intrinsic - 1.0 / ((step + 2) as f32).sqrt()).abs() < 1e-7);
        }

        assert!(
            agent.learn_scheduled(3).is_empty(),
            "zero ratio disables even the initial scheduled update"
        );
        let mut initial_actor_parameter = vec![0.0; 3];
        agent
            .behavior_train
            .read_param("behavior.actor.out.bias", &mut initial_actor_parameter);
        let report = agent.learn().expect("nine frames fill one tiny sequence");
        assert_learner_transfer_ledger(&report, agent.config());
        assert_eq!(report.learner_step, 1);
        assert_eq!(report.replay_len, 9);
        assert!(report.world.total_loss.is_finite());
        assert!(report.world.raw_kl >= 0.0);
        assert!((report.world.dynamics_kl - report.world.raw_kl).abs() < 1e-6);
        assert!(report.world.representation_kl >= agent.config.free_nats);
        assert!(report.behavior.total_loss.is_finite());
        assert_eq!(report.behavior.actor_update_scale, 0.0);
        assert!(
            (report.world.reward_loss - (agent.config.value_bins as f32).ln()).abs() < 1e-3,
            "uniform initial reward logits should report full cross-entropy"
        );
        assert!(
            (report.behavior.policy_entropy - (agent.config.action_count as f32).ln()).abs() < 0.02,
            "initial raw actor entropy should be close to uniform"
        );
        assert!(report.behavior.weighted_policy_entropy <= report.behavior.policy_entropy);

        let mut world_parameter = vec![0.0; config_value_bins(&agent)];
        let mut world_momentum = vec![0.0; config_value_bins(&agent)];
        let mut behavior_parameter = vec![0.0; 3];
        let mut behavior_momentum = vec![0.0; 3];
        let mut slow_parameter = vec![0.0; config_value_bins(&agent)];
        agent
            .world_train
            .read_param("world.reward.out.bias", &mut world_parameter);
        agent
            .world_train
            .read_adam_m("world.reward.out.bias", &mut world_momentum);
        agent
            .behavior_train
            .read_param("behavior.actor.out.bias", &mut behavior_parameter);
        agent
            .behavior_train
            .read_adam_m("behavior.actor.out.bias", &mut behavior_momentum);
        assert_eq!(behavior_parameter, initial_actor_parameter);
        assert!(
            behavior_momentum.iter().any(|value| *value != 0.0),
            "the frozen actor should warm its optimizer moments"
        );
        agent
            .behavior_slow
            .read_param("behavior.value.out.bias", &mut slow_parameter);
        let return_state = agent.return_normalizer.state();
        let visitation_state =
            serde_json::to_vec(&agent.visitation.as_ref().unwrap().state()).unwrap();

        let checkpoint = std::env::temp_dir().join(format!(
            "kindle-dreamer-checkpoint-test-{}",
            std::process::id()
        ));
        if checkpoint.exists() {
            fs::remove_dir_all(&checkpoint).unwrap();
        }
        agent.save_checkpoint(&checkpoint).unwrap();
        drop(agent);

        let metadata = read_checkpoint_metadata(&checkpoint).unwrap();
        let mut restored = DreamerCore::restore_with_gpu(&checkpoint, gpu, metadata).unwrap();
        assert_eq!(restored.learner_step(), 1);
        assert_eq!(restored.environment_step(), 8);
        assert_eq!(restored.replay_len(), 0);
        let mut restored_world = vec![0.0; config_value_bins(&restored)];
        let mut restored_momentum = vec![0.0; config_value_bins(&restored)];
        let mut restored_behavior = vec![0.0; 3];
        let mut restored_slow = vec![0.0; config_value_bins(&restored)];
        restored
            .world_train
            .read_param("world.reward.out.bias", &mut restored_world);
        restored
            .world_train
            .read_adam_m("world.reward.out.bias", &mut restored_momentum);
        restored
            .behavior_train
            .read_param("behavior.actor.out.bias", &mut restored_behavior);
        restored
            .behavior_slow
            .read_param("behavior.value.out.bias", &mut restored_slow);
        assert_eq!(restored_world, world_parameter);
        assert_eq!(restored_momentum, world_momentum);
        assert_eq!(restored_behavior, behavior_parameter);
        assert_eq!(restored_slow, slow_parameter);
        assert_eq!(restored.return_normalizer.state(), return_state);
        assert_eq!(
            serde_json::to_vec(&restored.visitation.as_ref().unwrap().state()).unwrap(),
            visitation_state
        );

        restored.begin_episode(observation());
        for step in 0..8 {
            restored.act(ActionMode::Greedy, None);
            restored.observe(
                observation(),
                Reward::default(),
                FrameFlags {
                    is_last: step == 7,
                    is_terminal: step == 7,
                    ..FrameFlags::default()
                },
            );
        }
        let report = restored.learn().expect("refilled replay learns");
        assert_learner_transfer_ledger(&report, restored.config());
        assert_eq!(report.learner_step, 2);
        assert_eq!(report.behavior.actor_update_scale, 1.0);
        drop(restored);
        fs::remove_dir_all(checkpoint).unwrap();
    }

    #[test]
    #[ignore = "builds and runs all Dreamer GPU sessions"]
    fn tiny_future_prediction_is_causal_and_trains_through_previous_observations() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales = super::super::config::LossScales {
            reconstruction: 0.0,
            future_prediction: 0.25,
            reward: 0.0,
            continuation: 0.0,
            dynamics: 0.0,
            representation: 0.0,
            policy: 0.0,
            value: 0.0,
            replay_value: 0.0,
        };
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut left = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let mut right = DreamerCore::with_gpu(config.clone(), gpu);
        let mut control_config = config.clone();
        control_config.loss_scales.future_prediction = 0.0;
        control_config.loss_scales.reconstruction = 0.25;
        let control_graph = world::build_observation_prediction_graph(&control_config, 1);
        let mut control = build_session(&control_graph, &left.gpu, Mode::Inference, false);
        initialize_d3(&mut control, &control_graph, config.seed);
        for name in control.param_names() {
            if name == "world.decoder.trunk.weight" {
                continue;
            }
            let future_name = name.replacen("world.decoder.", "world.future_predictor.", 1);
            assert_eq!(
                control.read_params(&[name]),
                left.world_train.read_params(&[&future_name]),
                "matched downstream initialization for {name}"
            );
        }
        let observation = |value| Observation::from_vec(vec![value; Observation::LEN]);
        for agent in [&mut left, &mut right] {
            agent.begin_episode(observation(0.2));
            assert_eq!(
                agent.act(ActionMode::Greedy, Some(&[true, false, false])),
                0
            );
        }
        left.observe(observation(0.4), Reward::default(), FrameFlags::default());
        right.observe(observation(-0.4), Reward::default(), FrameFlags::default());
        assert_eq!(left.deter, right.deter);
        assert_ne!(left.encoded_observation, right.encoded_observation);
        assert_eq!(
            left.observation_prediction(),
            right.observation_prediction()
        );
        let (_, first_action) = left.prior_diagnostic_rollout(&[0]);
        let (_, second_action) = left.prior_diagnostic_rollout(&[1]);
        assert_ne!(first_action, second_action);

        for step in 2..=config.batch_length {
            left.act(ActionMode::Sample, None);
            left.observe(
                observation(step as f32 * 0.2),
                Reward::default(),
                FrameFlags::default(),
            );
        }
        let mut batch = left.replay.sample(&config, &mut left.rngs.replay).unwrap();
        let posterior = left.sample_posterior_batch(&batch, &mut InputWriteTiming::default());
        let behavior =
            left.imagine_and_target(&batch, &posterior, &mut InputWriteTiming::default());
        let metrics = left.train_world(&batch, &posterior, &behavior);
        assert!(metrics.future_prediction_loss > 0.0);
        assert_eq!(metrics.reconstruction_loss, 0.0);
        let name = "world.representation.encoder.patch0.weight";
        let mut gradient = vec![0.0; left.world_train.param_size(name).unwrap()];
        left.world_train.read_param_grad(name, &mut gradient);
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(
            gradient.iter().any(|value| value.abs() > 1e-7),
            "future loss must train the earlier encoder"
        );

        for flags in batch.flags.iter_mut().flatten() {
            flags.is_first = true;
        }
        let posterior = left.sample_posterior_batch(&batch, &mut InputWriteTiming::default());
        let behavior =
            left.imagine_and_target(&batch, &posterior, &mut InputWriteTiming::default());
        let reset = left.train_world(&batch, &posterior, &behavior);
        assert_eq!(reset.future_prediction_loss, 0.0);
        assert_eq!(reset.total_loss, 0.0);
        left.world_train.read_param_grad(name, &mut gradient);
        assert!(gradient.iter().all(|value| *value == 0.0));

        let checkpoint = std::env::temp_dir().join(format!(
            "kindle-future-checkpoint-test-{}",
            std::process::id()
        ));
        left.save_checkpoint(&checkpoint).unwrap();
        let parameter = "world.future_predictor.trunk.weight";
        let expected = left.world_train.read_params(&[parameter]);
        let metadata = read_checkpoint_metadata(&checkpoint).unwrap();
        let restored =
            DreamerCore::restore_with_gpu(&checkpoint, Arc::clone(&left.gpu), metadata).unwrap();
        assert_eq!(restored.config(), &config);
        assert_eq!(restored.world_train.read_params(&[parameter]), expected);
        assert!(
            restored
                .world_train
                .param_names()
                .iter()
                .all(|name| !name.starts_with("world.decoder."))
        );
        fs::remove_dir_all(checkpoint).unwrap();
    }

    #[test]
    #[ignore = "runs a sustained tiny-core GPU learning diagnostic"]
    fn tiny_core_learns_action_conditioned_reward() {
        let mut config = DreamerConfig::tiny(2);
        config.learning_rate = 1e-3;
        config.learning_rate_warmup = 0;
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut agent = DreamerCore::with_gpu(config, gpu);
        let observation = || Observation::from_vec(vec![0.0; Observation::LEN]);

        for _ in 0..512 {
            agent.begin_episode(observation());
            let action = agent.act(ActionMode::Sample, None);
            agent.observe(
                observation(),
                Reward {
                    extrinsic: f32::from(action == 1),
                    intrinsic: 0.0,
                },
                FrameFlags {
                    is_last: true,
                    is_terminal: true,
                    ..Default::default()
                },
            );
            let _ = agent.learn();
        }

        let mut optimal = 0;
        for _ in 0..100 {
            agent.begin_episode(observation());
            let action = agent.act(ActionMode::Greedy, None);
            optimal += usize::from(action == 1);
            agent.observe(
                observation(),
                Reward {
                    extrinsic: f32::from(action == 1),
                    intrinsic: 0.0,
                },
                FrameFlags {
                    is_last: true,
                    is_terminal: true,
                    ..FrameFlags::default()
                },
            );
        }
        assert!(
            optimal >= 90,
            "greedy policy chose the paying action {optimal}/100 times"
        );
    }

    #[test]
    #[ignore = "runs a sustained tiny-core GPU learning diagnostic"]
    fn tiny_core_learns_observation_conditioned_reward() {
        let mut config = DreamerConfig::tiny(2);
        // Make this a wiring canary rather than a small-scale benchmark: the
        // only trainable objective is reward prediction and every replay row
        // resets the RSSM, so class information can only arrive through the
        // observation encoder and posterior straight-through gradient.
        config.learning_rate = 1e-3;
        config.learning_rate_warmup = 0;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.continuation = 0.0;
        config.loss_scales.dynamics = 0.0;
        config.loss_scales.representation = 0.0;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut agent = DreamerCore::with_gpu(config, gpu);
        let observation = |class: usize| {
            let mut values = vec![0.0; Observation::LEN];
            for (index, value) in values.iter_mut().enumerate() {
                if index % 2 == class {
                    *value = 1.0;
                }
            }
            Observation::from_vec(values)
        };

        let size = agent.config.network();
        for index in 0..512 {
            let class = index % 2;
            agent.replay.push(
                ReplayFrame {
                    observation: observation(class),
                    previous_action: None,
                    reward: Reward {
                        extrinsic: class as f32,
                        intrinsic: 0.0,
                    },
                    flags: FrameFlags {
                        is_first: true,
                        is_last: true,
                        is_terminal: true,
                    },
                    deter: vec![0.0; size.deter].into_boxed_slice(),
                    stoch: vec![0.0; size.stoch * size.classes].into_boxed_slice(),
                },
                &agent.config,
            );
        }
        for update in 0..512 {
            let report = agent.learn().expect("pre-filled replay learns");
            if (update + 1) % 128 == 0 {
                eprintln!(
                    "update={} reward_loss={} rewarded={} unrewarded={}",
                    update + 1,
                    report.world.reward_loss,
                    report.world.rewarded_prediction_mean,
                    report.world.unrewarded_prediction_mean,
                );
            }
        }

        let predict = |agent: &mut DreamerCore, class: usize| {
            agent.posterior_live(observation(class).as_slice(), None, true);
            agent.posterior_reward_prediction()
        };
        let negative = (0..32).map(|_| predict(&mut agent, 0)).sum::<f32>() / 32.0;
        let positive = (0..32).map(|_| predict(&mut agent, 1)).sum::<f32>() / 32.0;
        assert!(
            positive > negative + 0.5,
            "observation-conditioned reward predictions did not separate: positive={positive}, negative={negative}"
        );
    }

    fn config_value_bins(agent: &DreamerCore) -> usize {
        agent.config.value_bins
    }

    #[test]
    #[ignore = "builds the full 12M baseline GPU sessions"]
    fn default_agent_sessions_build() {
        let agent = DreamerCore::new(DreamerConfig::new(4)).unwrap();
        assert_eq!(agent.config().model_size, super::super::ModelSize::Size12M);
    }
}
