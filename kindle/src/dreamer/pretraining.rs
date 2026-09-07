//! Action-aligned world learning from recorded frozen features, without behavior learning.

#[cfg(test)]
mod tests;

use std::{collections::HashSet, io, path::Path, sync::Arc, time::Instant};

use meganeura::{Mode, Session};
use rand::{SeedableRng, rngs::StdRng};

use super::{
    BLADE_REV, DREAMERV3_UPSTREAM_REV, DreamerConfig, MEGANEURA_REV,
    distributions::{TwoHotBins, sample_probabilities, softmax_unimix},
    readback::Readback,
    runtime::{build_session, configure_d3_optimizer, initialize_d3, sync_matching},
    world,
};
use crate::vision::{Observation, PerceptionIdentity};

/// Identity asserted by the dataset reader. The reader must verify its manifest
/// and feature contents; a matching feature shape alone does not establish identity.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PretrainingSource {
    pub dataset_sha256: String,
    pub perception: PerceptionIdentity,
    /// Ordered categorical controls, including the meaning of held inputs.
    pub action_names: Vec<String>,
    /// Clock units per second (e.g. 60 for Atari frames, 1e9 for nanoseconds).
    pub ticks_per_second: u64,
    /// The current RSSM has no duration input: variable-duration clips are rejected.
    pub action_ticks: u64,
}

impl PretrainingSource {
    pub(super) fn validate(&self, config: &DreamerConfig) -> io::Result<()> {
        self.perception.validate()?;
        require(
            self.dataset_sha256.len() == 64
                && self
                    .dataset_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()),
            "invalid dataset fingerprint",
        )?;
        require(
            self.action_ticks > 0 && self.ticks_per_second > 0,
            "action duration must be positive",
        )?;
        require(
            self.action_names.len() == config.action_count
                && self.action_names.iter().all(|name| !name.trim().is_empty())
                && self.action_names.iter().collect::<HashSet<_>>().len()
                    == self.action_names.len(),
            "action names must be complete, nonempty and unique",
        )
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PretrainingMetadata {
    pub format: u32,
    pub architecture: String,
    pub config: DreamerConfig,
    pub source: PretrainingSource,
    pub dreamerv3_revision: String,
    pub meganeura_revision: String,
    pub blade_revision: String,
    pub future_head_revision: Option<String>,
    pub updates: u64,
    pub sampled_observations: u64,
    pub sampled_transitions: u64,
    pub world_sha256: String,
}

/// A resulting observation and the action/reward that preceded it.
#[derive(Clone, Debug)]
pub struct PretrainingFrame {
    pub observation: Observation,
    pub game_tick: u64,
    pub previous_action: Option<usize>,
    pub reward: Option<f32>,
    /// Reset belief before this observation. Also true at a cold clip start,
    /// which need not be an actual game reset. No predictive loss on this row.
    pub is_first: bool,
    /// An observed episode boundary, not merely the end of a dataset window.
    pub is_last: bool,
    /// None means unobserved, not nonterminal. True implies is_last.
    pub is_terminal: Option<bool>,
}

#[derive(Debug, serde::Serialize)]
pub struct WorldPretrainingReport {
    pub update: u64,
    /// Cumulative sampled rows, including repeat visits. Not unique source data
    /// or online environment interactions.
    pub sampled_observations: u64,
    pub sampled_transitions: u64,
    pub reward_labels: usize,
    pub terminal_labels: usize,
    pub total_loss: f32,
    pub reconstruction_loss: f32,
    pub future_prediction_loss: f32,
    pub dynamics_kl: f32,
    pub representation_kl: f32,
    pub raw_kl: f32,
    pub reward_loss: f32,
    pub continuation_loss: f32,
    pub posterior_seconds: f64,
    pub train_seconds: f64,
    pub sync_seconds: f64,
    pub total_seconds: f64,
}

struct Batch {
    observations: Vec<Vec<f32>>,
    actions: Vec<Vec<f32>>,
    keep: Vec<Vec<f32>>,
    reward_targets: Vec<Vec<f32>>,
    continuation_targets: Vec<Vec<f32>>,
    reward_weights: Vec<Vec<f32>>,
    continuation_weights: Vec<Vec<f32>>,
    transitions: usize,
    reward_labels: usize,
    terminal_labels: usize,
}

impl Batch {
    fn new(
        config: &DreamerConfig,
        duration: u64,
        clips: &[&[PretrainingFrame]],
    ) -> io::Result<Self> {
        require(
            duration > 0 && clips.len() == config.batch_size,
            "wrong duration or clip count",
        )?;
        for clip in clips {
            require(
                clip.len() == config.batch_length && clip[0].is_first,
                "clips need T observations and a cold initial state",
            )?;
            for (time, frame) in clip.iter().enumerate() {
                require(
                    frame
                        .observation
                        .as_slice()
                        .iter()
                        .all(|value| value.is_finite()),
                    "nonfinite observation",
                )?;
                require(frame.reward.is_none_or(f32::is_finite), "nonfinite reward")?;
                require(
                    frame.is_terminal != Some(true) || frame.is_last,
                    "terminal frame must end the episode",
                )?;
                if frame.is_first {
                    require(
                        frame.previous_action.is_none() && frame.reward.is_none(),
                        "reset has no preceding action or reward",
                    )?;
                } else {
                    require(
                        frame
                            .previous_action
                            .is_some_and(|action| action < config.action_count),
                        "missing or invalid executed action",
                    )?;
                    require(
                        time > 0
                            && frame.game_tick.checked_sub(clip[time - 1].game_tick)
                                == Some(duration),
                        "observation gap or unsupported action duration",
                    )?;
                }
                if time > 0 {
                    require(
                        frame.is_first == clip[time - 1].is_last,
                        "episode boundary and next reset disagree",
                    )?;
                }
            }
        }
        let rows = config.batch_size;
        let allocate = |width| vec![vec![0.0; rows * width]; config.batch_length];
        let mut batch = Self {
            observations: allocate(config.observation_dim()),
            actions: allocate(config.action_count),
            keep: allocate(1),
            reward_targets: allocate(config.value_bins),
            continuation_targets: allocate(1),
            reward_weights: allocate(1),
            continuation_weights: allocate(1),
            transitions: 0,
            reward_labels: 0,
            terminal_labels: 0,
        };
        let bins = TwoHotBins::new(config.value_bins);
        for (row, clip) in clips.iter().enumerate() {
            for (time, frame) in clip.iter().enumerate() {
                slice_mut(&mut batch.observations[time], row, config.observation_dim())
                    .copy_from_slice(frame.observation.as_slice());
                if let Some(action) = frame.previous_action {
                    batch.actions[time][row * config.action_count + action] = 1.0;
                    batch.transitions += 1;
                }
                batch.keep[time][row] = f32::from(!frame.is_first);
                if let Some(reward) = frame.reward {
                    bins.encode(
                        reward,
                        slice_mut(&mut batch.reward_targets[time], row, config.value_bins),
                    );
                    batch.reward_weights[time][row] = 1.0;
                    batch.reward_labels += 1;
                }
                if let Some(terminal) = frame.is_terminal {
                    batch.continuation_targets[time][row] = if terminal {
                        0.0
                    } else {
                        config.continuation_discount()
                    };
                    batch.continuation_weights[time][row] = 1.0;
                    batch.terminal_labels += 1;
                }
            }
        }
        Ok(batch)
    }
}

/// Separate offline optimizer and counters. Owns only world training and posterior
/// sessions: no actor, critic, imagination, online replay or environment scheduler.
pub struct WorldPretrainer {
    config: DreamerConfig,
    source: PretrainingSource,
    train: Session,
    observe: Session,
    readback: Readback,
    rng: StdRng,
    updates: u64,
    sampled_observations: u64,
    sampled_transitions: u64,
    failed: bool,
}

impl WorldPretrainer {
    pub fn new(
        mut config: DreamerConfig,
        source: PretrainingSource,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        config.train_ratio = 0.0;
        config.loss_scales.policy = 0.0;
        config.loss_scales.value = 0.0;
        config.loss_scales.replay_value = 0.0;
        config
            .check()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        source.validate(&config)?;
        require(
            !config.visitation_bonus && config.intrinsic_reward_scale == 0.0,
            "offline labels must not include a new intrinsic reward",
        )?;
        require(
            config.extrinsic_reward_scale == 1.0,
            "offline reward labels are used without scaling",
        )?;
        let gpu = Arc::new(crate::init_gpu_context()?);
        Ok(Self::with_gpu(config, source, gpu))
    }

    pub(super) fn with_gpu(
        config: DreamerConfig,
        source: PretrainingSource,
        gpu: Arc<blade_graphics::Context>,
    ) -> Self {
        let mut training = config.clone();
        training.batch_size = config.world_microbatch_size();
        training.world_microbatch_size = Some(training.batch_size);
        let graph = world::build_pretraining_graph(&training, config.world_backprop_length);
        let observe_graph = world::build_observe_graph(&config, config.batch_size);
        let mut train = build_session(&graph, &gpu, Mode::Training, config.skip_full_optimize);
        let mut observe = build_session(&observe_graph, &gpu, Mode::Inference, false);
        initialize_d3(&mut train, &graph, config.seed);
        sync_matching(&train, &mut observe, "world.");
        let seed = config.seed ^ 0x7072_6574_7261_696e;
        Self {
            config,
            source,
            train,
            observe,
            readback: Readback::new(gpu),
            rng: StdRng::seed_from_u64(seed),
            updates: 0,
            sampled_observations: 0,
            sampled_transitions: 0,
            failed: false,
        }
    }

    pub fn config(&self) -> &DreamerConfig {
        &self.config
    }

    /// One update on independent contiguous clips. Validation finishes before any
    /// session input, RNG, counter or optimizer is touched. The first observation
    /// of each clip supplies a cold posterior, not an invented incoming transition.
    pub fn learn(&mut self, clips: &[&[PretrainingFrame]]) -> io::Result<WorldPretrainingReport> {
        let started = Instant::now();
        require(
            !self.failed,
            "pretrainer failed a numerical check; discard this runtime",
        )?;
        let batch = Batch::new(&self.config, self.source.action_ticks, clips)?;
        let rows = self.config.batch_size;
        let size = self.config.network();
        let stochastic_width = size.stoch * size.classes;
        let stage = Instant::now();
        let mut deter = vec![vec![0.0; rows * size.deter]];
        let mut stoch = vec![vec![0.0; rows * stochastic_width]];
        for time in 0..self.config.batch_length {
            self.observe.set_input("previous_deter", &deter[time]);
            self.observe.set_input("previous_stoch", &stoch[time]);
            self.observe
                .set_input("previous_action", &batch.actions[time]);
            self.observe
                .set_input("observation", &batch.observations[time]);
            set_keep(&mut self.observe, &batch.keep[time], &self.config, None);
            self.observe.step();
            let mut next_deter = vec![0.0; rows * size.deter];
            let mut logits = vec![0.0; rows * stochastic_width];
            self.readback
                .read(&self.observe, &mut [(0, &mut next_deter), (1, &mut logits)]);
            let mut samples = vec![0.0; logits.len()];
            let mut probabilities = vec![0.0; size.classes];
            for (logit, sample) in logits
                .chunks_exact(size.classes)
                .zip(samples.chunks_exact_mut(size.classes))
            {
                softmax_unimix(logit, self.config.unimix, &mut probabilities);
                sample[sample_probabilities(&probabilities, &mut self.rng)] = 1.0;
            }
            deter.push(next_deter);
            stoch.push(samples);
        }
        let posterior_seconds = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        let micro = self.config.world_microbatch_size();
        let length = self.config.world_backprop_length;
        let passes = (rows / micro) * (self.config.batch_length / length);
        self.train.set_grad_accumulate(passes.try_into().unwrap());
        self.train.zero_grad();
        let mut losses = [0.0; 9];
        let mut pass = 0;
        for start in (0..self.config.batch_length).step_by(length) {
            for first in (0..rows).step_by(micro) {
                self.train.set_input(
                    "initial_deter",
                    row_slice(&deter[start], first, micro, size.deter),
                );
                self.train.set_input(
                    "initial_stoch",
                    row_slice(&stoch[start], first, micro, stochastic_width),
                );
                for local in 0..length {
                    let time = start + local;
                    set_keep(
                        &mut self.train,
                        &batch.keep[time][first..first + micro],
                        &self.config,
                        Some(local),
                    );
                    for (name, values, width) in [
                        (
                            "observation",
                            &batch.observations[time],
                            self.config.observation_dim(),
                        ),
                        (
                            "previous_action",
                            &batch.actions[time],
                            self.config.action_count,
                        ),
                        ("posterior_sample", &stoch[time + 1], stochastic_width),
                        (
                            "reward_target",
                            &batch.reward_targets[time],
                            self.config.value_bins,
                        ),
                        ("continuation_target", &batch.continuation_targets[time], 1),
                        ("reward_weight", &batch.reward_weights[time], 1),
                        ("continuation_weight", &batch.continuation_weights[time], 1),
                    ] {
                        self.train.set_input(
                            &format!("{name}_{local}"),
                            row_slice(values, first, micro, width),
                        );
                    }
                }
                pass += 1;
                if pass == passes {
                    configure_d3_optimizer(
                        &mut self.train,
                        &self.config,
                        self.updates,
                        self.config.learning_rate,
                    );
                } else {
                    self.train.clear_optimizer();
                }
                self.train.step();
                let mut values = [0.0; 9];
                let mut outputs = values
                    .iter_mut()
                    .enumerate()
                    .map(|(index, value)| (index, std::slice::from_mut(value)))
                    .collect::<Vec<_>>();
                self.readback.read(&self.train, &mut outputs);
                for (loss, value) in losses.iter_mut().zip(values) {
                    *loss += value / passes as f32;
                }
            }
        }
        self.train.clear_grad_accumulate();
        if !losses.iter().all(|value| value.is_finite()) {
            self.failed = true;
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "nonfinite pretraining loss; discard this runtime",
            ));
        }
        let train_seconds = stage.elapsed().as_secs_f64();
        let stage = Instant::now();
        sync_matching(&self.train, &mut self.observe, "world.");
        let sync_seconds = stage.elapsed().as_secs_f64();
        self.updates += 1;
        self.sampled_observations += (rows * self.config.batch_length) as u64;
        self.sampled_transitions += batch.transitions as u64;
        Ok(WorldPretrainingReport {
            update: self.updates,
            sampled_observations: self.sampled_observations,
            sampled_transitions: self.sampled_transitions,
            reward_labels: batch.reward_labels,
            terminal_labels: batch.terminal_labels,
            total_loss: losses[world::LOSS_TOTAL],
            reconstruction_loss: losses[world::LOSS_RECONSTRUCTION],
            future_prediction_loss: losses[world::LOSS_FUTURE_PREDICTION],
            dynamics_kl: losses[world::LOSS_DYNAMICS],
            representation_kl: losses[world::LOSS_REPRESENTATION],
            raw_kl: losses[world::RAW_KL],
            reward_loss: losses[world::LOSS_REWARD],
            continuation_loss: losses[world::LOSS_CONTINUATION],
            posterior_seconds,
            train_seconds,
            sync_seconds,
            total_seconds: started.elapsed().as_secs_f64(),
        })
    }

    /// A separate artifact type, deliberately not an online checkpoint or exact
    /// offline resume. World-only initialization must validate the complete bundle.
    pub fn export_world(
        &mut self,
        directory: impl AsRef<Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let directory = directory.as_ref();
        require(!self.failed, "cannot save a numerically failed pretrainer")?;
        std::fs::create_dir(directory)?;
        let weights = directory.join("world.safetensors");
        self.train.save_checkpoint(&weights)?;
        let metadata = PretrainingMetadata {
            format: 1,
            architecture: "dreamerv3-world-pretraining".into(),
            config: self.config.clone(),
            source: self.source.clone(),
            dreamerv3_revision: DREAMERV3_UPSTREAM_REV.into(),
            meganeura_revision: MEGANEURA_REV.into(),
            blade_revision: BLADE_REV.into(),
            future_head_revision: (self.config.loss_scales.future_prediction > 0.0)
                .then(|| world::FUTURE_HEAD_REVISION.into()),
            updates: self.updates,
            sampled_observations: self.sampled_observations,
            sampled_transitions: self.sampled_transitions,
            world_sha256: crate::vision::checkpoint_sha256(&weights)?,
        };
        let output = std::fs::File::create_new(directory.join("pretraining.json"))?;
        serde_json::to_writer_pretty(output, &metadata)?;
        Ok(())
    }
}

fn require(condition: bool, message: &str) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidInput, message))
    }
}

fn slice_mut(values: &mut [f32], row: usize, width: usize) -> &mut [f32] {
    &mut values[row * width..(row + 1) * width]
}

fn row_slice(values: &[f32], first: usize, rows: usize, width: usize) -> &[f32] {
    &values[first * width..(first + rows) * width]
}

fn set_keep(session: &mut Session, keep: &[f32], config: &DreamerConfig, time: Option<usize>) {
    let size = config.network();
    for (name, width) in [
        ("keep_deter", size.deter),
        ("keep_stoch", size.stoch * size.classes),
        ("keep_action", config.action_count),
    ] {
        let name = match time {
            Some(time) => format!("{name}_{time}"),
            None => name.to_owned(),
        };
        let values = keep
            .iter()
            .flat_map(|&value| std::iter::repeat_n(value, width))
            .collect::<Vec<_>>();
        session.set_input(&name, &values);
    }
}
