//! Synchronous vector collection for one learner. No independent model replicas.

use super::*;

struct LiveStream {
    deter: Vec<f32>,
    stoch: Vec<f32>,
    feature: Vec<f32>,
    policy_rng: StdRng,
    posterior_rng: StdRng,
    active: bool,
    needs_reset: bool,
    pending_action: Option<usize>,
}

struct VectorCore {
    learner: DreamerCore,
    streams: Vec<LiveStream>,
    observe: Session,
    policy: Session,
}

fn check_capacity(config: &DreamerConfig, streams: usize) -> Result<(), &'static str> {
    if streams == 0 {
        return Err("at least one environment is required");
    }
    let needed = (config.replay_context + config.batch_length - 1)
        .checked_mul(streams)
        .and_then(|context| context.checked_add(config.replay_warmup_sequences()));
    if needed.is_none_or(|needed| needed > config.replay_capacity) {
        return Err("replay capacity cannot warm up this many independent streams");
    }
    Ok(())
}

impl VectorCore {
    fn new(mut learner: DreamerCore, streams: usize) -> Self {
        check_capacity(&learner.config, streams).unwrap();
        assert_eq!(learner.replay_len(), 0);
        learner.collection_streams = streams;
        learner.replay = SequenceReplay::with_streams(learner.config.replay_capacity, streams);
        let config = &learner.config;
        let mut observe = build_session(
            &world::build_observe_graph(config, streams),
            &learner.gpu,
            Mode::Inference,
            false,
        );
        let mut policy = build_session(
            &behavior::build_actor_inference_graph(config, streams),
            &learner.gpu,
            Mode::Inference,
            false,
        );
        sync_matching(&learner.world_train, &mut observe, "world.");
        sync_matching(&learner.behavior_train, &mut policy, "behavior.actor.");
        let size = config.network();
        let streams = (0..streams)
            .map(|stream| {
                let seed = config.seed.wrapping_add(stream as u64);
                let rngs = if learner.environment_step == 0 && learner.learner_step == 0 {
                    DreamerRngs::new(seed)
                } else {
                    DreamerRngs::resumed(seed, learner.learner_step, learner.environment_step)
                };
                LiveStream {
                    deter: vec![0.0; size.deter],
                    stoch: vec![0.0; size.stoch * size.classes],
                    feature: vec![0.0; config.feature_dim()],
                    policy_rng: rngs.policy,
                    posterior_rng: rngs.live_posterior,
                    active: false,
                    needs_reset: false,
                    pending_action: None,
                }
            })
            .collect();
        Self {
            learner,
            streams,
            observe,
            policy,
        }
    }

    fn check_arrivals(&self, arrivals: impl Iterator<Item = (usize, FrameFlags, Reward)>) {
        let mut seen = vec![false; self.streams.len()];
        for (id, flags, reward) in arrivals {
            assert!(id < seen.len() && !seen[id], "invalid or repeated stream");
            seen[id] = true;
            assert!(reward.extrinsic.is_finite() && reward.intrinsic.is_finite());
            assert!(!flags.is_terminal || flags.is_last);
            let stream = &self.streams[id];
            if flags.is_first {
                assert!(!flags.is_last && !flags.is_terminal);
                assert!(
                    stream.pending_action.is_none(),
                    "cannot reset a pending action"
                );
                assert!(
                    !stream.active || stream.needs_reset,
                    "reset requires an episode boundary"
                );
            } else {
                assert!(stream.pending_action.is_some(), "act must precede observe");
            }
        }
    }

    fn ingest(&mut self, arrivals: Vec<(usize, Observation, FrameFlags, Reward)>) -> Vec<Reward> {
        self.check_arrivals(arrivals.iter().map(|a| (a.0, a.2, a.3)));
        if arrivals.is_empty() {
            return Vec::new();
        }
        let config = &self.learner.config;
        let size = config.network();
        let rows = self.streams.len();
        let stochastic_width = size.stoch * size.classes;
        let mut observations = vec![0.0; rows * config.observation_dim()];
        let mut actions = vec![0.0; rows * config.action_count];
        let mut keep_deter = vec![0.0; rows * size.deter];
        let mut keep_stoch = vec![0.0; rows * stochastic_width];
        let mut keep_action = vec![0.0; rows * config.action_count];
        for (id, observation, flags, _) in &arrivals {
            observations[id * config.observation_dim()..(id + 1) * config.observation_dim()]
                .copy_from_slice(observation.as_slice());
            if !flags.is_first {
                actions[id * config.action_count + self.streams[*id].pending_action.unwrap()] = 1.0;
                keep_deter[id * size.deter..(id + 1) * size.deter].fill(1.0);
                keep_stoch[id * stochastic_width..(id + 1) * stochastic_width].fill(1.0);
                keep_action[id * config.action_count..(id + 1) * config.action_count].fill(1.0);
            }
        }
        let previous_deter: Vec<_> = self
            .streams
            .iter()
            .flat_map(|s| s.deter.iter().copied())
            .collect();
        let previous_stoch: Vec<_> = self
            .streams
            .iter()
            .flat_map(|s| s.stoch.iter().copied())
            .collect();
        for (name, values) in [
            ("previous_deter", &previous_deter),
            ("previous_stoch", &previous_stoch),
            ("previous_action", &actions),
            ("observation", &observations),
            ("keep_deter", &keep_deter),
            ("keep_stoch", &keep_stoch),
            ("keep_action", &keep_action),
        ] {
            self.observe.set_input(name, values);
        }
        self.observe.step();
        let mut deter = vec![0.0; rows * size.deter];
        let mut logits = vec![0.0; rows * stochastic_width];
        self.learner
            .readback
            .read(&self.observe, &mut [(0, &mut deter), (1, &mut logits)]);
        let mut rewards = Vec::with_capacity(arrivals.len());
        for (id, observation, flags, mut reward) in arrivals {
            let stream = &mut self.streams[id];
            stream
                .deter
                .copy_from_slice(&deter[id * size.deter..(id + 1) * size.deter]);
            stream.stoch = sample_latents(
                &logits[id * stochastic_width..(id + 1) * stochastic_width],
                1,
                size.stoch,
                size.classes,
                config.unimix,
                &mut stream.posterior_rng,
            );
            stream.feature = join_features(&stream.deter, &stream.stoch, 1, config);
            assert!(stream.feature.iter().all(|v| v.is_finite()));
            if let Some(bonus) = &mut self.learner.visitation {
                let intrinsic = bonus.observe(&observation);
                if !flags.is_first {
                    reward.intrinsic += intrinsic;
                }
            }
            let previous_action = stream.pending_action.take();
            self.learner.replay.push_stream(
                id,
                ReplayFrame {
                    observation,
                    previous_action,
                    reward,
                    flags,
                    deter: stream.deter.clone().into_boxed_slice(),
                    stoch: stream.stoch.clone().into_boxed_slice(),
                },
                config,
            );
            if !flags.is_first {
                self.learner.environment_step += 1;
                self.learner
                    .train_scheduler
                    .observe(config.train_ratio / (config.batch_size * config.batch_length) as f32);
            }
            stream.active = true;
            stream.needs_reset = flags.is_last;
            rewards.push(reward);
        }
        rewards
    }

    fn act(&mut self, mode: ActionMode) -> Vec<usize> {
        for stream in &self.streams {
            assert!(
                stream.active && !stream.needs_reset,
                "begin every episode before act"
            );
            assert!(
                stream.pending_action.is_none(),
                "observe every pending action first"
            );
        }
        let features: Vec<_> = self
            .streams
            .iter()
            .flat_map(|s| s.feature.iter().copied())
            .collect();
        self.policy.set_input("feature", &features);
        self.policy.step();
        self.policy.wait();
        let config = &self.learner.config;
        let mut logits = vec![0.0; self.streams.len() * config.action_count];
        self.policy.read_output_by_index(0, &mut logits);
        self.streams
            .iter_mut()
            .zip(logits.chunks_exact(config.action_count))
            .map(|(stream, logits)| {
                let mut probabilities = vec![0.0; config.action_count];
                softmax_unimix(logits, config.actor_unimix, &mut probabilities);
                let action = match mode {
                    ActionMode::Sample => {
                        sample_probabilities(&probabilities, &mut stream.policy_rng)
                    }
                    ActionMode::Greedy => argmax(&probabilities),
                };
                stream.pending_action = Some(action);
                action
            })
            .collect()
    }

    fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        let mut reports = self.learner.learn_scheduled(maximum_updates);
        if let Some(last) = reports.last_mut() {
            let started = Instant::now();
            sync_matching(&self.learner.world_train, &mut self.observe, "world.");
            let world = started.elapsed().as_secs_f64();
            let started = Instant::now();
            sync_matching(
                &self.learner.behavior_train,
                &mut self.policy,
                "behavior.actor.",
            );
            let behavior = started.elapsed().as_secs_f64();
            last.timing.world_sync_seconds += world;
            last.timing.behavior_sync_seconds += behavior;
            last.timing.total_seconds += world + behavior;
        }
        reports
    }
}

/// Multiple independent environments using one frozen encoder and Dreamer learner.
/// Dense perception, posterior and policy inference are batched on the GPU.
pub struct VectorDreamerAgent {
    perception: Perception,
    core: VectorCore,
}

impl VectorDreamerAgent {
    pub fn new(
        config: DreamerConfig,
        streams: usize,
        encoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_perception(config, PerceptionKind::LeVJepa, streams, encoder_checkpoint)
    }

    pub fn with_perception(
        config: DreamerConfig,
        kind: PerceptionKind,
        streams: usize,
        encoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        config.check()?;
        check_capacity(&config, streams)?;
        let identity = kind.identity(crate::vision::checkpoint_sha256(
            encoder_checkpoint.as_ref(),
        )?);
        let gpu = Arc::new(crate::init_gpu_context()?);
        let perception = Perception::load_batched(
            kind,
            encoder_checkpoint.as_ref(),
            streams,
            Arc::clone(&gpu),
            None,
        )?;
        let mut learner = DreamerCore::with_gpu(config, gpu);
        learner.perception_identity = Some(identity);
        Ok(Self {
            perception,
            core: VectorCore::new(learner, streams),
        })
    }

    pub fn restore(
        checkpoint: impl AsRef<Path>,
        streams: usize,
        encoder_checkpoint: impl AsRef<Path>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if streams == 0 {
            return Err("at least one environment is required".into());
        }
        let metadata = read_checkpoint_metadata(checkpoint.as_ref())?;
        check_capacity(&metadata.config, streams)?;
        let identity = metadata
            .perception
            .as_ref()
            .ok_or("missing perception identity")?;
        identity.verify_file(encoder_checkpoint.as_ref())?;
        let gpu = Arc::new(crate::init_gpu_context()?);
        let perception = Perception::load_batched(
            identity.kind,
            encoder_checkpoint.as_ref(),
            streams,
            Arc::clone(&gpu),
            None,
        )?;
        let learner = DreamerCore::restore_with_gpu(checkpoint.as_ref(), gpu, metadata)?;
        Ok(Self {
            perception,
            core: VectorCore::new(learner, streams),
        })
    }

    pub fn config(&self) -> &DreamerConfig {
        self.core.learner.config()
    }
    pub fn provenance(&self) -> ModelProvenance {
        self.core.learner.provenance()
    }
    pub fn gpu_device(&self) -> crate::GpuDeviceInfo {
        self.core.learner.gpu_device()
    }
    pub fn trainable_parameter_counts(&self) -> (usize, usize) {
        self.core.learner.trainable_parameter_counts()
    }
    pub fn learner_step(&self) -> u64 {
        self.core.learner.learner_step()
    }
    pub fn environment_step(&self) -> u64 {
        self.core.learner.environment_step()
    }
    pub fn replay_len(&self) -> usize {
        self.core.learner.replay_len()
    }
    pub fn cpu_worker_threads(&self) -> usize {
        self.core.learner.cpu_worker_threads()
    }
    pub fn stream_count(&self) -> usize {
        self.core.streams.len()
    }
    pub fn training_debt(&self) -> f32 {
        self.core.learner.train_scheduler.credit
    }

    pub fn begin_episodes(&mut self, frames: &[(usize, RgbFrame)]) {
        let flags = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        self.core
            .check_arrivals(frames.iter().map(|(id, _)| (*id, flags, Reward::default())));
        let arrivals: Vec<_> = frames
            .iter()
            .map(|(id, frame)| (*id, frame, true))
            .collect();
        let observations = self.perception.encode_frames_rgb8(&arrivals);
        self.core.ingest(
            frames
                .iter()
                .zip(observations)
                .map(|((id, _), observation)| (*id, observation, flags, Reward::default()))
                .collect(),
        );
    }

    pub fn act(&mut self, mode: ActionMode) -> Vec<usize> {
        self.core.act(mode)
    }

    pub fn observe(&mut self, transitions: &[(usize, Transition)]) -> Vec<Reward> {
        self.core
            .check_arrivals(transitions.iter().map(|(id, t)| (*id, t.flags(), t.reward)));
        let arrivals: Vec<_> = transitions
            .iter()
            .map(|(id, t)| (*id, &t.frame, false))
            .collect();
        let observations = self.perception.encode_frames_rgb8(&arrivals);
        self.core.ingest(
            transitions
                .iter()
                .zip(observations)
                .map(|((id, t), observation)| (*id, observation, t.flags(), t.reward))
                .collect(),
        )
    }

    pub fn learn_scheduled(&mut self, maximum_updates: usize) -> Vec<LearnReport> {
        self.core.learn_scheduled(maximum_updates)
    }
    pub fn save_checkpoint(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        self.core.learner.save_checkpoint(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> DreamerConfig {
        let mut config = DreamerConfig::tiny(3);
        config.batch_size = 2;
        config.batch_length = 4;
        config.world_backprop_length = 4;
        config.world_microbatch_size = Some(2);
        config.train_ratio = 8.0;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        config
    }

    fn observation(stream: usize, time: usize) -> Observation {
        Observation::from_vec(
            (0..Observation::LEN)
                .map(|i| ((i + stream * 17 + time * 31) % 101) as f32 / 101.0)
                .collect(),
        )
    }

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());
        for (&a, &b) in left.iter().zip(right) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }

    #[test]
    #[ignore = "requires GPU; compares batched live states to independent serial streams"]
    fn vector_live_beliefs_and_policy_match_serial() {
        let config = config();
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let learner = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let mut vector = VectorCore::new(learner, 3);
        let mut serial: Vec<_> = (0..3)
            .map(|id| {
                let mut core = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
                // Same model initialization, independent live RNGs as in the vector.
                core.rngs = DreamerRngs::new(config.seed + id);
                core
            })
            .collect();
        let first = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        vector.ingest(
            (0..3)
                .map(|id| (id, observation(id, 0), first, Reward::default()))
                .collect(),
        );
        for (id, core) in serial.iter_mut().enumerate() {
            core.begin_episode(observation(id, 0));
        }
        for time in 1..20 {
            let actions = vector.act(ActionMode::Sample);
            let mut arrivals = Vec::new();
            for (id, core) in serial.iter_mut().enumerate() {
                assert_eq!(actions[id], core.act(ActionMode::Sample, None));
                let last = time % (3 + id) == 0;
                let flags = FrameFlags {
                    is_last: last,
                    is_terminal: last && id != 1,
                    ..Default::default()
                };
                let reward = Reward {
                    extrinsic: actions[id] as f32,
                    intrinsic: 0.0,
                };
                core.observe(observation(id, time), reward, flags);
                arrivals.push((id, observation(id, time), flags, reward));
            }
            vector.ingest(arrivals);
            let mut resets = Vec::new();
            for (id, core) in serial.iter_mut().enumerate() {
                assert_close(core.latent_feature(), &vector.streams[id].feature);
                if core.needs_reset {
                    core.begin_episode(observation(id, 100 + time));
                    resets.push((id, observation(id, 100 + time), first, Reward::default()));
                }
            }
            vector.ingest(resets);
            for (id, core) in serial.iter().enumerate() {
                assert_close(core.latent_feature(), &vector.streams[id].feature);
            }
            assert_eq!(vector.learner.environment_step, 3 * time as u64);
        }
    }

    #[test]
    #[ignore = "requires GPU; includes scheduled updates and checkpoint restore"]
    fn vector_one_matches_serial_learning_and_checkpoint() {
        let config = config();
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut serial = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let learner = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
        let mut vector = VectorCore::new(learner, 1);
        let first = FrameFlags {
            is_first: true,
            ..Default::default()
        };
        serial.begin_episode(observation(0, 0));
        vector.ingest(vec![(0, observation(0, 0), first, Reward::default())]);
        for time in 1..24 {
            assert_eq!(
                vector.act(ActionMode::Sample)[0],
                serial.act(ActionMode::Sample, None)
            );
            let flags = FrameFlags {
                is_last: time % 5 == 0,
                is_terminal: time % 10 == 0,
                ..Default::default()
            };
            let reward = Reward {
                extrinsic: (time % 3) as f32 - 1.0,
                intrinsic: 0.0,
            };
            serial.observe(observation(0, time), reward, flags);
            vector.ingest(vec![(0, observation(0, time), flags, reward)]);
            let a = serial.learn_scheduled(usize::MAX);
            let b = vector.learn_scheduled(usize::MAX);
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(&b) {
                assert_eq!(a.world.total_loss, b.world.total_loss);
                assert_eq!(a.behavior.total_loss, b.behavior.total_loss);
            }
            assert_close(&serial.feature, &vector.streams[0].feature);
            if flags.is_last {
                serial.begin_episode(observation(0, time + 100));
                vector.ingest(vec![(
                    0,
                    observation(0, time + 100),
                    first,
                    Reward::default(),
                )]);
            }
        }
        assert!(vector.learner.learner_step > 0);
        assert_eq!(serial.learner_step, vector.learner.learner_step);
        assert_eq!(
            serial.train_scheduler.credit,
            vector.learner.train_scheduler.credit
        );
        let names = serial.world_train.param_names();
        assert_eq!(
            serial.world_train.read_params(&names),
            vector.learner.world_train.read_params(&names)
        );
        let checkpoint =
            std::env::temp_dir().join(format!("kindle-vector-test-{}", std::process::id()));
        assert!(!checkpoint.exists());
        vector.learner.save_checkpoint(&checkpoint).unwrap();
        let metadata = read_checkpoint_metadata(&checkpoint).unwrap();
        assert_eq!(metadata.collection_streams, 1);
        let learner = DreamerCore::restore_with_gpu(&checkpoint, gpu, metadata).unwrap();
        let mut restored = VectorCore::new(learner, 2);
        assert_eq!(restored.learner.replay_len(), 0);
        assert_eq!(restored.learner.environment_step, 23);
        restored.ingest(
            (0..2)
                .map(|id| (id, observation(id, 0), first, Reward::default()))
                .collect(),
        );
        assert_eq!(restored.act(ActionMode::Sample).len(), 2);
        fs::remove_dir_all(checkpoint).unwrap();
    }

    #[test]
    fn vector_scheduler_counts_actions_not_ticks_or_resets() {
        let mut scheduler = D3TrainScheduler::default();
        for _ in 0..400 {
            scheduler.observe(0.25);
        }
        assert!(!scheduler.update_due(false));
        assert!(scheduler.update_due(true));
        scheduler.consume_update();
        for _ in 0..3 {
            for _ in 0..8 {
                scheduler.observe(0.25);
            }
            for _ in 0..2 {
                assert!(scheduler.update_due(true));
                scheduler.consume_update();
            }
            assert!(!scheduler.update_due(true));
        }
        assert_eq!(scheduler.credit, 0.0);
    }

    #[test]
    fn vector_capacity_includes_each_stream_context() {
        let mut config = config();
        config.replay_capacity = config.replay_warmup_frames();
        assert!(check_capacity(&config, 1).is_ok());
        assert!(check_capacity(&config, 2).is_err());
        assert!(check_capacity(&config, 0).is_err());
        assert!(check_capacity(&config, usize::MAX).is_err());
        config.replay_capacity += config.replay_context + config.batch_length - 1;
        assert!(check_capacity(&config, 2).is_ok());
    }

    #[test]
    #[ignore = "requires GPU and pinned DINOv3 weights; full pixel-vector path and restore"]
    fn dino_pixel_vector_matches_serial_and_restores_recorded_frontend() {
        let weights = std::env::var_os("KINDLE_DINOV3_WEIGHTS").expect("set KINDLE_DINOV3_WEIGHTS");
        assert_eq!(
            crate::vision::checkpoint_sha256(Path::new(&weights)).unwrap(),
            crate::vision::VITS16_CHECKPOINT_SHA256
        );
        let mut config = config();
        config.train_ratio = 0.0;
        let mut vector = VectorDreamerAgent::with_perception(
            config.clone(),
            PerceptionKind::DinoV3,
            2,
            &weights,
        )
        .unwrap();
        let mut serial: Vec<_> = (0..2)
            .map(|id| {
                let mut agent = DreamerAgent::with_perception(
                    config.clone(),
                    PerceptionKind::DinoV3,
                    &weights,
                    None,
                )
                .unwrap();
                agent.core.rngs = DreamerRngs::new(config.seed + id);
                agent
            })
            .collect();
        let frame = |id: usize, time: usize| {
            RgbFrame::new(
                64,
                64,
                (0..64 * 64 * 3)
                    .map(|i| ((i * 37 + id * 71 + time * 13) % 251) as u8)
                    .collect(),
            )
        };
        vector.begin_episodes(&[(1, frame(1, 0)), (0, frame(0, 0))]);
        for (id, agent) in serial.iter_mut().enumerate() {
            agent.begin_episode(&frame(id, 0));
            assert_close(
                agent.core.latent_feature(),
                &vector.core.streams[id].feature,
            );
        }
        for time in 1..12 {
            let actions = vector.act(ActionMode::Sample);
            let mut arrivals = Vec::new();
            for (id, agent) in serial.iter_mut().enumerate() {
                assert_eq!(actions[id], agent.act(ActionMode::Sample, None));
                let last = time % (3 + id) == 0;
                let transition = Transition {
                    frame: frame(id, time),
                    reward: Reward {
                        extrinsic: (time % 3) as f32 - 1.0,
                        intrinsic: 0.0,
                    },
                    terminated: last && id == 0,
                    truncated: last && id == 1,
                };
                agent.observe(&transition);
                arrivals.push((id, transition));
            }
            arrivals.reverse();
            vector.observe(&arrivals);
            let mut resets = Vec::new();
            for (id, agent) in serial.iter_mut().enumerate() {
                assert_close(
                    agent.core.latent_feature(),
                    &vector.core.streams[id].feature,
                );
                if agent.core.needs_reset {
                    agent.begin_episode(&frame(id, 100 + time));
                    resets.push((id, frame(id, 100 + time)));
                }
            }
            vector.begin_episodes(&resets);
            for (id, agent) in serial.iter().enumerate() {
                assert_close(
                    agent.core.latent_feature(),
                    &vector.core.streams[id].feature,
                );
            }
            assert!(vector.learn_scheduled(usize::MAX).is_empty());
            assert_eq!(vector.environment_step(), 2 * time as u64);
            assert_eq!(vector.training_debt(), 0.0);
        }
        assert_eq!(
            vector.provenance().perception.as_ref().unwrap().kind,
            PerceptionKind::DinoV3
        );
        let checkpoint =
            std::env::temp_dir().join(format!("kindle-dino-vector-test-{}", std::process::id()));
        assert!(!checkpoint.exists());
        vector.save_checkpoint(&checkpoint).unwrap();
        drop(vector);
        drop(serial);
        let mut restored = VectorDreamerAgent::restore(&checkpoint, 2, &weights).unwrap();
        assert_eq!(
            restored.provenance().perception.as_ref().unwrap().kind,
            PerceptionKind::DinoV3
        );
        assert_eq!(restored.environment_step(), 22);
        assert_eq!(restored.learner_step(), 0);
        assert_eq!(restored.replay_len(), 0);
        restored.begin_episodes(&[(0, frame(0, 20)), (1, frame(1, 20))]);
        assert_eq!(restored.act(ActionMode::Sample).len(), 2);
        fs::remove_dir_all(checkpoint).unwrap();
    }
}
