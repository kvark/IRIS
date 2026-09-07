use super::*;
use crate::{PretrainingFrame, PretrainingSource, WorldPretrainer};

pub(super) struct PretrainedBundle {
    pub path: std::path::PathBuf,
    pub source: PretrainingSource,
}

impl Drop for PretrainedBundle {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        let online = self.path.join("online");
        if online.exists() {
            for name in [
                "world.safetensors",
                "behavior.safetensors",
                "slow_value.safetensors",
                "metadata.json",
            ] {
                fs::remove_file(online.join(name)).unwrap();
            }
            fs::remove_dir(online).unwrap();
        }
        for name in ["pretraining.json", "world.safetensors"] {
            fs::remove_file(self.path.join(name)).unwrap();
        }
        fs::remove_dir(&self.path).unwrap();
    }
}

pub(super) fn pretrained_bundle(
    target: &DreamerConfig,
    gpu: &Arc<blade_graphics::Context>,
) -> PretrainedBundle {
    let mut config = target.clone();
    config.seed = 7;
    config.train_ratio = 0.0;
    config.loss_scales.policy = 0.0;
    config.loss_scales.value = 0.0;
    config.loss_scales.replay_value = 0.0;
    let source = PretrainingSource {
        dataset_sha256: "a".repeat(64),
        perception: PerceptionKind::LeVJepa.identity("b".repeat(64)),
        action_names: (0..config.action_count)
            .map(|index| format!("action-{index}"))
            .collect(),
        ticks_per_second: 60,
        action_ticks: 4,
    };
    let clips = (0..config.batch_size)
        .map(|row| {
            (0..config.batch_length)
                .map(|time| PretrainingFrame {
                    observation: Observation::from_vec(vec![
                        (row + time) as f32 / 10.0;
                        Observation::LEN
                    ]),
                    game_tick: time as u64 * 4,
                    previous_action: (time > 0).then_some((row + time) % config.action_count),
                    reward: (time > 0).then_some(1.0),
                    is_first: time == 0,
                    is_last: false,
                    is_terminal: Some(false),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut trainer = WorldPretrainer::with_gpu(config, source.clone(), Arc::clone(gpu));
    let clips = clips.iter().map(Vec::as_slice).collect::<Vec<_>>();
    trainer.learn(&clips).unwrap();
    trainer.learn(&clips).unwrap();
    let tick = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "kindle-trained-world-{}-{tick}",
        std::process::id()
    ));
    trainer.export_world(&path).unwrap();
    PretrainedBundle { path, source }
}

pub(super) fn target() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.seed = 99;
    config.loss_scales.reconstruction = 0.25;
    config.loss_scales.future_prediction = 0.25;
    config
}

pub(super) fn snapshot(session: &Session) -> std::collections::BTreeMap<String, Vec<f32>> {
    let names = session.param_names();
    let mut values = std::collections::BTreeMap::new();
    for (name, data) in names.iter().zip(session.read_params(&names)) {
        if session.has_param_grad(name) {
            let mut m = vec![0.0; data.len()];
            let mut v = m.clone();
            session.read_adam_m(name, &mut m);
            session.read_adam_v(name, &mut v);
            values.insert(format!("adam_m.{name}"), m);
            values.insert(format!("adam_v.{name}"), v);
        }
        values.insert((*name).into(), data);
    }
    values
}

#[test]
#[ignore = "initializes from actually trained world weights and verifies fresh state, heads, moments and checkpoint lineage on GPU"]
fn world_initialization_preserves_fresh_behavior_and_survives_full_restore() {
    let config = target();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let bundle = pretrained_bundle(&config, &gpu);
    let load = || WorldInitialization::load(&bundle.path, &config, &bundle.source).unwrap();
    let source =
        meganeura::data::safetensors::SafeTensorsModel::load(bundle.path.join("world.safetensors"))
            .unwrap();
    let mut core = DreamerCore::with_gpu(config.clone(), Arc::clone(&gpu));
    core.ensure_world_prediction_live();
    core.ensure_behavior_value_live();
    let names = core
        .world_train
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let references = names.iter().map(String::as_str).collect::<Vec<_>>();
    let before = core.world_train.read_params(&references);
    let behavior_before = [
        &core.behavior_train,
        &core.behavior_online,
        &core.behavior_slow,
        &core.policy_live,
        core.behavior_value_live.as_ref().unwrap(),
    ]
    .map(snapshot);
    let policy_rng = core.rngs.policy.clone().random::<u64>();
    let posterior_rng = core.rngs.live_posterior.clone().random::<u64>();
    let normalizer = core.return_normalizer.state();
    core.initialize_world(load()).unwrap();
    let mut changed = 0;
    for (index, name) in names.iter().enumerate() {
        let actual = core.world_train.read_params(&[name]).pop().unwrap();
        if super::super::initialization::transferred(name) {
            changed += usize::from(actual != before[index]);
            assert_eq!(
                actual,
                source.tensor_f32(name).unwrap(),
                "wrong imported parameter: {name}"
            );
        } else {
            assert_eq!(actual, before[index], "fresh head changed: {name}");
        }
        if core.world_train.has_param_grad(name) {
            let mut m = vec![0.0; actual.len()];
            let mut v = m.clone();
            core.world_train.read_adam_m(name, &mut m);
            core.world_train.read_adam_v(name, &mut v);
            assert!(
                m.iter().chain(&v).all(|&value| value == 0.0),
                "imported optimizer state: {name}"
            );
        }
    }
    assert!(changed > 0, "fixture did not exercise a weight change");
    assert_eq!(
        [
            &core.behavior_train,
            &core.behavior_online,
            &core.behavior_slow,
            &core.policy_live,
            core.behavior_value_live.as_ref().unwrap(),
        ]
        .map(snapshot),
        behavior_before
    );
    for session in [
        &core.world_observe_batch,
        &core.world_observe_live,
        &core.world_transition,
        &core.world_transition_live,
        &core.world_heads,
        &core.world_heads_live,
        core.world_prediction_live.as_ref().unwrap(),
    ] {
        let names = session.param_names();
        assert_eq!(
            session.read_params(&names),
            core.world_train.read_params(&names)
        );
    }
    assert_eq!(core.rngs.policy.clone().random::<u64>(), policy_rng);
    assert_eq!(
        core.rngs.live_posterior.clone().random::<u64>(),
        posterior_rng
    );
    assert_eq!(core.return_normalizer.state(), normalizer);
    assert_eq!(
        (core.environment_step, core.learner_step, core.replay_len()),
        (0, 0, 0)
    );
    assert!(!core.active && core.pending_action.is_none());
    assert!(
        core.deter
            .iter()
            .chain(&core.stoch)
            .all(|&value| value == 0.0)
    );
    assert!(core.initialize_world(load()).is_err());
    let online = bundle.path.join("online");
    core.save_checkpoint(&online).unwrap();
    let metadata = read_checkpoint_metadata(&online).unwrap();
    assert_eq!(metadata.format, INITIALIZED_CHECKPOINT_FORMAT);
    assert_eq!(
        metadata
            .world_initialization
            .as_ref()
            .unwrap()
            .source_updates(),
        2
    );
    let mut restored = DreamerCore::restore_with_gpu(&online, Arc::clone(&gpu), metadata).unwrap();
    assert_eq!(restored.provenance(), core.provenance());
    assert_eq!(
        restored.world_train.read_params(&references),
        core.world_train.read_params(&references)
    );
    assert!(
        restored.initialize_world(load()).is_err(),
        "a restored zero-counter runtime is not fresh"
    );
    let mut active = DreamerCore::with_gpu(config.clone(), gpu);
    active.begin_episode(Observation::from_vec(vec![0.0; Observation::LEN]));
    assert_eq!((active.environment_step, active.learner_step), (0, 0));
    let before = snapshot(&active.world_train);
    assert!(active.initialize_world(load()).is_err());
    assert_eq!(snapshot(&active.world_train), before);
}
