use super::*;
use crate::vision::PerceptionKind;

#[test]
fn online_graph_fingerprints_match_original_control() {
    use sha2::{Digest, Sha256};
    // Checked byte-for-byte against world.rs at 190a80c, before removing the
    // temporary reference module. Includes node order, constants and outputs.
    for (model, rows, length, reconstruction, prediction, expected) in [
        (
            super::super::ModelSize::Tiny,
            2,
            4,
            1.0,
            0.0,
            "d7edf86989c5893d5b757ca5aee414346b40985a976b603702487e198174b4e2",
        ),
        (
            super::super::ModelSize::Tiny,
            2,
            4,
            0.0,
            0.25,
            "88bdac6664e53452d53d2344e9c1bf20cccea92d9199ab20f8367beb5e8edeb7",
        ),
        (
            super::super::ModelSize::Tiny,
            2,
            4,
            0.25,
            0.25,
            "f03c7aed1b20ea608668d638c0d6762be678c9e8cd2907aed5ba7accd7e513cb",
        ),
        (
            super::super::ModelSize::Size12M,
            16,
            64,
            0.0,
            0.25,
            "053b16bf9bac61387f900fb26b421c18420f2b482a65ac180ca9efc7d140d7e8",
        ),
        (
            super::super::ModelSize::Size12M,
            4,
            64,
            0.0,
            0.25,
            "81d47e86b51a1a1f8aa5c9e194e1d11f7784a8837a10057090f74e97f0709b0e",
        ),
    ] {
        let mut config = if model == super::super::ModelSize::Tiny {
            DreamerConfig::tiny(18)
        } else {
            DreamerConfig::new(18)
        };
        config.model_size = model;
        config.batch_size = rows;
        config.batch_length = length;
        config.world_backprop_length = length;
        config.loss_scales.reconstruction = reconstruction;
        config.loss_scales.future_prediction = prediction;
        let actual = world::build_training_graph(&config, length);
        let encode = |graph: &meganeura::Graph| {
            serde_json::to_vec(&(graph.nodes(), graph.outputs())).unwrap()
        };
        let after = encode(&actual);
        assert_eq!(
            format!("{:x}", Sha256::digest(&after)),
            expected,
            "online graph changed for {model:?}/B{rows}/T{length}/{reconstruction}/{prediction}"
        );
    }
}

fn source(config: &DreamerConfig) -> PretrainingSource {
    PretrainingSource {
        dataset_sha256: "a".repeat(64),
        perception: PerceptionKind::LeVJepa.identity("b".repeat(64)),
        action_names: (0..config.action_count)
            .map(|index| format!("action-{index}"))
            .collect(),
        ticks_per_second: 60,
        action_ticks: 4,
    }
}

fn config() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 0.25;
    config.loss_scales.replay_value = 0.0;
    config.loss_scales.policy = 0.0;
    config.loss_scales.value = 0.0;
    config.train_ratio = 0.0;
    config
}

fn clips(config: &DreamerConfig) -> Vec<Vec<PretrainingFrame>> {
    (0..config.batch_size)
        .map(|row| {
            (0..config.batch_length)
                .map(|time| PretrainingFrame {
                    observation: Observation::from_vec(vec![
                        (row * 10 + time) as f32 / 100.0;
                        Observation::LEN
                    ]),
                    game_tick: (100 * row + time) as u64 * 4,
                    previous_action: (time > 0).then_some((row + time) % config.action_count),
                    reward: None,
                    is_first: time == 0,
                    is_last: false,
                    is_terminal: None,
                })
                .collect()
        })
        .collect()
}

fn pack(config: &DreamerConfig, clips: &[Vec<PretrainingFrame>]) -> io::Result<Batch> {
    Batch::new(
        config,
        4,
        &clips.iter().map(Vec::as_slice).collect::<Vec<_>>(),
    )
}

#[test]
fn offline_rows_keep_actions_histories_and_unknown_labels_distinct() {
    let config = config();
    let mut clips = clips(&config);
    clips[0][1].reward = Some(0.0);
    clips[1][1].reward = Some(1.0);
    clips[1][1].is_last = true;
    clips[1][1].is_terminal = Some(false); // Observed timeout, not terminal.
    clips[1][2].is_first = true;
    clips[1][2].previous_action = None;
    clips[1][2].game_tick = 0;
    clips[1][3].game_tick = 4;
    clips[1][3].is_last = true;
    clips[1][3].is_terminal = Some(true);
    let batch = pack(&config, &clips).unwrap();
    assert_eq!(batch.transitions, 5);
    assert_eq!((batch.reward_labels, batch.terminal_labels), (2, 2));
    assert_eq!(
        batch.keep,
        [
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0]
        ]
    );
    assert_eq!(batch.actions[1], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(batch.actions[2][3..], [0.0; 3]);
    assert_eq!(batch.reward_weights[1], [1.0, 1.0]);
    assert_eq!(batch.reward_weights[2], [0.0, 0.0]);
    assert_eq!(batch.reward_targets[1][config.value_bins / 2], 1.0);
    assert!(batch.reward_targets[2].iter().all(|&value| value == 0.0));
    assert_eq!(batch.continuation_weights[1], [0.0, 1.0]);
    assert_eq!(
        batch.continuation_targets[1],
        [0.0, config.continuation_discount()]
    );
    assert_eq!(batch.continuation_weights[3], [0.0, 1.0]);
    assert_eq!(batch.continuation_targets[3], [0.0, 0.0]);
    assert_eq!(batch.observations[3][0], 0.03);
    assert_eq!(batch.observations[3][Observation::LEN], 0.13);
}

#[test]
fn offline_ingestion_rejects_unknown_actions_gaps_and_inconsistent_boundaries() {
    let config = config();
    let original = clips(&config);
    let invalid: &[fn(&mut Vec<Vec<PretrainingFrame>>)] = &[
        |clips| {
            clips.pop();
        },
        |clips| {
            clips[0].pop();
        },
        |clips| clips[0][0].is_first = false,
        |clips| clips[0][1].previous_action = None,
        |clips| clips[0][1].previous_action = Some(3),
        |clips| clips[0][0].previous_action = Some(0),
        |clips| clips[0][0].reward = Some(0.0),
        |clips| clips[0][1].reward = Some(f32::NAN),
        |clips| clips[0][1].game_tick = 5,
        |clips| clips[0][2].game_tick = 1,
        |clips| clips[0][1].is_terminal = Some(true),
        |clips| clips[0][1].is_last = true,
        |clips| {
            clips[0][1].is_first = true;
            clips[0][1].previous_action = None;
        },
    ];
    for (index, corrupt) in invalid.iter().enumerate() {
        let mut clips = original.clone();
        corrupt(&mut clips);
        let error = pack(&config, &clips)
            .err()
            .unwrap_or_else(|| panic!("accepted invalid case {index}"));
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
}

#[test]
fn offline_source_requires_exact_feature_and_action_namespaces() {
    let config = config();
    let original = source(&config);
    original.validate(&config).unwrap();
    let invalid: &[fn(&mut PretrainingSource)] = &[
        |source| source.dataset_sha256 = "A".repeat(64),
        |source| {
            source.dataset_sha256.pop();
        },
        |source| source.perception.encoding_revision.push_str("-different"),
        |source| source.action_ticks = 0,
        |source| source.ticks_per_second = 0,
        |source| source.action_names[1] = source.action_names[0].clone(),
        |source| source.action_names[0] = " ".into(),
        |source| {
            source.action_names.pop();
        },
    ];
    for corrupt in invalid {
        let mut source = original.clone();
        corrupt(&mut source);
        assert!(source.validate(&config).is_err());
    }
}

#[test]
fn offline_graph_has_masks_but_no_behavior_parameters_or_targets() {
    use meganeura::graph::Op;
    let config = config();
    let graph = world::build_pretraining_graph(&config, config.batch_length);
    let mut inputs = HashSet::new();
    for node in graph.nodes() {
        match &node.op {
            Op::Parameter { name } => assert!(name.starts_with("world."), "{name}"),
            Op::Input { name } => {
                inputs.insert(name.as_str());
            }
            _ => {}
        }
    }
    assert!(!inputs.iter().any(|name| name.starts_with("replay_")));
    for time in 0..config.batch_length {
        assert!(inputs.contains(format!("reward_weight_{time}").as_str()));
        assert!(inputs.contains(format!("continuation_weight_{time}").as_str()));
    }
    let (plan, _) = meganeura::compile_training_graph(&graph);
    assert!(!plan.dispatches.is_empty());
}

#[test]
fn invalid_pretraining_configuration_fails_before_gpu_initialization() {
    let config = config();
    let mut invalid = config.clone();
    invalid.batch_size = 0;
    assert!(WorldPretrainer::new(invalid, source(&config)).is_err());
    let mut invalid = config.clone();
    invalid.intrinsic_reward_scale = 1.0;
    assert!(WorldPretrainer::new(invalid, source(&config)).is_err());
    let mut invalid = config.clone();
    invalid.extrinsic_reward_scale = 0.5;
    assert!(WorldPretrainer::new(invalid, source(&config)).is_err());
    let mut invalid = source(&config);
    invalid.action_ticks = 0;
    assert!(WorldPretrainer::new(config, invalid).is_err());
}

#[test]
#[ignore = "trains offline world parameters, checks masked heads and full-row/microbatch parity on GPU"]
fn offline_world_updates_preserve_unlabeled_heads_and_microbatch_parity() {
    let config = config();
    let gpu = Arc::new(crate::init_gpu_context().unwrap());
    let mut full = WorldPretrainer::with_gpu(config.clone(), source(&config), Arc::clone(&gpu));
    let mut micro_config = config.clone();
    micro_config.world_microbatch_size = Some(1);
    let mut micro = WorldPretrainer::with_gpu(micro_config, source(&config), gpu);
    let names = full
        .train
        .param_names()
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(names.iter().all(|name| name.starts_with("world.")));
    let before = full
        .train
        .read_params(&names.iter().map(String::as_str).collect::<Vec<_>>());
    let mut clips = clips(&config);
    let invalid = {
        let mut data = clips.clone();
        data[0][1].previous_action = None;
        data
    };
    assert!(
        full.learn(&invalid.iter().map(Vec::as_slice).collect::<Vec<_>>())
            .is_err()
    );
    assert_eq!(full.updates, 0);
    for update in 1..=3 {
        let refs = clips.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let report = full.learn(&refs).unwrap();
        micro.learn(&refs).unwrap();
        assert_eq!(report.update, update);
        assert_eq!(report.sampled_observations, update * 8);
        assert_eq!(report.sampled_transitions, update * 6);
        assert_eq!((report.reward_labels, report.terminal_labels), (0, 0));
        assert_eq!((report.reward_loss, report.continuation_loss), (0.0, 0.0));
        assert!(report.future_prediction_loss > 0.0);
    }
    let mut changed = 0;
    for (index, name) in names.iter().enumerate() {
        let full_values = full.train.read_params(&[name.as_str()]).pop().unwrap();
        let micro_values = micro.train.read_params(&[name.as_str()]).pop().unwrap();
        if name.starts_with("world.reward.") || name.starts_with("world.continuation.") {
            assert_eq!(full_values, before[index], "unlabeled head changed: {name}");
        } else {
            changed += usize::from(full_values != before[index]);
        }
        assert!(
            full_values
                .iter()
                .zip(&micro_values)
                .all(|(a, b)| (a - b).abs() < 1e-4 * a.abs().max(b.abs()).max(1.0)),
            "microbatch mismatch: {name}"
        );
    }
    assert!(changed > 0);
    for clip in &mut clips {
        for frame in &mut clip[1..] {
            frame.reward = Some(0.0);
        }
    }
    let report = full
        .learn(&clips.iter().map(Vec::as_slice).collect::<Vec<_>>())
        .unwrap();
    assert_eq!(report.reward_labels, 6);
    assert!(
        report.reward_loss > 0.0,
        "known zero must differ from an unknown reward"
    );
    let name = "world.reward.out.weight";
    let index = names.iter().position(|n| n == name).unwrap();
    assert_ne!(
        full.train.read_params(&[name]).pop().unwrap(),
        before[index]
    );
}
