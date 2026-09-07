use super::*;
use crate::vision::PerceptionKind;
type TensorData = BTreeMap<String, Vec<f32>>;

fn target() -> DreamerConfig {
    let mut config = DreamerConfig::tiny(3);
    config.loss_scales.reconstruction = 0.0;
    config.loss_scales.future_prediction = 0.25;
    config
}

fn metadata() -> PretrainingMetadata {
    let mut config = target();
    config.train_ratio = 0.0;
    config.loss_scales.policy = 0.0;
    config.loss_scales.value = 0.0;
    config.loss_scales.replay_value = 0.0;
    PretrainingMetadata {
        format: 1,
        architecture: "dreamerv3-world-pretraining".into(),
        config,
        source: PretrainingSource {
            dataset_sha256: "a".repeat(64),
            perception: PerceptionKind::LeVJepa.identity("b".repeat(64)),
            action_names: vec!["NOOP".into(), "LEFT".into(), "RIGHT".into()],
            ticks_per_second: 60,
            action_ticks: 4,
        },
        dreamerv3_revision: DREAMERV3_UPSTREAM_REV.into(),
        meganeura_revision: MEGANEURA_REV.into(),
        blade_revision: BLADE_REV.into(),
        future_head_revision: Some(world::FUTURE_HEAD_REVISION.into()),
        updates: 1,
        sampled_observations: 8,
        sampled_transitions: 6,
        world_sha256: "c".repeat(64),
    }
}

fn tensors(config: &DreamerConfig) -> BTreeMap<String, Vec<f32>> {
    parameter_schema(config)
        .into_iter()
        .flat_map(|(name, shape)| {
            let length = shape.iter().product();
            [
                (name.clone(), vec![0.125; length]),
                (format!("adam_m.{name}"), vec![0.0; length]),
                (format!("adam_v.{name}"), vec![0.0; length]),
            ]
        })
        .collect()
}

fn encode(
    tensors: &BTreeMap<String, Vec<f32>>,
    change: impl FnOnce(&mut serde_json::Value),
) -> Vec<u8> {
    let mut data = Vec::new();
    let mut header = serde_json::json!({});
    for (name, values) in tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        header[name] = serde_json::json!({"dtype": "F32", "shape": [values.len()], "data_offsets": [start,data.len()]});
    }
    change(&mut header);
    let mut header = serde_json::to_vec(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut encoded = (header.len() as u64).to_le_bytes().to_vec();
    encoded.extend(header);
    encoded.extend(data);
    encoded
}

struct Bundle(std::path::PathBuf);

impl Bundle {
    fn new(mut metadata: PretrainingMetadata, tensors: Vec<u8>) -> Self {
        let tick = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "kindle-world-initialization-{}-{tick}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        metadata.world_sha256 = format!("{:x}", Sha256::digest(&tensors));
        std::fs::write(
            path.join("pretraining.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();
        std::fs::write(path.join("world.safetensors"), tensors).unwrap();
        Self(path)
    }
}

impl Drop for Bundle {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        for name in ["pretraining.json", "world.safetensors"] {
            std::fs::remove_file(self.0.join(name)).unwrap();
        }
        std::fs::remove_dir(&self.0).unwrap();
    }
}

#[test]
fn world_initialization_requires_declared_source_actions_timing_and_complete_training_metadata() {
    let original = metadata();
    validate_metadata(&original, &target(), &original.source).unwrap();
    let invalid: &[fn(&mut PretrainingMetadata)] = &[
        |m| m.format = 3,
        |m| m.architecture = "dreamerv3-visual-features".into(),
        |m| m.meganeura_revision = "different".into(),
        |m| m.dreamerv3_revision = "different".into(),
        |m| m.blade_revision = "different".into(),
        |m| m.source.dataset_sha256 = "d".repeat(64),
        |m| m.source.perception.checkpoint_sha256 = "d".repeat(64),
        |m| m.source.action_names.swap(1, 2),
        |m| m.source.action_ticks = 1,
        |m| m.source.ticks_per_second = 30,
        |m| m.config.loss_scales.policy = 1.0,
        |m| m.config.loss_scales.replay_value = 0.3,
        |m| m.future_head_revision = None,
        |m| m.updates = 0,
        |m| m.updates = u64::MAX,
        |m| m.sampled_observations = 7,
        |m| m.sampled_transitions = 0,
        |m| m.sampled_transitions = 7,
        |m| m.world_sha256 = "invalid".into(),
    ];
    for (index, corrupt) in invalid.iter().enumerate() {
        let mut changed = original.clone();
        corrupt(&mut changed);
        assert!(
            validate_metadata(&changed, &target(), &original.source).is_err(),
            "accepted case {index}"
        );
    }
    let mut incompatible = target();
    incompatible.loss_scales.future_prediction = 0.0;
    assert!(validate_metadata(&original, &incompatible, &original.source).is_err());
}

#[test]
fn bundle_loading_copies_only_world_dynamics_representation_and_predictor() {
    let metadata = metadata();
    let tensors = tensors(&metadata.config);
    let bundle = Bundle::new(metadata.clone(), encode(&tensors, |_| {}));
    let loaded = WorldInitialization::load(&bundle.0, &target(), &metadata.source).unwrap();
    assert_eq!(loaded.provenance.source(), &metadata.source);
    assert_eq!(loaded.provenance.source_updates(), 1);
    assert_eq!(
        loaded.parameters.len(),
        parameter_schema(&target())
            .keys()
            .filter(|name| transferred(name))
            .count()
    );
    for (name, values) in &loaded.parameters {
        assert!(transferred(name));
        assert_eq!(values, &tensors[name]);
    }
    let encoded = serde_json::to_vec(loaded.provenance()).unwrap();
    let decoded: WorldInitializationProvenance = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(loaded.provenance(), &decoded);
    decoded
        .validate(&target(), Some(&metadata.source.perception))
        .unwrap();
    assert!(decoded.validate(&target(), None).is_err());
    let wrong = PerceptionKind::DinoV3.identity("b".repeat(64));
    assert!(decoded.validate(&target(), Some(&wrong)).is_err());
    std::fs::write(bundle.0.join("world.safetensors"), b"changed after export").unwrap();
    let error = WorldInitialization::load(&bundle.0, &target(), &metadata.source)
        .err()
        .unwrap();
    assert!(error.to_string().contains("fingerprint mismatch"));
}

#[test]
fn even_uncopied_heads_and_moments_must_be_complete_finite_and_f32() {
    let schema = parameter_schema(&target());
    let original = tensors(&target());
    let invalid: &[fn(&mut TensorData)] = &[
        |t| {
            t.remove("world.reward.out.weight");
        },
        |t| {
            t.remove("adam_m.world.reward.out.weight");
        },
        |t| {
            t.remove("adam_v.world.reward.out.weight");
        },
        |t| {
            let values = t.remove("world.reward.out.weight").unwrap();
            t.insert("unexpected".into(), values);
        },
        |t| t.get_mut("world.reward.out.weight").unwrap()[0] = f32::NAN,
        |t| t.get_mut("adam_m.world.reward.out.weight").unwrap()[0] = f32::INFINITY,
        |t| t.get_mut("adam_v.world.reward.out.weight").unwrap()[0] = -0.1,
    ];
    for corrupt in invalid {
        let mut tensors = original.clone();
        corrupt(&mut tensors);
        let model = SafeTensorsModel::from_bytes(encode(&tensors, |_| {})).unwrap();
        assert!(validate_tensors(&model, &schema).is_err());
    }
    for field in ["dtype", "shape"] {
        let encoded = encode(&original, |header| {
            let tensor = &mut header["world.reward.out.weight"];
            tensor[field] = match field {
                "dtype" => serde_json::json!("I32"),
                _ => serde_json::json!([1, original["world.reward.out.weight"].len()]),
            };
        });
        let model = SafeTensorsModel::from_bytes(encoded).unwrap();
        assert!(validate_tensors(&model, &schema).is_err());
    }
}

#[test]
fn schema_preflight_is_independent_of_batch_and_time_axes() {
    for mut config in [target(), DreamerConfig::new(18)] {
        config.loss_scales.replay_value = 0.0;
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        config.world_backprop_length = config.batch_length;
        let graph = world::build_pretraining_graph(&config, config.batch_length);
        let full = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
                _ => None,
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(full, parameter_schema(&config));
    }
}
