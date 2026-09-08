//! Require complete training state around the backend's checkpoint loader.

use std::collections::HashSet;
use std::io;
use std::path::Path;

use meganeura::{Session, data::safetensors::SafeTensorsModel};

pub(super) const WORLD: &str = "world.safetensors";
pub(super) const BEHAVIOR: &str = "behavior.safetensors";
pub(super) const SLOW_VALUE: &str = "slow_value.safetensors";

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub(super) struct TensorFingerprints {
    world: String,
    behavior: String,
    slow_value: String,
}

impl TensorFingerprints {
    pub fn read(directory: &Path) -> io::Result<Self> {
        Ok(Self {
            world: crate::vision::checkpoint_sha256(&directory.join(WORLD))?,
            behavior: crate::vision::checkpoint_sha256(&directory.join(BEHAVIOR))?,
            slow_value: crate::vision::checkpoint_sha256(&directory.join(SLOW_VALUE))?,
        })
    }

    pub fn verify(&self, directory: &Path) -> io::Result<()> {
        for (name, expected) in [
            (WORLD, &self.world),
            (BEHAVIOR, &self.behavior),
            (SLOW_VALUE, &self.slow_value),
        ] {
            let actual = crate::vision::checkpoint_sha256(&directory.join(name))?;
            if actual != *expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("checkpoint {name} SHA-256 mismatch; damaged or mixed save"),
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn load_session(
    session: &mut Session,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Even logical-format checkpoints may omit optimizer moments. A training
    // restore must not silently keep their initialized values. Winograd caches
    // are derived execution storage, not saved parameters; identify them from
    // the plan, never from a user parameter's name.
    let model = SafeTensorsModel::load(path.to_path_buf())?;
    let caches: HashSet<_> = session
        .plan()
        .derived_params
        .iter()
        .filter_map(|(buffer, _, transform)| {
            matches!(
                transform,
                meganeura::graph::ParamTransform::Winograd3x3 { .. }
            )
            .then_some(*buffer)
        })
        .collect();
    let mut required = Vec::new();
    for (name, buffer) in &session.plan().param_buffers {
        if caches.contains(buffer) {
            continue;
        }
        required.push(name.to_owned());
        if session.has_param_grad(name) {
            required.push(format!("adam_m.{name}"));
            required.push(format!("adam_v.{name}"));
        }
    }
    check_tensor_names(model.tensor_info().keys().map(String::as_str), &required)?;
    drop(model);
    session.load_checkpoint(path)?;
    Ok(())
}

fn check_tensor_names<'a>(
    present: impl Iterator<Item = &'a str>,
    required: &[String],
) -> io::Result<()> {
    let present = present.collect::<HashSet<_>>();
    for name in required {
        if !present.contains(name.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("checkpoint missing required tensor {name}"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_restore_requires_parameters_and_both_optimizer_moments() {
        let names = ["weight", "adam_m.weight", "adam_v.weight"];
        let required = names.map(str::to_owned);
        check_tensor_names(names.into_iter(), &required).unwrap();
        for omitted in names {
            let error =
                check_tensor_names(names.into_iter().filter(|name| *name != omitted), &required)
                    .unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            assert!(error.to_string().contains(omitted));
        }
    }

    #[test]
    fn inference_restore_may_read_a_subset_of_a_training_checkpoint() {
        check_tensor_names(
            ["weight", "adam_m.weight", "adam_v.weight"].into_iter(),
            &["weight".to_owned()],
        )
        .unwrap();
    }

    #[test]
    #[ignore = "requires GPU; restores logical weights without saved Winograd caches"]
    fn logical_restore_regenerates_shared_winograd_cache() {
        use super::super::runtime::build_session;
        use meganeura::{Graph, Mode};
        use std::sync::Arc;

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = Graph::new();
        let input = graph.input("input", &[64 * 8 * 8]);
        let other = graph.input("other", &[64 * 8 * 8]);
        let kernel = graph.parameter("kernel:winograd", &[64 * 64 * 9]);
        let first = graph.conv2d(input, kernel, 1, 64, 8, 8, 64, 3, 3, 1, 1);
        let second = graph.conv2d(other, kernel, 1, 64, 8, 8, 64, 3, 3, 1, 1);
        graph.set_outputs(vec![first, second]);
        let mut source = build_session(&graph, &gpu, Mode::Inference, false);
        assert_eq!(source.plan().derived_params.len(), 1);
        let weights: Vec<_> = (0..64 * 64 * 9)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.0005)
            .collect();
        source.set_parameter("kernel:winograd", &weights);
        let path = std::env::temp_dir().join(format!(
            "kindle-shared-winograd-{}.safetensors",
            std::process::id()
        ));
        assert!(!path.exists());
        source.save_checkpoint(&path).unwrap();
        let saved = SafeTensorsModel::load(path.clone()).unwrap();
        assert_eq!(saved.tensor_info().len(), 1);
        assert!(saved.tensor_info().contains_key("kernel:winograd"));
        let mut restored = build_session(&graph, &gpu, Mode::Inference, false);
        load_session(&mut restored, &path).unwrap();
        let input: Vec<_> = (0..64 * 8 * 8).map(|i| (i % 31) as f32 / 31.0).collect();
        let other: Vec<_> = input.iter().map(|v| 0.2 - v).collect();
        for session in [&mut source, &mut restored] {
            session.set_input("input", &input);
            session.set_input("other", &other);
            session.step();
            session.wait();
        }
        for index in 0..2 {
            let mut expected = vec![0.0; input.len()];
            let mut actual = expected.clone();
            source.read_output_by_index(index, &mut expected);
            restored.read_output_by_index(index, &mut actual);
            assert_eq!(actual, expected);
        }
        std::fs::remove_file(path).unwrap();
    }
}
