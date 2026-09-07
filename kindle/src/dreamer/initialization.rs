//! Strict dynamics-only initialization, deliberately separate from training restore.

#[cfg(test)]
mod tests;

use std::{collections::BTreeMap, io, path::Path};

use meganeura::{data::safetensors::SafeTensorsModel, graph::Op};
use sha2::{Digest, Sha256};

use super::{
    BLADE_REV, DREAMERV3_UPSTREAM_REV, DreamerConfig, MEGANEURA_REV,
    pretraining::{PretrainingMetadata, PretrainingSource},
    world,
};
use crate::vision::PerceptionIdentity;

const REVISION: &str = "fresh-agent-dynamics-representation-predictor-v1";
type Parameters = Vec<(String, Vec<f32>)>;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldInitializationProvenance {
    revision: String,
    /// Fingerprint of the exact source metadata bytes read during initialization.
    metadata_sha256: String,
    metadata: PretrainingMetadata,
}

impl WorldInitializationProvenance {
    pub fn source(&self) -> &PretrainingSource {
        &self.metadata.source
    }
    pub fn source_updates(&self) -> u64 {
        self.metadata.updates
    }

    pub(super) fn validate(
        &self,
        target: &DreamerConfig,
        perception: Option<&PerceptionIdentity>,
    ) -> io::Result<()> {
        require(
            self.revision == REVISION && valid_hash(&self.metadata_sha256),
            "invalid world-initialization provenance",
        )?;
        validate_metadata(&self.metadata, target, self.source())?;
        require(
            perception == Some(&self.source().perception),
            "world initialization and target perception differ",
        )
    }
}

/// CPU-validated weights bound to one target configuration and explicit source
/// contract. Load this before constructing a GPU agent. It can be applied only
/// once to a new runtime, never as a permissive partial checkpoint restore.
pub struct WorldInitialization {
    pub(super) target: DreamerConfig,
    pub(super) provenance: WorldInitializationProvenance,
    pub(super) parameters: Parameters,
}

impl WorldInitialization {
    pub fn provenance(&self) -> &WorldInitializationProvenance {
        &self.provenance
    }

    pub fn load(
        directory: impl AsRef<Path>,
        target: &DreamerConfig,
        expected: &PretrainingSource,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let directory = directory.as_ref();
        let encoded = std::fs::read(directory.join("pretraining.json"))?;
        let metadata: PretrainingMetadata = serde_json::from_slice(&encoded)?;
        validate_metadata(&metadata, target, expected)?;
        let weights = std::fs::read(directory.join("world.safetensors"))?;
        require(
            format!("{:x}", Sha256::digest(&weights)) == metadata.world_sha256,
            "pretraining tensor fingerprint mismatch",
        )?;
        // Parse and hash the same bytes. Do not reopen the file after validation.
        let model = SafeTensorsModel::from_bytes(weights)?;
        let parameters = validate_tensors(&model, &parameter_schema(&metadata.config))?;
        let provenance = WorldInitializationProvenance {
            revision: REVISION.into(),
            metadata_sha256: format!("{:x}", Sha256::digest(&encoded)),
            metadata,
        };
        provenance.validate(target, Some(&expected.perception))?;
        Ok(Self {
            target: target.clone(),
            provenance,
            parameters,
        })
    }
}

fn validate_tensors(
    model: &SafeTensorsModel,
    schema: &BTreeMap<String, Vec<usize>>,
) -> Result<Parameters, Box<dyn std::error::Error>> {
    require(
        model.tensor_info().len() == 3 * schema.len(),
        "incomplete or unexpected pretraining tensors",
    )?;
    let mut parameters = Vec::new();
    for (name, shape) in schema {
        for prefix in ["", "adam_m.", "adam_v."] {
            let key = format!("{prefix}{name}");
            let info = model.tensor_info().get(&key).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing pretraining tensor {key}"),
                )
            })?;
            // The pinned backend exports flattened F32 parameter/moment buffers.
            require(
                info.shape == [shape.iter().product::<usize>()],
                &format!("wrong exported tensor shape: {key}"),
            )?;
            let values = model.tensor_f32(&key)?;
            require(
                values.iter().all(|value| value.is_finite()),
                &format!("nonfinite pretraining tensor: {key}"),
            )?;
            if prefix == "adam_v." {
                require(
                    values.iter().all(|&value| value >= 0.0),
                    &format!("negative second moment: {key}"),
                )?;
            }
            if prefix.is_empty() && transferred(name) {
                parameters.push((name.clone(), values));
            }
        }
    }
    require(!parameters.is_empty(), "no compatible world parameters")?;
    Ok(parameters)
}

fn require(condition: bool, message: &str) -> io::Result<()> {
    if condition {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidData, message))
    }
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

pub(super) fn transferred(name: &str) -> bool {
    [
        "world.dynamics.",
        "world.representation.",
        "world.future_predictor.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

fn parameter_schema(config: &DreamerConfig) -> BTreeMap<String, Vec<usize>> {
    // Parameter shapes do not depend on row/sequence counts. Avoid constructing
    // a production-length graph merely to inspect its weight schema.
    let mut config = config.clone();
    config.batch_size = 1;
    config.batch_length = 2;
    config.world_backprop_length = 1;
    config.world_microbatch_size = Some(1);
    config.loss_scales.replay_value = 0.0;
    let graph = world::build_pretraining_graph(&config, 1);
    graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Parameter { name } => Some((name.clone(), node.ty.shape.clone())),
            _ => None,
        })
        .collect()
}

fn validate_metadata(
    metadata: &PretrainingMetadata,
    target: &DreamerConfig,
    expected: &PretrainingSource,
) -> io::Result<()> {
    metadata.config.check().map_err(io::Error::other)?;
    target.check().map_err(io::Error::other)?;
    metadata.source.validate(&metadata.config)?;
    expected.validate(target)?;
    require(
        metadata.format == 1 && metadata.architecture == "dreamerv3-world-pretraining",
        "unsupported pretraining artifact",
    )?;
    require(
        metadata.dreamerv3_revision == DREAMERV3_UPSTREAM_REV
            && metadata.meganeura_revision == MEGANEURA_REV
            && metadata.blade_revision == BLADE_REV,
        "pretraining backend or algorithm revision differs",
    )?;
    require(
        &metadata.source == expected,
        "pretraining source, encoder, action order or timing contract differs",
    )?;
    let config = &metadata.config;
    require(
        config.train_ratio == 0.0
            && config.loss_scales.policy == 0.0
            && config.loss_scales.value == 0.0
            && config.loss_scales.replay_value == 0.0
            && !config.visitation_bonus
            && config.intrinsic_reward_scale == 0.0
            && config.extrinsic_reward_scale == 1.0,
        "not a world-only source configuration",
    )?;
    require(
        metadata.future_head_revision.as_deref()
            == (config.loss_scales.future_prediction > 0.0).then_some(world::FUTURE_HEAD_REVISION),
        "pretraining predictor revision differs",
    )?;
    let rows = metadata.updates.checked_mul(config.batch_size as u64);
    let observations = rows.and_then(|rows| rows.checked_mul(config.batch_length as u64));
    require(
        metadata.updates > 0
            && observations == Some(metadata.sampled_observations)
            && metadata.sampled_transitions > 0
            && rows.is_some_and(|rows| {
                metadata.sampled_transitions <= metadata.sampled_observations - rows
            }),
        "inconsistent or empty offline training counters",
    )?;
    require(
        valid_hash(&metadata.world_sha256),
        "invalid pretraining tensor fingerprint",
    )?;
    let selected = |config| {
        parameter_schema(config)
            .into_iter()
            .filter(|(name, _)| transferred(name))
            .collect::<BTreeMap<_, _>>()
    };
    require(
        selected(config) == selected(target),
        "source and target dynamics/representation/predictor schemas differ",
    )
}
