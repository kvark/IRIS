//! Meganeura session construction, D3 initialization, and model syncing.

use std::collections::HashMap;
use std::sync::Arc;

use meganeura::graph::Op;
use meganeura::{Graph, Mode, Session, SessionConfig};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::config::DreamerConfig;

pub(crate) fn build_session(
    graph: &Graph,
    gpu: &Arc<blade_graphics::Context>,
    mode: Mode,
    skip_full_optimize: bool,
) -> Session {
    meganeura::build(
        graph,
        SessionConfig {
            mode,
            gpu: Some(Arc::clone(gpu)),
            skip_full_optimize: mode == Mode::Training && skip_full_optimize,
            ..SessionConfig::default()
        },
    )
    .0
}

pub(crate) fn initialize_d3(session: &mut Session, graph: &Graph, seed: u64) {
    let shapes = graph
        .nodes()
        .iter()
        .filter_map(|node| match &node.op {
            Op::Parameter { name } => Some((name.as_str(), node.ty.shape.as_slice())),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let parameters = session
        .param_names()
        .into_iter()
        .map(|name| {
            let size = session.param_size(name).expect("known parameter size");
            (name.to_owned(), size)
        })
        .collect::<Vec<_>>();
    for (name, size) in parameters {
        let shape = shapes
            .get(name.as_str())
            .copied()
            .unwrap_or_else(|| panic!("parameter {name} missing from source graph"));
        let values = if name.ends_with(".norm.weight") {
            vec![1.0; size]
        } else if name.ends_with(".bias") {
            vec![0.0; size]
        } else if name == "world.reward.out.weight" || name == "behavior.value.out.weight" {
            // D3 initializes reward and value predictions to exactly zero.
            vec![0.0; size]
        } else {
            let output_scale = if name == "behavior.actor.out.weight" {
                0.01
            } else {
                1.0
            };
            let fan_in = d3_fan_in(shape);
            // Match reconstruction's spatial-head initialization in the causal
            // ablation. Only the trunk's smaller deterministic input differs.
            let initialization_name = match name.strip_prefix("world.future_predictor.") {
                Some(suffix) => format!("world.decoder.{suffix}"),
                None => name.clone(),
            };
            truncated_normal(&initialization_name, size, fan_in, output_scale, seed)
        };
        session.set_parameter(&name, &values);
    }
}

fn d3_fan_in(shape: &[usize]) -> usize {
    match shape {
        [] | [_] => 1,
        [input, _] => *input,
        _ => shape[..shape.len() - 1].iter().product(),
    }
}

fn truncated_normal(
    name: &str,
    size: usize,
    fan_in: usize,
    output_scale: f32,
    seed: u64,
) -> Vec<f32> {
    let mut hash = seed ^ 0xcbf2_9ce4_8422_2325;
    for byte in name.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut rng = StdRng::seed_from_u64(hash);
    let scale = 1.1368 / (fan_in as f32).sqrt() * output_scale;
    let mut output = Vec::with_capacity(size);
    while output.len() < size {
        let u1 = rng.random_range(f32::EPSILON..1.0);
        let u2 = rng.random::<f32>();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        for normal in [radius * angle.cos(), radius * angle.sin()] {
            if normal.abs() <= 2.0 {
                output.push(normal * scale);
                if output.len() == size {
                    break;
                }
            }
        }
    }
    output
}

pub(crate) fn sync_matching(source: &Session, target: &mut Session, prefix: &str) {
    let names = source
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .filter(|name| source.param_size(name) == target.param_size(name))
        .collect::<Vec<_>>();
    let values = source.read_params(&names);
    for (name, values) in names.into_iter().zip(values) {
        target.set_parameter(name, &values);
    }
}

/// Read overlapping weights once, preserving each target's upload order and
/// `set_parameter` handling of backend-derived weight layouts.
pub(crate) fn sync_matching_many(source: &Session, targets: &mut [&mut Session], prefix: &str) {
    let names = source
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .filter(|name| {
            targets
                .iter()
                .any(|target| source.param_size(name) == target.param_size(name))
        })
        .collect::<Vec<_>>();
    let values = source.read_params(&names);
    for target in targets {
        for (name, values) in names.iter().zip(&values) {
            if target.param_size(name) == Some(values.len()) {
                target.set_parameter(name, values);
            }
        }
    }
}

pub(crate) fn ema_matching(source: &Session, target: &mut Session, prefix: &str, rate: f32) {
    assert!((0.0..0.5).contains(&rate));
    let names = source
        .param_names()
        .into_iter()
        .filter(|name| name.starts_with(prefix))
        .filter(|name| source.param_size(name) == target.param_size(name))
        .collect::<Vec<_>>();
    let source_values = source.read_params(&names);
    let target_values = target.read_params(&names);
    for ((name, source_values), mut target_values) in
        names.into_iter().zip(source_values).zip(target_values)
    {
        for (target, source) in target_values.iter_mut().zip(source_values) {
            *target = rate * source + (1.0 - rate) * *target;
        }
        target.set_parameter(name, &target_values);
    }
}

pub(crate) fn configure_d3_optimizer(
    session: &mut Session,
    config: &DreamerConfig,
    learner_step: u64,
    learning_rate: f32,
) {
    session.set_laprop(
        d3_learning_rate(learning_rate, config.learning_rate_warmup, learner_step),
        config.optimizer_beta1,
        config.optimizer_beta2,
        config.optimizer_epsilon,
    );
    session.set_adaptive_grad_clip(config.agc, config.agc_pmin);
}

fn d3_learning_rate(learning_rate: f32, warmup: u64, learner_step: u64) -> f32 {
    let multiplier = if warmup == 0 {
        1.0
    } else {
        (learner_step as f32 / warmup as f32).min(1.0)
    };
    learning_rate * multiplier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_is_deterministic_and_bounded() {
        let left = truncated_normal("weight", 10_000, 100, 1.0, 7);
        let right = truncated_normal("weight", 10_000, 100, 1.0, 7);
        assert_eq!(left, right);
        let bound = 2.0 * 1.1368 / 10.0;
        assert!(left.iter().all(|value| value.abs() <= bound));
    }

    #[test]
    fn truncation_changes_with_experiment_seed() {
        let seed_zero = truncated_normal("weight", 10_000, 100, 1.0, 0);
        let seed_one = truncated_normal("weight", 10_000, 100, 1.0, 1);
        assert_ne!(seed_zero, seed_one);
    }

    #[test]
    fn grouped_block_linear_uses_upstream_fan_in() {
        assert_eq!(d3_fan_in(&[8, 96, 64]), 768);
        assert_eq!(d3_fan_in(&[96, 64]), 96);
    }

    #[test]
    fn learning_rate_warmup_matches_optax_step_indexing() {
        let mut config = DreamerConfig::tiny(2);
        config.learning_rate = 4e-5;
        config.learning_rate_warmup = 1_000;
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 0), 0.0);
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 500), 2e-5);
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 1_000), 4e-5);
        assert_eq!(d3_learning_rate(config.learning_rate, 1_000, 2_000), 4e-5);
    }

    #[test]
    fn world_inference_has_overlapping_logical_weights() {
        use super::super::world;

        let mut config = DreamerConfig::new(18);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        let starts = config.batch_size * config.batch_length;
        let graphs = [
            world::build_observe_graph(&config, config.batch_size),
            world::build_observe_graph(&config, 1),
            world::build_transition_graph(&config, starts),
            world::build_transition_graph(&config, 1),
            world::build_imagination_head_graph(&config, starts),
            world::build_head_graph(&config, 1),
        ];
        let mut repeated = 0;
        let mut unique = HashMap::new();
        for graph in &graphs {
            for node in graph.nodes() {
                if let Op::Parameter { name } = &node.op {
                    let elements: usize = node.ty.shape.iter().product();
                    repeated += elements;
                    if let Some(previous) = unique.insert(name, elements) {
                        assert_eq!(previous, elements, "{name}");
                    }
                }
            }
        }
        let unique: usize = unique.values().sum();
        assert!(repeated > 2 * unique);
        eprintln!(
            "six world inference graphs: repeated={} bytes, union={} bytes; logical F32 weights, not compiled caches or measured transfers",
            4 * repeated,
            4 * unique,
        );
    }

    #[test]
    #[ignore = "requires GPU; compares parameter fan-out with independent serial uploads"]
    fn parameter_fanout_matches_serial_subsets_and_updates() {
        fn graph(parameters: &[(&str, usize)]) -> Graph {
            let mut graph = Graph::new();
            let outputs = parameters
                .iter()
                .map(|(name, size)| graph.parameter(name, &[*size]))
                .collect();
            graph.set_outputs(outputs);
            graph
        }

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let source_graph = graph(&[
            ("world.shared", 4),
            ("world.only_a", 2),
            ("world.only_b", 3),
            ("world.mismatch", 2),
            ("behavior.value", 3),
        ]);
        let targets = [
            graph(&[
                ("world.shared", 4),
                ("world.only_a", 2),
                ("world.mismatch", 3),
                ("behavior.value", 3),
                ("world.target_only", 4),
            ]),
            graph(&[("world.only_b", 3), ("world.shared", 4)]),
        ];
        let mut source = build_session(&source_graph, &gpu, Mode::Inference, false);
        let mut serial: Vec<_> = targets
            .iter()
            .map(|graph| build_session(graph, &gpu, Mode::Inference, false))
            .collect();
        let mut fanout: Vec<_> = targets
            .iter()
            .map(|graph| build_session(graph, &gpu, Mode::Inference, false))
            .collect();
        for session in serial.iter_mut().chain(&mut fanout) {
            for (name, _) in session.plan().param_buffers.clone() {
                let size = session.param_size(&name).unwrap();
                session.set_parameter(&name, &vec![-7.0; size]);
            }
        }
        for update in 1..=2 {
            for (index, node) in source_graph.nodes().iter().enumerate() {
                if let Op::Parameter { name } = &node.op {
                    let values = vec![update as f32 + index as f32 * 0.25; node.ty.shape[0]];
                    source.set_parameter(name, &values);
                }
            }
            let source_before = source.read_params(&source.param_names());
            for target in &mut serial {
                sync_matching(&source, target, "world.");
            }
            let mut targets: Vec<_> = fanout.iter_mut().collect();
            sync_matching_many(&source, &mut targets, "world.");
            sync_matching_many(&source, &mut targets, "absent.");
            sync_matching_many(&source, &mut [], "world.");
            for (serial, fanout) in serial.iter().zip(&fanout) {
                assert_eq!(serial.param_names(), fanout.param_names());
                let names = serial.param_names();
                assert_eq!(serial.read_params(&names), fanout.read_params(&names));
            }
            assert_eq!(source_before, source.read_params(&source.param_names()));
            for (target, names) in [
                (&fanout[0], ["world.shared", "world.only_a"]),
                (&fanout[1], ["world.shared", "world.only_b"]),
            ] {
                assert_eq!(target.read_params(&names), source.read_params(&names));
            }
            for name in ["world.mismatch", "behavior.value", "world.target_only"] {
                let values = fanout[0].read_params(&[name]);
                assert!(values[0].iter().all(|value| *value == -7.0));
            }
        }
    }

    #[test]
    #[ignore = "requires GPU; verifies fan-out refreshes tied Winograd weight caches"]
    fn parameter_fanout_preserves_derived_cache_refresh() {
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let mut graph = Graph::new();
        let input = graph.input("input", &[64 * 8 * 8]);
        let other = graph.input("other", &[64 * 8 * 8]);
        let kernel = graph.parameter("world.kernel:winograd", &[64 * 64 * 9]);
        let first = graph.conv2d(input, kernel, 1, 64, 8, 8, 64, 3, 3, 1, 1);
        let second = graph.conv2d(other, kernel, 1, 64, 8, 8, 64, 3, 3, 1, 1);
        graph.set_outputs(vec![first, second]);
        let mut source = build_session(&graph, &gpu, Mode::Inference, false);
        let mut serial = build_session(&graph, &gpu, Mode::Inference, false);
        let mut first = build_session(&graph, &gpu, Mode::Inference, false);
        let mut second = build_session(&graph, &gpu, Mode::Inference, false);
        assert_eq!(first.plan().derived_params.len(), 1);
        let input: Vec<_> = (0..64 * 8 * 8).map(|i| (i % 31) as f32 / 31.0).collect();
        let other: Vec<_> = input.iter().map(|v| 0.2 - v).collect();
        for update in 1..=2 {
            let weights: Vec<_> = (0..64 * 64 * 9)
                .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.0005 * update as f32)
                .collect();
            source.set_parameter("world.kernel:winograd", &weights);
            sync_matching(&source, &mut serial, "world.");
            sync_matching_many(&source, &mut [&mut first, &mut second], "world.");
            for session in [&mut source, &mut serial, &mut first, &mut second] {
                session.set_input("input", &input);
                session.set_input("other", &other);
                session.step();
                session.wait();
            }
            for index in 0..2 {
                let mut expected = vec![0.0; input.len()];
                serial.read_output_by_index(index, &mut expected);
                assert!(expected.iter().all(|value| value.is_finite()));
                assert!(expected.iter().any(|value| *value != 0.0));
                for session in [&source, &first, &second] {
                    let mut actual = vec![0.0; expected.len()];
                    session.read_output_by_index(index, &mut actual);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
