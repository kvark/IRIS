//! Categorical DreamerV3 world model graphs.

use meganeura::{Graph, graph::NodeId};

use super::config::DreamerConfig;
use super::networks::{
    MlpHead, ObservationDecoder, Prior, Representation, RssmCore, categorical_kl, feature,
    mixed_probabilities, scale, slice_columns, straight_through_sample, sum,
    weighted_cross_entropy,
};
use crate::vision::{OBSERVATION_CHANNELS, OBSERVATION_GRID};

pub const LOSS_TOTAL: usize = 0;
pub const LOSS_RECONSTRUCTION: usize = 1;
pub const LOSS_DYNAMICS: usize = 2;
pub const LOSS_REPRESENTATION: usize = 3;
pub const LOSS_REWARD: usize = 4;
pub const LOSS_CONTINUATION: usize = 5;
pub const LOSS_REPLAY_VALUE: usize = 6;
pub const RAW_KL: usize = 7;
pub const LOSS_FUTURE_PREDICTION: usize = 8;
pub const FUTURE_HEAD_REVISION: &str = "spatial-deterministic-v1";

struct Dynamics {
    core: RssmCore,
    prior: Prior,
}

impl Dynamics {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        Self {
            core: RssmCore::new(graph, config),
            prior: Prior::new(graph, config),
        }
    }
}

struct WorldHeads {
    reward: MlpHead,
    continuation: MlpHead,
}

impl WorldHeads {
    fn new(graph: &mut Graph, config: &DreamerConfig) -> Self {
        let units = config.network().units;
        Self {
            reward: MlpHead::new(
                graph,
                "world.reward",
                config.feature_dim(),
                units,
                1,
                config.value_bins,
            ),
            continuation: MlpHead::new(
                graph,
                "world.continuation",
                config.feature_dim(),
                units,
                1,
                1,
            ),
        }
    }

    fn forward(&self, graph: &mut Graph, state: NodeId) -> (NodeId, NodeId) {
        let reward = self.reward.forward(graph, state);
        let continuation = self.continuation.forward(graph, state);
        let continuation = graph.sigmoid(continuation);
        (reward, continuation)
    }
}

struct WorldModel {
    dynamics: Dynamics,
    representation: Representation,
    decoder: Option<ObservationDecoder>,
    future_predictor: Option<ObservationDecoder>,
    heads: WorldHeads,
    /// The behavior optimizer owns these parameters. They are frozen in this
    /// graph so replay-value gradients only shape the posterior/RSSM path.
    replay_value: Option<MlpHead>,
}

impl WorldModel {
    fn new(graph: &mut Graph, config: &DreamerConfig, pretraining: bool) -> Self {
        let units = config.network().units;
        Self {
            dynamics: Dynamics::new(graph, config),
            representation: Representation::new(graph, config),
            decoder: (config.loss_scales.reconstruction > 0.0).then(|| {
                ObservationDecoder::new(graph, config, "world.decoder", config.feature_dim())
            }),
            future_predictor: (config.loss_scales.future_prediction > 0.0)
                .then(|| future_predictor(graph, config)),
            heads: WorldHeads::new(graph, config),
            replay_value: (!pretraining).then(|| {
                MlpHead::new(
                    graph,
                    "behavior.value",
                    config.feature_dim(),
                    units,
                    3,
                    config.value_bins,
                )
            }),
        }
    }
}

fn future_predictor(graph: &mut Graph, config: &DreamerConfig) -> ObservationDecoder {
    ObservationDecoder::new(
        graph,
        config,
        "world.future_predictor",
        config.network().deter,
    )
}

/// Full sequence loss with externally sampled hard posterior states.
///
/// The hard samples are produced immediately before this graph runs by the
/// observe graph. Inside this graph they are combined with posterior
/// probabilities using `hard + probs - stop_gradient(probs)`, exactly D3's
/// straight-through categorical estimator.
pub fn build_training_graph(config: &DreamerConfig, length: usize) -> Graph {
    build_training_graph_grouped(config, length, length)
}

/// Action-conditioned offline learning, without a critic or replay-value targets.
///
/// Each reward/continuation target has an explicit nonnegative weight. Missing
/// labels use weight zero, not a fabricated zero reward or nonterminal target.
/// Losses average over B*T, including masked rows, as the online objectives do.
pub(super) fn build_pretraining_graph(config: &DreamerConfig, length: usize) -> Graph {
    assert_eq!(config.loss_scales.replay_value, 0.0);
    build_sequence_graph(config, length, length, true)
}

// The RSSM and posterior stay sequential. Only row-independent operations are
// grouped across time; their RMS norms never mix rows. A one-step grouping is
// retained privately as the numerical reference for this execution change.
fn build_training_graph_grouped(
    config: &DreamerConfig,
    length: usize,
    time_batch_length: usize,
) -> Graph {
    build_sequence_graph(config, length, time_batch_length, false)
}

fn build_sequence_graph(
    config: &DreamerConfig,
    length: usize,
    time_batch_length: usize,
    pretraining: bool,
) -> Graph {
    config.validate();
    assert!(length > 0 && length <= config.batch_length);
    assert!(time_batch_length > 0 && length.is_multiple_of(time_batch_length));
    let batch = config.batch_size;
    let size = config.network();
    let patches = OBSERVATION_GRID * OBSERVATION_GRID;

    let mut graph = Graph::new();
    let model = WorldModel::new(&mut graph, config, pretraining);
    let mut deter = graph.input("initial_deter", &[batch, size.deter]);
    let mut stoch = graph.input("initial_stoch", &[batch * size.stoch, size.classes]);

    let observations = (0..length)
        .map(|time| {
            graph.input(
                &format!("observation_{time}"),
                &[batch * patches, OBSERVATION_CHANNELS],
            )
        })
        .collect::<Vec<_>>();
    let mut encodings = Vec::with_capacity(length);
    for chunk in observations.chunks(time_batch_length) {
        let observation = stack_time(&mut graph, chunk, batch, config.observation_dim());
        let encoded =
            model
                .representation
                .encoder
                .forward(&mut graph, observation, batch * chunk.len());
        split_time(
            &mut graph,
            encoded,
            chunk.len(),
            batch,
            model.representation.encoder.output_dim(),
            &mut encodings,
        );
    }
    let mut head_inputs = Vec::with_capacity(length);

    let mut reconstruction_losses = Vec::with_capacity(length);
    let mut future_prediction_losses = Vec::with_capacity(length);
    let mut dynamics_losses = Vec::with_capacity(length);
    let mut representation_losses = Vec::with_capacity(length);
    let mut raw_kl_metrics = Vec::with_capacity(length);
    let mut reward_losses = Vec::with_capacity(length);
    let mut continuation_losses = Vec::with_capacity(length);
    let mut replay_value_losses = Vec::with_capacity(length);

    for time in 0..length {
        let action = graph.input(
            &format!("previous_action_{time}"),
            &[batch, config.action_count],
        );
        let keep_deter = graph.input(&format!("keep_deter_{time}"), &[batch, size.deter]);
        let keep_stoch = graph.input(
            &format!("keep_stoch_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let keep_action = graph.input(
            &format!("keep_action_{time}"),
            &[batch, config.action_count],
        );
        let hard_sample = graph.input(
            &format!("posterior_sample_{time}"),
            &[batch * size.stoch, size.classes],
        );
        let masked_deter = graph.mul(deter, keep_deter);
        let masked_stoch = graph.mul(stoch, keep_stoch);
        let masked_action = graph.mul(action, keep_action);
        deter = model.dynamics.core.forward(
            &mut graph,
            masked_deter,
            masked_stoch,
            masked_action,
            batch,
        );
        let prior_logits = model.dynamics.prior.forward(&mut graph, deter, batch);
        let posterior_logits =
            model
                .representation
                .posterior
                .forward(&mut graph, deter, encodings[time], batch);
        let probabilities = mixed_probabilities(
            &mut graph,
            posterior_logits,
            batch * size.stoch,
            size.classes,
            config.unimix,
        );
        stoch = straight_through_sample(&mut graph, hard_sample, probabilities);

        let state = feature(&mut graph, deter, stoch, batch, config);
        head_inputs.push(HeadInputs {
            observation: observations[time],
            deter,
            state,
            keep_action,
        });

        let dynamics_kl = categorical_kl(
            &mut graph,
            posterior_logits,
            prior_logits,
            batch,
            size.stoch,
            size.classes,
            config.unimix,
            true,
            false,
            config.dynamics_free_nats(),
        );
        dynamics_losses.push(dynamics_kl.loss);
        raw_kl_metrics.push(dynamics_kl.raw);
        let representation_kl = categorical_kl(
            &mut graph,
            posterior_logits,
            prior_logits,
            batch,
            size.stoch,
            size.classes,
            config.unimix,
            false,
            true,
            config.free_nats,
        );
        representation_losses.push(representation_kl.loss);
    }

    for (group, chunk) in head_inputs.chunks(time_batch_length).enumerate() {
        let rows = batch * chunk.len();
        let HeadInputs {
            observation,
            deter,
            state,
            keep_action,
        } = HeadInputs::stack(&mut graph, chunk, config);
        let mut target = |name, width| {
            let inputs = (group * time_batch_length..(group + 1) * time_batch_length)
                .map(|time| graph.input(&format!("{name}_{time}"), &[batch, width]))
                .collect::<Vec<_>>();
            stack_time(&mut graph, &inputs, batch, width)
        };
        let reward_target = target("reward_target", config.value_bins);
        let continuation_target = target("continuation_target", 1);
        let replay_targets = (!pretraining).then(|| {
            (
                target("replay_value_target", config.value_bins),
                target("replay_slow_target", config.value_bins),
                target("replay_value_weight", 1),
            )
        });
        let label_weights =
            pretraining.then(|| (target("reward_weight", 1), target("continuation_weight", 1)));
        let reward_weight = match label_weights {
            Some((reward, _)) => reward,
            None => graph.constant(vec![1.0; rows], &[rows, 1]),
        };

        if let Some(predictor) = &model.future_predictor {
            // Each row uses deter_t, before posterior_t consumes observation_t.
            let prediction = predictor.forward(&mut graph, deter, rows);
            let prediction = graph.reshape(prediction, &[rows, config.observation_dim()]);
            let target = graph.stop_gradient(observation);
            let negative_target = graph.neg(target);
            let residual = graph.add(prediction, negative_target);
            let squared = graph.mul(residual, residual);
            let per_row = graph.sum_inner(squared);
            // Reset observations have no preceding action-conditioned state.
            // Keep valid sequence/microbatch boundaries and normalize over B*T.
            let keep = graph.sum_inner(keep_action);
            let normalization =
                graph.constant(vec![1.0 / config.action_count as f32; rows], &[rows, 1]);
            let keep = graph.mul(keep, normalization);
            let masked = graph.mul(per_row, keep);
            future_prediction_losses.push(graph.mean_all(masked));
        }
        if let Some(decoder) = &model.decoder {
            let reconstruction = decoder.forward(&mut graph, state, rows);
            let reconstruction = graph.reshape(reconstruction, &[rows, config.observation_dim()]);
            let negative_target = graph.neg(observation);
            let residual = graph.add(reconstruction, negative_target);
            let squared = graph.mul(residual, residual);
            let reconstruction_loss = graph.sum_all(squared);
            // Sum event dimensions and average B*T, as in the D3 control.
            reconstruction_losses.push(scale(&mut graph, reconstruction_loss, 1.0 / rows as f32));
        }

        let (reward, continuation) = if pretraining {
            (
                model.heads.reward.forward(&mut graph, state),
                model.heads.continuation.forward(&mut graph, state),
            )
        } else {
            model.heads.forward(&mut graph, state)
        };
        // Keep the reduction explicit because this value is composed into the
        // joint loss and exported as a metric. The fused backend loss stores
        // per-row partials and is only scalar when used as a terminal output.
        reward_losses.push(weighted_cross_entropy(
            &mut graph,
            reward,
            reward_target,
            reward_weight,
        ));
        let continuation_loss = if let Some((_, weight)) = label_weights {
            weighted_binary_cross_entropy(&mut graph, continuation, continuation_target, weight)
        } else {
            mean_binary_cross_entropy(&mut graph, continuation, continuation_target, rows)
        };
        continuation_losses.push(continuation_loss);
        if let Some((replay_value_target, replay_slow_target, replay_value_weight)) = replay_targets
        {
            let replay_value_state = if config.replay_value_gradient {
                state
            } else {
                graph.stop_gradient(state)
            };
            let value = model
                .replay_value
                .as_ref()
                .unwrap()
                .forward_frozen(&mut graph, replay_value_state);
            let value_target =
                weighted_cross_entropy(&mut graph, value, replay_value_target, replay_value_weight);
            let slow_target =
                weighted_cross_entropy(&mut graph, value, replay_slow_target, replay_value_weight);
            replay_value_losses.push(graph.add(value_target, slow_target));
        }
    }

    let average = 1.0 / length as f32;
    let head_average = time_batch_length as f32 * average;
    let reconstruction = sum_or_zero(&mut graph, &reconstruction_losses);
    let reconstruction = scale(&mut graph, reconstruction, head_average);
    let future_prediction = sum_or_zero(&mut graph, &future_prediction_losses);
    let future_prediction = scale(&mut graph, future_prediction, head_average);
    let dynamics = sum(&mut graph, &dynamics_losses);
    let dynamics = scale(&mut graph, dynamics, average);
    let representation = sum(&mut graph, &representation_losses);
    let representation = scale(&mut graph, representation, average);
    let raw_kl = sum(&mut graph, &raw_kl_metrics);
    let raw_kl = scale(&mut graph, raw_kl, average);
    let reward = sum(&mut graph, &reward_losses);
    let reward = scale(&mut graph, reward, head_average);
    let continuation = sum(&mut graph, &continuation_losses);
    let continuation = scale(&mut graph, continuation, head_average);
    let replay_value = sum_or_zero(&mut graph, &replay_value_losses);
    let replay_value = scale(&mut graph, replay_value, head_average);

    let scales = config.loss_scales;
    let weighted = [
        scale(&mut graph, reconstruction, scales.reconstruction),
        scale(&mut graph, future_prediction, scales.future_prediction),
        scale(&mut graph, dynamics, scales.dynamics),
        scale(&mut graph, representation, scales.representation),
        scale(&mut graph, reward, scales.reward),
        scale(&mut graph, continuation, scales.continuation),
        scale(&mut graph, replay_value, scales.replay_value),
    ];
    let total = sum(&mut graph, &weighted);
    graph.set_outputs(vec![
        total,
        reconstruction,
        dynamics,
        representation,
        reward,
        continuation,
        replay_value,
        raw_kl,
        future_prediction,
    ]);
    graph
}

struct HeadInputs {
    observation: NodeId,
    deter: NodeId,
    state: NodeId,
    keep_action: NodeId,
}

fn mean_binary_cross_entropy(
    graph: &mut Graph,
    prediction: NodeId,
    target: NodeId,
    rows: usize,
) -> NodeId {
    // The backend BCE kernel emits one partial per 256 rows, not a final
    // scalar reduction. Compose only scalar-sized pieces into the joint loss.
    let mut losses = Vec::with_capacity(rows.div_ceil(256));
    for start in (0..rows).step_by(256) {
        let count = (rows - start).min(256);
        let prediction = slice_columns(graph, prediction, 1, rows, start, count);
        let target = slice_columns(graph, target, 1, rows, start, count);
        let loss = graph.bce_loss(prediction, target);
        losses.push(scale(graph, loss, count as f32 / rows as f32));
    }
    sum(graph, &losses)
}

fn weighted_binary_cross_entropy(
    graph: &mut Graph,
    logits: NodeId,
    target: NodeId,
    weight: NodeId,
) -> NodeId {
    // BCE from logits has a scalar reduction for any row count. Unlike the
    // online fused probability loss, individual unlabeled rows can be masked.
    let softplus = graph.softplus(logits, 1.0);
    let product = graph.mul(logits, target);
    let negative_product = graph.neg(product);
    let per_row = graph.add(softplus, negative_product);
    let weighted = graph.mul(per_row, weight);
    graph.mean_all(weighted)
}

impl HeadInputs {
    fn stack(graph: &mut Graph, steps: &[Self], config: &DreamerConfig) -> Self {
        let mut stack = |field: fn(&Self) -> NodeId, width| {
            let values = steps.iter().map(field).collect::<Vec<_>>();
            stack_time(graph, &values, config.batch_size, width)
        };
        Self {
            observation: stack(|step| step.observation, config.observation_dim()),
            deter: stack(|step| step.deter, config.network().deter),
            state: stack(|step| step.state, config.feature_dim()),
            keep_action: stack(|step| step.keep_action, config.action_count),
        }
    }
}

// Balanced trees avoid quadratic copying/padding in forward and backward.
// Row order is time-major throughout: all B rows at t, then all B rows at t+1.
fn stack_time(graph: &mut Graph, steps: &[NodeId], batch: usize, width: usize) -> NodeId {
    assert!(!steps.is_empty());
    if steps.len() == 1 {
        return graph.reshape(steps[0], &[batch, width]);
    }
    let middle = steps.len() / 2;
    let left = stack_time(graph, &steps[..middle], batch, width);
    let right = stack_time(graph, &steps[middle..], batch, width);
    let left_rows = middle * batch;
    let right_rows = (steps.len() - middle) * batch;
    let left = graph.reshape(left, &[left_rows * width]);
    let right = graph.reshape(right, &[right_rows * width]);
    let joined = graph.concat(
        left,
        right,
        1,
        left_rows as u32,
        right_rows as u32,
        width as u32,
    );
    graph.reshape(joined, &[steps.len() * batch, width])
}

fn split_time(
    graph: &mut Graph,
    input: NodeId,
    length: usize,
    batch: usize,
    width: usize,
    output: &mut Vec<NodeId>,
) {
    assert!(length > 0);
    if length == 1 {
        output.push(graph.reshape(input, &[batch, width]));
        return;
    }
    let middle = length / 2;
    let left_rows = middle * batch;
    let right_rows = (length - middle) * batch;
    let input = graph.reshape(input, &[length * batch * width]);
    let left = graph.split_a(input, 1, left_rows as u32, right_rows as u32, width as u32);
    let right = graph.split_b(input, 1, left_rows as u32, right_rows as u32, width as u32);
    split_time(graph, left, middle, batch, width, output);
    split_time(graph, right, length - middle, batch, width, output);
}

fn sum_or_zero(graph: &mut Graph, values: &[NodeId]) -> NodeId {
    if values.is_empty() {
        graph.constant(vec![0.0], &[1])
    } else {
        sum(graph, values)
    }
}

/// One posterior update used both by the live actor and the pre-training
/// categorical sampling pass.
///
/// Outputs: next deterministic state, posterior logits, prior logits, and the
/// trainable observation-encoder output used by the posterior.
pub fn build_observe_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let patches = OBSERVATION_GRID * OBSERVATION_GRID;
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let representation = Representation::new(&mut graph, config);
    let previous_deter = graph.input("previous_deter", &[batch, size.deter]);
    let previous_stoch = graph.input("previous_stoch", &[batch * size.stoch, size.classes]);
    let previous_action = graph.input("previous_action", &[batch, config.action_count]);
    let observation = graph.input("observation", &[batch * patches, OBSERVATION_CHANNELS]);
    let keep_deter = graph.input("keep_deter", &[batch, size.deter]);
    let keep_stoch = graph.input("keep_stoch", &[batch * size.stoch, size.classes]);
    let keep_action = graph.input("keep_action", &[batch, config.action_count]);
    let previous_deter = graph.mul(previous_deter, keep_deter);
    let previous_stoch = graph.mul(previous_stoch, keep_stoch);
    let previous_action = graph.mul(previous_action, keep_action);
    let deter = dynamics.core.forward(
        &mut graph,
        previous_deter,
        previous_stoch,
        previous_action,
        batch,
    );
    let prior = dynamics.prior.forward(&mut graph, deter, batch);
    let encoded = representation
        .encoder
        .forward(&mut graph, observation, batch);
    let posterior = representation
        .posterior
        .forward(&mut graph, deter, encoded, batch);
    graph.set_outputs(vec![deter, posterior, prior, encoded]);
    graph
}

/// One prior transition for imagination.
///
/// Outputs: next deterministic state and prior logits. Categorical sampling
/// happens on the CPU between this graph and the next imagined step.
pub fn build_transition_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let dynamics = Dynamics::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let action = graph.input("action", &[batch, config.action_count]);
    let next_deter = dynamics
        .core
        .forward(&mut graph, deter, stoch, action, batch);
    let prior = dynamics.prior.forward(&mut graph, next_deter, batch);
    graph.set_outputs(vec![next_deter, prior]);
    graph
}

/// Reward and continuation predictions for posterior or imagined states.
pub fn build_head_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    let heads = WorldHeads::new(&mut graph, config);
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let state = feature(&mut graph, deter, stoch, batch, config);
    let (reward, continuation) = heads.forward(&mut graph, state);
    graph.set_outputs(vec![reward, continuation]);
    graph
}

/// Evaluate the future head when enabled, otherwise posterior reconstruction.
/// Forecasts read only the deterministic state, including in auxiliary runs.
/// This graph is constructed lazily by diagnostics.
pub fn build_observation_prediction_graph(config: &DreamerConfig, batch: usize) -> Graph {
    config.validate();
    assert!(batch > 0);
    let size = config.network();
    let mut graph = Graph::new();
    if config.loss_scales.future_prediction > 0.0 {
        let predictor = future_predictor(&mut graph, config);
        let deter = graph.input("deter", &[batch, size.deter]);
        graph.input("stoch", &[batch * size.stoch, size.classes]);
        let observation = predictor.forward(&mut graph, deter, batch);
        graph.set_outputs(vec![observation]);
        return graph;
    }
    assert!(
        config.loss_scales.reconstruction > 0.0,
        "no observation prediction head"
    );
    let decoder =
        ObservationDecoder::new(&mut graph, config, "world.decoder", config.feature_dim());
    let deter = graph.input("deter", &[batch, size.deter]);
    let stoch = graph.input("stoch", &[batch * size.stoch, size.classes]);
    let state = feature(&mut graph, deter, stoch, batch, config);
    let observation = decoder.forward(&mut graph, state, batch);
    graph.set_outputs(vec![observation]);
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_graphs_expose_d3_shapes() {
        let config = DreamerConfig::tiny(3);
        let size = config.network();
        let training = build_training_graph(&config, config.world_backprop_length);
        assert_eq!(training.outputs().len(), 9);
        let grouped_weight = training
            .nodes()
            .iter()
            .find_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name }
                    if name == "world.dynamics.core.dynhid0.weight" =>
                {
                    Some(node.ty.shape.as_slice())
                }
                _ => None,
            })
            .expect("grouped RSSM weight");
        assert_eq!(
            grouped_weight,
            [
                size.blocks,
                size.deter / size.blocks + 3 * size.hidden,
                size.deter / size.blocks,
            ]
        );
        let observe = build_observe_graph(&config, 2);
        assert_eq!(
            observe.node(observe.outputs()[0]).ty.shape,
            vec![2, size.deter]
        );
        assert_eq!(
            observe.node(observe.outputs()[1]).ty.shape,
            vec![2 * size.stoch, size.classes]
        );
        assert_eq!(
            observe.node(observe.outputs()[3]).ty.shape,
            vec![2, OBSERVATION_GRID * OBSERVATION_GRID * size.vision_depth]
        );
        let transition = build_transition_graph(&config, 5);
        assert_eq!(
            transition.node(transition.outputs()[0]).ty.shape,
            vec![5, size.deter]
        );
        let heads = build_head_graph(&config, 5);
        assert_eq!(
            heads.node(heads.outputs()[0]).ty.shape,
            vec![5, config.value_bins]
        );
        assert_eq!(heads.node(heads.outputs()[1]).ty.shape, vec![5, 1]);
        let decoder = build_observation_prediction_graph(&config, 5);
        assert_eq!(
            decoder.node(decoder.outputs()[0]).ty.shape,
            vec![
                5 * OBSERVATION_GRID * OBSERVATION_GRID,
                OBSERVATION_CHANNELS
            ]
        );
    }

    #[test]
    fn prediction_only_omits_decoder_parameters() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.reconstruction = 0.0;
        config.loss_scales.future_prediction = 0.25;
        let graph = build_training_graph(&config, config.world_backprop_length);
        let names = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name } => Some(name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(!names.iter().any(|name| name.starts_with("world.decoder.")));
        assert!(
            names
                .iter()
                .any(|name| name.starts_with("world.future_predictor."))
        );
    }

    #[test]
    fn future_and_reconstruction_heads_share_spatial_parameterization() {
        let mut config = DreamerConfig::tiny(3);
        config.loss_scales.future_prediction = 0.25;
        let graph = build_training_graph(&config, config.world_backprop_length);
        let parameters = graph
            .nodes()
            .iter()
            .filter_map(|node| match &node.op {
                meganeura::graph::Op::Parameter { name } => Some((name, &node.ty.shape)),
                _ => None,
            })
            .collect::<std::collections::HashMap<_, _>>();
        for (name, shape) in &parameters {
            if let Some(suffix) = name.strip_prefix("world.decoder.") {
                let future = parameters[&format!("world.future_predictor.{suffix}")];
                if suffix == "trunk.weight" {
                    assert_eq!(future[0], config.network().deter);
                    assert_eq!(shape[0], config.feature_dim());
                    assert_eq!(future[1..], shape[1..]);
                } else {
                    assert_eq!(future, *shape);
                }
            }
        }
    }

    #[test]
    fn tiny_world_training_graph_autodiffs_and_compiles() {
        let config = DreamerConfig::tiny(3);
        let graph = build_training_graph(&config, config.world_backprop_length);
        let (plan, _) = meganeura::compile_training_graph(&graph);
        assert!(!plan.dispatches.is_empty());
    }

    #[test]
    #[ignore = "checks masked continuation loss and gradients beyond one GPU workgroup"]
    fn masked_continuation_loss_has_zero_unlabeled_gradient_and_complete_reduction() {
        use super::super::runtime::build_session;
        use meganeura::Mode;
        use std::sync::Arc;

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        for rows in [513, 1024] {
            let mut graph = Graph::new();
            let logits = graph.parameter("logits", &[rows, 1]);
            let target = graph.input("target", &[rows, 1]);
            let weight = graph.input("weight", &[rows, 1]);
            let loss = weighted_binary_cross_entropy(&mut graph, logits, target, weight);
            graph.set_outputs(vec![loss]);
            let mut session = build_session(&graph, &gpu, Mode::Training, false);
            let logits = (0..rows)
                .map(|row| (row as f32 * 0.17).sin() * 40.0)
                .collect::<Vec<_>>();
            let target = (0..rows)
                .map(|row| (row % 7) as f32 / 6.0)
                .collect::<Vec<_>>();
            let weights = (0..rows)
                .map(|row| f32::from(row % 3 != 0))
                .collect::<Vec<_>>();
            session.set_parameter("logits", &logits);
            session.set_input("target", &target);
            session.set_input("weight", &weights);
            session.step();
            session.wait();
            let mut actual_loss = [0.0];
            let mut gradient = vec![0.0; rows];
            session.read_output_by_index(0, &mut actual_loss);
            session.read_param_grad("logits", &mut gradient);
            let mut expected_loss = 0.0_f64;
            for row in 0..rows {
                let x = f64::from(logits[row]);
                let target = f64::from(target[row]);
                let weight = f64::from(weights[row]);
                expected_loss +=
                    weight * (x.max(0.0) + (-x.abs()).exp().ln_1p() - x * target) / rows as f64;
                let expected = weight * (1.0 / (1.0 + (-x).exp()) - target) / rows as f64;
                assert!((f64::from(gradient[row]) - expected).abs() < 1e-8);
                if weight == 0.0 {
                    assert_eq!(gradient[row], 0.0);
                }
            }
            assert!((f64::from(actual_loss[0]) - expected_loss).abs() < 1e-5);
        }
    }

    #[test]
    #[ignore = "checks a composed continuation loss beyond one GPU workgroup"]
    fn large_composed_continuation_loss_covers_every_row() {
        use super::super::runtime::build_session;
        use meganeura::Mode;
        use std::sync::Arc;

        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        for rows in [513, 1024] {
            let mut graph = Graph::new();
            let logits = graph.parameter("logits", &[rows, 1]);
            let prediction = graph.sigmoid(logits);
            let target = graph.input("target", &[rows, 1]);
            let loss = mean_binary_cross_entropy(&mut graph, prediction, target, rows);
            let loss = scale(&mut graph, loss, 0.3);
            graph.set_outputs(vec![loss]);
            let mut session = build_session(&graph, &gpu, Mode::Training, false);
            let logits = (0..rows)
                .map(|row| (row as f32 * 0.17).sin() * 2.0)
                .collect::<Vec<_>>();
            let target = (0..rows)
                .map(|row| (row % 7) as f32 / 6.0)
                .collect::<Vec<_>>();
            session.set_parameter("logits", &logits);
            session.set_input("target", &target);
            session.step();
            session.wait();
            let mut actual_loss = [0.0];
            let mut actual_gradient = vec![0.0; rows];
            session.read_output_by_index(0, &mut actual_loss);
            session.read_param_grad("logits", &mut actual_gradient);
            let mut expected_loss = 0.0_f64;
            for ((logit, target), actual) in logits.iter().zip(&target).zip(&actual_gradient) {
                let p = 1.0 / (1.0 + (-f64::from(*logit)).exp());
                let t = f64::from(*target);
                expected_loss -= 0.3 * (t * p.ln() + (1.0 - t) * (1.0 - p).ln()) / rows as f64;
                let expected_gradient = 0.3 * (p - t) / rows as f64;
                assert!((f64::from(*actual) - expected_gradient).abs() < 1e-8);
            }
            assert!((f64::from(actual_loss[0]) - expected_loss).abs() < 1e-6);
        }
    }

    #[test]
    #[ignore = "compares temporal batching losses and every parameter gradient on GPU"]
    fn temporal_batching_matches_serial_losses_and_gradients() {
        use super::super::runtime::{build_session, initialize_d3};
        use meganeura::{Mode, Session};
        use std::sync::Arc;

        fn fill_inputs(session: &mut Session, config: &DreamerConfig) {
            let batch = config.batch_size;
            let size = config.network();
            let signal = |count: usize, time: usize| {
                (0..count)
                    .map(|index| ((index * 17 + time * 37) as f32 * 0.013).sin() * 0.3)
                    .collect::<Vec<_>>()
            };
            let one_hot = |rows: usize, width: usize, offset: usize| {
                (0..rows * width)
                    .map(|index| f32::from(index % width == (index / width + offset) % width))
                    .collect::<Vec<_>>()
            };
            session.set_input("initial_deter", &signal(batch * size.deter, 7));
            session.set_input(
                "initial_stoch",
                &one_hot(batch * size.stoch, size.classes, 1),
            );
            for time in 0..config.batch_length {
                for (name, width) in [
                    ("keep_deter", size.deter),
                    ("keep_stoch", size.stoch * size.classes),
                    ("keep_action", config.action_count),
                ] {
                    let keep = (0..batch * width)
                        .map(|index| f32::from(time != index / width))
                        .collect::<Vec<_>>();
                    session.set_input(&format!("{name}_{time}"), &keep);
                }
                session.set_input(
                    &format!("observation_{time}"),
                    &signal(batch * config.observation_dim(), time),
                );
                session.set_input(
                    &format!("previous_action_{time}"),
                    &one_hot(batch, config.action_count, time),
                );
                session.set_input(
                    &format!("posterior_sample_{time}"),
                    &one_hot(batch * size.stoch, size.classes, time + 2),
                );
                for (name, offset) in [
                    ("reward_target", 2 * time),
                    ("replay_value_target", time + 1),
                    ("replay_slow_target", time + 3),
                ] {
                    session.set_input(
                        &format!("{name}_{time}"),
                        &one_hot(batch, config.value_bins, offset),
                    );
                }
                session.set_input(
                    &format!("continuation_target_{time}"),
                    &(0..batch)
                        .map(|row| {
                            f32::from((time + row) % 3 != 0) * config.continuation_discount()
                        })
                        .collect::<Vec<_>>(),
                );
                session.set_input(
                    &format!("replay_value_weight_{time}"),
                    &(0..batch)
                        .map(|row| {
                            if time + 1 == config.batch_length {
                                0.0
                            } else {
                                (row + 1) as f32 / batch as f32
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }

        // Opt into the production B16/T64 graph on a GPU with enough memory
        // for both reference and grouped sessions. Larger matmuls can select
        // different forward kernels; derivative operands remain F32.
        let full = std::env::var_os("KINDLE_FULL_WORLD_PARITY").is_some();
        let configs = if full {
            let mut config = DreamerConfig::new(18);
            config.world_backprop_length = config.batch_length;
            config.loss_scales.reconstruction = 0.0;
            config.loss_scales.future_prediction = 0.25;
            vec![config]
        } else {
            [(3, 0.0, true, 1.0), (4, 0.75, false, 0.0)]
                .map(
                    |(length, reconstruction, replay_value_gradient, free_nats)| {
                        let mut config = DreamerConfig::tiny(3);
                        config.batch_size = 3;
                        config.batch_length = length;
                        config.world_backprop_length = length;
                        config.loss_scales.future_prediction = 0.25;
                        config.loss_scales.reconstruction = reconstruction;
                        config.replay_value_gradient = replay_value_gradient;
                        config.free_nats = free_nats;
                        config
                    },
                )
                .to_vec()
        };
        let loss_tolerance = if full { 3e-4 } else { 3e-5 };
        let gradient_tolerance = if full { 3e-3 } else { 3e-4 };
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        for config in configs {
            let length = config.batch_length;
            let mut sessions = [1, length].map(|group| {
                let graph = build_training_graph_grouped(&config, length, group);
                let mut session = build_session(&graph, &gpu, Mode::Training, false);
                initialize_d3(&mut session, &graph, config.seed);
                // Exercise the input gradients of heads that D3 initializes to zero.
                for name in ["world.reward.out.weight", "behavior.value.out.weight"] {
                    let values = (0..session.param_size(name).unwrap())
                        .map(|index| (index as f32 * 0.17).sin() * 0.02)
                        .collect::<Vec<_>>();
                    session.set_parameter(name, &values);
                }
                fill_inputs(&mut session, &config);
                session.clear_optimizer();
                session.step();
                session.wait();
                session
            });
            let [serial, grouped] = &mut sessions;
            for metric in 0..9 {
                let mut left = [0.0];
                let mut right = [0.0];
                serial.read_output_by_index(metric, &mut left);
                grouped.read_output_by_index(metric, &mut right);
                let tolerance = loss_tolerance * left[0].abs().max(1.0);
                assert!(
                    (left[0] - right[0]).abs() <= tolerance,
                    "T={length}, metric {metric}: {left:?} vs {right:?}"
                );
            }
            let mut names = serial.param_names();
            let mut grouped_names = grouped.param_names();
            names.sort_unstable();
            grouped_names.sort_unstable();
            assert_eq!(names, grouped_names);
            let mut worst_relative = 0.0_f64;
            for name in names {
                assert_eq!(serial.param_size(name), grouped.param_size(name));
                assert_eq!(serial.has_param_grad(name), grouped.has_param_grad(name));
                if name.starts_with("behavior.") {
                    assert!(!serial.has_param_grad(name), "replay critic remains frozen");
                }
                if !serial.has_param_grad(name) {
                    continue;
                }
                let mut left = vec![0.0; serial.param_size(name).unwrap()];
                let mut right = vec![0.0; left.len()];
                serial.read_param_grad(name, &mut left);
                grouped.read_param_grad(name, &mut right);
                assert!(left.iter().chain(&right).all(|value| value.is_finite()));
                let squared = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| f64::from(*value).powi(2))
                        .sum::<f64>()
                };
                let difference = left
                    .iter()
                    .zip(&right)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm = squared(&left).max(squared(&right)).sqrt();
                let relative = difference / norm.max(1e-12);
                worst_relative = worst_relative.max(relative);
                assert!(
                    difference <= gradient_tolerance * norm + 1e-7,
                    "T={length}, {name}: gradient relative L2={relative}, norm={norm}"
                );
            }
            eprintln!("T={length}, worst per-parameter gradient relative L2={worst_relative}");
        }
    }
}
