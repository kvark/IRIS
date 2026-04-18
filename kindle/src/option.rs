//! L1 option-policy graph (Phase G).
//!
//! A small MLP that picks an option index from the current latent.
//! Goals are NOT decoded from the network — they're a fixed lookup
//! table of orthogonal vectors stored CPU-side on the Agent. This
//! keeps goals meaningful from step 1 (distinct directions in latent
//! space) while L1 learns WHICH option to pick for each state.
//!
//! Phase G v4 adds a learned-termination head: a sigmoid output
//! β(z_t) representing the probability of ending the current option
//! at state `z_t`, trained with a binary cross-entropy loss against a
//! termination target supplied by the agent. When the config flag
//! `learned_termination` is off, the termination head is still
//! compiled (for graph stability) but its gradient is gated off
//! agent-side and options still run for a fixed `option_horizon`
//! steps.
//!
//! See `docs/phase-g-l1-options.md` for the full design.

use meganeura::graph::Graph;
use meganeura::nn;

/// Build the option-policy training graph.
///
/// Inputs:
/// - `"z"`: `[batch_size, latent_dim]` — current encoder latent
/// - `"option_taken"`: `[batch_size, num_options]` — advantage-weighted
///   one-hot of the option that was taken (for training; zeroed during
///   inference)
/// - `"option_return"`: `[batch_size, 1]` — accumulated return over the
///   option window (value-head target)
/// - `"termination_target"`: `[batch_size, 1]` — BCE target for the
///   learned-termination head (0 = continue, 1 = terminate now).
///   Zero-filled at inference time.
///
/// Outputs:
/// - `[0]`: combined loss (cross-entropy + value MSE + termination BCE)
/// - `[1]`: option logits `[batch_size, num_options]`
/// - `[2]`: option value `[batch_size, 1]`
/// - `[3]`: termination probability β(z_t) `[batch_size, 1]`, post-sigmoid
pub fn build_option_graph(
    latent_dim: usize,
    num_options: usize,
    hidden_dim: usize,
    batch_size: usize,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let option_taken = g.input("option_taken", &[batch_size, num_options]);
    let option_return = g.input("option_return", &[batch_size, 1]);
    let termination_target = g.input("termination_target", &[batch_size, 1]);

    let trunk = nn::Linear::new(&mut g, "option.trunk", latent_dim, hidden_dim);
    let h = trunk.forward(&mut g, z);
    let h = g.relu(h);

    let option_head = nn::Linear::no_bias(&mut g, "option.head", hidden_dim, num_options);
    let option_logits = option_head.forward(&mut g, h);

    let value_head = nn::Linear::no_bias(&mut g, "option.value", hidden_dim, 1);
    let option_value = value_head.forward(&mut g, h);

    // Learned termination head (Phase G v4): sigmoid gives β(z_t) ∈ (0,1).
    // A constant `-3.0` bias is added to the logit so β ≈ σ(−3) ≈ 0.047
    // at initialization — the agent starts out strongly preferring to
    // continue the current option, and only raises β when the
    // termination BCE target consistently pushes it up in specific
    // states. Without this bias, Xavier-init logits hover near zero so
    // β ≈ 0.5 from step 1 and bernoulli(β) fires on half of all steps —
    // mean option length collapses to ~2 and options never run long
    // enough to develop distinct behaviour.
    let term_head = nn::Linear::no_bias(&mut g, "option.term", hidden_dim, 1);
    let term_raw = term_head.forward(&mut g, h);
    // Constant [1]-shaped bias broadcast across the batch via bias_add.
    let term_bias = g.scalar(-3.0);
    let term_logit = g.bias_add(term_raw, term_bias);
    let term_prob = g.sigmoid(term_logit);

    let policy_loss = g.cross_entropy_loss(option_logits, option_taken);
    let value_loss = g.mse_loss(option_value, option_return);
    let term_loss = g.bce_loss(term_prob, termination_target);
    let pv_loss = g.add(policy_loss, value_loss);
    let total_loss = g.add(pv_loss, term_loss);

    g.set_outputs(vec![total_loss, option_logits, option_value, term_prob]);
    g
}

/// Build a fixed goal lookup table: [num_options, option_dim].
/// Each option gets an orthogonal unit-scale direction. Options beyond
/// `option_dim` wrap around with alternating sign.
pub fn build_goal_table(num_options: usize, option_dim: usize) -> Vec<f32> {
    let mut table = vec![0.0f32; num_options * option_dim];
    for o in 0..num_options {
        let dim = o % option_dim;
        let sign = if (o / option_dim).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        table[o * option_dim + dim] = sign * 0.5;
    }
    table
}
