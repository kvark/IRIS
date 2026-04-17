//! L1 option-policy graph (Phase G).
//!
//! A small MLP that picks an option index from the current latent.
//! Goals are NOT decoded from the network — they're a fixed lookup
//! table of orthogonal vectors stored CPU-side on the Agent. This
//! keeps goals meaningful from step 1 (distinct directions in latent
//! space) while L1 learns WHICH option to pick for each state.
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
///
/// Outputs:
/// - `[0]`: combined loss (cross-entropy + value MSE)
/// - `[1]`: option logits `[batch_size, num_options]`
/// - `[2]`: option value `[batch_size, 1]`
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

    let trunk = nn::Linear::new(&mut g, "option.trunk", latent_dim, hidden_dim);
    let h = trunk.forward(&mut g, z);
    let h = g.relu(h);

    let option_head = nn::Linear::no_bias(&mut g, "option.head", hidden_dim, num_options);
    let option_logits = option_head.forward(&mut g, h);

    let value_head = nn::Linear::no_bias(&mut g, "option.value", hidden_dim, 1);
    let option_value = value_head.forward(&mut g, h);

    let policy_loss = g.cross_entropy_loss(option_logits, option_taken);
    let value_loss = g.mse_loss(option_value, option_return);
    let total_loss = g.add(policy_loss, value_loss);

    g.set_outputs(vec![total_loss, option_logits, option_value]);
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
