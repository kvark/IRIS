//! L1 option-policy graph (Phase G).
//!
//! A small MLP that picks an option index from the current latent and
//! decodes the selected option into a goal-latent that conditions L0.
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
/// - `[3]`: goal latents `[batch_size, num_options * option_dim]` —
///   flattened; reshape on CPU to `[N, num_options, option_dim]` and
///   index by the selected option to get the per-lane goal-latent.
pub fn build_option_graph(
    latent_dim: usize,
    num_options: usize,
    option_dim: usize,
    hidden_dim: usize,
    batch_size: usize,
) -> Graph {
    let mut g = Graph::new();
    let z = g.input("z", &[batch_size, latent_dim]);
    let option_taken = g.input("option_taken", &[batch_size, num_options]);
    let option_return = g.input("option_return", &[batch_size, 1]);

    // Shared trunk.
    let trunk = nn::Linear::new(&mut g, "option.trunk", latent_dim, hidden_dim);
    let h = trunk.forward(&mut g, z);
    let h = g.relu(h);

    // Option head: categorical logits over num_options.
    let option_head = nn::Linear::no_bias(&mut g, "option.head", hidden_dim, num_options);
    let option_logits = option_head.forward(&mut g, h);

    // Value head: predicts expected option return.
    let value_head = nn::Linear::no_bias(&mut g, "option.value", hidden_dim, 1);
    let option_value = value_head.forward(&mut g, h);

    // Goal decoder: one goal-latent per option, flattened.
    let goal_dim = num_options * option_dim;
    let goal_dec = nn::Linear::no_bias(&mut g, "option.goal_dec", hidden_dim, goal_dim);
    let goals = goal_dec.forward(&mut g, h);

    // Losses — same per-row advantage-weighting trick as L0: callers
    // feed `option_taken = advantage · one_hot(o)`, so the cross-entropy
    // gradient per row is scaled by the lane's advantage. Non-terminating
    // lanes feed zeros and contribute no gradient.
    let policy_loss = g.cross_entropy_loss(option_logits, option_taken);
    let value_loss = g.mse_loss(option_value, option_return);
    let total_loss = g.add(policy_loss, value_loss);

    g.set_outputs(vec![total_loss, option_logits, option_value, goals]);
    g
}
