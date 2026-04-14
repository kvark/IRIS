//! Tier 2: E-graph optimization parity tests.
//!
//! Verify that meganeura's e-graph optimizer produces numerically identical
//! results to the unoptimized path. If outputs diverge, it indicates a
//! meganeura optimization bug.
//!
//! These tests require a GPU.

use meganeura::{Graph, build_session, build_session_unoptimized};

const ATOL: f32 = 1e-4;

/// Compare outputs of optimized vs unoptimized sessions for the same graph.
fn check_parity(
    graph: &Graph,
    param_inits: &[(&str, &[f32])],
    input_data: &[(&str, &[f32])],
    output_len: usize,
) {
    let mut session_opt = build_session(graph);
    let mut session_unopt = build_session_unoptimized(graph);

    for &(name, data) in param_inits {
        session_opt.set_parameter(name, data);
        session_unopt.set_parameter(name, data);
    }

    for &(name, data) in input_data {
        session_opt.set_input(name, data);
        session_unopt.set_input(name, data);
    }

    // Run forward only (lr=0 so params don't change)
    session_opt.set_learning_rate(0.0);
    session_unopt.set_learning_rate(0.0);

    session_opt.step();
    session_opt.wait();
    session_unopt.step();
    session_unopt.wait();

    let out_opt = session_opt.read_output(output_len);
    let out_unopt = session_unopt.read_output(output_len);

    for (i, (&a, &b)) in out_opt.iter().zip(out_unopt.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < ATOL,
            "output[{i}] diverged: optimized={a:.6}, unoptimized={b:.6}, diff={diff:.6}"
        );
    }
}

/// Parity check for the encoder (linear + relu + norm + linear).
#[test]
#[ignore] // requires GPU
fn opt_parity_encoder() {
    let batch = 4;
    let obs_dim = 8;
    let latent_dim = 4;
    let hidden_dim = 6;

    let mut g = Graph::new();
    let x = g.input("x", &[batch, obs_dim]);
    let target = g.input("target", &[batch, latent_dim]);

    let encoder = iris::encoder::Encoder::new(&mut g, obs_dim, latent_dim, hidden_dim);
    let z = encoder.forward(&mut g, x);
    let loss = g.mse_loss(z, target);
    g.set_outputs(vec![loss]);

    let fc1_w: Vec<f32> = (0..obs_dim * hidden_dim)
        .map(|i| (i as f32 * 0.05) - 0.2)
        .collect();
    let fc1_b = vec![0.01f32; hidden_dim];
    let norm_w = vec![1.0f32; hidden_dim];
    let fc2_w: Vec<f32> = (0..hidden_dim * latent_dim)
        .map(|i| (i as f32 * 0.08) - 0.1)
        .collect();

    let x_data: Vec<f32> = (0..batch * obs_dim).map(|i| i as f32 * 0.1).collect();
    let target_data = vec![0.0f32; batch * latent_dim];

    check_parity(
        &g,
        &[
            ("encoder.fc1.weight", &fc1_w),
            ("encoder.fc1.bias", &fc1_b),
            ("encoder.norm.weight", &norm_w),
            ("encoder.fc2.weight", &fc2_w),
        ],
        &[("x", &x_data), ("target", &target_data)],
        1, // scalar loss
    );
}

/// Parity check for the world model (linear layers + relu).
#[test]
#[ignore] // requires GPU
fn opt_parity_world_model() {
    let batch = 4;
    let latent_dim = 4;
    let action_dim = 3;
    let hidden_dim = 6;
    let input_dim = latent_dim + action_dim;

    let mut g = Graph::new();
    let za = g.input("za", &[batch, input_dim]);
    let target = g.input("target", &[batch, latent_dim]);

    let wm = iris::world_model::WorldModel::new(&mut g, latent_dim, action_dim, hidden_dim);
    let z_hat = wm.forward(&mut g, za);
    let loss = g.mse_loss(z_hat, target);
    g.set_outputs(vec![loss]);

    let fc1_w: Vec<f32> = (0..input_dim * hidden_dim)
        .map(|i| (i as f32 * 0.04) - 0.15)
        .collect();
    let fc1_b = vec![0.0f32; hidden_dim];
    let fc2_w: Vec<f32> = (0..hidden_dim * hidden_dim)
        .map(|i| (i as f32 * 0.03) - 0.1)
        .collect();
    let fc2_b = vec![0.0f32; hidden_dim];
    let fc_out_w: Vec<f32> = (0..hidden_dim * latent_dim)
        .map(|i| (i as f32 * 0.06) - 0.2)
        .collect();

    let za_data: Vec<f32> = (0..batch * input_dim).map(|i| i as f32 * 0.1).collect();
    let target_data = vec![0.0f32; batch * latent_dim];

    check_parity(
        &g,
        &[
            ("world_model.fc1.weight", &fc1_w),
            ("world_model.fc1.bias", &fc1_b),
            ("world_model.fc2.weight", &fc2_w),
            ("world_model.fc2.bias", &fc2_b),
            ("world_model.fc_out.weight", &fc_out_w),
        ],
        &[("za", &za_data), ("target", &target_data)],
        1,
    );
}
