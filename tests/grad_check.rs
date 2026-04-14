//! Tier 1: Finite-difference gradient verification.
//!
//! For each graph primitive IRIS uses, compare meganeura's analytical gradients
//! (from autodiff) against numerical gradients (perturb → forward → diff).
//!
//! Tolerance: relative error < 1e-3 for f32.
//!
//! These tests require a GPU.

use meganeura::{Graph, build_session};

const EPS: f32 = 1e-4;
const REL_TOL: f32 = 1e-3;

fn check_gradient(
    graph: &Graph,
    param_name: &str,
    param_data: &[f32],
    input_name: &str,
    input_data: &[f32],
    label_name: &str,
    label_data: &[f32],
) {
    // Get analytical gradient
    let mut session = build_session(graph);
    session.set_parameter(param_name, param_data);
    session.set_input(input_name, input_data);
    session.set_input(label_name, label_data);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();

    let mut analytical_grad = vec![0.0f32; param_data.len()];
    session.read_param_grad(param_name, &mut analytical_grad);

    // Check a subset of elements (full check is expensive)
    let check_count = param_data.len().min(10);
    let stride = param_data.len() / check_count;

    for i in (0..param_data.len()).step_by(stride.max(1)).take(check_count) {
        let num_grad = {
            // For numerical gradient, we need the full input set
            // Build sessions with both inputs set
            let loss_plus = {
                let mut s = build_session(graph);
                let mut p = param_data.to_vec();
                p[i] += EPS;
                s.set_parameter(param_name, &p);
                s.set_input(input_name, input_data);
                s.set_input(label_name, label_data);
                s.set_learning_rate(0.0);
                s.step();
                s.wait();
                s.read_loss()
            };
            let loss_minus = {
                let mut s = build_session(graph);
                let mut p = param_data.to_vec();
                p[i] -= EPS;
                s.set_parameter(param_name, &p);
                s.set_input(input_name, input_data);
                s.set_input(label_name, label_data);
                s.set_learning_rate(0.0);
                s.step();
                s.wait();
                s.read_loss()
            };
            (loss_plus - loss_minus) / (2.0 * EPS)
        };

        let anal = analytical_grad[i];
        let rel_err = (anal - num_grad).abs() / (anal.abs() + num_grad.abs() + 1e-8);

        assert!(
            rel_err < REL_TOL,
            "gradient mismatch for {param_name}[{i}]: analytical={anal:.6}, numerical={num_grad:.6}, rel_err={rel_err:.6}"
        );
    }
}

/// Gradient check for a simple linear layer (matmul + bias_add) with MSE loss.
#[test]
#[ignore] // requires GPU
fn grad_check_linear_mse() {
    let batch = 4;
    let input_dim = 3;
    let output_dim = 2;

    let mut g = Graph::new();
    let x = g.input("x", &[batch, input_dim]);
    let target = g.input("target", &[batch, output_dim]);
    let w = g.parameter("w", &[input_dim, output_dim]);
    let b = g.parameter("b", &[output_dim]);

    let mm = g.matmul(x, w);
    let y = g.bias_add(mm, b);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss]);

    let w_data: Vec<f32> = (0..input_dim * output_dim)
        .map(|i| (i as f32 * 0.1) - 0.3)
        .collect();
    let _b_data = vec![0.1f32; output_dim];
    let x_data: Vec<f32> = (0..batch * input_dim)
        .map(|i| (i as f32 * 0.2) - 0.5)
        .collect();
    let target_data: Vec<f32> = (0..batch * output_dim)
        .map(|i| i as f32 * 0.05)
        .collect();

    // Check weight gradient
    check_gradient(&g, "w", &w_data, "x", &x_data, "target", &target_data);
}

/// Gradient check for ReLU activation.
#[test]
#[ignore] // requires GPU
fn grad_check_relu() {
    let batch = 4;
    let dim = 3;

    let mut g = Graph::new();
    let x = g.input("x", &[batch, dim]);
    let target = g.input("target", &[batch, dim]);
    let w = g.parameter("w", &[dim, dim]);

    let h = g.matmul(x, w);
    let h = g.relu(h);
    let loss = g.mse_loss(h, target);
    g.set_outputs(vec![loss]);

    let w_data: Vec<f32> = (0..dim * dim).map(|i| (i as f32 * 0.3) - 0.5).collect();
    let x_data: Vec<f32> = (0..batch * dim).map(|i| (i as f32 * 0.2) - 0.3).collect();
    let target_data = vec![0.5f32; batch * dim];

    check_gradient(&g, "w", &w_data, "x", &x_data, "target", &target_data);
}

/// Gradient check for a two-layer MLP (encoder-like).
#[test]
#[ignore] // requires GPU
fn grad_check_mlp() {
    let batch = 4;
    let input_dim = 4;
    let hidden_dim = 3;
    let output_dim = 2;

    let mut g = Graph::new();
    let x = g.input("x", &[batch, input_dim]);
    let target = g.input("target", &[batch, output_dim]);

    let w1 = g.parameter("w1", &[input_dim, hidden_dim]);
    let b1 = g.parameter("b1", &[hidden_dim]);
    let w2 = g.parameter("w2", &[hidden_dim, output_dim]);

    let h = g.matmul(x, w1);
    let h = g.bias_add(h, b1);
    let h = g.relu(h);
    let y = g.matmul(h, w2);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss]);

    let w1_data: Vec<f32> = (0..input_dim * hidden_dim)
        .map(|i| (i as f32 * 0.1) - 0.2)
        .collect();
    let _b1_data = vec![0.0f32; hidden_dim];
    let _w2_data: Vec<f32> = (0..hidden_dim * output_dim)
        .map(|i| (i as f32 * 0.15) - 0.1)
        .collect();
    let x_data: Vec<f32> = (0..batch * input_dim)
        .map(|i| i as f32 * 0.1)
        .collect();
    let target_data = vec![0.0f32; batch * output_dim];

    check_gradient(
        &g,
        "w1",
        &w1_data,
        "x",
        &x_data,
        "target",
        &target_data,
    );
}
