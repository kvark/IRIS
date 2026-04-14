//! Tier 3: Training convergence canaries.
//!
//! Small, well-understood problems with known convergence behavior.
//! These verify that the full training pipeline (graph → autodiff →
//! compile → GPU execution → optimizer) is wired correctly.
//!
//! These tests require a GPU.

/// Canary 3: next-step prediction on a deterministic sequence.
///
/// A trivial world model task: given an observation `x`, predict `x + delta`.
/// The world model should drive prediction error below a threshold within
/// a bounded number of steps.
///
/// Expected: MSE loss < 0.01 within 2000 steps.
/// Failure signal: broken world model training, optimizer, or gradient flow.
#[test]
#[ignore] // requires GPU
fn canary_next_step_prediction() {
    use meganeura::{Graph, build_session};
    use meganeura::nn;

    let batch = 1;
    let dim = 4;
    let hidden = 8;
    let steps = 2000;

    // Build a simple prediction graph: input -> MLP -> output, MSE loss
    let mut g = Graph::new();
    let x = g.input("x", &[batch, dim]);
    let target = g.input("target", &[batch, dim]);

    let fc1 = nn::Linear::new(&mut g, "fc1", dim, hidden);
    let fc2 = nn::Linear::new(&mut g, "fc2", hidden, dim);

    let h = fc1.forward(&mut g, x);
    let h = g.relu(h);
    let y = fc2.forward(&mut g, h);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);

    // Xavier init
    let scale1 = (2.0 / (dim + hidden) as f32).sqrt();
    let w1: Vec<f32> = (0..dim * hidden)
        .map(|i| ((i as f32 * 0.618034).fract() - 0.5) * 2.0 * scale1)
        .collect();
    session.set_parameter("fc1.weight", &w1);
    session.set_parameter("fc1.bias", &vec![0.0f32; hidden]);

    let scale2 = (2.0 / (hidden + dim) as f32).sqrt();
    let w2: Vec<f32> = (0..hidden * dim)
        .map(|i| ((i as f32 * 0.381966).fract() - 0.5) * 2.0 * scale2)
        .collect();
    session.set_parameter("fc2.weight", &w2);
    session.set_parameter("fc2.bias", &vec![0.0f32; dim]);

    // Training data: predict x + 0.1
    let inputs: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect())
        .collect();
    let targets: Vec<Vec<f32>> = inputs
        .iter()
        .map(|x| x.iter().map(|v| v + 0.1).collect())
        .collect();

    let mut final_loss = f32::MAX;
    for step in 0..steps {
        let idx = step % inputs.len();
        session.set_input("x", &inputs[idx]);
        session.set_input("target", &targets[idx]);
        // Must set LR before each step — meganeura clears it after use
        session.set_learning_rate(1e-3);
        session.step();
        session.wait();
        final_loss = session.read_loss();
    }

    assert!(
        final_loss < 0.01,
        "canary 3 failed: next-step prediction loss {final_loss:.4} > 0.01 after {steps} steps"
    );
}

/// Canary: IRIS encoder + world model convergence.
///
/// Verifies that the full IRIS training graph (encoder → world model → MSE)
/// reduces prediction error on a deterministic observation sequence.
#[test]
#[ignore] // requires GPU
fn canary_iris_world_model() {
    use iris::{Agent, AgentConfig};
    use iris::env::{
        Action, Environment, HomeostaticProvider, HomeostaticVariable, Observation, StepResult,
    };

    struct ConstantEnv {
        state: usize,
        patterns: Vec<Vec<f32>>,
    }

    impl ConstantEnv {
        fn new() -> Self {
            // 4 deterministic observations that cycle
            Self {
                state: 0,
                patterns: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.0, 1.0],
                ],
            }
        }
    }

    impl HomeostaticProvider for ConstantEnv {
        fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
            &[]
        }
    }

    impl Environment for ConstantEnv {
        fn observation_dim(&self) -> usize { 4 }
        fn num_actions(&self) -> usize { 2 }
        fn observe(&self) -> Observation {
            Observation::new(self.patterns[self.state].clone())
        }
        fn step(&mut self, _action: &Action) -> StepResult {
            self.state = (self.state + 1) % self.patterns.len();
            StepResult {
                observation: self.observe(),
                homeostatic: vec![],
            }
        }
        fn reset(&mut self) { self.state = 0; }
    }

    let mut env = ConstantEnv::new();
    let config = AgentConfig {
        obs_dim: 4,
        action_dim: 2,
        latent_dim: 4,
        hidden_dim: 16,
        buffer_capacity: 500,
        batch_size: 1,
        learning_rate: 1e-3,
        ..AgentConfig::default()
    };

    let mut agent = Agent::new(config);
    let mut rng = rand::thread_rng();

    // Collect early loss
    let warmup = 50;
    for _ in 0..warmup {
        let obs = env.observe();
        let action = agent.act(&mut rng);
        env.step(&action);
        agent.observe(&obs, &action, &env, &mut rng);
    }
    let early_loss = agent.diagnostics().loss_world_model;

    // Train more
    for _ in warmup..500 {
        let obs = env.observe();
        let action = agent.act(&mut rng);
        env.step(&action);
        agent.observe(&obs, &action, &env, &mut rng);
    }
    let late_loss = agent.diagnostics().loss_world_model;

    assert!(
        late_loss < early_loss || late_loss < 0.1,
        "IRIS world model didn't converge: early={early_loss:.4}, late={late_loss:.4}"
    );
}
