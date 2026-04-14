//! Run the IRIS agent on Acrobot.
//!
//! Run: `cargo run --example acrobot`

use iris::envs::acrobot::Acrobot;
use iris::{Agent, AgentConfig, Environment};
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("IRIS Acrobot");
    println!("=============");

    let mut env = Acrobot::new();
    let config = AgentConfig {
        obs_dim: 6,
        action_dim: 3,
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-4,
        warmup_steps: 200,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graphs)...");
    let mut agent = Agent::new(config);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let num_steps = 5000;

    for step in 0..num_steps {
        let obs = env.observe();
        let action = agent.act(&obs, &mut rng);
        env.step(&action);
        agent.observe(&obs, &action, &env, &mut rng);

        if (step + 1) % 1000 == 0 {
            let d = agent.diagnostics();
            println!(
                "step {:>5} | wm={:.4} cr={:.4} pi={:.4} | r={:.3} ent={:.2} H={:.1} | buf={}",
                d.step, d.loss_world_model, d.loss_credit, d.loss_policy,
                d.reward_mean, d.policy_entropy, d.h_eff, d.buffer_len,
            );
        }
    }

    let d = agent.diagnostics();
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(&d).unwrap());
}
