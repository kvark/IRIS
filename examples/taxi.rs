//! Run the IRIS agent on Taxi.
//!
//! Run: `cargo run --example taxi`

use iris::envs::taxi::{NUM_ACTIONS, OBS_DIM, Taxi};
use iris::{Agent, AgentConfig, Environment};
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("IRIS Taxi");
    println!("==========");

    let mut env = Taxi::new();
    let config = AgentConfig {
        obs_dim: OBS_DIM,
        action_dim: NUM_ACTIONS,
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 8,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 3e-5,
        lr_credit: 1e-5,
        lr_policy: 1.5e-5,
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
                "step {:>5} | wm={:.4} cr={:.4} pi={:.4} | r={:.3} ent={:.2} H={:.1} drift={:.3} | buf={}",
                d.step,
                d.loss_world_model,
                d.loss_credit,
                d.loss_policy,
                d.reward_mean,
                d.policy_entropy,
                d.h_eff,
                d.repr_drift,
                d.buffer_len,
            );
        }
    }

    let d = agent.diagnostics();
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(&d).unwrap());
}
