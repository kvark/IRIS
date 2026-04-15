//! Run the kindle agent in the 5x5 grid world.
//!
//! Run: `cargo run --example grid_world`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::grid_world::{GridWorld, NUM_ACTIONS, OBS_DIM};
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("kindle Grid World");
    println!("================");

    let mut env = GridWorld::new();
    let adapter = Box::new(GenericAdapter::discrete(0, OBS_DIM, NUM_ACTIONS));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 16,
        history_len: 8,
        buffer_capacity: 1000,
        batch_size: 1,
        learning_rate: 1e-3,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graphs)...");
    let mut agent = Agent::new(config, vec![adapter]);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let num_steps = 500;

    for step in 0..num_steps {
        let obs = env.observe();
        let action = agent
            .act(std::slice::from_ref(&obs), &mut rng)
            .remove(0);
        env.step(&action);
        let env_ref: &dyn Environment = &env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        if (step + 1) % 100 == 0 {
            let diags = agent.diagnostics();
            let d = &diags[0];
            println!(
                "step {:>4} | wm={:.4} cr={:.4} pi={:.4} | r={:.3} ent={:.2} H={:.1} | buf={}",
                d.step,
                d.loss_world_model,
                d.loss_credit,
                d.loss_policy,
                d.reward_mean,
                d.policy_entropy,
                d.h_eff,
                d.buffer_len,
            );
        }
    }

    let d = &agent.diagnostics()[0];
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(d).unwrap());
}
