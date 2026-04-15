//! Run the kindle agent on MountainCar.
//!
//! Run: `cargo run --example mountain_car`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::mountain_car::MountainCar;
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("kindle MountainCar");
    println!("=================");

    let mut env = MountainCar::new();
    let adapter = Box::new(GenericAdapter::discrete(2, 2, 3));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graphs)...");
    let mut agent = Agent::new(config, vec![adapter]);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let num_steps = 5000;

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

        if (step + 1) % 1000 == 0 {
            let diags = agent.diagnostics();
            let d = &diags[0];
            println!(
                "step {:>5} | wm={:.4} cr={:.4} pi={:.4} | r={:.3} ent={:.2} H={:.1} | buf={}",
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
