//! Run the kindle agent on Pendulum (continuous action).
//!
//! Run: `cargo run --example pendulum`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::pendulum::Pendulum;
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("kindle Pendulum (continuous)");
    println!("===========================");

    let mut env = Pendulum::new();
    let adapter = Box::new(GenericAdapter::continuous(5, 3, 1, 0.5));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-4,
        lr_credit: 3e-5,
        lr_policy: 5e-5,
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
                "step {:>5} | wm={:.4} cr={:.4} pi={:.4} | r={:.3} drift={:.3} | buf={}",
                d.step,
                d.loss_world_model,
                d.loss_credit,
                d.loss_policy,
                d.reward_mean,
                d.repr_drift,
                d.buffer_len,
            );
        }
    }

    let d = &agent.diagnostics()[0];
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(d).unwrap());
}
