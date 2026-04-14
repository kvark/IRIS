//! Run the IRIS agent on CartPole.
//!
//! Run: `cargo run --example cart_pole`

use iris::envs::cart_pole::CartPole;
use iris::{Agent, AgentConfig, Environment};
use rand::SeedableRng;

fn main() {
    env_logger::init();

    println!("IRIS CartPole");
    println!("==============");

    let mut env = CartPole::new();
    let config = AgentConfig {
        obs_dim: 4,
        action_dim: 2,
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graphs)...");
    let mut agent = Agent::new(config);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let num_steps = 2000;

    for step in 0..num_steps {
        let obs = env.observe();
        let action = agent.act(&mut rng);
        env.step(&action);
        agent.observe(&obs, &action, &env, &mut rng);

        if (step + 1) % 500 == 0 {
            let d = agent.diagnostics();
            println!(
                "step {:>4} | wm={:.4} cr={:.4} | r={:.3} (s={:.3} n={:.3} h={:.3}) | H_eff={:.1} buf={}",
                d.step, d.loss_world_model, d.loss_credit,
                d.reward_mean, d.reward_surprise, d.reward_novelty, d.reward_homeo,
                d.h_eff, d.buffer_len,
            );
        }
    }

    let d = agent.diagnostics();
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(&d).unwrap());
}
