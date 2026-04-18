//! Diagnose WM regression on CartPole + MountainCar.
//!
//! Runs four configurations per env:
//!   A) L0-only (num_options=1), 5000 steps
//!   B) L1 (num_options=4), 5000 steps
//!   C) L0-only, 10000 steps
//!   D) L1, 10000 steps
//!
//! For each: prints the WM loss trajectory at 500-step checkpoints so
//! we can tell if the regression at ~2000 steps is a transient local
//! peak, a slow monotonic divergence, or L1-specific.
//!
//! Run: `cargo run --release --example wm_probe`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::{cart_pole::CartPole, mountain_car::MountainCar};
use rand::SeedableRng;

fn run(
    name: &str,
    l1: bool,
    total_steps: usize,
    make_env: &dyn Fn() -> (Box<dyn Environment>, Box<dyn kindle::EnvAdapter>),
) {
    let (mut env, adapter) = make_env();
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        num_options: if l1 { 4 } else { 1 },
        option_horizon: 10,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mode = if l1 { "L1" } else { "L0" };
    print!("  {:>12} {} steps={:>5} | wm@", name, mode, total_steps);

    let mut checkpoints = Vec::new();
    for step in 0..total_steps {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        env.step(&action);
        let env_ref: &dyn Environment = &*env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );
        if (step + 1) % 500 == 0 {
            let wm = agent.diagnostics()[0].loss_world_model;
            checkpoints.push((step + 1, wm));
        }
    }
    for (s, wm) in &checkpoints {
        print!(" {}={:.3}", s, wm);
    }
    println!();
}

fn main() {
    env_logger::init();
    println!("WM regression probe — CartPole + MountainCar at long horizons\n");
    let make_cartpole: &dyn Fn() -> (Box<dyn Environment>, Box<dyn kindle::EnvAdapter>) = &|| {
        (
            Box::new(CartPole::new()) as Box<dyn Environment>,
            Box::new(GenericAdapter::discrete(1, 4, 2)) as Box<dyn kindle::EnvAdapter>,
        )
    };
    let make_mountain: &dyn Fn() -> (Box<dyn Environment>, Box<dyn kindle::EnvAdapter>) = &|| {
        (
            Box::new(MountainCar::new()) as Box<dyn Environment>,
            Box::new(GenericAdapter::discrete(2, 2, 3)) as Box<dyn kindle::EnvAdapter>,
        )
    };

    println!("CartPole:");
    run("CartPole", false, 5000, make_cartpole);
    run("CartPole", true, 5000, make_cartpole);
    run("CartPole", false, 10_000, make_cartpole);
    run("CartPole", true, 10_000, make_cartpole);

    println!("\nMountainCar:");
    run("MountainCar", false, 5000, make_mountain);
    run("MountainCar", true, 5000, make_mountain);
    run("MountainCar", false, 10_000, make_mountain);
    run("MountainCar", true, 10_000, make_mountain);
}
