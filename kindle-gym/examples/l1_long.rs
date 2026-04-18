//! Longer-horizon L1 option-diversity test.
//!
//! Run each env for 10k steps under L1 and measure whether L0
//! differentiates actions across options. Tail window = last 3k steps
//! so transient behaviour doesn't mask the converged policy.
//!
//! Supports sweeping `goal_bonus_alpha` + `entropy_beta` to explore
//! the space where collapse vs diversification tips.
//!
//! Run: `cargo run --release --example l1_long`

use kindle::{Action, Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::{
    acrobot::Acrobot, cart_pole::CartPole, grid_world::GridWorld, mountain_car::MountainCar,
    pendulum::Pendulum, random_walk::RandomWalk, taxi::Taxi,
};
use rand::SeedableRng;

const STEPS: usize = 10_000;
const TAIL: usize = 3_000;
const NUM_OPTIONS: usize = 4;

fn run(
    name: &str,
    env: Box<dyn Environment>,
    adapter: Box<dyn kindle::EnvAdapter>,
    alpha: f32,
    entropy_beta: f32,
) -> (usize, f32) {
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        num_options: NUM_OPTIONS,
        option_horizon: 10,
        goal_bonus_alpha: alpha,
        entropy_beta,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut env = env;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut per_option_tail: Vec<Vec<u32>> = (0..NUM_OPTIONS).map(|_| vec![0u32; 6]).collect();
    let mut final_entropy = 0.0f32;
    let mut wm_late = 0.0f32;

    for step in 0..STEPS {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let d = agent.diagnostics()[0].clone();
        let opt = d.current_option as usize;
        let a_idx = match &action {
            Action::Discrete(i) => *i,
            Action::Continuous(v) if !v.is_empty() => {
                if v[0] > 0.0 {
                    0
                } else {
                    1
                }
            }
            _ => 0,
        };
        if step >= STEPS - TAIL && opt < NUM_OPTIONS && a_idx < 6 {
            per_option_tail[opt][a_idx] += 1;
        }

        env.step(&action);
        let env_ref: &dyn Environment = &*env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );
        if step == STEPS - 1 {
            final_entropy = d.policy_entropy;
            wm_late = d.loss_world_model;
        }
    }

    let mut distinct_late = std::collections::HashSet::new();
    for o in 0..NUM_OPTIONS {
        let total: u32 = per_option_tail[o].iter().sum();
        if total == 0 {
            continue;
        }
        let (best_a, _) = per_option_tail[o]
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .unwrap();
        distinct_late.insert(best_a);
    }
    print!(
        "  {:12} α={:.1} β={:.2} | ent={:.2} wm={:.3} | ",
        name, alpha, entropy_beta, final_entropy, wm_late
    );
    for o in 0..NUM_OPTIONS {
        let total: u32 = per_option_tail[o].iter().sum();
        let (best_a, best_c) = per_option_tail[o]
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .map(|(i, &c)| (i, c))
            .unwrap_or((0, 0));
        let pct = if total > 0 {
            100.0 * best_c as f32 / total as f32
        } else {
            0.0
        };
        print!("o{o}=a{}({:.0}%) ", best_a, pct);
    }
    println!("[distinct={}]", distinct_late.len());
    (distinct_late.len(), final_entropy)
}

fn main() {
    env_logger::init();
    println!("L1 long-horizon diversity sweep ({STEPS} steps, tail={TAIL})\n");

    type Factory = Box<dyn Fn() -> (Box<dyn Environment>, Box<dyn kindle::EnvAdapter>)>;
    let mut envs: Vec<(&'static str, Factory)> = Vec::new();
    use kindle_gym::*;
    envs.push((
        "GridWorld",
        Box::new(|| {
            (
                Box::new(GridWorld::new()) as Box<dyn Environment>,
                Box::new(GenericAdapter::discrete(
                    0,
                    grid_world::OBS_DIM,
                    grid_world::NUM_ACTIONS,
                )) as Box<dyn kindle::EnvAdapter>,
            )
        }),
    ));
    envs.push((
        "CartPole",
        Box::new(|| {
            (
                Box::new(CartPole::new()),
                Box::new(GenericAdapter::discrete(1, 4, 2)),
            )
        }),
    ));
    envs.push((
        "MountainCar",
        Box::new(|| {
            (
                Box::new(MountainCar::new()),
                Box::new(GenericAdapter::discrete(2, 2, 3)),
            )
        }),
    ));
    envs.push((
        "Acrobot",
        Box::new(|| {
            (
                Box::new(Acrobot::new()),
                Box::new(GenericAdapter::discrete(3, 6, 3)),
            )
        }),
    ));
    envs.push((
        "Taxi",
        Box::new(|| {
            (
                Box::new(Taxi::new()),
                Box::new(GenericAdapter::discrete(
                    4,
                    taxi::OBS_DIM,
                    taxi::NUM_ACTIONS,
                )),
            )
        }),
    ));
    envs.push((
        "RandomWalk",
        Box::new(|| {
            (
                Box::new(RandomWalk::new(10)),
                Box::new(GenericAdapter::discrete(5, 10, 2)),
            )
        }),
    ));
    envs.push((
        "Pendulum",
        Box::new(|| {
            (
                Box::new(Pendulum::new()),
                Box::new(GenericAdapter::continuous(6, 3, 1, 0.5)),
            )
        }),
    ));

    let settings = [(1.0f32, 0.01f32), (5.0, 0.01), (5.0, 0.1), (10.0, 0.1)];

    for &(alpha, beta) in &settings {
        println!("-- α={alpha:.1} β={beta:.2} --");
        for (name, factory) in &envs {
            let (env, adapter) = factory();
            run(name, env, adapter, alpha, beta);
        }
        println!();
    }
}
