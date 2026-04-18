//! Sweep `goal_bonus_alpha` on all 7 envs to see whether the L1
//! option-collapse can be broken by scaling the goal-reach pressure.
//!
//! For each (env, alpha) pair: trains 3000 steps, reports
//!   - late wm loss (sanity: training still converges)
//!   - goal_distance (late) — is L0 actually reaching the goal?
//!   - distinct_modal_actions across the 4 options (main signal)
//!   - option histogram variance (are options used differently?)
//!
//! Run: `cargo run --release --example l1_alpha_sweep`

use kindle::{Action, Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::{
    acrobot::Acrobot, cart_pole::CartPole, grid_world::GridWorld, mountain_car::MountainCar,
    pendulum::Pendulum, random_walk::RandomWalk, taxi::Taxi,
};
use rand::SeedableRng;

struct EnvRun {
    env: Box<dyn Environment>,
    adapter: Box<dyn kindle::EnvAdapter>,
}

fn make(name: &str) -> EnvRun {
    use kindle_gym::*;
    match name {
        "GridWorld" => EnvRun {
            env: Box::new(GridWorld::new()),
            adapter: Box::new(GenericAdapter::discrete(
                0,
                grid_world::OBS_DIM,
                grid_world::NUM_ACTIONS,
            )),
        },
        "CartPole" => EnvRun {
            env: Box::new(CartPole::new()),
            adapter: Box::new(GenericAdapter::discrete(1, 4, 2)),
        },
        "MountainCar" => EnvRun {
            env: Box::new(MountainCar::new()),
            adapter: Box::new(GenericAdapter::discrete(2, 2, 3)),
        },
        "Acrobot" => EnvRun {
            env: Box::new(Acrobot::new()),
            adapter: Box::new(GenericAdapter::discrete(3, 6, 3)),
        },
        "Taxi" => EnvRun {
            env: Box::new(Taxi::new()),
            adapter: Box::new(GenericAdapter::discrete(
                4,
                taxi::OBS_DIM,
                taxi::NUM_ACTIONS,
            )),
        },
        "RandomWalk" => EnvRun {
            env: Box::new(RandomWalk::new(10)),
            adapter: Box::new(GenericAdapter::discrete(5, 10, 2)),
        },
        "Pendulum" => EnvRun {
            env: Box::new(Pendulum::new()),
            adapter: Box::new(GenericAdapter::continuous(6, 3, 1, 0.5)),
        },
        _ => panic!(),
    }
}

fn run_one(run: EnvRun, alpha: f32, steps: usize) -> (f32, f32, usize, f32) {
    let num_options = 4;
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        num_options,
        option_horizon: 10,
        goal_bonus_alpha: alpha,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![run.adapter]);
    let mut env = run.env;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut per_option: Vec<Vec<u32>> = (0..num_options).map(|_| vec![0u32; 6]).collect();
    let mut goal_dist_sum = 0.0f32;
    let mut goal_dist_count = 0u32;
    let mut wm_late = 0.0f32;

    for step in 0..steps {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let d = agent.diagnostics()[0].clone();
        let opt = d.current_option as usize;
        if opt < num_options {
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
            if a_idx < per_option[opt].len() {
                per_option[opt][a_idx] += 1;
            }
        }

        env.step(&action);
        let env_ref: &dyn Environment = &*env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        if step >= steps - 200 {
            goal_dist_sum += agent.diagnostics()[0].goal_distance;
            goal_dist_count += 1;
        }
        if step == steps - 1 {
            wm_late = agent.diagnostics()[0].loss_world_model;
        }
    }

    let mut distinct = std::collections::HashSet::new();
    for actions in &per_option {
        let total: u32 = actions.iter().sum();
        if total == 0 {
            continue;
        }
        let (best_a, _) = actions.iter().enumerate().max_by_key(|&(_, c)| *c).unwrap();
        distinct.insert(best_a);
    }

    // Option-usage variance: higher = more asymmetric allocation.
    let option_counts: Vec<u32> = per_option.iter().map(|v| v.iter().sum()).collect();
    let total: u32 = option_counts.iter().sum();
    let mean = total as f32 / num_options as f32;
    let var: f32 = option_counts
        .iter()
        .map(|&c| (c as f32 - mean).powi(2))
        .sum::<f32>()
        / num_options as f32;

    (
        wm_late,
        goal_dist_sum / goal_dist_count.max(1) as f32,
        distinct.len(),
        var.sqrt(),
    )
}

fn main() {
    env_logger::init();
    println!("goal_bonus_alpha sweep — distinct_modal_actions (higher is better)\n");
    let envs = [
        "GridWorld",
        "CartPole",
        "MountainCar",
        "Acrobot",
        "Taxi",
        "RandomWalk",
        "Pendulum",
    ];
    let alphas = [0.0f32, 0.1, 0.5, 1.0, 2.0];
    let steps = 3000;

    println!(
        "{:>12} | {:>6} | {:>8} | {:>9} | {:>8} | {:>9}",
        "env", "alpha", "wm_late", "goal_dist", "distinct", "opt_stdev"
    );
    for env_name in &envs {
        for &alpha in &alphas {
            let run = make(env_name);
            let (wm, gd, distinct, stdev) = run_one(run, alpha, steps);
            println!(
                "{:>12} | {:>6.2} | {:>8.3} | {:>9.2} | {:>8} | {:>9.1}",
                env_name, alpha, wm, gd, distinct, stdev
            );
        }
        println!();
    }
}
