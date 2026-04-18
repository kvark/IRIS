//! Compare fixed-horizon options vs learned-termination (Phase G v4).
//!
//! For each of the 7 envs, runs two 3k-step trainings with identical
//! config except for `AgentConfig.learned_termination`. Reports:
//!   - distinct_modal_actions across the 4 options
//!   - mean realized option length (how long did options actually run
//!     on average with learned termination?)
//!   - termination-prob mean at the end of training (how confident is
//!     the agent about terminating?)
//!   - wm loss late, homeo_dev drop
//!
//! Run: `cargo run --release --example l1_learned_term`

use kindle::{Action, Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::{
    acrobot::Acrobot, cart_pole::CartPole, grid_world::GridWorld, mountain_car::MountainCar,
    pendulum::Pendulum, random_walk::RandomWalk, taxi::Taxi,
};
use rand::SeedableRng;

const STEPS: usize = 3000;
const NUM_OPTIONS: usize = 4;
const OPTION_HORIZON: usize = 10;

fn homeo_dev(env: &dyn Environment) -> f32 {
    let vars = env.homeostatic_variables();
    if vars.is_empty() {
        return 0.0;
    }
    vars.iter()
        .map(|v| (v.value - v.target).abs() / v.tolerance.max(1e-6))
        .sum::<f32>()
        / vars.len() as f32
}

#[allow(dead_code)]
struct RunOut {
    distinct: usize,
    mean_option_length: f32,
    homeo_drop: f32,
    wm_late: f32,
}

fn run(
    env: Box<dyn Environment>,
    adapter: Box<dyn kindle::EnvAdapter>,
    learned: bool,
) -> RunOut {
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        num_options: NUM_OPTIONS,
        option_horizon: OPTION_HORIZON,
        learned_termination: learned,
        ..AgentConfig::default()
    };
    let mut agent = Agent::new(config, vec![adapter]);
    let mut env = env;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut per_option_tail: Vec<Vec<u32>> =
        (0..NUM_OPTIONS).map(|_| vec![0u32; 6]).collect();
    let mut last_option: i32 = -1;
    let mut current_length: u32 = 0;
    let mut option_lengths: Vec<u32> = Vec::new();

    let mut homeo_early = 0.0f32;
    let mut homeo_late = 0.0f32;
    let window = 200usize;
    let mut wm_late = 0.0f32;

    for step in 0..STEPS {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let d = agent.diagnostics()[0].clone();
        let opt = d.current_option as i32;

        if last_option < 0 {
            last_option = opt;
            current_length = 1;
        } else if opt != last_option {
            option_lengths.push(current_length);
            last_option = opt;
            current_length = 1;
        } else {
            current_length += 1;
        }

        let a_idx = match &action {
            Action::Discrete(i) => *i,
            Action::Continuous(v) if !v.is_empty() => if v[0] > 0.0 { 0 } else { 1 },
            _ => 0,
        };
        if step >= STEPS - 1000 && (opt as usize) < NUM_OPTIONS && a_idx < 6 {
            per_option_tail[opt as usize][a_idx] += 1;
        }

        env.step(&action);
        let env_ref: &dyn Environment = &*env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        if step < window {
            homeo_early += homeo_dev(env_ref);
        }
        if step >= STEPS - window {
            homeo_late += homeo_dev(env_ref);
        }
        if step == STEPS - 1 {
            wm_late = d.loss_world_model;
        }
    }
    option_lengths.push(current_length);

    let mut distinct = std::collections::HashSet::new();
    for row in &per_option_tail {
        let total: u32 = row.iter().sum();
        if total == 0 { continue; }
        let (best_a, _) = row.iter().enumerate().max_by_key(|&(_, c)| *c).unwrap();
        distinct.insert(best_a);
    }

    let mean_len = if option_lengths.is_empty() {
        0.0
    } else {
        option_lengths.iter().map(|&x| x as f32).sum::<f32>() / option_lengths.len() as f32
    };

    RunOut {
        distinct: distinct.len(),
        mean_option_length: mean_len,
        homeo_drop: (homeo_early - homeo_late) / window as f32,
        wm_late,
    }
}

fn main() {
    env_logger::init();
    println!(
        "learned-termination comparison ({STEPS} steps, num_options={NUM_OPTIONS}, \
         option_horizon={OPTION_HORIZON})\n"
    );
    println!(
        "{:>12} | {:>24} | {:>24}",
        "env", "fixed (learn_term=off)", "learned (learn_term=on)"
    );
    println!(
        "{:>12} | {:>6} {:>7} {:>9} | {:>6} {:>7} {:>9}",
        "", "dist", "meanL", "Δhomeo", "dist", "meanL", "Δhomeo"
    );

    type Factory = Box<dyn Fn() -> (Box<dyn Environment>, Box<dyn kindle::EnvAdapter>)>;
    use kindle_gym::*;
    let envs: Vec<(&'static str, Factory)> = vec![
        ("GridWorld", Box::new(|| (
            Box::new(GridWorld::new()) as Box<dyn Environment>,
            Box::new(GenericAdapter::discrete(0, grid_world::OBS_DIM, grid_world::NUM_ACTIONS))
                as Box<dyn kindle::EnvAdapter>,
        ))),
        ("CartPole", Box::new(|| (
            Box::new(CartPole::new()), Box::new(GenericAdapter::discrete(1, 4, 2)),
        ))),
        ("MountainCar", Box::new(|| (
            Box::new(MountainCar::new()), Box::new(GenericAdapter::discrete(2, 2, 3)),
        ))),
        ("Acrobot", Box::new(|| (
            Box::new(Acrobot::new()), Box::new(GenericAdapter::discrete(3, 6, 3)),
        ))),
        ("Taxi", Box::new(|| (
            Box::new(Taxi::new()),
            Box::new(GenericAdapter::discrete(4, taxi::OBS_DIM, taxi::NUM_ACTIONS)),
        ))),
        ("RandomWalk", Box::new(|| (
            Box::new(RandomWalk::new(10)), Box::new(GenericAdapter::discrete(5, 10, 2)),
        ))),
        ("Pendulum", Box::new(|| (
            Box::new(Pendulum::new()),
            Box::new(GenericAdapter::continuous(6, 3, 1, 0.5)),
        ))),
    ];

    for (name, factory) in &envs {
        let (e1, a1) = factory();
        let fixed = run(e1, a1, false);
        let (e2, a2) = factory();
        let learned = run(e2, a2, true);
        println!(
            "{:>12} | {:>6} {:>7.2} {:>+9.2} | {:>6} {:>7.2} {:>+9.2}",
            name,
            fixed.distinct, fixed.mean_option_length, fixed.homeo_drop,
            learned.distinct, learned.mean_option_length, learned.homeo_drop,
        );
    }
}
