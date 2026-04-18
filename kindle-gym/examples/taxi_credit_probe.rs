//! Investigate: why does Taxi regress when the L1 credit assigner is
//! on vs off?
//!
//! Runs three configs on Taxi (seed 42, 3k steps) and reports
//! early/late homeo_dev + distinct_modal_actions:
//!   A) L0-only (num_options=1)
//!   B) L1 without credit assigner (num_options=4, option_history_len=1)
//!   C) L1 with credit assigner (num_options=4, option_history_len=8)
//!
//! Run: `cargo run --release --example taxi_credit_probe`

use kindle::{Action, Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::taxi::{Taxi, NUM_ACTIONS, OBS_DIM};
use rand::SeedableRng;

const STEPS: usize = 3000;
const WINDOW: usize = 200;

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

fn run(label: &str, num_options: usize, option_history_len: usize, learned_term: bool) {
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
        option_history_len,
        learned_termination: learned_term,
        ..AgentConfig::default()
    };
    let adapter = Box::new(GenericAdapter::discrete(4, OBS_DIM, NUM_ACTIONS));
    let mut agent = Agent::new(config, vec![adapter]);
    let mut env = Taxi::new();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut per_option_tail: Vec<Vec<u32>> = (0..num_options.max(1))
        .map(|_| vec![0u32; 6])
        .collect();
    let mut homeo_sum_early = 0.0f32;
    let mut homeo_sum_late = 0.0f32;
    let mut wm_late = 0.0f32;
    let mut h_eff_l1_late = 0.0f32;

    for step in 0..STEPS {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let d = agent.diagnostics()[0].clone();
        let opt = d.current_option as usize;
        let a_idx = match &action {
            Action::Discrete(i) => *i,
            _ => 0,
        };
        if step >= STEPS - 1000 && opt < num_options && a_idx < 6 {
            per_option_tail[opt][a_idx] += 1;
        }

        env.step(&action);
        let env_ref: &dyn Environment = &env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        if step < WINDOW {
            homeo_sum_early += homeo_dev(env_ref);
        }
        if step >= STEPS - WINDOW {
            homeo_sum_late += homeo_dev(env_ref);
        }
        if step == STEPS - 1 {
            wm_late = d.loss_world_model;
            h_eff_l1_late = d.h_eff_l1;
        }
    }

    let mut distinct = std::collections::HashSet::new();
    for row in &per_option_tail {
        let total: u32 = row.iter().sum();
        if total == 0 {
            continue;
        }
        let (best_a, _) = row.iter().enumerate().max_by_key(|&(_, c)| *c).unwrap();
        distinct.insert(best_a);
    }

    println!(
        "  {:40} | wm_late={:.3} h_eff_l1={:.2} homeo {:.2}→{:.2} (Δ={:+.2}) distinct={}",
        label,
        wm_late,
        h_eff_l1_late,
        homeo_sum_early / WINDOW as f32,
        homeo_sum_late / WINDOW as f32,
        (homeo_sum_late - homeo_sum_early) / WINDOW as f32,
        distinct.len(),
    );
}

fn main() {
    env_logger::init();
    println!("Taxi L1-credit probe ({STEPS} steps, seed 42)\n");
    println!(
        "{:42} | {:16} {:16} {:20} distinct",
        "config", "wm", "h_eff_l1", "homeo_dev"
    );
    run("A) L0-only (num_options=1)", 1, 8, false);
    run("B) L1 no-credit (history_len=1)", 4, 1, false);
    run("C) L1 + credit (history_len=8)", 4, 8, false);
    run("D) L1 + credit + learned-term", 4, 8, true);
}
