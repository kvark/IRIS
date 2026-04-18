//! L1 diagnostic across all 7 gymnasium envs.
//!
//! Produces four outputs the user asked for:
//!   1. Convergence of the world-model and homeostatic-deviation signal
//!      on every env (does training progress?).
//!   2. L1 options as reasonable commands — histogram of option selections
//!      across the run + a text translation of each option's modal L0
//!      action, so each "option index" reads as the verb it is driving
//!      L0 to execute.
//!   3. Self-improvement — homeostatic deviation drop (early vs late) per
//!      env, and for CartPole the estimated episode length trend.
//!   4. Skill transfer — pretrain on CartPole, then measure the first 1k
//!      steps on MountainCar warm-started vs cold-started. Compares wm
//!      loss + homeo improvement.
//!
//! Run: `cargo run --release --example l1_diagnostic`

use kindle::{Action, Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::{
    acrobot::Acrobot, cart_pole::CartPole, grid_world::GridWorld, mountain_car::MountainCar,
    pendulum::Pendulum, random_walk::RandomWalk, taxi::Taxi,
};
use rand::SeedableRng;

/// Human-readable text labels for each env's discrete action set.
/// For continuous envs, we bucket the first action dim (sign of torque).
fn action_text(env_id: u32, action: &Action) -> &'static str {
    match (env_id, action) {
        // GridWorld: 4 actions (y-up, y-down, x-left, x-right from source)
        (0, Action::Discrete(0)) => "N",
        (0, Action::Discrete(1)) => "S",
        (0, Action::Discrete(2)) => "W",
        (0, Action::Discrete(3)) => "E",
        // CartPole: 2 actions
        (1, Action::Discrete(0)) => "pushL",
        (1, Action::Discrete(1)) => "pushR",
        // MountainCar: 3 actions
        (2, Action::Discrete(0)) => "accelL",
        (2, Action::Discrete(1)) => "coast",
        (2, Action::Discrete(2)) => "accelR",
        // Acrobot: 3 actions
        (3, Action::Discrete(0)) => "torqueN",
        (3, Action::Discrete(1)) => "torque0",
        (3, Action::Discrete(2)) => "torqueP",
        // Taxi: 6 actions
        (4, Action::Discrete(0)) => "S",
        (4, Action::Discrete(1)) => "N",
        (4, Action::Discrete(2)) => "E",
        (4, Action::Discrete(3)) => "W",
        (4, Action::Discrete(4)) => "pickup",
        (4, Action::Discrete(5)) => "dropoff",
        // RandomWalk: 2 actions
        (5, Action::Discrete(0)) => "left",
        (5, Action::Discrete(1)) => "right",
        // Pendulum: continuous
        (6, Action::Continuous(v)) if !v.is_empty() && v[0] > 0.0 => "torqueP",
        (6, Action::Continuous(_)) => "torqueN",
        _ => "?",
    }
}

/// Text label for each L1 option (orthogonal goal direction in latent space).
fn option_text(opt: u32, option_dim: usize, num_options: usize) -> String {
    // Goal table maps option o → ±0.5 in latent dim (o % option_dim),
    // sign alternates in each wrap-around. See option::build_goal_table.
    let dim = (opt as usize) % option_dim;
    let wrap = (opt as usize) / option_dim;
    let sign = if wrap.is_multiple_of(2) { "+" } else { "-" };
    let _ = num_options;
    format!("goal{sign}z{dim}")
}

/// Homeostatic deviation magnitude: Σ |value − target| / tolerance.
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

struct EnvRun {
    name: &'static str,
    env_id: u32,
    env: Box<dyn Environment>,
    adapter: Box<dyn kindle::EnvAdapter>,
}

fn make_runs() -> Vec<EnvRun> {
    use kindle_gym::*;
    vec![
        EnvRun {
            name: "GridWorld",
            env_id: 0,
            env: Box::new(GridWorld::new()),
            adapter: Box::new(GenericAdapter::discrete(
                0,
                grid_world::OBS_DIM,
                grid_world::NUM_ACTIONS,
            )),
        },
        EnvRun {
            name: "CartPole",
            env_id: 1,
            env: Box::new(CartPole::new()),
            adapter: Box::new(GenericAdapter::discrete(1, 4, 2)),
        },
        EnvRun {
            name: "MountainCar",
            env_id: 2,
            env: Box::new(MountainCar::new()),
            adapter: Box::new(GenericAdapter::discrete(2, 2, 3)),
        },
        EnvRun {
            name: "Acrobot",
            env_id: 3,
            env: Box::new(Acrobot::new()),
            adapter: Box::new(GenericAdapter::discrete(3, 6, 3)),
        },
        EnvRun {
            name: "Taxi",
            env_id: 4,
            env: Box::new(Taxi::new()),
            adapter: Box::new(GenericAdapter::discrete(
                4,
                taxi::OBS_DIM,
                taxi::NUM_ACTIONS,
            )),
        },
        EnvRun {
            name: "RandomWalk",
            env_id: 5,
            env: Box::new(RandomWalk::new(10)),
            adapter: Box::new(GenericAdapter::discrete(5, 10, 2)),
        },
        EnvRun {
            name: "Pendulum",
            env_id: 6,
            env: Box::new(Pendulum::new()),
            adapter: Box::new(GenericAdapter::continuous(6, 3, 1, 0.5)),
        },
    ]
}

fn agent_config(num_options: usize) -> AgentConfig {
    AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 16,
        buffer_capacity: 5000,
        batch_size: 1,
        learning_rate: 1e-3,
        warmup_steps: 200,
        num_options,
        option_horizon: 10,
        ..AgentConfig::default()
    }
}

/// Stats collected during one training run.
#[allow(dead_code)]
struct RunStats {
    wm_early: f32,
    wm_late: f32,
    homeo_early: f32,
    homeo_late: f32,
    reward_mean: f32,
    /// Counts of option indices chosen this run.
    option_histogram: Vec<u32>,
    /// For each option, histogram over MAX_ACTION_DIM entries. `[o][a]` is
    /// the number of times action `a` was issued while option `o` was active.
    per_option_action: Vec<Vec<u32>>,
    steps: usize,
    final_entropy: f32,
}

fn run_l1_on_env(run: EnvRun, steps: usize, num_options: usize) -> RunStats {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let config = agent_config(num_options);
    let option_dim = if config.option_dim == 0 {
        config.latent_dim
    } else {
        config.option_dim
    };
    let mut agent = Agent::new(config.clone(), vec![run.adapter]);
    let mut env = run.env;

    let window = 200usize;
    let mut homeo_sum_early = 0.0f32;
    let mut homeo_sum_late = 0.0f32;
    let mut reward_sum_late = 0.0f32;
    let mut wm_early = f32::NAN;
    let mut wm_late = f32::NAN;
    let mut final_entropy = 0.0f32;

    let mut option_hist = vec![0u32; num_options];
    let mut per_option_action: Vec<Vec<u32>> = (0..num_options).map(|_| vec![0u32; 6]).collect();

    let mut text_samples: Vec<(usize, u32, &'static str)> = Vec::new();

    for step in 0..steps {
        let obs = env.observe();
        let action = agent.act(std::slice::from_ref(&obs), &mut rng).remove(0);
        let d_pre = agent.diagnostics()[0].clone();
        let opt = d_pre.current_option as usize;
        if opt < num_options {
            option_hist[opt] += 1;
            let a_idx = match &action {
                Action::Discrete(i) => *i,
                Action::Continuous(v) if !v.is_empty() => {
                    if v[0] > 0.0 {
                        0
                    } else {
                        1
                    }
                }
                Action::Continuous(_) => 0,
            };
            if a_idx < per_option_action[opt].len() {
                per_option_action[opt][a_idx] += 1;
            }
        }

        if step < 8 && text_samples.len() < 8 {
            text_samples.push((step, d_pre.current_option, action_text(run.env_id, &action)));
        }

        env.step(&action);
        let env_ref: &dyn Environment = &*env;
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        let dev = homeo_dev(env_ref);
        if step < window {
            homeo_sum_early += dev;
        }
        if step >= steps - window {
            homeo_sum_late += dev;
            reward_sum_late += agent.diagnostics()[0].reward_mean;
        }

        let d = agent.diagnostics()[0].clone();
        if step == window {
            wm_early = d.loss_world_model;
        }
        if step == steps - 1 {
            wm_late = d.loss_world_model;
            final_entropy = d.policy_entropy;
        }
    }

    let final_h_eff_l1 = agent.diagnostics()[0].h_eff_l1;
    println!(
        "  {}: wm {:.3}→{:.3}  homeo_dev {:.2}→{:.2}  reward_mean(late) {:+.2}  entropy {:.2}  h_eff_l1 {:.2}",
        run.name,
        wm_early,
        wm_late,
        homeo_sum_early / window as f32,
        homeo_sum_late / window as f32,
        reward_sum_late / window as f32,
        final_entropy,
        final_h_eff_l1,
    );

    // Option histogram and per-option modal action.
    let total: u32 = option_hist.iter().sum();
    print!("    option histogram: ");
    for (o, &c) in option_hist.iter().enumerate() {
        let pct = if total > 0 {
            100.0 * c as f32 / total as f32
        } else {
            0.0
        };
        print!(
            "[{}={} {:>4.1}%] ",
            option_text(o as u32, option_dim, num_options),
            c,
            pct
        );
    }
    println!();

    print!("    per-option modal L0 action: ");
    for (o, actions) in per_option_action.iter().enumerate().take(num_options) {
        let (best_a, best_c) = actions
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| *c)
            .map(|(i, &c)| (i, c))
            .unwrap_or((0, 0));
        let total_o: u32 = actions.iter().sum();
        let pct = if total_o > 0 {
            100.0 * best_c as f32 / total_o as f32
        } else {
            0.0
        };
        let act = match run.env_id {
            6 => Action::Continuous(vec![if best_a == 0 { 1.0 } else { -1.0 }]),
            _ => Action::Discrete(best_a),
        };
        print!(
            "[{}→{} {:.0}%] ",
            option_text(o as u32, option_dim, num_options),
            action_text(run.env_id, &act),
            pct
        );
    }
    println!();

    print!("    translated first steps: ");
    for (s, opt, act) in &text_samples {
        print!(
            "t={:<2}:{}/{}  ",
            s,
            option_text(*opt, option_dim, num_options),
            act
        );
    }
    println!();

    RunStats {
        wm_early,
        wm_late,
        homeo_early: homeo_sum_early / window as f32,
        homeo_late: homeo_sum_late / window as f32,
        reward_mean: reward_sum_late / window as f32,
        option_histogram: option_hist,
        per_option_action,
        steps,
        final_entropy,
    }
}

/// Transfer test: train on src_run for `pretrain` steps, then evaluate on
/// tgt_run by `switch_lane`-ing the adapter. Returns (warm_wm_first200,
/// warm_homeo_late, cold_wm_first200, cold_homeo_late).
fn skill_transfer(
    src: EnvRun,
    tgt_factory: impl Fn() -> EnvRun,
    pretrain: usize,
    target_steps: usize,
) -> (f32, f32, f32, f32) {
    println!(
        "\n  [warm] pretrain on {} for {} steps...",
        src.name, pretrain
    );
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let config = agent_config(4);
    let mut agent = Agent::new(config.clone(), vec![src.adapter]);
    let mut env = src.env;

    for _ in 0..pretrain {
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
    }
    let d = agent.diagnostics()[0].clone();
    println!(
        "  [warm] after pretrain: wm={:.3} ent={:.2}",
        d.loss_world_model, d.policy_entropy
    );

    // Swap to target env — the compiled GPU graphs stay the same.
    let tgt = tgt_factory();
    println!(
        "  [warm] switching lane 0 → {} (env_id={})",
        tgt.name, tgt.env_id
    );
    agent.switch_lane(0, tgt.adapter);
    let mut env = tgt.env;

    let mut warm_wm_sum = 0.0f32;
    let mut warm_wm_count = 0;
    let mut warm_homeo_sum = 0.0f32;
    let late_window = 200usize.min(target_steps / 2);

    for step in 0..target_steps {
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
        if step < 200 {
            warm_wm_sum += agent.diagnostics()[0].loss_world_model;
            warm_wm_count += 1;
        }
        if step >= target_steps - late_window {
            warm_homeo_sum += homeo_dev(env_ref);
        }
    }
    let warm_wm = warm_wm_sum / warm_wm_count.max(1) as f32;
    let warm_homeo = warm_homeo_sum / late_window as f32;

    // Cold baseline on same target.
    println!("  [cold] baseline on target from scratch...");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let tgt2 = tgt_factory();
    let mut agent = Agent::new(config, vec![tgt2.adapter]);
    let mut env = tgt2.env;
    let mut cold_wm_sum = 0.0f32;
    let mut cold_wm_count = 0;
    let mut cold_homeo_sum = 0.0f32;

    for step in 0..target_steps {
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
        if step < 200 {
            cold_wm_sum += agent.diagnostics()[0].loss_world_model;
            cold_wm_count += 1;
        }
        if step >= target_steps - late_window {
            cold_homeo_sum += homeo_dev(env_ref);
        }
    }
    let cold_wm = cold_wm_sum / cold_wm_count.max(1) as f32;
    let cold_homeo = cold_homeo_sum / late_window as f32;

    (warm_wm, warm_homeo, cold_wm, cold_homeo)
}

fn main() {
    env_logger::init();

    println!("=== kindle L1 diagnostic — all 7 gym envs, 2000 steps each ===\n");
    println!("Legend: wm = world model loss (early→late window=200)");
    println!("        homeo_dev = mean |x-target|/tolerance over homeo vars");
    println!("        option_text: goal±z<k> = L1 option with goal ±0.5 in latent dim k\n");

    let num_options = 4;
    let steps = 2000;

    let mut all_stats: Vec<(String, RunStats)> = Vec::new();
    for run in make_runs() {
        let name = run.name.to_string();
        println!("\n-- {} --", name);
        let stats = run_l1_on_env(run, steps, num_options);
        let _ = (stats.steps, stats.final_entropy); // silence dead-field warn
        all_stats.push((name, stats));
    }

    println!("\n=== Convergence + self-improvement summary ===");
    println!(
        "{:12} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | note",
        "env", "wm_early", "wm_late", "wm_drop", "homeo_e", "homeo_l"
    );
    for (name, s) in &all_stats {
        let wm_drop = if s.wm_early > 0.0 {
            s.wm_early - s.wm_late
        } else {
            0.0
        };
        let converged = s.wm_late < s.wm_early * 0.5 || s.wm_late < 0.1;
        let improved = s.homeo_late < s.homeo_early * 0.9;
        let note = match (converged, improved) {
            (true, true) => "wm-converge + task-improve",
            (true, false) => "wm-converge only",
            (false, true) => "task-improve only",
            (false, false) => "neither",
        };
        println!(
            "{:12} | {:>8.3} | {:>8.3} | {:>+8.3} | {:>8.2} | {:>8.2} | {}",
            name, s.wm_early, s.wm_late, wm_drop, s.homeo_early, s.homeo_late, note
        );
    }

    println!("\n=== L1 commands as text (per env, modal L0 action per option) ===");
    for (name, s) in &all_stats {
        let env_id = match name.as_str() {
            "GridWorld" => 0u32,
            "CartPole" => 1,
            "MountainCar" => 2,
            "Acrobot" => 3,
            "Taxi" => 4,
            "RandomWalk" => 5,
            "Pendulum" => 6,
            _ => 255,
        };
        print!("  {:12}: ", name);
        let mut distinct_modal = std::collections::HashSet::new();
        for (o, actions) in s.per_option_action.iter().enumerate() {
            let (best_a, best_c) = actions
                .iter()
                .enumerate()
                .max_by_key(|&(_, c)| *c)
                .map(|(i, &c)| (i, c))
                .unwrap_or((0, 0));
            let total: u32 = actions.iter().sum();
            if total == 0 {
                continue;
            }
            let pct = 100.0 * best_c as f32 / total as f32;
            let act = match env_id {
                6 => Action::Continuous(vec![if best_a == 0 { 1.0 } else { -1.0 }]),
                _ => Action::Discrete(best_a),
            };
            let txt = action_text(env_id, &act);
            distinct_modal.insert(txt);
            print!(
                "{}→{} ({:.0}%)  ",
                option_text(o as u32, 8, num_options),
                txt,
                pct
            );
        }
        let div = distinct_modal.len();
        print!("  [distinct_modal_actions={}]", div);
        println!();
    }

    println!("\n=== Skill transfer: CartPole → MountainCar (pretrain=2000, target=1500) ===");
    let src = make_runs()
        .into_iter()
        .find(|r| r.name == "CartPole")
        .unwrap();
    let (w_wm, w_homeo, c_wm, c_homeo) = skill_transfer(
        src,
        || {
            make_runs()
                .into_iter()
                .find(|r| r.name == "MountainCar")
                .unwrap()
        },
        2000,
        1500,
    );
    println!(
        "  warm (CartPole-pretrained): wm_first200={:.3}  homeo_late={:.2}",
        w_wm, w_homeo
    );
    println!(
        "  cold (from-scratch)      : wm_first200={:.3}  homeo_late={:.2}",
        c_wm, c_homeo
    );
    let wm_ratio = if c_wm > 0.0 { w_wm / c_wm } else { 1.0 };
    let homeo_ratio = if c_homeo > 0.0 {
        w_homeo / c_homeo
    } else {
        1.0
    };
    let transfer = if wm_ratio < 0.8 || homeo_ratio < 0.8 {
        "TRANSFER (warm better)"
    } else if wm_ratio > 1.25 || homeo_ratio > 1.25 {
        "INTERFERENCE (warm worse)"
    } else {
        "NEUTRAL"
    };
    println!(
        "  verdict: {} (wm ratio w/c = {:.2}, homeo ratio = {:.2})",
        transfer, wm_ratio, homeo_ratio
    );

    println!("\n=== Skill transfer: MountainCar → Acrobot (pretrain=2000, target=1500) ===");
    let src = make_runs()
        .into_iter()
        .find(|r| r.name == "MountainCar")
        .unwrap();
    let (w_wm, w_homeo, c_wm, c_homeo) = skill_transfer(
        src,
        || {
            make_runs()
                .into_iter()
                .find(|r| r.name == "Acrobot")
                .unwrap()
        },
        2000,
        1500,
    );
    println!(
        "  warm (MountainCar-pretrained): wm_first200={:.3}  homeo_late={:.2}",
        w_wm, w_homeo
    );
    println!(
        "  cold (from-scratch)         : wm_first200={:.3}  homeo_late={:.2}",
        c_wm, c_homeo
    );
    let wm_ratio = if c_wm > 0.0 { w_wm / c_wm } else { 1.0 };
    let homeo_ratio = if c_homeo > 0.0 {
        w_homeo / c_homeo
    } else {
        1.0
    };
    let transfer = if wm_ratio < 0.8 || homeo_ratio < 0.8 {
        "TRANSFER (warm better)"
    } else if wm_ratio > 1.25 || homeo_ratio > 1.25 {
        "INTERFERENCE (warm worse)"
    } else {
        "NEUTRAL"
    };
    println!(
        "  verdict: {} (wm ratio w/c = {:.2}, homeo ratio = {:.2})",
        transfer, wm_ratio, homeo_ratio
    );
}
