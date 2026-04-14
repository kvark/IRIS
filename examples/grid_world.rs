//! Toy grid world environment for early IRIS development.
//!
//! A 5x5 grid where the agent starts at (0,0). Three food sources
//! replenish energy. The agent's energy decays each step — maintaining
//! energy is the homeostatic challenge.
//!
//! Run: `cargo run --example grid_world`

use iris::env::{
    Action, Environment, HomeostaticProvider, HomeostaticVariable, Observation, StepResult,
};
use iris::{Agent, AgentConfig};
use rand::SeedableRng;

const WIDTH: usize = 5;
const HEIGHT: usize = 5;
const OBS_DIM: usize = WIDTH * HEIGHT + 1; // one-hot position + energy
const NUM_ACTIONS: usize = 4; // up, down, left, right

struct GridWorld {
    pos: (usize, usize),
    energy: f32,
    food: Vec<(usize, usize)>,
    homeo: Vec<HomeostaticVariable>,
}

impl GridWorld {
    fn new() -> Self {
        let food = vec![(1, 3), (3, 1), (4, 4)];
        let mut w = Self {
            pos: (0, 0),
            energy: 1.0,
            food,
            homeo: Vec::new(),
        };
        w.update_homeo();
        w
    }

    fn update_homeo(&mut self) {
        self.homeo = vec![HomeostaticVariable {
            value: self.energy,
            target: 0.6,
            tolerance: 0.3,
        }];
    }

    fn make_obs(&self) -> Observation {
        let mut data = vec![0.0f32; OBS_DIM];
        data[self.pos.1 * WIDTH + self.pos.0] = 1.0;
        data[WIDTH * HEIGHT] = self.energy;
        Observation::new(data)
    }
}

impl HomeostaticProvider for GridWorld {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for GridWorld {
    fn observation_dim(&self) -> usize {
        OBS_DIM
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    fn observe(&self) -> Observation {
        self.make_obs()
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let Action::Discrete(a) = action else {
            panic!("grid world uses discrete actions");
        };

        // Move
        match a {
            0 if self.pos.1 > 0 => self.pos.1 -= 1,         // up
            1 if self.pos.1 < HEIGHT - 1 => self.pos.1 += 1, // down
            2 if self.pos.0 > 0 => self.pos.0 -= 1,         // left
            3 if self.pos.0 < WIDTH - 1 => self.pos.0 += 1,  // right
            _ => {}                                           // wall or invalid
        }

        // Energy dynamics
        self.energy = (self.energy - 0.05).max(0.0);
        if self.food.contains(&self.pos) {
            self.energy = (self.energy + 0.3).min(1.0);
        }

        self.update_homeo();

        StepResult {
            observation: self.make_obs(),
            homeostatic: self.homeo.clone(),
        }
    }

    fn reset(&mut self) {
        self.pos = (0, 0);
        self.energy = 1.0;
        self.update_homeo();
    }
}

fn main() {
    env_logger::init();

    println!("IRIS Grid World — Phase 1");
    println!("=========================");

    let mut env = GridWorld::new();
    let config = AgentConfig {
        obs_dim: OBS_DIM,
        action_dim: NUM_ACTIONS,
        latent_dim: 8,
        hidden_dim: 16,
        history_len: 8,
        buffer_capacity: 1000,
        batch_size: 1,
        learning_rate: 1e-3,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graph)...");
    let mut agent = Agent::new(config);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let num_steps = 200;

    for step in 0..num_steps {
        let obs = env.observe();
        let action = agent.act(&mut rng);
        let _result = env.step(&action);

        agent.observe(&obs, &action, &env, &mut rng);

        if (step + 1) % 50 == 0 {
            let diag = agent.diagnostics();
            println!(
                "step {:>4} | wm={:.4} cr={:.4} | r={:.3} (s={:.3} n={:.3} h={:.3}) | H_eff={:.1} buf={}",
                diag.step,
                diag.loss_world_model,
                diag.loss_credit,
                diag.reward_mean,
                diag.reward_surprise,
                diag.reward_novelty,
                diag.reward_homeo,
                diag.h_eff,
                diag.buffer_len,
            );
        }
    }

    let diag = agent.diagnostics();
    println!("\nfinal diagnostics:");
    println!("{}", serde_json::to_string_pretty(&diag).unwrap());
}
