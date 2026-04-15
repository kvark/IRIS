//! Run the kindle agent across multiple environments, hopping between them
//! on a schedule. Tests cross-env transfer: the agent's encoder, world
//! model, credit assigner, and policy all persist through env switches;
//! only the adapter changes.
//!
//! Run: `cargo run --example multi_env`

use kindle::{Agent, AgentConfig, Environment, GenericAdapter};
use kindle_gym::{cart_pole::CartPole, grid_world::GridWorld, mountain_car::MountainCar};
use rand::SeedableRng;

/// Wraps a concrete environment as a `&mut dyn Environment` target.
/// Having the three envs co-exist lets us switch between them cheaply.
struct Envs {
    grid: GridWorld,
    cart: CartPole,
    mountain: MountainCar,
}

#[derive(Clone, Copy)]
enum Which {
    Grid,
    Cart,
    Mountain,
}

impl Envs {
    fn new() -> Self {
        Self {
            grid: GridWorld::new(),
            cart: CartPole::new(),
            mountain: MountainCar::new(),
        }
    }

    fn observe(&self, which: Which) -> kindle::Observation {
        match which {
            Which::Grid => self.grid.observe(),
            Which::Cart => self.cart.observe(),
            Which::Mountain => self.mountain.observe(),
        }
    }

    fn step(&mut self, which: Which, action: &kindle::Action) {
        match which {
            Which::Grid => {
                self.grid.step(action);
            }
            Which::Cart => {
                self.cart.step(action);
            }
            Which::Mountain => {
                self.mountain.step(action);
            }
        }
    }

    fn env(&self, which: Which) -> &dyn Environment {
        match which {
            Which::Grid => &self.grid,
            Which::Cart => &self.cart,
            Which::Mountain => &self.mountain,
        }
    }
}

fn main() {
    env_logger::init();

    println!("kindle Multi-Env (Grid → Cart → Mountain rotating)");
    println!("==================================================");

    let mut envs = Envs::new();

    // Start in GridWorld
    let adapter = Box::new(GenericAdapter::discrete(
        0,
        kindle_gym::grid_world::OBS_DIM,
        kindle_gym::grid_world::NUM_ACTIONS,
    ));
    let config = AgentConfig {
        latent_dim: 8,
        hidden_dim: 32,
        history_len: 8,
        buffer_capacity: 5000,
        batch_size: 1,
        // Conservative LR — works across all three envs
        learning_rate: 1e-4,
        lr_credit: 3e-5,
        lr_policy: 5e-5,
        warmup_steps: 200,
        ..AgentConfig::default()
    };

    println!("building agent (compiling graphs — universal sizes)...");
    let mut agent = Agent::new(config, vec![adapter]);
    println!("agent ready");

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Hop between envs every `hop_every` steps. Full cycle: Grid → Cart
    // → Mountain → Grid → ... Each env gets `hop_every` consecutive
    // steps to let the agent adapt before the next switch.
    let hop_every = 500;
    let total_steps = 3000;
    let order = [Which::Grid, Which::Cart, Which::Mountain];

    let _ = &envs; // keep envs alive for the loop

    for step in 0..total_steps {
        let phase = (step / hop_every) % order.len();
        let which = order[phase];

        // At each phase boundary, swap the adapter.
        if step > 0 && step % hop_every == 0 {
            let adapter: Box<dyn kindle::EnvAdapter> = match which {
                Which::Grid => Box::new(GenericAdapter::discrete(
                    0,
                    kindle_gym::grid_world::OBS_DIM,
                    kindle_gym::grid_world::NUM_ACTIONS,
                )),
                Which::Cart => Box::new(GenericAdapter::discrete(1, 4, 2)),
                Which::Mountain => Box::new(GenericAdapter::discrete(2, 2, 3)),
            };
            agent.switch_lane(0, adapter);
            let name = match which {
                Which::Grid => "GridWorld",
                Which::Cart => "CartPole",
                Which::Mountain => "MountainCar",
            };
            println!(
                "\n--- step {step}: hopping to {name} (env_id={}) ---",
                agent.env_id(0)
            );
        }

        let obs = envs.observe(which);
        let action = agent
            .act(std::slice::from_ref(&obs), &mut rng)
            .remove(0);
        envs.step(which, &action);
        let env_ref: &dyn Environment = envs.env(which);
        agent.observe(
            std::slice::from_ref(&obs),
            std::slice::from_ref(&action),
            std::slice::from_ref(&env_ref),
            &mut rng,
        );

        if (step + 1) % 250 == 0 {
            let diags = agent.diagnostics();
            let d = &diags[0];
            println!(
                "step {:>4} env={} | wm={:.4} cr={:.4} pi={:.4} | r={:.3} drift={:.3} buf={}",
                d.step,
                d.env_id,
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
