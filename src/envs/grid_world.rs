//! 5x5 grid world with homeostatic energy.
//!
//! The agent navigates a grid with food sources that replenish energy.
//! Energy decays each step. The homeostatic challenge is maintaining energy
//! within a healthy range.

use crate::env::*;

pub const WIDTH: usize = 5;
pub const HEIGHT: usize = 5;
pub const OBS_DIM: usize = WIDTH * HEIGHT + 1;
pub const NUM_ACTIONS: usize = 4;

pub struct GridWorld {
    pub pos: (usize, usize),
    pub energy: f32,
    food: Vec<(usize, usize)>,
    homeo: Vec<HomeostaticVariable>,
}

impl GridWorld {
    pub fn new() -> Self {
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
        let &Action::Discrete(a) = action else {
            panic!("grid world uses discrete actions");
        };

        match a {
            0 if self.pos.1 > 0 => self.pos.1 -= 1,
            1 if self.pos.1 < HEIGHT - 1 => self.pos.1 += 1,
            2 if self.pos.0 > 0 => self.pos.0 -= 1,
            3 if self.pos.0 < WIDTH - 1 => self.pos.0 += 1,
            _ => {}
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_world_food_replenishes_energy() {
        let mut env = GridWorld::new();
        // Walk to food at (1,3): right, down, down, down
        env.step(&Action::Discrete(3)); // right to (1,0)
        env.step(&Action::Discrete(1)); // down to (1,1)
        env.step(&Action::Discrete(1)); // down to (1,2)
        let pre_food = env.energy;
        env.step(&Action::Discrete(1)); // down to (1,3) — food!
        assert!(env.energy > pre_food, "food should replenish energy");
    }

    #[test]
    fn grid_world_energy_decays() {
        let mut env = GridWorld::new();
        let e0 = env.energy;
        env.step(&Action::Discrete(0)); // move (away from food)
        assert!(env.energy < e0);
    }
}
