//! 1D random walk environment.
//!
//! The agent walks on an integer line [0, size). Actions: step left, step right.
//! Observation: one-hot position vector.
//! Homeostatic: distance from center (target 0, tolerance 2).
//!
//! This is the Canary 3 environment: a world model should learn to predict
//! next-step transitions perfectly on this deterministic environment.

use crate::env::*;

pub struct RandomWalk {
    pos: usize,
    size: usize,
    homeo: Vec<HomeostaticVariable>,
}

impl RandomWalk {
    pub fn new(size: usize) -> Self {
        let mut env = Self {
            pos: size / 2,
            size,
            homeo: Vec::new(),
        };
        env.update_homeo();
        env
    }

    fn update_homeo(&mut self) {
        let center = self.size as f32 / 2.0;
        self.homeo = vec![HomeostaticVariable {
            value: (self.pos as f32 - center).abs(),
            target: 0.0,
            tolerance: 2.0,
        }];
    }
}

impl HomeostaticProvider for RandomWalk {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for RandomWalk {
    fn observation_dim(&self) -> usize {
        self.size
    }

    fn num_actions(&self) -> usize {
        2
    }

    fn observe(&self) -> Observation {
        let mut data = vec![0.0f32; self.size];
        data[self.pos] = 1.0;
        Observation::new(data)
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let &Action::Discrete(a) = action else {
            panic!("RandomWalk uses discrete actions");
        };

        match a {
            0 if self.pos > 0 => self.pos -= 1,
            1 if self.pos < self.size - 1 => self.pos += 1,
            _ => {}
        }

        self.update_homeo();

        StepResult {
            observation: self.observe(),
            homeostatic: self.homeo.clone(),
        }
    }

    fn reset(&mut self) {
        self.pos = self.size / 2;
        self.update_homeo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_walk_basics() {
        let mut env = RandomWalk::new(10);
        assert_eq!(env.observe().dim(), 10);
        assert_eq!(env.num_actions(), 2);

        // Start at center (5), move right
        env.step(&Action::Discrete(1));
        let obs = env.observe();
        assert_eq!(obs.data[6], 1.0);

        // Move left twice
        env.step(&Action::Discrete(0));
        env.step(&Action::Discrete(0));
        let obs = env.observe();
        assert_eq!(obs.data[4], 1.0);
    }

    #[test]
    fn random_walk_clamps_at_edges() {
        let mut env = RandomWalk::new(5);
        env.pos = 0;
        env.step(&Action::Discrete(0)); // try to go left at 0
        assert_eq!(env.pos, 0);

        env.pos = 4;
        env.step(&Action::Discrete(1)); // try to go right at end
        assert_eq!(env.pos, 4);
    }
}
