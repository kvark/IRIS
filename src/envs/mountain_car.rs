//! MountainCar: drive an underpowered car up a steep hill.
//!
//! Ported from OpenAI Gymnasium `MountainCar-v0`.
//!
//! The car is on a 1D track between two hills. The engine is too weak to
//! climb directly — the agent must learn to build momentum by rocking
//! back and forth. This makes it a good test for exploration-driven
//! intrinsic rewards (novelty bonus for reaching new positions).
//!
//! Observation (2-dim, normalized): `[position, velocity]`
//! Actions (3): push left, no push, push right.
//! Homeostatic: position (target = goal at 0.5, tolerance 0.2).

use crate::env::*;

const MIN_POSITION: f32 = -1.2;
const MAX_POSITION: f32 = 0.6;
const MAX_SPEED: f32 = 0.07;
const GOAL_POSITION: f32 = 0.5;
const FORCE: f32 = 0.001;
const GRAVITY: f32 = 0.0025;

pub struct MountainCar {
    position: f32,
    velocity: f32,
    step_count: usize,
    max_steps: usize,
    homeo: Vec<HomeostaticVariable>,
}

impl MountainCar {
    pub fn new() -> Self {
        let mut env = Self {
            position: -0.5,
            velocity: 0.0,
            step_count: 0,
            max_steps: 200,
            homeo: Vec::new(),
        };
        env.update_homeo();
        env
    }

    fn update_homeo(&mut self) {
        self.homeo = vec![HomeostaticVariable {
            value: self.position,
            target: GOAL_POSITION,
            tolerance: 0.2,
        }];
    }
}

impl HomeostaticProvider for MountainCar {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for MountainCar {
    fn observation_dim(&self) -> usize {
        2
    }

    fn num_actions(&self) -> usize {
        3
    }

    fn observe(&self) -> Observation {
        // Normalize to roughly [-1, 1]
        let pos_norm = (self.position - MIN_POSITION) / (MAX_POSITION - MIN_POSITION) * 2.0 - 1.0;
        let vel_norm = self.velocity / MAX_SPEED;
        Observation::new(vec![pos_norm, vel_norm])
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let &Action::Discrete(a) = action else {
            panic!("MountainCar uses discrete actions");
        };

        // Physics: v += (action - 1) * force - cos(3 * position) * gravity
        self.velocity += (a as f32 - 1.0) * FORCE - (3.0 * self.position).cos() * GRAVITY;
        self.velocity = self.velocity.clamp(-MAX_SPEED, MAX_SPEED);

        self.position += self.velocity;
        self.position = self.position.clamp(MIN_POSITION, MAX_POSITION);

        // Bounce off left wall
        if self.position <= MIN_POSITION && self.velocity < 0.0 {
            self.velocity = 0.0;
        }

        self.step_count += 1;

        // Auto-reset on goal or truncation (continual learning)
        let reached_goal = self.position >= GOAL_POSITION;
        let truncated = self.step_count >= self.max_steps;
        if reached_goal || truncated {
            self.reset();
        }

        self.update_homeo();

        StepResult {
            observation: self.observe(),
            homeostatic: self.homeo.clone(),
        }
    }

    fn reset(&mut self) {
        self.position = -0.5;
        self.velocity = 0.0;
        self.step_count = 0;
        self.update_homeo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mountain_car_basics() {
        let mut env = MountainCar::new();
        assert_eq!(env.observation_dim(), 2);
        assert_eq!(env.num_actions(), 3);

        // Push right for a while
        for _ in 0..50 {
            env.step(&Action::Discrete(2));
        }
        // Should have moved right from -0.5
        assert!(env.position > -0.5);
    }

    #[test]
    fn mountain_car_auto_resets() {
        let mut env = MountainCar::new();
        // Run 250 steps — should auto-reset at 200
        for _ in 0..250 {
            env.step(&Action::Discrete(1)); // no push
        }
        // After reset, step_count should be < 200
        assert!(env.step_count < 200);
    }

    #[test]
    fn mountain_car_momentum() {
        let mut env = MountainCar::new();
        // Rock back and forth: left then right, within one episode
        for _ in 0..30 {
            env.step(&Action::Discrete(0)); // push left
        }
        for _ in 0..30 {
            env.step(&Action::Discrete(2)); // push right
        }
        // Should have moved from starting position
        assert!(env.position != -0.5 || env.velocity.abs() > 0.0);
    }
}
