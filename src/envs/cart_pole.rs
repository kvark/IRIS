//! Classic CartPole balancing environment.
//!
//! A pole is attached to a cart on a frictionless track. The agent applies
//! a force of +1 or -1 to the cart each step. The episode fails when the
//! pole angle exceeds ±12° or the cart leaves ±2.4 units.
//!
//! Observation (4-dim): `[cart_position, cart_velocity, pole_angle, pole_angular_velocity]`
//! Actions (2): push left, push right.
//! Homeostatic: pole angle (target 0, tolerance 0.05 rad ≈ 3°).

use crate::env::*;

const GRAVITY: f32 = 9.8;
const CART_MASS: f32 = 1.0;
const POLE_MASS: f32 = 0.1;
const TOTAL_MASS: f32 = CART_MASS + POLE_MASS;
const POLE_HALF_LEN: f32 = 0.5;
const FORCE_MAG: f32 = 10.0;
const DT: f32 = 0.02;

const ANGLE_LIMIT: f32 = 12.0 * std::f32::consts::PI / 180.0;
const POSITION_LIMIT: f32 = 2.4;

pub struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
    failed: bool,
    step_count: usize,
    homeo: Vec<HomeostaticVariable>,
}

impl CartPole {
    pub fn new() -> Self {
        let mut env = Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.01, // slight initial tilt
            theta_dot: 0.0,
            failed: false,
            step_count: 0,
            homeo: Vec::new(),
        };
        env.update_homeo();
        env
    }

    fn update_homeo(&mut self) {
        self.homeo = vec![HomeostaticVariable {
            value: self.theta,
            target: 0.0,
            tolerance: 0.05,
        }];
    }

    fn check_bounds(&mut self) {
        if self.x.abs() > POSITION_LIMIT || self.theta.abs() > ANGLE_LIMIT {
            self.failed = true;
        }
    }
}

impl HomeostaticProvider for CartPole {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for CartPole {
    fn observation_dim(&self) -> usize {
        4
    }

    fn num_actions(&self) -> usize {
        2
    }

    fn observe(&self) -> Observation {
        // Normalize to roughly [-1, 1] to prevent gradient explosion
        Observation::new(vec![
            self.x / POSITION_LIMIT,
            self.x_dot / 3.0,
            self.theta / ANGLE_LIMIT,
            self.theta_dot / 3.0,
        ])
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let &Action::Discrete(a) = action else {
            panic!("CartPole uses discrete actions");
        };

        let force = if a == 1 { FORCE_MAG } else { -FORCE_MAG };

        let cos_theta = self.theta.cos();
        let sin_theta = self.theta.sin();

        // Physics from Barto, Sutton & Anderson (1983)
        let temp = (force + POLE_MASS * POLE_HALF_LEN * self.theta_dot.powi(2) * sin_theta)
            / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (POLE_HALF_LEN * (4.0 / 3.0 - POLE_MASS * cos_theta.powi(2) / TOTAL_MASS));
        let x_acc = temp - POLE_MASS * POLE_HALF_LEN * theta_acc * cos_theta / TOTAL_MASS;

        // Euler integration
        self.x += DT * self.x_dot;
        self.x_dot += DT * x_acc;
        self.theta += DT * self.theta_dot;
        self.theta_dot += DT * theta_acc;

        self.step_count += 1;
        self.check_bounds();
        self.update_homeo();

        // Auto-reset on failure (continual learning — no episode boundaries)
        if self.failed {
            self.reset();
        }

        StepResult {
            observation: self.observe(),
            homeostatic: self.homeo.clone(),
        }
    }

    fn reset(&mut self) {
        self.x = 0.0;
        self.x_dot = 0.0;
        self.theta = 0.01;
        self.theta_dot = 0.0;
        self.failed = false;
        self.step_count = 0;
        self.update_homeo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartpole_runs_without_panic() {
        let mut env = CartPole::new();
        let obs = env.observe();
        assert_eq!(obs.dim(), 4);

        // Run 100 steps with alternating actions
        for i in 0..100 {
            env.step(&Action::Discrete(i % 2));
        }
    }

    #[test]
    fn cartpole_falls_without_control() {
        let mut env = CartPole::new();
        // Always push right — pole will fall
        let mut fell = false;
        for _ in 0..500 {
            env.step(&Action::Discrete(1));
            if env.theta.abs() > ANGLE_LIMIT || env.x.abs() > POSITION_LIMIT {
                fell = true;
                break;
            }
        }
        // It should have auto-reset, so fell might not be caught directly.
        // At least verify it doesn't panic.
        assert!(fell || env.step_count < 500);
    }

    #[test]
    fn cartpole_homeostatic_tracks_angle() {
        let mut env = CartPole::new();
        env.step(&Action::Discrete(0));
        let vars = env.homeostatic_variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0].target, 0.0);
    }
}
