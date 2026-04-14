//! Pendulum from OpenAI Gymnasium — inverted pendulum swing-up.
//!
//! The pendulum starts in a random position and must be swung upright.
//! Continuous 1-D torque action in [-2, 2].
//!
//! Physics: `θ̈ = (3g)/(2l)·sin(θ) + (3)/(m·l²)·u`, Euler integration dt=0.05
//!
//! Observation (3-dim, normalized): `[cos θ, sin θ, θ̇ / max_speed]`
//! Actions (continuous, 1-dim, clamped to [-1, 1] then scaled to [-2, 2])
//! Homeostatic: angle magnitude (target 0, tolerance 0.1 rad)

use crate::env::*;
use std::f32::consts::PI;

const G: f32 = 10.0;
const M: f32 = 1.0;
const L: f32 = 1.0;
const DT: f32 = 0.05;
const MAX_SPEED: f32 = 8.0;
const MAX_TORQUE: f32 = 2.0;

pub struct Pendulum {
    theta: f32,
    theta_dot: f32,
    step_count: usize,
    max_steps: usize,
    homeo: Vec<HomeostaticVariable>,
}

impl Pendulum {
    pub fn new() -> Self {
        let mut env = Self {
            theta: PI, // hanging down
            theta_dot: 0.0,
            step_count: 0,
            max_steps: 200,
            homeo: Vec::new(),
        };
        env.update_homeo();
        env
    }

    fn normalize_angle(x: f32) -> f32 {
        let mut a = (x + PI) % (2.0 * PI);
        if a < 0.0 {
            a += 2.0 * PI;
        }
        a - PI
    }

    fn update_homeo(&mut self) {
        let theta_norm = Self::normalize_angle(self.theta);
        self.homeo = vec![HomeostaticVariable {
            value: theta_norm.abs(),
            target: 0.0,
            tolerance: 0.1,
        }];
    }
}

impl HomeostaticProvider for Pendulum {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for Pendulum {
    fn observation_dim(&self) -> usize {
        3
    }

    fn num_actions(&self) -> usize {
        1 // continuous action dim
    }

    fn observe(&self) -> Observation {
        Observation::new(vec![
            self.theta.cos(),
            self.theta.sin(),
            self.theta_dot / MAX_SPEED,
        ])
    }

    #[allow(clippy::needless_borrowed_reference)]
    fn step(&mut self, action: &Action) -> StepResult {
        let &Action::Continuous(ref a) = action else {
            panic!("Pendulum uses continuous actions");
        };
        // Clamp to [-1, 1] and scale to [-MAX_TORQUE, MAX_TORQUE]
        let u = a.first().copied().unwrap_or(0.0).clamp(-1.0, 1.0) * MAX_TORQUE;

        // Physics: θ̈ = (3g)/(2l) sin(θ) + 3/(m·l²) u
        let theta_acc = (3.0 * G) / (2.0 * L) * self.theta.sin() + 3.0 / (M * L * L) * u;

        self.theta_dot += theta_acc * DT;
        self.theta_dot = self.theta_dot.clamp(-MAX_SPEED, MAX_SPEED);
        self.theta += self.theta_dot * DT;
        self.theta = Self::normalize_angle(self.theta);

        self.step_count += 1;
        if self.step_count >= self.max_steps {
            self.reset();
        }
        self.update_homeo();

        StepResult {
            observation: self.observe(),
            homeostatic: self.homeo.clone(),
        }
    }

    fn reset(&mut self) {
        self.theta = PI;
        self.theta_dot = 0.0;
        self.step_count = 0;
        self.update_homeo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pendulum_basics() {
        let env = Pendulum::new();
        assert_eq!(env.observation_dim(), 3);
        let obs = env.observe();
        // theta = π: cos(π) = -1, sin(π) ≈ 0
        assert!((obs.data[0] + 1.0).abs() < 0.01);
    }

    #[test]
    fn pendulum_gravity_swings() {
        let mut env = Pendulum::new();
        // Slightly perturb from straight down
        env.theta = PI - 0.3;
        let zero = Action::Continuous(vec![0.0]);
        for _ in 0..20 {
            env.step(&zero);
        }
        // Should have built angular velocity due to gravity
        assert!(env.theta_dot.abs() > 0.0);
    }

    #[test]
    fn pendulum_torque_clamping() {
        let mut env = Pendulum::new();
        // Way out-of-range action should be clamped internally
        let huge = Action::Continuous(vec![100.0]);
        for _ in 0..5 {
            env.step(&huge);
        }
        assert!(env.theta_dot.is_finite());
    }
}
