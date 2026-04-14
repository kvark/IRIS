//! Acrobot: swing a two-link robot arm to reach a target height.
//!
//! Ported from OpenAI Gymnasium `Acrobot-v1` (Sutton & Barto variant).
//!
//! A two-link pendulum hangs downward. Only the second joint is actuated.
//! The goal is to swing the tip above a height threshold. This requires
//! learning multi-step momentum strategies — a strong test for credit
//! assignment over long horizons.
//!
//! Observation (6-dim): `[cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁, θ̇₂]`
//! Actions (3): torque -1, 0, +1 on the actuated joint.
//! Homeostatic: tip height (target = above threshold, tolerance 0.1).

use crate::env::*;
use std::f32::consts::PI;

const M1: f32 = 1.0;
const M2: f32 = 1.0;
const L1: f32 = 1.0;
const _L2: f32 = 1.0; // unused in physics (lc2 used instead) but kept for reference
const LC1: f32 = 0.5;
const LC2: f32 = 0.5;
const I1: f32 = 1.0;
const I2: f32 = 1.0;
const G: f32 = 9.8;
const DT: f32 = 0.2;
const MAX_VEL1: f32 = 4.0 * PI;
const MAX_VEL2: f32 = 9.0 * PI;

pub struct Acrobot {
    theta1: f32,
    theta2: f32,
    dtheta1: f32,
    dtheta2: f32,
    step_count: usize,
    max_steps: usize,
    homeo: Vec<HomeostaticVariable>,
}

impl Acrobot {
    pub fn new() -> Self {
        let mut env = Self {
            theta1: 0.05,
            theta2: 0.05,
            dtheta1: 0.0,
            dtheta2: 0.0,
            step_count: 0,
            max_steps: 500,
            homeo: Vec::new(),
        };
        env.update_homeo();
        env
    }

    fn tip_height(&self) -> f32 {
        -self.theta1.cos() - (self.theta2 + self.theta1).cos()
    }

    fn update_homeo(&mut self) {
        self.homeo = vec![HomeostaticVariable {
            value: -self.tip_height(), // negate so higher tip → lower value → closer to target
            target: -1.0,             // tip above threshold
            tolerance: 0.1,
        }];
    }

    /// Compute derivatives: [dθ₁, dθ₂, ddθ₁, ddθ₂]
    fn derivatives(&self, theta1: f32, theta2: f32, dtheta1: f32, dtheta2: f32, torque: f32) -> [f32; 4] {
        let d1 = M1 * LC1 * LC1
            + M2 * (L1 * L1 + LC2 * LC2 + 2.0 * L1 * LC2 * theta2.cos())
            + I1
            + I2;
        let d2 = M2 * (LC2 * LC2 + L1 * LC2 * theta2.cos()) + I2;

        let phi2 = M2 * LC2 * G * (theta1 + theta2 - PI / 2.0).cos();
        let phi1 = -M2 * L1 * LC2 * dtheta2 * dtheta2 * theta2.sin()
            - 2.0 * M2 * L1 * LC2 * dtheta2 * dtheta1 * theta2.sin()
            + (M1 * LC1 + M2 * L1) * G * (theta1 - PI / 2.0).cos()
            + phi2;

        let ddtheta2 = (torque + (d2 / d1) * phi1
            - M2 * L1 * LC2 * dtheta1 * dtheta1 * theta2.sin()
            - phi2)
            / (M2 * LC2 * LC2 + I2 - d2 * d2 / d1);
        let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

        [dtheta1, dtheta2, ddtheta1, ddtheta2]
    }

    /// RK4 integration step.
    fn rk4_step(&mut self, torque: f32) {
        let s = [self.theta1, self.theta2, self.dtheta1, self.dtheta2];

        let k1 = self.derivatives(s[0], s[1], s[2], s[3], torque);

        let s2: Vec<f32> = s.iter().zip(k1.iter()).map(|(si, ki)| si + DT / 2.0 * ki).collect();
        let k2 = self.derivatives(s2[0], s2[1], s2[2], s2[3], torque);

        let s3: Vec<f32> = s.iter().zip(k2.iter()).map(|(si, ki)| si + DT / 2.0 * ki).collect();
        let k3 = self.derivatives(s3[0], s3[1], s3[2], s3[3], torque);

        let s4: Vec<f32> = s.iter().zip(k3.iter()).map(|(si, ki)| si + DT * ki).collect();
        let k4 = self.derivatives(s4[0], s4[1], s4[2], s4[3], torque);

        for i in 0..4 {
            let ds = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            match i {
                0 => self.theta1 += DT * ds,
                1 => self.theta2 += DT * ds,
                2 => self.dtheta1 += DT * ds,
                3 => self.dtheta2 += DT * ds,
                _ => unreachable!(),
            }
        }

        self.theta1 = wrap_angle(self.theta1);
        self.theta2 = wrap_angle(self.theta2);
        self.dtheta1 = self.dtheta1.clamp(-MAX_VEL1, MAX_VEL1);
        self.dtheta2 = self.dtheta2.clamp(-MAX_VEL2, MAX_VEL2);
    }
}

fn wrap_angle(x: f32) -> f32 {
    let mut a = (x + PI) % (2.0 * PI);
    if a < 0.0 {
        a += 2.0 * PI;
    }
    a - PI
}

impl HomeostaticProvider for Acrobot {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for Acrobot {
    fn observation_dim(&self) -> usize {
        6
    }

    fn num_actions(&self) -> usize {
        3
    }

    fn observe(&self) -> Observation {
        // [cos θ₁, sin θ₁, cos θ₂, sin θ₂, θ̇₁ (normalized), θ̇₂ (normalized)]
        Observation::new(vec![
            self.theta1.cos(),
            self.theta1.sin(),
            self.theta2.cos(),
            self.theta2.sin(),
            self.dtheta1 / MAX_VEL1,
            self.dtheta2 / MAX_VEL2,
        ])
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let &Action::Discrete(a) = action else {
            panic!("Acrobot uses discrete actions");
        };

        let torque = a as f32 - 1.0; // {0,1,2} → {-1, 0, +1}
        self.rk4_step(torque);
        self.step_count += 1;

        // Auto-reset on goal or truncation
        let reached_goal = self.tip_height() > 1.0;
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
        self.theta1 = 0.05;
        self.theta2 = 0.05;
        self.dtheta1 = 0.0;
        self.dtheta2 = 0.0;
        self.step_count = 0;
        self.update_homeo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acrobot_basics() {
        let env = Acrobot::new();
        assert_eq!(env.observation_dim(), 6);
        assert_eq!(env.num_actions(), 3);

        let obs = env.observe();
        // cos and sin of small angles
        assert!((obs.data[0] - 1.0).abs() < 0.01); // cos(0.05) ≈ 1
        assert!((obs.data[1] - 0.05).abs() < 0.01); // sin(0.05) ≈ 0.05
    }

    #[test]
    fn acrobot_physics_runs() {
        let mut env = Acrobot::new();
        // Apply torque for 100 steps
        for _ in 0..100 {
            env.step(&Action::Discrete(2)); // +1 torque
        }
        // Should have built angular momentum
        assert!(env.dtheta1.abs() > 0.0 || env.dtheta2.abs() > 0.0);
    }

    #[test]
    fn wrap_angle_test() {
        assert!((wrap_angle(0.0)).abs() < 1e-6);
        assert!((wrap_angle(PI) - PI).abs() < 1e-5 || (wrap_angle(PI) + PI).abs() < 1e-5);
        assert!((wrap_angle(3.0 * PI) - PI).abs() < 1e-5 || (wrap_angle(3.0 * PI) + PI).abs() < 1e-5);
    }
}
