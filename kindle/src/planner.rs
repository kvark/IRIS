//! Model-based planning with the world model.
//!
//! Kindle's world model `W(z_t, a_t) → z_{t+1}` is an MLP trained
//! alongside the policy. This module exposes a CPU-side rollout of
//! `W` so we can simulate candidate action sequences without touching
//! the real environment. Planning selects the sequence whose
//! predicted trajectory maximizes a novelty score — pulling the
//! actual policy toward regions of the latent space the WM hasn't
//! mapped well yet.
//!
//! The WM weights are pulled from the live meganeura session
//! periodically (via `refresh_from_session`). Between refreshes the
//! CPU rollout operates on frozen copies, so planning can be called
//! many times per training step without racing the graph.

use meganeura::Session;

/// CPU-side copy of the world model's parameters, sufficient to
/// replicate its forward pass on arbitrary `(z, action)` inputs.
///
/// Layout matches `WorldModel` in `world_model.rs`:
///   h = relu(z @ W_z + b_z + a @ W_a)
///   h = relu(h @ W_fc2 + b_fc2)
///   z_hat = h @ W_fc_out
pub struct WmRollout {
    pub latent_dim: usize,
    pub action_dim: usize,
    pub hidden_dim: usize,
    w_z: Vec<f32>,
    b_z: Vec<f32>,
    w_a: Vec<f32>,
    w_fc2: Vec<f32>,
    b_fc2: Vec<f32>,
    w_fc_out: Vec<f32>,
}

impl WmRollout {
    pub fn new(latent_dim: usize, action_dim: usize, hidden_dim: usize) -> Self {
        Self {
            latent_dim,
            action_dim,
            hidden_dim,
            w_z: vec![0.0; latent_dim * hidden_dim],
            b_z: vec![0.0; hidden_dim],
            w_a: vec![0.0; action_dim * hidden_dim],
            w_fc2: vec![0.0; hidden_dim * hidden_dim],
            b_fc2: vec![0.0; hidden_dim],
            w_fc_out: vec![0.0; hidden_dim * latent_dim],
        }
    }

    /// Pull the current WM weights out of the WM session into the
    /// CPU cache. Call periodically (e.g. every `refresh_interval`
    /// observe steps) so plans reflect the WM as it's trained.
    pub fn refresh_from_session(&mut self, wm_session: &Session) {
        wm_session.read_param("world_model.z_proj.weight", &mut self.w_z);
        wm_session.read_param("world_model.z_proj.bias", &mut self.b_z);
        wm_session.read_param("world_model.a_proj.weight", &mut self.w_a);
        wm_session.read_param("world_model.fc2.weight", &mut self.w_fc2);
        wm_session.read_param("world_model.fc2.bias", &mut self.b_fc2);
        wm_session.read_param("world_model.fc_out.weight", &mut self.w_fc_out);
    }

    /// One WM forward step: `(z, action_one_hot) → z_hat_next`.
    /// Caller provides `out` with capacity `latent_dim`.
    #[allow(clippy::needless_range_loop)]
    pub fn forward_step(&self, z: &[f32], action_one_hot: &[f32], out: &mut [f32]) {
        debug_assert_eq!(z.len(), self.latent_dim);
        debug_assert_eq!(action_one_hot.len(), self.action_dim);
        debug_assert_eq!(out.len(), self.latent_dim);

        // h = z @ W_z (+ b_z) + a @ W_a, then relu.
        let mut h = vec![0.0f32; self.hidden_dim];
        for k in 0..self.hidden_dim {
            let mut acc = self.b_z[k];
            for i in 0..self.latent_dim {
                acc += z[i] * self.w_z[i * self.hidden_dim + k];
            }
            for j in 0..self.action_dim {
                acc += action_one_hot[j] * self.w_a[j * self.hidden_dim + k];
            }
            h[k] = acc.max(0.0);
        }

        // h2 = relu(h @ W_fc2 + b_fc2).
        let mut h2 = vec![0.0f32; self.hidden_dim];
        for k in 0..self.hidden_dim {
            let mut acc = self.b_fc2[k];
            for j in 0..self.hidden_dim {
                acc += h[j] * self.w_fc2[j * self.hidden_dim + k];
            }
            h2[k] = acc.max(0.0);
        }

        // z_hat = h2 @ W_fc_out (no bias).
        for i in 0..self.latent_dim {
            let mut acc = 0.0f32;
            for k in 0..self.hidden_dim {
                acc += h2[k] * self.w_fc_out[k * self.latent_dim + i];
            }
            out[i] = acc;
        }
    }

    /// Roll out a K-action sequence starting from `z0`. Writes the
    /// predicted latents into `trajectory` (shape `K * latent_dim`,
    /// one row per step). The caller owns the action encoder —
    /// `one_hot` is a scratch `action_dim`-length buffer reused
    /// each step.
    pub fn rollout(
        &self,
        z0: &[f32],
        actions: &[u32],
        one_hot: &mut [f32],
        trajectory: &mut [f32],
    ) {
        debug_assert_eq!(z0.len(), self.latent_dim);
        debug_assert_eq!(one_hot.len(), self.action_dim);
        debug_assert_eq!(trajectory.len(), actions.len() * self.latent_dim);

        let mut z_cur = z0.to_vec();
        for (step, &a) in actions.iter().enumerate() {
            // One-hot encode this step's action.
            for slot in one_hot.iter_mut() {
                *slot = 0.0;
            }
            let a_idx = a as usize;
            if a_idx < self.action_dim {
                one_hot[a_idx] = 1.0;
            }
            let out_row = &mut trajectory[step * self.latent_dim..(step + 1) * self.latent_dim];
            self.forward_step(&z_cur, one_hot, out_row);
            z_cur.copy_from_slice(out_row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construct a WmRollout with known weights and check that
    /// `forward_step` produces the expected output. Weights chosen
    /// so the ReLUs don't trigger.
    #[test]
    fn forward_step_matches_manual_math() {
        let mut wm = WmRollout::new(2, 2, 2);
        // W_z: identity → h_z = z.
        wm.w_z = vec![1.0, 0.0, 0.0, 1.0];
        wm.b_z = vec![0.5, 0.5];
        // W_a: identity → h_a = a.
        wm.w_a = vec![1.0, 0.0, 0.0, 1.0];
        // W_fc2: identity → h2 = h.
        wm.w_fc2 = vec![1.0, 0.0, 0.0, 1.0];
        wm.b_fc2 = vec![0.0, 0.0];
        // W_fc_out: identity → z_hat = h2.
        wm.w_fc_out = vec![1.0, 0.0, 0.0, 1.0];

        let z = vec![0.3, 0.7];
        let a = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];
        wm.forward_step(&z, &a, &mut out);
        // h = relu([0.3 + 0.5 + 1.0, 0.7 + 0.5 + 0.0]) = relu([1.8, 1.2]) = [1.8, 1.2]
        // h2 = relu([1.8, 1.2]) = [1.8, 1.2]
        // z_hat = [1.8, 1.2]
        assert!((out[0] - 1.8).abs() < 1e-5, "out[0]: {}", out[0]);
        assert!((out[1] - 1.2).abs() < 1e-5, "out[1]: {}", out[1]);
    }

    #[test]
    fn relu_zeros_negative_pre_activations() {
        let mut wm = WmRollout::new(2, 2, 2);
        // W_z: identity, b_z: -2 so pre-relu is negative.
        wm.w_z = vec![1.0, 0.0, 0.0, 1.0];
        wm.b_z = vec![-2.0, -2.0];
        wm.w_a = vec![0.0, 0.0, 0.0, 0.0];
        wm.w_fc2 = vec![1.0, 0.0, 0.0, 1.0];
        wm.b_fc2 = vec![0.0, 0.0];
        wm.w_fc_out = vec![1.0, 0.0, 0.0, 1.0];

        let z = vec![0.1, 0.1];
        let a = vec![1.0, 0.0];
        let mut out = vec![0.0; 2];
        wm.forward_step(&z, &a, &mut out);
        // Pre-relu: [0.1 - 2.0, 0.1 - 2.0] = [-1.9, -1.9] → relu → [0, 0].
        assert_eq!(out, vec![0.0, 0.0]);
    }

    #[test]
    fn rollout_chains_k_steps() {
        // Identity WM: z_hat = z + a (no relu effect because we pick
        // positive inputs). So rollout with actions [0, 1, 0] on
        // z0=[1,0] should give sequence [2,0], [2,1], [3,1].
        let mut wm = WmRollout::new(2, 2, 2);
        wm.w_z = vec![1.0, 0.0, 0.0, 1.0];
        wm.b_z = vec![0.0, 0.0];
        wm.w_a = vec![1.0, 0.0, 0.0, 1.0];
        wm.w_fc2 = vec![1.0, 0.0, 0.0, 1.0];
        wm.b_fc2 = vec![0.0, 0.0];
        wm.w_fc_out = vec![1.0, 0.0, 0.0, 1.0];

        let z0 = vec![1.0, 0.0];
        let actions = vec![0u32, 1, 0];
        let mut one_hot = vec![0.0; 2];
        let mut traj = vec![0.0; 3 * 2];
        wm.rollout(&z0, &actions, &mut one_hot, &mut traj);
        // Step 0 (a=0): h = [1+1, 0+0] = [2, 0] → out [2, 0]
        // Step 1 (a=1): h = [2+0, 0+1] = [2, 1] → out [2, 1]
        // Step 2 (a=0): h = [2+1, 1+0] = [3, 1] → out [3, 1]
        assert!((traj[0] - 2.0).abs() < 1e-5);
        assert!((traj[1] - 0.0).abs() < 1e-5);
        assert!((traj[2] - 2.0).abs() < 1e-5);
        assert!((traj[3] - 1.0).abs() < 1e-5);
        assert!((traj[4] - 3.0).abs() < 1e-5);
        assert!((traj[5] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rollout_with_zero_length_is_noop() {
        let wm = WmRollout::new(2, 2, 2);
        let z0 = vec![1.0, 2.0];
        let actions: Vec<u32> = vec![];
        let mut one_hot = vec![0.0; 2];
        let mut traj: Vec<f32> = vec![];
        wm.rollout(&z0, &actions, &mut one_hot, &mut traj);
        // No steps taken — trajectory is empty, no panic.
        assert!(traj.is_empty());
    }

    #[test]
    fn new_allocates_correct_sizes() {
        let wm = WmRollout::new(16, 6, 32);
        assert_eq!(wm.w_z.len(), 16 * 32);
        assert_eq!(wm.b_z.len(), 32);
        assert_eq!(wm.w_a.len(), 6 * 32);
        assert_eq!(wm.w_fc2.len(), 32 * 32);
        assert_eq!(wm.b_fc2.len(), 32);
        assert_eq!(wm.w_fc_out.len(), 32 * 16);
    }
}
