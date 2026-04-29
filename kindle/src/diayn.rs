//! DIAYN-style mutual-information bonus over L1 options.
//!
//! Eysenbach et al. 2018, "Diversity is All You Need". Trains a
//! **discriminator** `q(option | z)` to predict which option produced
//! a given latent `z`. The intrinsic reward is the per-step
//! log-likelihood of the taken option under the discriminator, minus
//! the (uniform) prior `log(1/num_options)`:
//!
//!   r_intrinsic_t  =  log q(option_t | z_t)  +  log(num_options)
//!
//! Gradient of the policy then pushes each option toward producing
//! z trajectories that are MAXIMALLY DISTINGUISHABLE from the others
//! — i.e., each option learns a distinct skill, even in absence of
//! any extrinsic reward signal.
//!
//! In kindle this composes with the existing L1 options framework:
//! the option index that L1 selects is the "skill" index here. With
//! `num_options >= 2` and `diayn_reward_alpha > 0`, the discriminator
//! is built and its bonus is added per-step.
//!
//! CPU implementation mirrors `rnd.rs`. The discriminator is a small
//! 2-layer MLP with softmax + cross-entropy training. At kindle's
//! latent_dim ≈ 16 and num_options ≤ 8, the per-step cost is
//! negligible and the dependency-free Rust impl is simpler than a
//! meganeura graph for this size.

/// 2-layer MLP with softmax output for discriminator.
struct DiscriminatorMlp {
    in_dim: usize,
    hidden_dim: usize,
    num_classes: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

impl DiscriminatorMlp {
    fn new(in_dim: usize, hidden_dim: usize, num_classes: usize, seed: u64) -> Self {
        let w1 = xavier(hidden_dim, in_dim, seed);
        let b1 = vec![0.0; hidden_dim];
        let w2 = xavier(num_classes, hidden_dim, seed.wrapping_add(1));
        let b2 = vec![0.0; num_classes];
        Self {
            in_dim,
            hidden_dim,
            num_classes,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Forward → softmax probabilities `[num_classes]`.
    fn forward_softmax(&self, z: &[f32]) -> Vec<f32> {
        debug_assert_eq!(z.len(), self.in_dim);
        let mut h = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.in_dim..(j + 1) * self.in_dim];
            for k in 0..self.in_dim {
                acc += row[k] * z[k];
            }
            h[j] = acc.max(0.0);
        }
        let mut logits = vec![0.0; self.num_classes];
        for c in 0..self.num_classes {
            let mut acc = self.b2[c];
            let row = &self.w2[c * self.hidden_dim..(c + 1) * self.hidden_dim];
            for j in 0..self.hidden_dim {
                acc += row[j] * h[j];
            }
            logits[c] = acc;
        }
        softmax(&logits)
    }

    /// One SGD step against a one-hot target. Returns
    /// `(probs, mean_log_lik_of_target)`.
    fn train_step(&mut self, z: &[f32], target_class: usize, lr: f32) -> (Vec<f32>, f32) {
        debug_assert_eq!(z.len(), self.in_dim);
        debug_assert!(target_class < self.num_classes);

        // Forward (need pre-activations + mask for backprop).
        let mut h_pre = vec![0.0; self.hidden_dim];
        let mut mask = vec![false; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.in_dim..(j + 1) * self.in_dim];
            for k in 0..self.in_dim {
                acc += row[k] * z[k];
            }
            h_pre[j] = acc;
            mask[j] = acc > 0.0;
        }
        let h: Vec<f32> = h_pre
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        let mut logits = vec![0.0; self.num_classes];
        for c in 0..self.num_classes {
            let mut acc = self.b2[c];
            let row = &self.w2[c * self.hidden_dim..(c + 1) * self.hidden_dim];
            for j in 0..self.hidden_dim {
                acc += row[j] * h[j];
            }
            logits[c] = acc;
        }
        let probs = softmax(&logits);

        // CE loss gradient w.r.t. logits: probs[c] - one_hot[c]
        let mut d_logits = probs.clone();
        d_logits[target_class] -= 1.0;

        // Backprop into w2 and b2.
        let mut d_h = vec![0.0; self.hidden_dim];
        for c in 0..self.num_classes {
            self.b2[c] -= lr * d_logits[c];
            let row_off = c * self.hidden_dim;
            for j in 0..self.hidden_dim {
                self.w2[row_off + j] -= lr * d_logits[c] * h[j];
                if mask[j] {
                    d_h[j] += d_logits[c] * self.w2[row_off + j];
                }
            }
        }

        // Backprop into w1 and b1 (only through ReLU-active hidden units).
        for j in 0..self.hidden_dim {
            if !mask[j] {
                continue;
            }
            self.b1[j] -= lr * d_h[j];
            let row_off = j * self.in_dim;
            for k in 0..self.in_dim {
                self.w1[row_off + k] -= lr * d_h[j] * z[k];
            }
        }

        let log_lik_target = probs[target_class].max(1e-30).ln();
        (probs, log_lik_target)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
}

/// Same splitmix64-based Xavier init pattern as `rnd::xavier`.
/// Handles full-range u64 seeds correctly (avoids f64 precision loss).
fn xavier(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (6.0 / (rows + cols) as f32).sqrt();
    let n = rows * cols;
    (0..n as u64)
        .map(|i| {
            let mut x = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            x ^= x >> 33;
            let u = (x >> 40) as f32 / (1u64 << 24) as f32;
            (u * PI * 2.0).sin() * scale
        })
        .collect()
}

/// DIAYN state: a single discriminator over (z → option_idx).
pub struct DiaynState {
    pub num_options: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    pub last_log_lik: f32,
    discriminator: DiscriminatorMlp,
}

impl DiaynState {
    pub fn new(
        latent_dim: usize,
        num_options: usize,
        hidden_dim: usize,
        lr: f32,
        seed: u64,
    ) -> Self {
        Self {
            num_options,
            hidden_dim,
            lr,
            last_log_lik: -(num_options as f32).ln(), // log(1/K) at init
            discriminator: DiscriminatorMlp::new(latent_dim, hidden_dim, num_options, seed),
        }
    }

    /// Train discriminator on (z, option_idx) and return the intrinsic
    /// reward: `log q(option_idx | z) + log(num_options)`.
    /// Reward is positive when the discriminator predicts the option
    /// better than uniform, negative when worse.
    pub fn step(&mut self, z: &[f32], option_idx: usize) -> f32 {
        let (_probs, log_lik) = self.discriminator.train_step(z, option_idx, self.lr);
        self.last_log_lik = log_lik;
        // Reward: log q(option | z) - log p(option) where p is uniform = 1/K
        // = log q(option | z) + log(K)
        log_lik + (self.num_options as f32).ln()
    }

    /// Pure forward — same reward formula but doesn't train.
    pub fn reward(&self, z: &[f32], option_idx: usize) -> f32 {
        let probs = self.discriminator.forward_softmax(z);
        let log_lik = probs[option_idx].max(1e-30).ln();
        log_lik + (self.num_options as f32).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diayn_init_reward_near_zero() {
        // At init, discriminator should be near uniform → reward near 0.
        let d = DiaynState::new(4, 4, 16, 0.01, 42);
        let z = vec![0.5, -0.3, 0.1, 0.2];
        let r = d.reward(&z, 2);
        // log(1/4) + log(4) = 0; init has small randomness.
        assert!(r.abs() < 0.5, "init reward should be near 0, got {}", r);
    }

    #[test]
    fn diayn_learns_to_distinguish_distinct_z_per_option() {
        // Train on synthetic data where each option's z is in a
        // separate quadrant; discriminator should learn to predict
        // the right option and reward should rise above 0.
        let mut d = DiaynState::new(2, 4, 16, 0.05, 7);
        // Distinct centroids per option.
        let centroids = vec![
            vec![1.0, 0.0],  // option 0
            vec![0.0, 1.0],  // option 1
            vec![-1.0, 0.0], // option 2
            vec![0.0, -1.0], // option 3
        ];
        let mut last_rewards = vec![0.0; 4];
        for _epoch in 0..2000 {
            for opt in 0..4 {
                let r = d.step(&centroids[opt], opt);
                last_rewards[opt] = r;
            }
        }
        // After training, each option's reward should be well above 0
        // (discriminator learned to separate them).
        for opt in 0..4 {
            assert!(
                last_rewards[opt] > 0.5,
                "option {} reward {} should be > 0.5 after training",
                opt,
                last_rewards[opt]
            );
        }
    }

    #[test]
    fn diayn_indistinguishable_z_gives_low_average_reward() {
        // If all options share the same z, the discriminator can't
        // learn to consistently predict ONE option (each batch tries
        // to push the prediction toward a different target). The
        // discriminator oscillates near uniform, so the AVERAGE reward
        // across all options is near 0 (= log(1/K) + log(K)).
        let mut d = DiaynState::new(2, 4, 16, 0.005, 13);
        let z = vec![0.5, 0.5];
        for _ in 0..500 {
            for opt in 0..4 {
                d.step(&z, opt);
            }
        }
        // Average reward across all option queries — should be near 0
        // because softmax(uniform_logits) gives 1/K to each → log(1/K)
        // + log(K) = 0 exactly.
        let avg_r: f32 = (0..4).map(|opt| d.reward(&z, opt)).sum::<f32>() / 4.0;
        assert!(
            avg_r.abs() < 0.5,
            "indistinguishable z should give ~0 average reward across options, got {}",
            avg_r
        );
    }
}
