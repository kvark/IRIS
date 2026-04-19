//! M6 learnable reward: outcome-value head (CPU).
//!
//! Small MLP `R̂(z_t) → scalar` trained by MSE against full-episode
//! Monte-Carlo returns, centered by a running EMA baseline. Acts as
//! a fifth reward primitive alongside the four frozen ones
//! (surprise / novelty / homeostatic / order). See
//! `docs/phase-m6-learnable-reward.md` for the design.
//!
//! CPU implementation rationale: at kindle's shapes (latent_dim=16 →
//! hidden=32 → 1 ≈ 560 parameters) the MLP is trivially fast in
//! Rust. A dedicated GPU session would force awkward batch-size
//! choices — per-step inference at N lanes vs. per-episode training
//! at L trajectory rows — and cost more in dispatch overhead than
//! the compute saves. Plain SGD suffices; stop-grad into the
//! encoder is automatic at the CPU boundary.

pub struct OutcomeHead {
    pub latent_dim: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    w1: Vec<f32>, // [hidden_dim, latent_dim] row-major
    b1: Vec<f32>, // [hidden_dim]
    w2: Vec<f32>, // [hidden_dim]
    b2: f32,
    pub last_loss: f32,
}

impl OutcomeHead {
    pub fn new(latent_dim: usize, hidden_dim: usize, lr: f32, seed: u64) -> Self {
        Self {
            latent_dim,
            hidden_dim,
            lr,
            w1: xavier(hidden_dim, latent_dim, seed),
            b1: vec![0.0; hidden_dim],
            w2: xavier(1, hidden_dim, seed.wrapping_add(1)),
            b2: 0.0,
            last_loss: 0.0,
        }
    }

    /// Forward for a single latent `z`, returning `R̂(z)`.
    pub fn forward(&self, z: &[f32]) -> f32 {
        debug_assert_eq!(z.len(), self.latent_dim);
        let mut out = self.b2;
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.latent_dim..(j + 1) * self.latent_dim];
            for k in 0..self.latent_dim {
                acc += row[k] * z[k];
            }
            if acc > 0.0 {
                out += self.w2[j] * acc;
            }
        }
        out
    }

    /// Train on one episode's trajectory with a single averaged SGD
    /// step. All rows share the same scalar `target` (the centered
    /// episode return). Returns the mean squared loss.
    #[allow(clippy::needless_range_loop)]
    pub fn train_batch(&mut self, zs: &[Vec<f32>], target: f32) -> f32 {
        if zs.is_empty() {
            return 0.0;
        }
        let n = zs.len() as f32;
        let inv_n = 1.0 / n;

        let mut gw1 = vec![0.0f32; self.hidden_dim * self.latent_dim];
        let mut gb1 = vec![0.0f32; self.hidden_dim];
        let mut gw2 = vec![0.0f32; self.hidden_dim];
        let mut gb2 = 0.0f32;
        let mut loss_sum = 0.0f32;

        let mut h = vec![0.0f32; self.hidden_dim];
        let mut mask = vec![false; self.hidden_dim];

        for z in zs {
            // Forward
            for j in 0..self.hidden_dim {
                let mut acc = self.b1[j];
                let row = &self.w1[j * self.latent_dim..(j + 1) * self.latent_dim];
                for k in 0..self.latent_dim {
                    acc += row[k] * z[k];
                }
                mask[j] = acc > 0.0;
                h[j] = if mask[j] { acc } else { 0.0 };
            }
            let mut y = self.b2;
            for j in 0..self.hidden_dim {
                y += self.w2[j] * h[j];
            }

            // MSE on 0.5·(y−t)^2 gives dL/dy = (y−t).
            let d_y = y - target;
            loss_sum += d_y * d_y;

            // Backprop
            gb2 += d_y;
            for j in 0..self.hidden_dim {
                gw2[j] += d_y * h[j];
                if !mask[j] {
                    continue;
                }
                let d_h = d_y * self.w2[j];
                gb1[j] += d_h;
                let row = &mut gw1[j * self.latent_dim..(j + 1) * self.latent_dim];
                for k in 0..self.latent_dim {
                    row[k] += d_h * z[k];
                }
            }
        }

        let lr = self.lr;
        for j in 0..self.hidden_dim {
            self.w2[j] -= lr * gw2[j] * inv_n;
            self.b1[j] -= lr * gb1[j] * inv_n;
            let off = j * self.latent_dim;
            for k in 0..self.latent_dim {
                self.w1[off + k] -= lr * gw1[off + k] * inv_n;
            }
        }
        self.b2 -= lr * gb2 * inv_n;

        let mean_loss = loss_sum * inv_n;
        self.last_loss = mean_loss;
        mean_loss
    }
}

fn xavier(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (6.0 / (rows + cols) as f32).sqrt();
    let n = rows * cols;
    (0..n)
        .map(|i| {
            let h =
                ((seed as f64 + i as f64 * 1.234_567) * 0.618_033_988_749_895).fract() as f32;
            (h * PI * 2.0).sin() * scale
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_head_fits_constant_target() {
        let mut head = OutcomeHead::new(8, 16, 1e-2, 42);
        let z = vec![0.1f32; 8];
        let zs = vec![z.clone(); 32];
        let target = 2.0;
        // A few epochs should drive the loss near zero on a constant target.
        let mut last = 0.0;
        for _ in 0..200 {
            last = head.train_batch(&zs, target);
        }
        assert!(last < 1e-2, "outcome head failed to fit constant: {last}");
        let pred = head.forward(&z);
        assert!((pred - target).abs() < 0.1, "pred {pred} far from target {target}");
    }
}
