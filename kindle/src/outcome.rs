//! M6 learnable reward: outcome-value head (CPU).
//!
//! Small MLP `R̂(window_z) → scalar` trained by MSE against
//! Monte-Carlo episode targets, centered by a running EMA baseline.
//! Acts as a fifth reward primitive alongside the four frozen ones
//! (surprise / novelty / homeostatic / order).
//!
//! Window support (M6 v2, 2026-04-19): the head accepts the
//! concatenation of the last `window` latents `[z_{t−window+1}, …,
//! z_t]`. `window == 1` reduces to the single-frame variant from
//! M6 v1 — this is the default and the back-compat path. `window
//! ≥ 2` adds trajectory-momentum information to the input, which
//! the 2026-04-19 mechanism check identified as the missing
//! ingredient for LunarLander: per-state R̂ couldn't discriminate
//! soft from crash even when the training target carried the
//! signal, because mid-flight z_t doesn't contain forward-
//! trajectory information. A windowed input carries "how the
//! trajectory has been evolving" — slow and upright vs. fast and
//! tumbling — which *is* outcome-predictive.
//!
//! CPU implementation rationale (unchanged from v1): at kindle's
//! shapes the MLP is trivially fast in Rust. Stop-grad into the
//! encoder is automatic at the CPU boundary.

pub struct OutcomeHead {
    pub latent_dim: usize,
    pub window: usize,
    pub hidden_dim: usize,
    pub lr: f32,
    /// Full input dim = `window * latent_dim`. Cached to avoid
    /// recomputing on every forward/backward.
    input_dim: usize,
    w1: Vec<f32>, // [hidden_dim, input_dim] row-major
    b1: Vec<f32>, // [hidden_dim]
    w2: Vec<f32>, // [hidden_dim]
    b2: f32,
    pub last_loss: f32,
}

impl OutcomeHead {
    pub fn new(latent_dim: usize, window: usize, hidden_dim: usize, lr: f32, seed: u64) -> Self {
        let window = window.max(1);
        let input_dim = latent_dim * window;
        Self {
            latent_dim,
            window,
            hidden_dim,
            lr,
            input_dim,
            w1: xavier(hidden_dim, input_dim, seed),
            b1: vec![0.0; hidden_dim],
            w2: xavier(1, hidden_dim, seed.wrapping_add(1)),
            b2: 0.0,
            last_loss: 0.0,
        }
    }

    /// Forward for one window. `input` must have length
    /// `window * latent_dim`, pre-flattened as
    /// `[z_{t−window+1} || … || z_t]`.
    pub fn forward(&self, input: &[f32]) -> f32 {
        debug_assert_eq!(input.len(), self.input_dim);
        let mut out = self.b2;
        for j in 0..self.hidden_dim {
            let mut acc = self.b1[j];
            let row = &self.w1[j * self.input_dim..(j + 1) * self.input_dim];
            for k in 0..self.input_dim {
                acc += row[k] * input[k];
            }
            if acc > 0.0 {
                out += self.w2[j] * acc;
            }
        }
        out
    }

    /// Train on a batch of pre-flattened windows against a shared
    /// scalar target. Thin wrapper over `train_batch_variable` that
    /// broadcasts the single target across every input.
    pub fn train_batch(&mut self, inputs: &[Vec<f32>], target: f32) -> f32 {
        if inputs.is_empty() {
            return 0.0;
        }
        let targets = vec![target; inputs.len()];
        self.train_batch_variable(inputs, &targets)
    }

    /// Train on a batch of pre-flattened windows with a per-sample
    /// target. Supports `OutcomeTarget::RewardToGo`, where each step
    /// of a completed episode gets its own target = Σ r_{k≥t}, so
    /// intra-episode windows carry differentiated supervision
    /// instead of the uniform episode-sum target. Returns mean loss.
    #[allow(clippy::needless_range_loop)]
    pub fn train_batch_variable(&mut self, inputs: &[Vec<f32>], targets: &[f32]) -> f32 {
        debug_assert_eq!(inputs.len(), targets.len());
        if inputs.is_empty() {
            return 0.0;
        }
        let n = inputs.len() as f32;
        let inv_n = 1.0 / n;

        let mut gw1 = vec![0.0f32; self.hidden_dim * self.input_dim];
        let mut gb1 = vec![0.0f32; self.hidden_dim];
        let mut gw2 = vec![0.0f32; self.hidden_dim];
        let mut gb2 = 0.0f32;
        let mut loss_sum = 0.0f32;

        let mut h = vec![0.0f32; self.hidden_dim];
        let mut mask = vec![false; self.hidden_dim];

        for (x, &target) in inputs.iter().zip(targets.iter()) {
            debug_assert_eq!(x.len(), self.input_dim);
            for j in 0..self.hidden_dim {
                let mut acc = self.b1[j];
                let row = &self.w1[j * self.input_dim..(j + 1) * self.input_dim];
                for k in 0..self.input_dim {
                    acc += row[k] * x[k];
                }
                mask[j] = acc > 0.0;
                h[j] = if mask[j] { acc } else { 0.0 };
            }
            let mut y = self.b2;
            for j in 0..self.hidden_dim {
                y += self.w2[j] * h[j];
            }

            let d_y = y - target;
            loss_sum += d_y * d_y;

            gb2 += d_y;
            for j in 0..self.hidden_dim {
                gw2[j] += d_y * h[j];
                if !mask[j] {
                    continue;
                }
                let d_h = d_y * self.w2[j];
                gb1[j] += d_h;
                let row = &mut gw1[j * self.input_dim..(j + 1) * self.input_dim];
                for k in 0..self.input_dim {
                    row[k] += d_h * x[k];
                }
            }
        }

        let lr = self.lr;
        for j in 0..self.hidden_dim {
            self.w2[j] -= lr * gw2[j] * inv_n;
            self.b1[j] -= lr * gb1[j] * inv_n;
            let off = j * self.input_dim;
            for k in 0..self.input_dim {
                self.w1[off + k] -= lr * gw1[off + k] * inv_n;
            }
        }
        self.b2 -= lr * gb2 * inv_n;

        let mean_loss = loss_sum * inv_n;
        self.last_loss = mean_loss;
        mean_loss
    }

    /// Build a window input from a trajectory of latents, ending at
    /// index `end` (inclusive). Left-pads with the trajectory's first
    /// frame when there's less than `window` history available — so
    /// early-episode forwards never fail but carry a degenerate
    /// (repeated-frame) signal until the real window fills.
    ///
    /// Returns `Some(flat_input)` when `trajectory` is non-empty and
    /// `end < trajectory.len()`, else `None`.
    pub fn build_window(&self, trajectory: &[Vec<f32>], end: usize) -> Option<Vec<f32>> {
        if trajectory.is_empty() || end >= trajectory.len() {
            return None;
        }
        let mut out = Vec::with_capacity(self.input_dim);
        let start_desired = end as i32 - self.window as i32 + 1;
        for j in 0..self.window {
            let idx_signed = start_desired + j as i32;
            let idx = if idx_signed < 0 { 0 } else { idx_signed as usize };
            let frame = &trajectory[idx];
            debug_assert_eq!(frame.len(), self.latent_dim);
            out.extend_from_slice(frame);
        }
        Some(out)
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
    fn outcome_head_fits_constant_target_window_1() {
        let mut head = OutcomeHead::new(8, 1, 16, 1e-2, 42);
        let z = vec![0.1f32; 8];
        let inputs = vec![z.clone(); 32];
        let target = 2.0;
        let mut last = 0.0;
        for _ in 0..200 {
            last = head.train_batch(&inputs, target);
        }
        assert!(last < 1e-2, "window-1 head failed to fit constant: {last}");
    }

    #[test]
    fn outcome_head_fits_constant_target_windowed() {
        let mut head = OutcomeHead::new(4, 3, 16, 1e-2, 42);
        // A three-step window means each input is 12 floats.
        let window = vec![0.05f32; 12];
        let inputs = vec![window.clone(); 32];
        let target = -1.5;
        let mut last = 0.0;
        for _ in 0..300 {
            last = head.train_batch(&inputs, target);
        }
        assert!(last < 1e-2, "windowed head failed to fit constant: {last}");
        let pred = head.forward(&window);
        assert!(
            (pred - target).abs() < 0.1,
            "windowed pred {pred} far from target {target}"
        );
    }

    #[test]
    fn build_window_left_pads() {
        let head = OutcomeHead::new(2, 3, 4, 1e-3, 0);
        let traj = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        // end=0, window=3 → left-pad with first frame twice.
        let w0 = head.build_window(&traj, 0).unwrap();
        assert_eq!(w0, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        // end=1, window=3 → one pad + two frames.
        let w1 = head.build_window(&traj, 1).unwrap();
        assert_eq!(w1, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn build_window_empty_or_oob() {
        let head = OutcomeHead::new(2, 3, 4, 1e-3, 0);
        let empty: Vec<Vec<f32>> = vec![];
        assert!(head.build_window(&empty, 0).is_none());
        let one = vec![vec![1.0, 2.0]];
        assert!(head.build_window(&one, 5).is_none());
    }

    #[test]
    fn windowed_head_discriminates_when_windows_differ() {
        // Sanity: the windowed head can learn to distinguish two
        // different windows that map to different targets — the
        // capability the 2026-04-19 mechanism check found was
        // missing from single-frame inputs.
        let mut head = OutcomeHead::new(2, 3, 8, 2e-2, 7);
        let window_a = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // "steady"
        let window_b = vec![0.0, 1.0, -0.5, 0.5, -1.0, 1.0]; // "swinging"
        for _ in 0..500 {
            head.train_batch(&[window_a.clone()], 1.0);
            head.train_batch(&[window_b.clone()], -1.0);
        }
        let pa = head.forward(&window_a);
        let pb = head.forward(&window_b);
        assert!(
            pa - pb > 1.0,
            "windowed head failed to separate distinguishable windows: {pa} vs {pb}"
        );
    }
}
