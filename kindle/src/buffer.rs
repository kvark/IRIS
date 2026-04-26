//! Circular experience buffer for continual learning.
//!
//! All modules draw from and write to a shared `ExperienceBuffer`.
//! There is no episodic boundary — experience accumulates continuously.

use hashbrown::HashMap;
use rand::Rng;

/// Fixed-capacity circular buffer.
pub struct RingBuffer<T> {
    data: Vec<T>,
    head: usize,
    len: usize,
}

impl<T: Clone + Default> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            head: 0,
            len: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Push a new item, overwriting the oldest when full.
    pub fn push(&mut self, item: T) {
        let cap = self.data.len();
        self.data[self.head] = item;
        self.head = (self.head + 1) % cap;
        if self.len < cap {
            self.len += 1;
        }
    }

    /// Get item by logical index (0 = oldest still in buffer).
    pub fn get(&self, index: usize) -> &T {
        assert!(index < self.len, "index out of bounds");
        let cap = self.data.len();
        let start = if self.len < cap { 0 } else { self.head };
        &self.data[(start + index) % cap]
    }

    /// Get mutable item by logical index (0 = oldest still in buffer).
    pub fn get_mut(&mut self, index: usize) -> &mut T {
        assert!(index < self.len, "index out of bounds");
        let cap = self.data.len();
        let start = if self.len < cap { 0 } else { self.head };
        &mut self.data[(start + index) % cap]
    }

    /// Get the most recent item.
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let cap = self.data.len();
        let idx = (self.head + cap - 1) % cap;
        Some(&self.data[idx])
    }

    /// Return the N most recent items, oldest first.
    pub fn recent(&self, n: usize) -> Vec<&T> {
        let n = n.min(self.len);
        let start = self.len.saturating_sub(n);
        (start..self.len).map(|i| self.get(i)).collect()
    }

    /// Sample `n` random indices (with replacement) from the buffer.
    pub fn sample_indices<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<usize> {
        (0..n).map(|_| rng.random_range(0..self.len)).collect()
    }
}

/// A single timestep stored in the experience buffer.
#[derive(Clone, Debug, Default)]
pub struct Transition {
    pub observation: Vec<f32>,
    pub latent: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub credit: f32,
    pub pred_error: f32,
    /// Value baseline V(s_t) at this step, cached from the act-time
    /// policy forward. Used by the n-step advantage path (see
    /// `AgentConfig::n_step`) to compute `R − V(s_old)` without a
    /// second forward on old state.
    pub value: f32,
    /// Probability π_old(a_t | s_t) of the action actually taken, cached
    /// at act-time under the policy that collected this transition. Used
    /// as the denominator in PPO's importance ratio; unused by the plain
    /// policy-gradient path. Always in (0, 1].
    pub prob_taken: f32,
    /// Full pre-softmax logits at action time (length MAX_ACTION_DIM).
    /// Populated only when AgentConfig::use_kl_ppo is on; otherwise empty.
    /// Used by the KL-penalty PPO path to compute KL(π_new ‖ π_old)
    /// exactly via softmax(old_logits) at training time.
    pub logits_at_action: Vec<f32>,
    /// L1 option index active at this step. Mirrors `lane.current_option`
    /// at push time; needed so the n-step training forward can feed the
    /// option_onehot that matches the old state (options can change
    /// within the n-step lookahead window at short horizons).
    pub option_idx: u32,
    /// Environment id (from the adapter active when this step was recorded).
    /// Used to stratify sampling and mark env-boundary resets.
    pub env_id: u32,
    /// Boundary flag: true on the first transition after `switch_env`, so
    /// the credit assigner and world model skip cross-env attribution.
    pub env_boundary: bool,
}

/// Key for the novelty visit-count map.
/// Discretizes a latent vector into a grid cell.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateKey(Vec<i32>);

impl StateKey {
    /// Quantize a latent vector into grid cells with the given resolution.
    pub fn from_latent(latent: &[f32], resolution: f32) -> Self {
        Self(
            latent
                .iter()
                .map(|&x| (x / resolution).round() as i32)
                .collect(),
        )
    }
}

/// The agent's sole persistent memory.
pub struct ExperienceBuffer {
    transitions: RingBuffer<Transition>,
    pub visit_counts: HashMap<StateKey, u32>,
    grid_resolution: f32,
}

impl ExperienceBuffer {
    pub fn new(capacity: usize, grid_resolution: f32) -> Self {
        Self {
            transitions: RingBuffer::new(capacity),
            visit_counts: HashMap::new(),
            grid_resolution,
        }
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Record a new transition and update the novelty visit count.
    pub fn push(&mut self, transition: Transition) {
        let key = StateKey::from_latent(&transition.latent, self.grid_resolution);
        *self.visit_counts.entry(key).or_insert(0) += 1;
        self.transitions.push(transition);
    }

    /// Get transition by logical index.
    pub fn get(&self, index: usize) -> &Transition {
        self.transitions.get(index)
    }

    /// Return the most recent transition.
    pub fn last(&self) -> Option<&Transition> {
        self.transitions.last()
    }

    /// Return the N most recent transitions.
    pub fn recent_window(&self, n: usize) -> Vec<&Transition> {
        self.transitions.recent(n)
    }

    /// Look up the visit count for a latent vector.
    pub fn visit_count(&self, latent: &[f32]) -> u32 {
        let key = StateKey::from_latent(latent, self.grid_resolution);
        self.visit_counts.get(&key).copied().unwrap_or(0)
    }

    /// Get a mutable reference to transition by logical index.
    pub fn get_mut(&mut self, index: usize) -> &mut Transition {
        self.transitions.get_mut(index)
    }

    /// Flatten a history window into a single vector for the credit assigner.
    ///
    /// Each timestep becomes `[latent..., action..., reward_normalized]`,
    /// concatenated across `n` steps. Rewards are normalized to zero mean
    /// and clamped to [-1, 1] to prevent gradient explosion.
    pub fn flatten_history(&self, n: usize) -> Option<Vec<f32>> {
        if self.len() < n {
            return None;
        }
        let window = self.recent_window(n);

        // Normalize rewards in the window
        let rewards: Vec<f32> = window.iter().map(|t| t.reward).collect();
        let mean = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let std = (rewards.iter().map(|r| (r - mean).powi(2)).sum::<f32>() / rewards.len() as f32)
            .sqrt()
            .max(1e-6);

        let mut flat = Vec::new();
        for (i, t) in window.iter().enumerate() {
            flat.extend_from_slice(&t.latent);
            flat.extend_from_slice(&t.action);
            flat.push(((rewards[i] - mean) / std).clamp(-1.0, 1.0));
        }
        Some(flat)
    }

    /// Flatten a history window starting at a specific index.
    pub fn flatten_history_at(&self, start: usize, n: usize) -> Option<Vec<f32>> {
        if start + n > self.len() {
            return None;
        }
        let mut flat = Vec::new();
        for i in start..start + n {
            let t = self.get(i);
            flat.extend_from_slice(&t.latent);
            flat.extend_from_slice(&t.action);
            flat.push(t.reward);
        }
        Some(flat)
    }

    /// Find a contrastive pair: two timesteps with similar latents but
    /// divergent rewards. Returns `(high_reward_idx, low_reward_idx)`.
    ///
    /// Searches a random subset of the buffer for efficiency.
    pub fn find_contrastive_pair<R: Rng>(
        &self,
        rng: &mut R,
        history_len: usize,
        _latent_dim: usize,
    ) -> Option<(usize, usize)> {
        if self.len() < history_len * 2 {
            return None;
        }

        let search_range = history_len..self.len();
        let num_candidates = 50.min(search_range.len());

        let mut best_pair: Option<(usize, usize)> = None;
        let mut best_score = 0.0f32;

        for _ in 0..num_candidates {
            let i = rng.random_range(search_range.clone());
            let j = rng.random_range(search_range.clone());
            if i == j {
                continue;
            }

            let ti = self.get(i);
            let tj = self.get(j);

            // Stratify by env_id: contrastive pairs must come from the
            // same environment, otherwise the signal "rewards differ" is
            // meaningless (different envs have different reward scales).
            if ti.env_id != tj.env_id {
                continue;
            }

            // Latent similarity (lower = more similar)
            let latent_dist: f32 = ti
                .latent
                .iter()
                .zip(tj.latent.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();

            // Reward divergence (higher = better pair)
            let reward_diff = (ti.reward - tj.reward).abs();

            // Score: high reward divergence relative to latent similarity
            let score = reward_diff / (latent_dist + 0.1);

            if score > best_score {
                best_score = score;
                if ti.reward >= tj.reward {
                    best_pair = Some((i, j));
                } else {
                    best_pair = Some((j, i));
                }
            }
        }

        best_pair
    }

    /// Compute contrastive credit target: high credit to timesteps where
    /// actions diverged between two history windows.
    ///
    /// Returns a softmax-normalized target of length `history_len`.
    pub fn contrastive_target(
        &self,
        high_idx: usize,
        low_idx: usize,
        history_len: usize,
    ) -> Vec<f32> {
        let mut divergence = vec![0.0f32; history_len];

        let high_start = high_idx.saturating_sub(history_len - 1);
        let low_start = low_idx.saturating_sub(history_len - 1);

        for (i, div) in divergence.iter_mut().enumerate() {
            let h_idx = high_start + i;
            let l_idx = low_start + i;
            if h_idx < self.len() && l_idx < self.len() {
                let th = self.get(h_idx);
                let tl = self.get(l_idx);
                *div = th
                    .action
                    .iter()
                    .zip(tl.action.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
            }
        }

        // Softmax normalize
        let max = divergence.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = divergence.iter().map(|&d| (d - max).exp()).sum();
        if exp_sum > 0.0 {
            divergence
                .iter_mut()
                .for_each(|d| *d = (*d - max).exp() / exp_sum);
        }
        divergence
    }

    /// Write credit values back to the most recent N transitions.
    pub fn write_credits(&mut self, credits: &[f32]) {
        let n = credits.len().min(self.len());
        let start = self.len() - n;
        for (i, &c) in credits.iter().enumerate() {
            self.get_mut(start + i).credit = c;
        }
    }

    /// Sample a batch for training: `replay_ratio` fraction from full history,
    /// the rest from the most recent window.
    pub fn sample_batch<R: Rng>(
        &self,
        rng: &mut R,
        batch_size: usize,
        replay_ratio: f32,
    ) -> Vec<&Transition> {
        if self.is_empty() {
            return Vec::new();
        }

        let replay_count = (batch_size as f32 * replay_ratio).round() as usize;
        let recent_count = batch_size - replay_count;

        let mut batch = Vec::with_capacity(batch_size);

        // Replay samples from full buffer history
        for _ in 0..replay_count.min(self.len()) {
            let idx = rng.random_range(0..self.len());
            batch.push(self.transitions.get(idx));
        }

        // Recent samples from the tail of the buffer
        let recent_start = self.len().saturating_sub(recent_count * 4);
        let recent_range = recent_start..self.len();
        if !recent_range.is_empty() {
            for _ in 0..recent_count {
                let idx = rng.random_range(recent_range.clone());
                batch.push(self.transitions.get(idx));
            }
        }

        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_and_get() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);
        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.len(), 3);
        assert_eq!(*rb.get(0), 10);
        assert_eq!(*rb.get(2), 30);

        // Overwrite oldest
        rb.push(40);
        assert_eq!(rb.len(), 3);
        assert_eq!(*rb.get(0), 20);
        assert_eq!(*rb.get(2), 40);
    }

    #[test]
    fn ring_buffer_recent() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(5);
        for i in 0..5 {
            rb.push(i);
        }
        let recent: Vec<_> = rb.recent(3).into_iter().copied().collect();
        assert_eq!(recent, vec![2, 3, 4]);
    }

    #[test]
    fn ring_buffer_last() {
        let mut rb: RingBuffer<i32> = RingBuffer::new(3);
        assert!(rb.last().is_none());
        rb.push(1);
        assert_eq!(*rb.last().unwrap(), 1);
        rb.push(2);
        assert_eq!(*rb.last().unwrap(), 2);
    }

    #[test]
    fn experience_buffer_visit_counts() {
        let mut buf = ExperienceBuffer::new(100, 0.1);
        let t = Transition {
            latent: vec![0.15, 0.25],
            ..Default::default()
        };
        buf.push(t.clone());
        buf.push(t.clone());

        assert_eq!(buf.visit_count(&[0.15, 0.25]), 2);
        assert_eq!(buf.visit_count(&[0.99, 0.99]), 0);
    }

    #[test]
    fn experience_buffer_sample_batch() {
        let mut buf = ExperienceBuffer::new(100, 0.1);
        for i in 0..50 {
            buf.push(Transition {
                reward: i as f32,
                ..Default::default()
            });
        }

        let mut rng = rand::rng();
        let batch = buf.sample_batch(&mut rng, 10, 0.2);
        assert_eq!(batch.len(), 10);
    }
}
