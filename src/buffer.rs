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
        let start = if self.len < cap {
            0
        } else {
            self.head
        };
        &self.data[(start + index) % cap]
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
        (0..n).map(|_| rng.gen_range(0..self.len)).collect()
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
}

/// Key for the novelty visit-count map.
/// Discretizes a latent vector into a grid cell.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StateKey(Vec<i32>);

impl StateKey {
    /// Quantize a latent vector into grid cells with the given resolution.
    pub fn from_latent(latent: &[f32], resolution: f32) -> Self {
        Self(latent.iter().map(|&x| (x / resolution).round() as i32).collect())
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

    /// Sample a batch for training: `replay_ratio` fraction from full history,
    /// the rest from the most recent window.
    pub fn sample_batch<R: Rng>(
        &self,
        rng: &mut R,
        batch_size: usize,
        replay_ratio: f32,
    ) -> Vec<&Transition> {
        if self.len() == 0 {
            return Vec::new();
        }

        let replay_count = (batch_size as f32 * replay_ratio).round() as usize;
        let recent_count = batch_size - replay_count;

        let mut batch = Vec::with_capacity(batch_size);

        // Replay samples from full buffer history
        for _ in 0..replay_count.min(self.len()) {
            let idx = rng.gen_range(0..self.len());
            batch.push(self.transitions.get(idx));
        }

        // Recent samples from the tail of the buffer
        let recent_start = self.len().saturating_sub(recent_count * 4);
        let recent_range = recent_start..self.len();
        if !recent_range.is_empty() {
            for _ in 0..recent_count {
                let idx = rng.gen_range(recent_range.clone());
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

        let mut rng = rand::thread_rng();
        let batch = buf.sample_batch(&mut rng, 10, 0.2);
        assert_eq!(batch.len(), 10);
    }
}
