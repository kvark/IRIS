//! Cross-episode state-action memory.
//!
//! Kindle's existing novelty primitive (`1 / sqrt(state_visits)`) is
//! per-state and gives an equal bonus to every action taken at a
//! visited state. This module adds *state-action* novelty: for the
//! same quantized state, actions that have been tried more rarely
//! receive a larger per-step bonus. The counts persist across
//! episodes and across lanes — the intended failure mode it targets
//! is "agent reaches the same attractor every episode and retries
//! the same handful of actions there, never exploring what other
//! actions from that state would do."
//!
//! Self-supervised, no task priors. The quantization key is the same
//! `StateKey` kindle already uses for state-novelty, so a per-state
//! action-histogram is consistent with the rest of the reward stack.
//!
//! Reward each step: `α / sqrt(1 + count(prev_state, prev_action))`.
//! The `+1` in the denominator keeps the bonus finite on the first
//! visit (α / 1 = α — strongest signal for never-tried pairs) and
//! decays as the pair becomes well-explored. The reward is emitted
//! using the PREVIOUS step's (state, action) because those are the
//! quantities that produced the transition being rewarded now —
//! action_t is sampled AFTER observe_t, so observe_t can only know
//! about action_{t-1}.

use crate::buffer::StateKey;
use hashbrown::HashMap;

/// Persistent per-(quantized_state, action) visit counter shared
/// across lanes and episodes.
pub struct StateActionMemory {
    pub grid_resolution: f32,
    counts: HashMap<(StateKey, u32), u32>,
}

impl StateActionMemory {
    pub fn new(grid_resolution: f32) -> Self {
        Self {
            grid_resolution: grid_resolution.max(1e-6),
            counts: HashMap::new(),
        }
    }

    /// Record one observation of `(z, action)` and return the
    /// updated count for that pair.
    pub fn observe(&mut self, z: &[f32], action: u32) -> u32 {
        let key = (StateKey::from_latent(z, self.grid_resolution), action);
        let c = self.counts.entry(key).or_insert(0);
        *c += 1;
        *c
    }

    /// Count for `(z, action)` — 0 if never observed.
    pub fn count(&self, z: &[f32], action: u32) -> u32 {
        let key = (StateKey::from_latent(z, self.grid_resolution), action);
        self.counts.get(&key).copied().unwrap_or(0)
    }

    /// Intrinsic reward: `α / sqrt(1 + count(z, action))`. Never
    /// -tried pairs (`count = 0`) score `α`; decays as the pair
    /// is revisited.
    pub fn reward(&self, z: &[f32], action: u32, alpha: f32) -> f32 {
        let c = self.count(z, action);
        alpha / ((c as f32 + 1.0).sqrt())
    }

    /// Number of distinct `(state, action)` pairs tracked.
    pub fn distinct_pairs(&self) -> usize {
        self.counts.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_tried_pair_scores_alpha() {
        let m = StateActionMemory::new(0.5);
        let r = m.reward(&[0.1, 0.2], 3, 0.5);
        // count=0 → α / sqrt(1) = α.
        assert!((r - 0.5).abs() < 1e-6, "reward: {r}");
    }

    #[test]
    fn count_increments_and_reward_decays() {
        let mut m = StateActionMemory::new(0.5);
        for _ in 0..3 {
            m.observe(&[0.1, 0.2], 3);
        }
        assert_eq!(m.count(&[0.1, 0.2], 3), 3);
        // α / sqrt(4) = α / 2.
        let r = m.reward(&[0.1, 0.2], 3, 1.0);
        assert!((r - 0.5).abs() < 1e-6, "reward after 3 observes: {r}");
    }

    #[test]
    fn different_actions_counted_separately() {
        let mut m = StateActionMemory::new(0.5);
        m.observe(&[0.0, 0.0], 1);
        m.observe(&[0.0, 0.0], 1);
        m.observe(&[0.0, 0.0], 2);
        assert_eq!(m.count(&[0.0, 0.0], 1), 2);
        assert_eq!(m.count(&[0.0, 0.0], 2), 1);
        assert_eq!(m.count(&[0.0, 0.0], 3), 0);
        assert_eq!(m.distinct_pairs(), 2);
    }

    #[test]
    fn quantization_merges_nearby_latents() {
        let mut m = StateActionMemory::new(0.5);
        // Both round to grid cell (0, 0).
        m.observe(&[0.05, 0.10], 1);
        m.observe(&[0.12, 0.22], 1);
        assert_eq!(m.count(&[0.08, 0.15], 1), 2);
        // Different cell:
        m.observe(&[0.80, 0.80], 1);
        assert_eq!(m.count(&[0.80, 0.80], 1), 1);
        assert_eq!(m.distinct_pairs(), 2);
    }

    #[test]
    fn reward_strictly_decreases_with_count() {
        let mut m = StateActionMemory::new(0.5);
        let mut prev = f32::INFINITY;
        for _ in 0..10 {
            let r = m.reward(&[0.0], 0, 1.0);
            assert!(r < prev, "expected decay: {r} < {prev}");
            prev = r;
            m.observe(&[0.0], 0);
        }
    }
}
