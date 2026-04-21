//! M8 delta-goal reward primitive.
//!
//! Self-supervised goal discovery from per-step latent deltas.
//! Maintains a rolling bank of latent positions where a significant
//! state change was just observed. Reward each step is proportional
//! to the inverse distance to the NEAREST goal in the bank — pulling
//! the policy toward regions where interesting transitions happen,
//! without any task-specific shaping.
//!
//! Differences from M7 (`approach::ApproachState`):
//! - Triggered per-step by latent delta magnitude, not by episode
//!   termination.
//! - Stores a bank of K goals (not one centroid); reward is
//!   `-α · min_i ‖z − g_i‖`, so the policy is pulled toward the
//!   nearest goal rather than toward a top-return average.
//! - No return-weighting: a "goal" is "a state the agent reached
//!   that differs substantially from the previous state", regardless
//!   of outcome value.
//!
//! Goal-farming mitigation: new candidates within `merge_radius` of
//! any existing goal are dropped, so the bank stays diverse rather
//! than collapsing onto one cluster the agent can oscillate near.

use std::collections::VecDeque;

/// Rolling bank of latent "goal" positions, each recorded when the
/// caller observed a step whose latent-delta exceeded
/// `delta_threshold`. Goal positions are stored in insertion order;
/// when the bank is full, the oldest entry is evicted.
pub struct DeltaGoalBank {
    pub latent_dim: usize,
    pub bank_size: usize,
    pub delta_threshold: f32,
    pub merge_radius: f32,
    goals: VecDeque<Vec<f32>>,
}

impl DeltaGoalBank {
    pub fn new(
        latent_dim: usize,
        bank_size: usize,
        delta_threshold: f32,
        merge_radius: f32,
    ) -> Self {
        Self {
            latent_dim,
            bank_size: bank_size.max(1),
            delta_threshold: delta_threshold.max(0.0),
            merge_radius: merge_radius.max(0.0),
            goals: VecDeque::with_capacity(bank_size.max(1)),
        }
    }

    /// Number of goals currently in the bank.
    pub fn len(&self) -> usize {
        self.goals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.goals.is_empty()
    }

    /// Consider `(z_prev, z_cur)` for a new goal entry. Returns
    /// `true` iff a new goal was added. Adds iff:
    /// 1. `‖z_cur − z_prev‖ ≥ delta_threshold` (significant change).
    /// 2. `z_cur` is NOT within `merge_radius` of any existing goal
    ///    (diversity — prevents one high-delta region from filling
    ///    the bank with near-duplicates).
    ///
    /// Callers pass `None` for `z_prev` on the first step of an
    /// episode or the agent's lifetime; that disables the check
    /// (no delta to compute).
    pub fn observe_delta(&mut self, z_prev: Option<&[f32]>, z_cur: &[f32]) -> bool {
        debug_assert_eq!(z_cur.len(), self.latent_dim);
        let Some(z_prev) = z_prev else {
            return false;
        };
        debug_assert_eq!(z_prev.len(), self.latent_dim);
        let delta = l2_distance(z_prev, z_cur);
        if delta < self.delta_threshold {
            return false;
        }
        for g in self.goals.iter() {
            if l2_distance(g, z_cur) < self.merge_radius {
                return false;
            }
        }
        if self.goals.len() == self.bank_size {
            self.goals.pop_front();
        }
        self.goals.push_back(z_cur.to_vec());
        true
    }

    /// Distance to the nearest goal in the bank. Returns `None`
    /// when the bank is empty.
    pub fn nearest_distance(&self, z: &[f32]) -> Option<f32> {
        debug_assert_eq!(z.len(), self.latent_dim);
        let mut best = f32::INFINITY;
        for g in self.goals.iter() {
            let d = l2_distance(g, z);
            if d < best {
                best = d;
            }
        }
        if best.is_finite() { Some(best) } else { None }
    }

    /// Reward at the current latent: `-α · min_i ‖z − g_i‖` clamped
    /// to `[-α · clamp, 0]`. Zero when the bank is empty.
    pub fn reward(&self, z: &[f32], alpha: f32, clamp: f32) -> f32 {
        let Some(d) = self.nearest_distance(z) else {
            return 0.0;
        };
        -alpha * d.min(clamp.max(0.0))
    }
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        acc += d * d;
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_records_nothing() {
        let mut b = DeltaGoalBank::new(3, 8, 0.5, 0.1);
        let added = b.observe_delta(None, &[1.0, 2.0, 3.0]);
        assert!(!added);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn small_delta_rejected() {
        let mut b = DeltaGoalBank::new(2, 8, 1.0, 0.1);
        let added = b.observe_delta(Some(&[0.0, 0.0]), &[0.3, 0.4]); // dist 0.5
        assert!(!added);
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn large_delta_recorded() {
        let mut b = DeltaGoalBank::new(2, 8, 1.0, 0.1);
        let added = b.observe_delta(Some(&[0.0, 0.0]), &[3.0, 4.0]); // dist 5
        assert!(added);
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn duplicates_merged_by_radius() {
        let mut b = DeltaGoalBank::new(2, 8, 0.5, 1.0);
        // First goal at (3,4).
        b.observe_delta(Some(&[0.0, 0.0]), &[3.0, 4.0]);
        // Second post-delta landing near (3.3, 4.1) — within
        // merge_radius=1.0 of the first; must not be recorded.
        let added = b.observe_delta(Some(&[10.0, 10.0]), &[3.3, 4.1]);
        assert!(!added);
        assert_eq!(b.len(), 1);
        // A landing far from the first is added.
        let added = b.observe_delta(Some(&[0.0, 0.0]), &[-3.0, -4.0]);
        assert!(added);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn bank_caps_and_evicts_oldest() {
        let mut b = DeltaGoalBank::new(1, 3, 0.1, 0.05);
        // 5 well-separated goals, bank size 3: oldest two should
        // be evicted, leaving the last three.
        for i in 0..5 {
            let prev = [i as f32 * 10.0];
            let cur = [i as f32 * 10.0 + 1.0];
            b.observe_delta(Some(&prev), &cur);
        }
        assert_eq!(b.len(), 3);
        // Nearest-distance to 41 (the last pushed cur) should be 0.
        let d = b.nearest_distance(&[41.0]).unwrap();
        assert!(d < 1e-5, "nearest: {d}");
        // Distance to 0 (the first pushed cur, evicted) should be
        // the distance to the oldest surviving goal (21.0).
        let d = b.nearest_distance(&[0.0]).unwrap();
        assert!((d - 21.0).abs() < 1e-3, "nearest to 0: {d}");
    }

    #[test]
    fn reward_is_negative_nearest_distance_clamped() {
        let mut b = DeltaGoalBank::new(2, 8, 0.1, 0.01);
        b.observe_delta(Some(&[0.0, 0.0]), &[3.0, 4.0]); // goal at (3,4)
        // At the goal: distance 0, reward 0.
        let r = b.reward(&[3.0, 4.0], 0.5, 10.0);
        assert!(r.abs() < 1e-6, "at-goal: {r}");
        // Close to the goal: distance 1, reward -0.5.
        let r = b.reward(&[3.0, 5.0], 0.5, 10.0);
        assert!((r + 0.5).abs() < 1e-6, "near-goal: {r}");
        // Far from the goal with tight clamp: reward = -α · clamp.
        let r = b.reward(&[100.0, 100.0], 0.5, 2.0);
        assert!((r + 1.0).abs() < 1e-6, "clamped far: {r}");
    }

    #[test]
    fn reward_is_zero_when_bank_empty() {
        let b = DeltaGoalBank::new(2, 8, 0.5, 0.1);
        assert_eq!(b.reward(&[1.0, 2.0], 0.5, 10.0), 0.0);
        assert!(b.nearest_distance(&[1.0, 2.0]).is_none());
    }

    #[test]
    fn nearest_picks_smallest_of_many() {
        let mut b = DeltaGoalBank::new(1, 8, 0.1, 0.01);
        for v in [0.0, 5.0, 10.0, 15.0, 20.0] {
            b.observe_delta(Some(&[v - 1.0]), &[v]);
        }
        // Query between the 5.0 and 10.0 goals, closer to 5.0.
        let d = b.nearest_distance(&[6.0]).unwrap();
        assert!((d - 1.0).abs() < 1e-4, "nearest: {d}");
    }
}
