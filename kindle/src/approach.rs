//! M7 approach-reward primitive.
//!
//! Self-supervised goal discovery: maintain a rolling buffer of
//! (terminal_latent, episode_return) pairs across lanes. Every
//! `approach_update_interval` episodes, recompute a single
//! prototype centroid as the mean of the top-`approach_top_frac`
//! fraction of terminals by return. The approach reward each step
//! is `α · (−‖z_t − centroid‖)`, clamped to avoid runaway
//! magnitudes. See `docs/phase-m7-approach-reward.md` for the full
//! design and the rationale for why this addresses the
//! approach-shaping gap the PPO ceiling test (commit 9a8dcce)
//! identified in kindle's reward class.

use std::collections::VecDeque;

/// One completed episode's terminal summary. Latent dim is implicit
/// (matches `AgentConfig::latent_dim`); callers push with the right
/// shape.
#[derive(Clone, Debug)]
pub struct TerminalEntry {
    pub z_end: Vec<f32>,
    pub r_episode: f32,
}

/// State for the M7 approach-reward primitive.
pub struct ApproachState {
    pub latent_dim: usize,
    pub buffer_size: usize,
    pub top_frac: f32,
    pub update_interval: usize,
    pub warmup_episodes: usize,
    /// Recent terminal entries (oldest at front, newest at back).
    pub buffer: VecDeque<TerminalEntry>,
    /// Current prototype centroid. `None` until first computation
    /// after warmup.
    pub centroid: Option<Vec<f32>>,
    /// Number of completed episodes seen so far. Used to trigger
    /// `update_interval` rolls and compare against `warmup_episodes`.
    pub episodes_seen: usize,
    /// Episodes since last centroid recompute.
    pub episodes_since_update: usize,
    /// Distance between the previous centroid and the current one,
    /// at the last recompute. Diagnostic — high jitter means the
    /// prototype hasn't converged.
    pub last_centroid_drift: f32,
    /// Centroid age in episodes since last recompute. 0 right after
    /// a recompute; grows until the next one.
    pub centroid_age: usize,
}

impl ApproachState {
    pub fn new(
        latent_dim: usize,
        buffer_size: usize,
        top_frac: f32,
        update_interval: usize,
        warmup_episodes: usize,
    ) -> Self {
        Self {
            latent_dim,
            buffer_size: buffer_size.max(1),
            top_frac: top_frac.clamp(0.01, 1.0),
            update_interval: update_interval.max(1),
            warmup_episodes: warmup_episodes.max(1),
            buffer: VecDeque::with_capacity(buffer_size.max(1)),
            centroid: None,
            episodes_seen: 0,
            episodes_since_update: 0,
            last_centroid_drift: 0.0,
            centroid_age: 0,
        }
    }

    /// Record one episode's terminal. Returns `true` if this call
    /// triggered a centroid recompute.
    pub fn push_terminal(&mut self, z_end: &[f32], r_episode: f32) -> bool {
        debug_assert_eq!(z_end.len(), self.latent_dim);
        if self.buffer.len() == self.buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(TerminalEntry {
            z_end: z_end.to_vec(),
            r_episode,
        });
        self.episodes_seen += 1;
        self.episodes_since_update += 1;
        self.centroid_age += 1;

        // Only recompute once we've passed warmup AND enough
        // episodes have elapsed since last update.
        if self.episodes_seen >= self.warmup_episodes
            && self.episodes_since_update >= self.update_interval
        {
            self.recompute_centroid();
            self.episodes_since_update = 0;
            self.centroid_age = 0;
            true
        } else {
            false
        }
    }

    /// Recompute the prototype centroid: mean of top-`top_frac`
    /// entries by `r_episode`.
    pub fn recompute_centroid(&mut self) {
        let n = self.buffer.len();
        if n == 0 {
            return;
        }
        // Sort indices descending by r_episode (do a partial sort
        // via full sort since buffer is small, ~100).
        let mut idxs: Vec<usize> = (0..n).collect();
        idxs.sort_by(|&a, &b| {
            self.buffer[b]
                .r_episode
                .partial_cmp(&self.buffer[a].r_episode)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let take = ((n as f32 * self.top_frac).ceil() as usize).max(1).min(n);
        let mut new_centroid = vec![0.0f32; self.latent_dim];
        for &i in idxs.iter().take(take) {
            for (d, &v) in new_centroid.iter_mut().zip(self.buffer[i].z_end.iter()) {
                *d += v;
            }
        }
        let scale = 1.0 / take as f32;
        for v in new_centroid.iter_mut() {
            *v *= scale;
        }
        // Measure drift relative to previous centroid, for
        // diagnostics.
        self.last_centroid_drift = match self.centroid.as_ref() {
            Some(old) => l2_distance(old, &new_centroid),
            None => 0.0,
        };
        self.centroid = Some(new_centroid);
    }

    /// Approach reward at the current state latent. Returns 0 when
    /// the centroid hasn't been seeded yet.
    pub fn reward(&self, z_t: &[f32], alpha: f32, distance_clamp: f32) -> f32 {
        debug_assert_eq!(z_t.len(), self.latent_dim);
        let Some(c) = self.centroid.as_ref() else {
            return 0.0;
        };
        let d = l2_distance(c, z_t);
        let d_clamped = d.min(distance_clamp.max(0.0));
        -alpha * d_clamped
    }

    /// Distance to the current centroid, without the `α` scaling.
    /// Returns 0 when the centroid hasn't been seeded.
    pub fn distance(&self, z_t: &[f32]) -> f32 {
        let Some(c) = self.centroid.as_ref() else {
            return 0.0;
        };
        l2_distance(c, z_t)
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
    fn centroid_empty_until_warmup() {
        let mut s = ApproachState::new(3, 10, 0.5, 2, 5);
        assert!(s.centroid.is_none());
        for i in 0..4 {
            s.push_terminal(&[i as f32, 0.0, 0.0], i as f32);
        }
        // Only 4 entries < warmup 5 → still no centroid.
        assert!(s.centroid.is_none());
    }

    #[test]
    fn centroid_is_mean_of_top_fraction() {
        let mut s = ApproachState::new(2, 10, 0.5, 1, 1);
        // 4 entries; top 50% by return should be the two highest.
        s.push_terminal(&[0.0, 0.0], -10.0);
        s.push_terminal(&[10.0, 10.0], 5.0); // top
        s.push_terminal(&[8.0, 8.0], 3.0); // top
        s.push_terminal(&[1.0, 1.0], -5.0);
        // The last push_terminal triggers a recompute (interval=1,
        // warmup=1). Top 2 entries: (10,10) and (8,8). Mean (9,9).
        let c = s.centroid.as_ref().unwrap();
        assert!((c[0] - 9.0).abs() < 1e-4, "centroid x: {}", c[0]);
        assert!((c[1] - 9.0).abs() < 1e-4, "centroid y: {}", c[1]);
    }

    #[test]
    fn approach_reward_is_negative_distance() {
        let mut s = ApproachState::new(2, 10, 1.0, 1, 1);
        s.push_terminal(&[3.0, 4.0], 10.0);
        // centroid should be (3,4) now — single entry, top_frac=1.0.
        let r = s.reward(&[0.0, 0.0], 0.5, 100.0);
        // distance = 5.0, alpha = 0.5 → reward = -2.5
        assert!((r + 2.5).abs() < 1e-4, "reward: {}", r);
    }

    #[test]
    fn buffer_caps_at_size() {
        let mut s = ApproachState::new(1, 3, 0.5, 1, 1);
        for i in 0..10 {
            s.push_terminal(&[i as f32], i as f32);
        }
        assert_eq!(s.buffer.len(), 3);
        // Last three are i=7,8,9 so buffer contains 7,8,9.
        assert!((s.buffer.front().unwrap().r_episode - 7.0).abs() < 1e-4);
        assert!((s.buffer.back().unwrap().r_episode - 9.0).abs() < 1e-4);
    }

    #[test]
    fn drift_tracked() {
        // top_frac=0.5 selects only the highest-return half, so the
        // centroid follows the best-outcome cluster even as older
        // low-return entries stay in the buffer.
        let mut s = ApproachState::new(1, 10, 0.5, 2, 1);
        s.push_terminal(&[0.0], 1.0);
        s.push_terminal(&[0.0], 1.0);
        let c_first = s.centroid.as_ref().unwrap().clone();
        assert_eq!(c_first, vec![0.0]);
        s.push_terminal(&[10.0], 5.0);
        s.push_terminal(&[10.0], 5.0);
        let c_second = s.centroid.as_ref().unwrap();
        // Top 2 of 4 are the two z=10, r=5 entries. Centroid = 10.
        assert!(
            (c_second[0] - 10.0).abs() < 1e-4,
            "centroid moved: {}",
            c_second[0]
        );
        assert!(
            s.last_centroid_drift > 0.0,
            "drift: {}",
            s.last_centroid_drift
        );
    }

    #[test]
    fn reward_zero_before_centroid() {
        let s = ApproachState::new(2, 10, 0.5, 2, 5);
        let r = s.reward(&[1.0, 1.0], 1.0, 100.0);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn distance_clamp_bounds_reward() {
        let mut s = ApproachState::new(1, 10, 1.0, 1, 1);
        s.push_terminal(&[0.0], 1.0);
        // z far from centroid: distance=1000, clamp=5, alpha=1 → r=-5
        let r = s.reward(&[1000.0], 1.0, 5.0);
        assert!((r + 5.0).abs() < 1e-4, "clamped: {}", r);
    }
}
