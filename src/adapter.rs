//! Universal action / observation adapters for cross-environment training.
//!
//! See `docs/universal-actions.md` for the design. The agent's core graphs
//! are built once for these fixed token sizes; each environment gets a
//! per-env adapter that translates between its native shapes and the
//! universal token space.

use crate::env::{Action, ActionKind, Observation};
use rand::{Rng, RngCore};

/// Maximum action dimension across all supported environments.
/// Discrete envs use the first `n` dims; continuous envs use the first `dim`.
pub const MAX_ACTION_DIM: usize = 6;

/// Observation token dimension. Raw observations of any size are projected
/// (padded with zeros initially) to this size.
pub const OBS_TOKEN_DIM: usize = 64;

/// Per-env task embedding dimension. The agent holds one `[TASK_DIM]`
/// learned vector per env_id, fed to the encoder as a second input so
/// the shared encoder can specialize its behaviour per environment.
pub const TASK_DIM: usize = 8;

/// Adapter translating between an environment's native obs/action shape
/// and the agent's universal token space.
///
/// Implementations must be trait-object safe (no generic methods) so that
/// `Agent::switch_env` can swap adapters at runtime without rebuilding
/// GPU graphs.
pub trait EnvAdapter: Send {
    /// Stable identifier for this environment (for env_id tagging).
    fn id(&self) -> u32;

    /// Native observation dimension of the wrapped environment.
    fn obs_dim(&self) -> usize;

    /// Native action kind of the wrapped environment.
    fn action_kind(&self) -> ActionKind;

    /// Project the env's observation into a fixed-size universal token.
    /// Default: zero-pad the raw observation to `OBS_TOKEN_DIM`.
    fn obs_to_token(&self, obs: &Observation, out: &mut [f32]) {
        assert_eq!(out.len(), OBS_TOKEN_DIM);
        out.fill(0.0);
        let n = obs.data.len().min(OBS_TOKEN_DIM);
        out[..n].copy_from_slice(&obs.data[..n]);
    }

    /// Encode an action into the universal action token.
    fn action_to_token(&self, action: &Action, out: &mut [f32]);

    /// Sample an action from the policy head output (shape `MAX_ACTION_DIM`).
    fn sample_action(&self, head: &[f32], rng: &mut dyn RngCore) -> Action;

    /// Entropy of the policy distribution over live dimensions.
    /// For continuous adapters, returns the fixed exploration scale as a proxy.
    fn head_entropy(&self, head: &[f32]) -> f32;
}

/// Zero-pad + identity adapter that works for any env matching the
/// universal sizes. Discrete: first `n` head dims as logits; Continuous:
/// first `dim` head dims as Gaussian means.
pub struct GenericAdapter {
    id: u32,
    obs_dim: usize,
    kind: ActionKind,
}

impl GenericAdapter {
    /// Adapter for a discrete env with `n` actions.
    pub fn discrete(id: u32, obs_dim: usize, n: usize) -> Self {
        assert!(
            n <= MAX_ACTION_DIM,
            "discrete n ({n}) exceeds MAX_ACTION_DIM ({MAX_ACTION_DIM})"
        );
        assert!(
            obs_dim <= OBS_TOKEN_DIM,
            "obs_dim ({obs_dim}) exceeds OBS_TOKEN_DIM ({OBS_TOKEN_DIM})"
        );
        Self {
            id,
            obs_dim,
            kind: ActionKind::Discrete { n },
        }
    }

    /// Adapter for a continuous env with `dim` action dims and Gaussian
    /// exploration noise of standard deviation `scale`.
    pub fn continuous(id: u32, obs_dim: usize, dim: usize, scale: f32) -> Self {
        assert!(dim <= MAX_ACTION_DIM);
        assert!(obs_dim <= OBS_TOKEN_DIM);
        Self {
            id,
            obs_dim,
            kind: ActionKind::Continuous { dim, scale },
        }
    }
}

impl EnvAdapter for GenericAdapter {
    fn id(&self) -> u32 {
        self.id
    }

    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn action_kind(&self) -> ActionKind {
        self.kind
    }

    fn action_to_token(&self, action: &Action, out: &mut [f32]) {
        assert_eq!(out.len(), MAX_ACTION_DIM);
        out.fill(0.0);
        #[allow(clippy::pattern_type_mismatch)]
        match (self.kind, action) {
            (ActionKind::Discrete { n }, Action::Discrete(i)) => {
                assert!(*i < n, "action index {i} >= n {n}");
                out[*i] = 1.0;
            }
            (ActionKind::Continuous { dim, .. }, Action::Continuous(v)) => {
                let copy_len = v.len().min(dim);
                out[..copy_len].copy_from_slice(&v[..copy_len]);
            }
            _ => panic!("action kind mismatch with adapter"),
        }
    }

    fn sample_action(&self, head: &[f32], rng: &mut dyn RngCore) -> Action {
        assert!(head.len() >= MAX_ACTION_DIM);
        match self.kind {
            ActionKind::Discrete { n } => Action::Discrete(sample_discrete(&head[..n], rng)),
            ActionKind::Continuous { dim, scale } => {
                Action::Continuous(sample_gaussian(&head[..dim], scale, rng))
            }
        }
    }

    fn head_entropy(&self, head: &[f32]) -> f32 {
        match self.kind {
            ActionKind::Discrete { n } => discrete_entropy(&head[..n]),
            ActionKind::Continuous { scale, .. } => scale,
        }
    }
}

/// Categorical sample from logits via softmax + inverse-CDF.
fn sample_discrete(logits: &[f32], rng: &mut dyn RngCore) -> usize {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    let u: f32 = rng.random_range(0.0..1.0);
    let mut cum = 0.0;
    for (i, &e) in exp.iter().enumerate() {
        cum += e / sum;
        if u < cum {
            return i;
        }
    }
    logits.len() - 1
}

/// Gaussian sample `μ + ε·scale` via Box–Muller.
fn sample_gaussian(mu: &[f32], scale: f32, rng: &mut dyn RngCore) -> Vec<f32> {
    use std::f32::consts::TAU;
    mu.iter()
        .map(|&m| {
            let u1: f32 = rng.random_range(1e-7..1.0);
            let u2: f32 = rng.random_range(0.0..1.0);
            let noise = (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos();
            m + scale * noise
        })
        .collect()
}

/// Shannon entropy of softmax(logits).
fn discrete_entropy(logits: &[f32]) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    -exp.iter()
        .filter(|&&e| e > 1e-10)
        .map(|&e| {
            let p = e / sum;
            p * p.ln()
        })
        .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_adapter_discrete_obs_to_token() {
        let adapter = GenericAdapter::discrete(0, 4, 3);
        let obs = Observation::new(vec![1.0, 2.0, 3.0, 4.0]);
        let mut token = vec![-1.0; OBS_TOKEN_DIM];
        adapter.obs_to_token(&obs, &mut token);
        assert_eq!(token[0], 1.0);
        assert_eq!(token[3], 4.0);
        assert_eq!(token[4], 0.0); // padded
        assert_eq!(token[OBS_TOKEN_DIM - 1], 0.0);
    }

    #[test]
    fn generic_adapter_discrete_action_to_token() {
        let adapter = GenericAdapter::discrete(0, 4, 3);
        let mut token = vec![99.0; MAX_ACTION_DIM];
        adapter.action_to_token(&Action::Discrete(1), &mut token);
        assert_eq!(token[0], 0.0);
        assert_eq!(token[1], 1.0);
        assert_eq!(token[2], 0.0);
        assert_eq!(token[3..], [0.0; 3]); // padded to MAX_ACTION_DIM
    }

    #[test]
    fn generic_adapter_continuous_action_to_token() {
        let adapter = GenericAdapter::continuous(1, 3, 2, 0.5);
        let mut token = vec![99.0; MAX_ACTION_DIM];
        adapter.action_to_token(&Action::Continuous(vec![0.5, -0.25]), &mut token);
        assert_eq!(token[0], 0.5);
        assert_eq!(token[1], -0.25);
        assert_eq!(token[2..], [0.0; 4]);
    }

    #[test]
    fn generic_adapter_sample_discrete_returns_valid_index() {
        let adapter = GenericAdapter::discrete(0, 4, 3);
        let head = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let mut rng = rand::rng();
        for _ in 0..20 {
            let Action::Discrete(i) = adapter.sample_action(&head, &mut rng) else {
                panic!("wrong action kind")
            };
            assert!(i < 3);
        }
    }

    #[test]
    fn generic_adapter_sample_continuous_matches_dim() {
        let adapter = GenericAdapter::continuous(2, 3, 2, 0.5);
        let head = vec![0.1, 0.2, 0.0, 0.0, 0.0, 0.0];
        let mut rng = rand::rng();
        let Action::Continuous(v) = adapter.sample_action(&head, &mut rng) else {
            panic!("wrong action kind")
        };
        assert_eq!(v.len(), 2);
    }

    #[test]
    #[should_panic]
    fn discrete_n_too_large_panics() {
        let _ = GenericAdapter::discrete(0, 4, MAX_ACTION_DIM + 1);
    }

    #[test]
    #[should_panic]
    fn obs_dim_too_large_panics() {
        let _ = GenericAdapter::discrete(0, OBS_TOKEN_DIM + 1, 2);
    }
}
