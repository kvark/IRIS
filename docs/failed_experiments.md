# Failed experiments — kept for the diagnostic trail

This document collects experiments whose code was removed because it
didn't work, but whose lessons are useful for future architectural
work. Each entry: what was tried, what symptoms appeared, what the
hypothesis was, why we stopped.

## PPO + end-to-end encoder (build_ppo_policy_graph_e2e)

**Removed** 2026-04-26. The function was in `kindle/src/policy.rs`
and dispatched in `agent.rs` when both `use_ppo=true` and
`end_to_end_encoder=true`. Across four debug rounds it never
converged on CartPole.

### Architecture
PPO clipped surrogate (`-mean(min(r·A, clip(r,1±ε)·A))`) on the same
encoder + policy + value graph as `build_policy_graph_e2e`. Entropy
term used `stop_gradient(z)` to keep its gradient out of the encoder
(same fix as plain e2e).

### Symptoms
With `--advantage-normalize` (the standard PPO setup), entropy
collapsed to 0 within ~5k env-steps regardless of LR. Without
advantage_normalize, entropy stayed healthy but mean return was
stuck at ~+19 (random baseline). Compare plain-PG e2e: solves
CartPole sustained at +218.

### Debug attempts (all unsuccessful)
| Variation | Mean return | Notes |
|-----------|-------------|-------|
| baseline | +9.6 | ent collapse step 5k |
| ppo_n_epochs=1 | +9.5 | no help |
| ppo_n_epochs=4 | +9.6 | no help |
| no advantage_normalize | +19.5 | stuck at random, ent stable |
| larger clip_eps=0.5 | +14.7 | wider clip, no help |
| stop_gradient on value head's z | +11 | mostly random, V doesn't shape encoder |
| + entropy_beta=0.1 | +16.8 | NO collapse, but no convergence |
| + entropy 0.05→0 anneal | +16.5 | no change from constant |
| + auxiliary CE loss | +11.6 | overfits on n_epochs=4, hurts |
| vlc=0 (value coef = 0, isolate policy) | +9.5 | confirms policy graph is broken |

### Hypothesis (unverified, would need meganeura instrumentation)
The PPO surrogate at small normalized advantages (mean(A) ≈ 0 by
construction) produces a per-element gradient with magnitude
proportional to A, but the gradient signal-to-noise ratio is too
low at kindle's per-step / small-batch update cadence. Plain
cross-entropy loss `-A·log_softmax(action)` has stronger
well-conditioned gradients on the same advantages, which is why the
plain-PG e2e graph converges where the PPO surrogate does not.
The clip mechanism only kicks in when ratio drifts from 1, but
ratio stays near 1 across a single rollout's epochs in our setup
because policy updates are tiny → clip never engages → no
trust-region behavior.

### Why we stopped
Every variation tested still failed to converge. Real diagnosis
requires per-op gradient norm inspection in meganeura (an
instrumentation feature that doesn't exist). Possibly the right
fix is a complete rewrite using KL-penalty PPO instead of the
clipped surrogate, or matching the surrogate to plain-PG's
log-prob form (essentially making it a soft-clipped CE loss).
Either is multi-day work and would need empirical comparison to
the working plain-PG e2e baseline.

### What replaces it in practice
The working sustained-solve path is **plain-PG e2e + `--lr-drop-on-solve`**.
The dynamic LR mutator (`Agent::set_learning_rate` /
`Agent::set_lr_policy`) provides the trust-region effect that PPO
would provide if it worked: once a sustained solve is detected, drop
the LR by 10×–1000× to prevent the post-solve crash. Empirically
this gets CartPole-v1 to mean +218 over 1.6M env-steps (officially
solved) and Acrobot-v1 to mean -128 (sustained improvement).

### Files removed
- `kindle/src/policy.rs::build_ppo_policy_graph_e2e` (~120 lines)
- The PPO-and-e2e dispatch branch in
  `kindle/src/agent.rs::Agent::new` (replaced with an explicit
  `assert!(!(use_ppo && end_to_end_encoder))`)

The PPO surrogate's gradient pathology in our setup is documented
here as a known issue that future PPO work would need to address.

## Auxiliary CE loss alongside PPO surrogate

Tried adding `cross_entropy_loss(logits, A·one_hot)` to the PPO
surrogate to provide a strong policy gradient even when the surrogate
gradient is weak. With ppo_n_epochs=4, the unclipped CE part overfits
the rollout batch (same advantage targets repeated 4×), and the policy
overshoots. Mean +11.6 — same as PPO baseline. Not worth keeping.

Reverted same day. The intuition (PPO needs a non-degenerate gradient
fallback) might still be right, but the right form is probably
matching the surrogate to plain-PG's log-prob structure rather than
adding CE on top.
