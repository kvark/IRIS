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

## Frozen π_old snapshot for KL-PPO

**Removed** 2026-04-26 (implemented + reverted same day). Designed to
address the empirical finding that kindle's KL-PPO has near-zero KL
between π_new and π_old — the per-transition `logits_at_action` stored
at action-time is too close to the current policy (only a few env-steps
of drift), so the KL trust region never activates.

### Architecture
A `capture_kl_snapshot_logits` method that ran a forward-only pass
(LR=0) on the current rollout's obs/task using the current policy
weights, capturing the resulting logits as a "frozen snapshot." The
K subsequent training-epoch calls then used this snapshot as the
old_logits input, so KL would grow monotonically as the policy
drifted from the captured point — what standard PPO does.

### Implementation sketch
- New method `capture_kl_snapshot_logits(n_step, rollout_length)` that
  set a `kl_snapshot_capture_pending` flag and re-entered
  `policy_step_rollout_batch` with the flag on.
- Inside `policy_step_rollout_batch`, when the flag was set: LR forced
  to 0, weights didn't move, but the forward pass produced logits.
- After the call: `read_output_by_index(1, &mut kl_old_logits_scratch)`
  to capture the snapshot.
- Inside the K-epoch dispatch, called the snapshot capture once before
  the K-loop. Removed the per-row refill of kl_old_logits_scratch from
  ripe.logits_at_action (the snapshot replaced it).

### Symptoms
Tested on CartPole seed=42, KL-PPO config (vlc=0.1, β=0.05,
rollout=5, epochs=4):
- Steps 5k-25k: V learns normally (V[+0.18, +27]), ent oscillates
  0.01-0.69, KL stays at 0.0001-0.004 (still tiny — snapshot didn't
  produce expected KL growth).
- Step 35k+: V drops to exactly 0.00, ent locks at 0.69 (uniform max),
  pi loss has small spikes. Mean +17.

V going to exactly 0.00 indicates either the value head's output
saturates at the scaled_tanh's zero crossing, OR the encoder produces
z=0 (degenerate). Couldn't quickly diagnose which.

### Hypotheses for the bug
1. The forward-only "snapshot" pass still triggers the SGD update at
   LR=0; meganeura's update path does the dispatch even with
   learning_rate=0.0. Adam's m/v moments may still update (need to
   verify). If m/v drift during snapshot passes, then on real training
   passes Adam's effective LR is wrong.
2. The early-return after snapshot capture skips loss bookkeeping
   (read_loss, last_policy_loss EMA update, watchdog). The watchdog
   tracks policy_loss_ema and re-inits the policy if loss explodes.
   Skipping the read + EMA update could mask a bad loss.
3. The kl_old_logits_scratch buffer was being read mid-step in the
   capture pass; possible race with the GPU-side scratch.

### Why we stopped
The bug needs grad-inspection iteration that adds another 1-2 hours.
Plus, even if the snapshot mechanism worked, the deeper architectural
issue is that kindle's training cadence (per-step or 5-step rollout
with 1-4 epochs) is fundamentally different from standard PPO's
"collect 2048 steps with frozen policy, then train 10 epochs of
mini-batches." The snapshot would help, but a full PPO refactor
(separate collect phase from train phase) is the principled fix.

### What replaces it in practice
Nothing — KL-PPO without the snapshot remains in the codebase
(behaves like plain PG with a tiny KL nudge). The frozen-snapshot
mechanism is documented here for future attempt.

### Files removed
- `Agent::capture_kl_snapshot_logits` (~30 lines)
- `Agent::kl_snapshot_capture_pending` field
- The conditional snapshot capture call in the K-epoch dispatch in
  `policy_step_batched`

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
