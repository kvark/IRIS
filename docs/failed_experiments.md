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

### Update 2026-04-27 — bug confirmed as NaN cascade in encoder
With the snapshot mechanism re-enabled and `--grad-debug-every 5000`,
the failure mode is precisely localized: between step 40k and 45k
on CartPole, ALL `policy_encoder.*` parameters become NaN (verified
via grad-inspection's bulk weight-norm read). The KL gradient flowing
through stop_gradient(z) → policy_encoder pushes the weights into
overflow within a few hundred K-cycles.

Tried adding NaN/Inf guard + soft clamp (±20) on captured logits in
`capture_kl_snapshot_logits`; result was V collapse moved EARLIER
to step 15k (the clamp itself caused gradient pathology — clamping
old_logits to ±20 made KL gradient larger for non-clamped new_logits).

Real fix likely needs one of:
- Stop-gradient on z for the KL term too (currently only for entropy)
- Numerically-stable KL formulation that doesn't blow up on
  near-deterministic π_old
- Frozen-weights snapshot in a SEPARATE session (not a shared-weights
  forward pass), so capturing doesn't perturb gradients

All require multi-day diagnostic work. Documented and parked.

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

## Standard-PPO scaling on kindle's K-epoch loop

**Tested** 2026-04-27 with the now-fixed KL-PPO snapshot mechanism
(`stop_gradient(z)` on KL term, encoder no longer NaN-cascades).
The hypothesis was that running KL-PPO at *standard* PPO scales —
larger rollouts, more epochs per rollout — would let the trust region
engage productively and outperform plain-PG e2e on CartPole.

### Configurations tried (CartPole-v1, 240k env-steps)
| rollout | history_len | epochs | lr_pi | mean | symptom |
|---------|-------------|--------|-------|------|---------|
| 64 | 32 (default) | 4 | 3e-4 | ~+9 | V=0 throughout (buffer too small) |
| 64 | 32 | 10 | 1e-4 | ~+9 | V=0 throughout |
| 16 | 64 | 10 | 1e-4 | +10.75 | ent collapses by step 10k, KL=0 |

### Diagnosis
The K-epoch loop in `policy_step_batched` calls
`policy_step_rollout_batch` K times on the *same* `lanes ×
rollout_length` batch — no mini-batching. Standard PPO mini-batches
within each epoch precisely to prevent overfitting to a single batch.
With K=10 and tiny rollouts, the policy overfits and entropy collapses
to 0 within a few hundred rollouts. Both π_old and π_new collapse to
the same deterministic policy → KL stays at 0 → trust region never
engages anyway.

The architectural difference between kindle's per-step / small-rollout
cadence and standard PPO's "collect 2048 steps with frozen policy,
train 10 epochs of mini-batches" is fundamental. A real PPO refactor
would require:
1. A separate collect phase that snapshots obs/action/return arrays
   into a fixed buffer (decoupled from the lane buffers)
2. A train phase that mini-batches over that buffer (e.g., M=64
   mini-batch over B=2048 collected transitions, K=10 epochs)
3. Snapshot capture happens once at the start of the train phase
   (already implemented), but the K-loop iterates over mini-batches,
   not the full batch

This is multi-day work and the empirical evidence so far suggests
trust-region PPO doesn't outperform plain-PG e2e + LR-drop on these
toy tasks. Parked until a task surfaces where the trust region's
benefit outweighs the implementation cost.

### LunarLander head-to-head (2026-04-27)
Tested KL-PPO snapshot vs plain-PG e2e at identical config and
budget on LunarLander-v3 (the original motivation for trust region —
its local-optimum plateau).

| Approach | Mean | LR drop event |
|----------|------|---------------|
| Plain-PG e2e | -179 | step 34k @ -151 (well-timed; sustained) |
| KL-PPO snapshot | -380 | step 8k @ -140 (premature; crash to -900s) |

Same flags except `--use-kl-ppo --kl-use-snapshot --kl-beta 0.05
--kl-target 0.02`. 80k policy steps × 8 lanes = 640k env-steps each.

KL stayed at 0.0000–0.0019 throughout the snapshot run; adaptive β
collapsed to floor 0.0001 immediately. Diagnosis: lr_policy=5e-5
across a 32-row batch × 2 epochs produces weight drift too small to
register meaningful KL(π_new ‖ snapshot). Trust region never engages.

To get a non-trivial KL drift signal in the snapshot setup, kindle
would need either (a) much higher lr_policy with the trust region
constraining commit (but raises the entropy-collapse risk seen on
CartPole), or (b) a real PPO refactor that snapshots over a much
larger collect buffer with mini-batched K-epoch training. Neither is
in scope for the current investigation; the snapshot mechanism stays
in the codebase as a research artifact behind `--kl-use-snapshot`.

### Bug fixed in passing
Found and fixed: `entropy_beta_input_present` was true for
`use_kl_ppo + e2e + entropy_beta > 0`, but the KL graph variant
doesn't include the entropy regularization branch (no `entropy_beta`
input). Set_input call panicked with "unknown input: entropy_beta".
Gate now also checks `!use_kl_ppo`. Note: this means KL-PPO has no
entropy regularization on its policy loss — adding one would require
extending build_kl_policy_graph_e2e.

## DIAYN-options head-to-head on LunarLander

**Tested** 2026-04-27. Same context as the KL-PPO failure: testing
whether DIAYN (mutual-info skill diversity, kindle `5b7fbc7`) breaks
the LunarLander -132 ceiling that plain-PG e2e + LR-drop established.

### Configurations (LunarLander-v3, 80k policy steps × 8 lanes)
| Config | Mean | Notes |
|--------|------|-------|
| Plain-PG e2e | **-179** | baseline (no options, no DIAYN) |
| Options=4, DIAYN α=0.0 | -221 | options alone hurt |
| Options=4, DIAYN α=0.1 | -214 | tiny lift over α=0 |
| Options=4, DIAYN α=0.3 | -243 | DIAYN reward overwhelms extrinsic |

### Diagnosis
- Options alone (4 per-option heads, no DIAYN) hurt by 42 points vs
  plain-PG. The per-option-head gradient dilution outweighs whatever
  benefit option diversity might provide, at this budget.
- DIAYN at α=0.1 partially compensates (small lift over α=0) but
  doesn't recover plain-PG's level — entropy stays near max (1.30+)
  throughout, indicating the policy never commits.
- DIAYN at α=0.3 makes things worse — the intrinsic reward signal
  becomes large enough to drown out extrinsic, and the policy is
  rewarded more for trajectory diversity than for landing.

This aligns with the structural finding from MountainCar+DIAYN
(memory `project_kindle_structural_cap.md`): "DIAYN gives the agent
diverse skills to try, but if NONE of those skills stochastically
reaches the goal frequently enough to bootstrap policy gradient, the
agent never converges." LunarLander's positive-reward (+100 for
successful landing) is rare enough at random behavior that even
8 lanes × 6300 episodes = ~50k episodes don't see enough successful
landings to outweigh the DIAYN bonus.

### Conclusion across all 2026-04-27 LunarLander head-to-heads
Plain-PG e2e + LR-drop dominates at 80k×8 budget. Trust region (KL-PPO
snapshot) and skill diversity (DIAYN-options) both fail to break
through.

### LR-drop strength sweep — 100x is the sweet spot
Subsequently tested whether the prior reported "-132 ceiling" was a
sustained mean or a transient peak window. Extended budget to 200k
policy steps (1.6M env-steps) at three LR-drop strengths:

| `--lr-drop-on-solve` | Mean / 1.6M env-steps |
|----------------------|-----------------------|
| 10 (the previous default in scripts) | -221 (peak window -110, decays after step 100k to -480) |
| **100** | **-160** (sustained, no decay; range -130..-170 across 60k+ steps) |
| 1000 | -163 (similar to 100; freezing too aggressively loses minor learning) |

The prior reported ceiling of "-132 over 1.2M env-steps" was the
PEAK WINDOW from a 10x LR-drop run that was stopped before the
post-decay phase. With 10x drop, the policy reaches that peak
around step 80-100k but decays afterward as accumulated noise
pushes it into worse basins.

Setting `--lr-drop-on-solve 100` (instead of 10) preserves the policy
at the peak window indefinitely. This is the new recommended config
for kindle on negative-reward envs where the LR-drop strategy applies.
1000x freezes too tightly and the policy can't adapt to lane-batch
variance; 100x is the empirical sweet spot.
