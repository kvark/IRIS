# Phase M6 — Learnable reward circuit

## Motivation (distilled from the LunarLander Tier 3 verdict)

All three Tier 3 architectural levers — L0 capacity, goal
representation, credit horizon — cap at the same ~5% / ~9% soft-rate
on LunarLander. The ceiling is not about *how* the agent routes the
signal; it's that the signal isn't in the reward circuit to begin
with. Kindle's four primitives (surprise, novelty, homeostatic,
order) are all **state-instantaneous** functions. LunarLander
success is a **sequential-ordering** property (legs before belly, in
the last few steps, at the end of a slow descent). A weighted sum
of state-instantaneous primitives, however tuned, can't express it.

M6 is the minimal architectural answer to this gap: add a
**learned** primitive whose training signal is episode outcome,
alongside (not replacing) the four frozen primitives. Kindle's
original M6 scope ("unfreeze the weights") turns out to be
necessary-but-insufficient — the 4-D weight space doesn't contain
a combination that expresses event-ordering. We need a *new
primitive with learnable structure*, not just learnable weights on
the existing four.

## Design principle

- **Add capacity for outcome-driven reward; don't redefine the
  existing primitives.** The four frozen primitives stay frozen.
  The new head is additive. If it's silent, behaviour is the
  pre-M6 agent.
- **Train only from self-observables.** The agent already receives
  `mark_boundary` / `env_boundary` as part of its sensor stream
  (the env tells it "a reset happened" the same way it tells it
  "the joint angle is 0.7"). No extrinsic reward enters the
  system — only the structure the agent is already exposed to.
- **Reducible to zero.** Behind a config flag (default off) so the
  pre-M6 canary / M5 stability result is preserved.

## The primitive: outcome-conditioned value prediction

A small MLP `R̂: z_t → scalar` trained to predict the *total
intrinsic return* over the episode that `z_t` belongs to:

```
R_episode  = Σ_{t ∈ episode} r_t      (sum of existing reward-circuit output)
target(t)  = R_episode − baseline_ema  (centered for variance reduction)
loss       = MSE(R̂(z_t), target(t))   over all t in the just-ended episode
```

This is a pure *outcome-prediction* head — it asks "what total
return did my trajectory earn when I was in state z_t?" and
trains on episode completions.

Crucially, `R̂(z_t)` can learn to discriminate *decelerated-slow-
upright* mid-flight states (which occasionally end in soft
landings, R_episode ≈ −15) from *out-of-control* mid-flight states
(which reliably end in fast crashes, R_episode ≈ −80). The four
frozen primitives cannot express this distinction because they
never see the terminal. `R̂` can, because MC-episode-return
training bakes the terminal into every z_t target.

### Why this isn't just "n-step returns" again

Tier-3 sequence credit tested n-step Monte-Carlo advantage on the
single shared value head. It capped identically to the other
Tier-3 levers. M6 is different in two ways:

1. **Separate head + separate LR.** `R̂` trains on its own MSE
   loss, on its own GPU graph, at its own learning rate. The
   joint-LR instability that killed commit 8f291e5 doesn't apply.
2. **Full-episode horizon, not sliding window.** n-step at 32
   steps with γ=0.99 still terminates inside an episode
   (LunarLander episodes are 100–300 steps). `R̂` trains on the
   *complete* episode return — no bootstrap, no truncation.

### How it enters the reward signal

Additively, as a fifth primitive:

```
r_t_augmented = r_t_base + α · R̂(z_t)
```

where `α` is a fixed scalar (default 0.1). The policy gradient
then sees states predicted to precede good outcomes as rewarding,
independent of whether any of the four frozen primitives fire on
that specific state.

Two guardrails:
- `R̂(z_t)` is clamped to `[−5, +5]` before multiplication by α.
  A runaway outcome prediction can't destabilize the policy.
- Gradients from `R̂` do **not** flow back into the encoder. The
  z input is detached the same way the world-model target is
  (BYOL-style stop-grad). This keeps the encoder learning solely
  from dynamics prediction, preserving the M1 ignition invariant.

## Changes to kindle

### AgentConfig

```rust
/// M6 learnable reward: weight α on the outcome-value head's
/// output when it's added as a fifth reward primitive. `0.0`
/// (default) disables M6 entirely — behaviour byte-identical to
/// pre-M6.
pub outcome_reward_alpha: f32,

/// M6: learning rate for the outcome-value head. Defaults to
/// `learning_rate × 0.3` (same scale as the credit head).
pub lr_outcome: Option<f32>,

/// M6: EMA rate for the baseline used to center episode-return
/// targets. Lower = smoother baseline, higher variance reduction
/// but slower drift. Default `0.05`.
pub outcome_baseline_ema: f32,
```

### New module: `kindle/src/outcome.rs`

```rust
pub fn build_outcome_graph(
    latent_dim: usize,
    hidden_dim: usize,
    batch_size: usize,
) -> Graph { /* z → MLP → scalar, MSE loss vs. "target" input */ }
```

Inputs:
- `z` : `[batch_size, latent_dim]`
- `target` : `[batch_size, 1]` — the centered episode return
  broadcast-copied across every batch row of the same episode.

Outputs:
- `loss` : scalar
- `r_hat` : `[batch_size, 1]`

Batch size = `min(buffer_capacity, episode_length_cap)`. For
LunarLander with episode_length ≈ 100–300, batch 256 fits.

### New per-lane state

```rust
struct Lane {
    /// Trajectory buffer for the current episode: the sequence
    /// of `z_t` vectors since the last `env_boundary`. Capped at
    /// `outcome_max_episode_len` (2000 — any env that exceeds
    /// gets its tail truncated; diagnostic warn on truncation).
    /// Cleared on episode start.
    outcome_ep_trajectory: Vec<Vec<f32>>,
    /// Running sum of `r_t` (base reward, no M6 bonus) since the
    /// last `env_boundary`.
    outcome_ep_return: f32,
    /// Running EMA baseline of completed episode returns.
    outcome_baseline: f32,
    /// Cached last `R̂(z_t)` for diagnostics + reward-bonus use.
    last_r_hat: f32,
}
```

### Training flow

At every step:
1. Push `z_t` onto `lane.outcome_ep_trajectory`.
2. Accumulate `r_t_base` into `lane.outcome_ep_return`.
3. Forward the outcome head on current batched z-stack (LR=0)
   to read `last_r_hat` for the bonus.

When `env_boundary` fires on a lane (episode end):
1. `R_ep = lane.outcome_ep_return`.
2. `target = R_ep − lane.outcome_baseline`.
3. Train the outcome head on *that lane's* trajectory: batched
   forward+backward over up to `outcome_max_episode_len` z's with
   the same centered target. Lanes whose episode hasn't ended
   this step contribute zero rows (LR-mask the grad).
4. Update `lane.outcome_baseline ← (1 − ema) · baseline + ema · R_ep`.
5. Clear `outcome_ep_trajectory` and `outcome_ep_return`.

The per-lane episode-end training dispatch is cheap — typical
episode length ≤ 300 z's at latent_dim=16 = 4.8k floats, one MLP
backward. Runs once per episode per lane, not per step. At 4
lanes × 1 episode / 200 steps × 100k steps = 2000 total dispatches.

### Reward integration

In `observe()` after computing the four existing primitives:

```rust
let r_base = ...;  // existing weighted sum
let r_m6 = if alpha > 0.0 {
    let capped = lane.last_r_hat.clamp(-5.0, 5.0);
    alpha * capped
} else { 0.0 };
let reward = r_base + r_m6;   // what the policy / credit / value heads see
let reward_for_outcome = r_base;  // what trains R̂ — no self-reference
```

The outcome head trains on `r_base` (not `r_base + r_m6`) so its
target can't chase its own output — stability guarantee.

## LunarLander success criterion

If M6 works the way the design claims:

- Peak-window soft-rate ≥ 15% consistently at 100k steps / 4 lanes
  / seed 42, with v3 shaping, `α = 0.1`, `n_step = 1`.
- `R̂(z_t)` diagnostics show a visible gap: mean-R̂ at mid-flight
  in "decelerating" states is noticeably higher than in
  "free-fall" states (measurable by bucketing states by `vy`).

Null result (≤ 9% peak-window) is a stronger finding than Tier 3's
null results were — if the direct outcome-prediction primitive
fails, LunarLander's intrinsic-reward budget is truly depleted and
option (1) in next-steps (accept the ceiling) becomes the answer.

## What M6 does not do

- **Does not unfreeze the four primitives.** Surprise / novelty /
  homeostatic / order keep their fixed weights and formulas.
- **Does not feed extrinsic reward.** The agent learns from
  `env_boundary` events, not from the gym `reward` field.
- **Does not break cross-env transfer.** The outcome head lives
  alongside the encoder and trains on each env's own episodic
  signal; the encoder stays unchanged.
- **Does not change M5 (stability) guarantees.** Behind a config
  flag; default `α = 0` is byte-identical to pre-M6.

## Rollout order

1. `outcome.rs` module + graph (smallest surface).
2. Session build + init in `Agent::new` gated on `alpha > 0`.
3. Per-lane trajectory tracking in `observe()`.
4. Episode-end training dispatch in `observe()`.
5. Reward-bonus plumbing + `last_r_hat` forward read.
6. Python binding: expose `outcome_reward_alpha` on `BatchAgent`.
7. LunarLander 100k eval vs. baseline (`α=0` canary, `α=0.1`).

Stopping points at each step: after (1–2), graph compiles + no
regressions on existing benches. After (3–5), `α=0` byte-parity.
After (7), either the ceiling lifts or we've cleanly null-tested
the last remaining Tier-4 lever.

## Implementation notes (2026-04-19)

Implemented as a **CPU** MLP rather than a second GPU session. At
kindle's shapes (latent_dim=16 → hidden=32 → 1 ≈ 560 params), a
dedicated GPU graph forces awkward batch-size choices (per-step
inference at N lanes vs. per-episode training at L trajectory
rows) and costs more in dispatch overhead than compute. The
stop-grad into the encoder is automatic at the CPU boundary —
there's no parameter path back through the encoder at all.

## Results (2026-04-19, 100k / 4 lanes / seed 42 / v3 shaping)

| config | cumulative soft | peak window |
|---|---|---|
| pre-M6 baseline | 5.38% | 9.5% |
| `α=0.1` | 5.53% | **10.0%** |
| `α=0.5` | 5.73% | 8.5% |

M6 moves the peak window up by ~0.5pp at `α=0.1` but doesn't
crack the 15% bar the design doc set. The head demonstrably
learns (MSE loss drops from ~20k → ~300 over 100k; `r_hat` drifts
signed across ±3). The bonus is ~6–30× smaller than the
per-step homeo penalty in absolute terms (`α · clamp(r_hat) ≤
±0.5` vs. homeo `≈ −3 to −15`), which likely explains why it
can't dominate policy learning.

### Interpretation

M6 as designed (additive bonus on instantaneous state, trained
on episode-MC return) is **necessary but not sufficient** for
LunarLander. The head learns the distribution of returns, but
the signal it emits is structurally still *state-instantaneous* —
R̂(z_t) is a function of a single latent. The missing ingredient
isn't "a learnable primitive" in the abstract; it's specifically
**event-ordering awareness**: the primitive needs the trajectory
as input, not just one frame.

### What would unlock LunarLander honestly

Three directions, ordered by scope:

1. **Reweight M6 relative to homeo.** Raise the per-step homeo-
   penalty clamp or drop its weight when M6 is active, so the
   learned signal isn't drowned. Low effort, narrow gain.
2. **Trajectory-conditioned M6.** Replace `R̂(z_t)` with
   `R̂(z_{t−k:t})` — a small causal-attention head over recent
   latents, predicting episode return from a window. This
   genuinely adds event-ordering expressiveness. Medium scope
   (needs a small attention graph, likely back to a GPU session).
3. **Accept the ceiling.** Per the Tier-3 verdict: pivot the
   success bar to the 6 envs where kindle is already
   demonstrating option differentiation and transfer.

Tier-3 and M6 together null out the "single-variable fix"
hypothesis class. The remaining hypotheses are architectural
(trajectory-conditioned reward, option-hierarchy credit) or
philosophical (accept the ceiling). The M6 code stays in — it's
a clean building block for (2) if we pursue it.
