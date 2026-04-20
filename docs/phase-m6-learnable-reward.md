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
| `α=0.1` clamp=5 | 5.53% | 10.0% |
| `α=0.5` clamp=5 | 5.73% | 8.5% |
| `α=5.0` clamp=20 | 4.90% | 8.5% |

The head demonstrably trains (MSE loss drops from ~20k → ~300;
`r_hat` drifts signed across ±3). Pushing α × clamp up 100× does
*not* help — it mildly hurts. That's the first clue.

### Mechanism check (2026-04-19, 30k steps instrumented)

Mean `R̂(z_t)` bucketed by the episode's eventual outcome:

```
outcome   episodes   ep_len   mean r_hat   early    mid     late
soft           80     87.2    +0.86        +0.60   +0.82   +1.14
crash        1219     91.7    +0.96        +0.76   +0.90   +1.20
timeout         1   1000.0    −1.10        −1.34   −1.06   −0.89

soft − crash gap: mean = −0.098  (inverted)
```

At `α=5.0, clamp=20` the inversion widens (soft−crash = −0.191)
rather than resolving.

**The head learns "later-in-episode = higher r_hat"**, not
"soft-trajectory = higher r_hat". Crashes and soft landings are
statistically indistinguishable to it. Loud M6 amplifies the
slight inversion and mildly hurts the cumulative soft-rate.

### What the mechanism check actually showed

The reward circuit's **episode-return signal does not distinguish
soft from crash** in kindle's intrinsic regime:

- Soft / crash episode lengths are near-identical (~87 vs. ~92 steps).
- Per-step homeo is −3 to −15, dominating a cumulative −400 to
  −1000 per episode.
- v3 shaping's `not_safely_landed * 1.0` term produces a +1
  bonus at one step of one episode — invisible against the
  cumulative magnitude.

So the Monte-Carlo target that M6 trains against — the sum of
existing reward-circuit outputs over an episode — contains no
soft-vs-crash gradient. M6 inherits the silence. It isn't a case
of "the primitive shape is wrong" or "the bonus is too quiet";
**there is nothing landing-related in the training target itself**.

### Revised understanding

The earlier write-up's conclusion ("M6 is state-instantaneous,
therefore can't express event-ordering") was the wrong
diagnosis. Episode-MC training *does* bake the terminal into every
per-state target — that part works. The real problem is that the
*content* of that target (cumulative intrinsic reward) has
almost no correlation with landing quality. Kindle's four
primitives emit virtually the same integral on a 90-step soft
landing and a 90-step crash; the terminal event is a rounding
error, not a gradient.

### Follow-up experiments (2026-04-19): (1) amplified shaping,
### (2) different target

Tested both directions I proposed. Both are now infrastructure
(v5 shaping in `lunar_lander_batch.py`; `OutcomeTarget` enum
with `EpisodeSum` and `TerminalReward` variants in agent.rs).

```
config                              cum soft    gap      loss
v3 + EpisodeSum     (prior)          5.53%     −0.098    12k
v5 + EpisodeSum     — (1)            4.78%     +0.032   300k
v3 + TerminalReward — (2)            5.00%     +0.011    15
v5 + TerminalReward — (1)+(2)        5.18%     −0.002    60
```

All three end up with a near-zero discrimination gap. v5 +
EpisodeSum nudged the gap positive (+0.032), the first
non-inverted result — but the cumulative soft-rate dropped
slightly. The head is fitting its target well (loss is 60 for
TerminalReward), but what it's fitting turns out not to be
useful per-state information.

### Revised revised diagnosis

Even with a target that carries quality signal (v5's terminal
homeo is distinctly worse for crashes; v3's terminal reward
differs by several reward units), the per-state R̂(z_t) still
can't discriminate. Why?

**Mid-flight z_t doesn't contain forward-trajectory information.**
Two identical mid-flight states can end soft or crash depending
on future policy decisions. A per-state predictor fit to
outcome labels converges on the expected value *averaged over
all possible futures from this state* — which for a
mostly-exploratory policy is nearly uniform across states.

This is **a pure information-theoretic limit of the input**, not
a shape problem or a target problem. No amount of target
engineering fixes it, because the information R̂ is asked to
provide simply isn't in its input.

Confirms the deeper shape of the issue I kept circling around:

- ❌ "M6 is state-instantaneous, can't express event-ordering"
    — wrong; training bakes trajectory into per-state targets.
- ❌ "M6's target doesn't differ between soft and crash"
    — we fixed this with v5 and TerminalReward; gap still zero.
- ✅ "The *input* z_t doesn't carry forward-trajectory info,
      so per-state R̂ always reduces to a state-average
      prediction."

### What M6 v2 would have to look like

Three real architectural directions to give R̂ the input it
needs, in order of scope:

A. **Action-conditioned R̂(z_t, a_t)** — adds "what the agent is
   doing" to the input. Becomes a Q-function. Large overlap with
   the existing policy/value path; arguable duplication.
B. **Trajectory-conditioned R̂(z_{t−k:t})** — small MLP over a
   concatenated window of recent latents. Captures momentum
   ("this trajectory is slow + upright → likely soft"). Distinct
   from the policy path. Natural kindle fit; CPU-side, shares
   M6 v1 infrastructure.
C. **Outcome-supervised encoder** — unfreeze the encoder under
   a gradient from M6's outcome loss. The encoder then learns
   representations where z_t itself carries outcome-relevant
   features, even from a single frame. Biggest departure from
   kindle's M1 ignition invariant.

## M6 v2 implementation (2026-04-19): windowed head

Implemented (B): `OutcomeHead` generalized to accept a window of
`k` concatenated latents, input dim = `k · latent_dim`.
`window == 1` is byte-identical to M6 v1 (verified on
l1_diagnostic across 7 envs). `AgentConfig::outcome_window`
defaults to 1; `BatchAgent(outcome_window=k)` and
`--outcome-window k` plumb through.

100k / 4 lanes / seed 42 / α=0.1 / clamp=5:

```
config                                         cum soft   gap
v5 + EpisodeSum   + window=1   (prior)          4.78%   +0.032
v3 + EpisodeSum   + window=16                   ~5%     −0.17
v5 + EpisodeSum   + window=16  (mid-train 30k)  —       +0.144
v5 + EpisodeSum   + window=16  (full 100k)      4.78%   +0.052
v3 + TerminalReward + window=16                 5.04%   +0.012
```

**The windowed head produces the first-ever non-trivial positive
discrimination gap on v5 + EpisodeSum: +0.144 at 30k training
steps.** Above random noise, above every prior (v1 + any variant)
result, and in the correct direction (soft > crash). Sanity test
on the architectural claim: yes — a windowed input CAN
discriminate where a single-frame input could not.

### Why the soft-rate still doesn't move

Two compounding issues identified from the 100k + instrumented
run:

1. **Class imbalance saturation.** The training set is ~20× more
   crash windows than soft windows. Shared-target-per-episode
   training collapses the head toward the dominant-class target;
   the discrimination gap *shrinks* with more training (+0.144 at
   30k → +0.052 at 100k).

2. **Intra-episode flatness of the predicted bonus.** With one
   target shared across every step of a given episode, the head
   fits to the episode-wide mean. Within an episode, `R̂` is
   essentially constant across early / mid / late windows. A
   constant shift to per-step reward within an episode is a
   **policy-gradient-invariant** — the optimal policy doesn't
   change under a constant reward baseline. So even a
   discriminating head produces no policy gradient direction
   when its output is per-step-flat.

The +0.144 gap is a signal about episode class (soft vs. crash)
but not a signal about *where in the episode to act differently*.
The policy needs per-state differential, not per-episode
differential.

### What v2 actually proved

- ✅ Trajectory input fixes the information deficit identified in
  the v1 mechanism check. Windowed R̂ can discriminate outcomes.
  Theory confirmed.
- ✅ The discrimination gap appears only when the target carries
  the signal (v5, not v3). Both halves of the 2026-04-19 picture
  are now empirically validated.
- ❌ Shared-episode targets + class imbalance produce a flat
  per-episode bias rather than per-step differential. That bias
  contributes no policy gradient, so cumulative soft-rate
  doesn't improve.

### M6 v3 implementation (2026-04-19): RTG × PotentialDelta

Implemented both:

1. **`OutcomeTarget::RewardToGo`** — back-accumulate per-step RTG
   targets at episode end, train the head with a variable
   per-sample target (`train_batch_variable` in outcome.rs).
   Each step in a completed episode gets its own supervision
   signal = `Σ_{k≥t} r_base_k − baseline`.
2. **`OutcomeBonus::PotentialDelta`** — swap the per-step bonus
   from `α · R̂_t` to `α · (R̂_t − R̂_{t-1})`. Classical Ng-et-al
   potential-based shaping, policy-invariant up to a constant.
   `prev_r_hat` cached on `Lane`, reset to 0 at `env_boundary`.

Both plumbed to `BatchAgent` and `lunar_lander_batch.py` as
string-keyed args. Clippy / unit tests clean (24 pass + 5
outcome-specific).

### Results (100k / 4 lanes / seed 42 / v5 shaping / window=16)

```
config                                          α    clamp   cum     gap     verdict
EpSum     + Raw            (M6 v2)              0.1    5    4.78%  +0.052   weak
EpSum     + PotentialDelta                      0.1    5    5.18%  +0.018   late +0.106
RTG       + Raw                                 0.1    5    4.96%  −0.158   clamp-saturated
RTG       + PotentialDelta                      0.1    5    5.18%  +0.322   ★ first pass
RTG       + PotentialDelta                      0.02  50    5.18%  +3.22    strong discrim
RTG       + Raw                                 0.02  50    5.41%  −1.05    policy-feedback flip
RTG       + PotentialDelta                      0.5   50    4.68%  +0.22    loud → policy-flip
```

### What v3 proved

✅ **Per-step RTG targets + PotentialDelta + windowed head IS the
architecture that discriminates.** `+3.22` gap is 22× the
original M6 v1 result and the first config to pass the verdict
threshold consistently. The theory ladder is now fully validated:
windowed input + per-step target + difference-based bonus
together produce a clean per-state quality signal.

❌ **Soft-rate plateau holds.** Every config lands in 4.68–5.41%,
indistinguishable from the 5.38% pre-M6 baseline at 4-lane /
100k / seed 42. The head discriminates; the policy doesn't
improve.

### Why discrimination doesn't translate

Observed in the logs:

- Under PotentialDelta, `r_hat` is approximately *constant within
  an episode* even under RTG training (soft 42.69 early → 41.73
  late, variation of 1.0 over 90 steps). The windowed inputs
  differ across the episode, and the per-step targets differ
  substantially, but the head learns to emit a roughly
  episode-constant output because MSE on wildly-scaled targets
  (0 → +500 within one episode) pulls the fit toward a single
  mean output per episode class.
- PotentialDelta of a near-constant sequence is near-zero per-step.
  The bonus DIFFERENCE between soft-state and crash-state
  trajectories is present in the cumulative sum (telescoping), but
  the per-step gradient signal to the policy is tiny.
- Raw bonus at magnitudes loud enough to drive the policy
  (`α=0.02 × clamp=50 → ±1/step`) causes a **policy feedback
  loop**: the bonus changes the trajectory distribution, the new
  trajectories retrain the head with different r_hat correlations,
  and within a few thousand steps the head's prediction *inverts*
  (crashes get higher r_hat because the bonus-modified policy
  actively pursues M6-predicted-valuable states that happen to be
  long-running pre-crash states).

### Revised revised revised diagnosis

At this point we've exhausted every combination in the M6-shape
design space:

- Frozen primitives → loud terminal shaping → episode-MC → terminal
  target → windowed input → per-step RTG targets → delta-based
  bonus. All composed, all tested.

The cumulative soft-rate is invariant to all of it at this budget.
Even the architecturally-strongest config (`RTG + PotentialDelta +
clamp=50 + α=0.02`, +3.22 discrimination gap) gives 5.18% cum, the
same as baseline.

**The plateau is not a discrimination problem anymore.** It's
something downstream: either the policy's optimization budget is
too small to convert a discrimination signal into improved
behavior, or the kindle-intrinsic policy-space genuinely has no
trajectory that reliably lands in ≤100k 4-lane steps.

### Honest stopping point

M6 v3 (RTG × PotentialDelta) is the architectural end of the
"learnable reward circuit" line. Code stays in — it's verifiably
correct, the mechanism is validated, and it's a clean platform
for future work. But the LunarLander soft-rate does not move.

## Ablation sweep (2026-04-19): the non-M6 cause of the plateau

After an exhausted reward-circuit hypothesis class and a user
challenge of "wait — homeo points at landing; why doesn't
tuning fix it?", ran a commitment-suppression ablation against
the policy optimizer stack instead of the reward. Key observation
that motivated it: **policy entropy stays at ln(num_actions) ≈
1.386 throughout all M6 runs on LunarLander** — the policy never
commits. On all 6 other gym envs kindle handles, entropy drops
cleanly (CartPole 0.59, MountainCar 1.08, Acrobot 1.05, etc.),
so the commitment failure is LunarLander-specific.

100k / 4 lanes / seed 42 / v3 shaping / no M6:

| # | ablation | entropy behavior | cum soft% |
|---|---|---|---|
| 0 | defaults | **pinned at 1.386 (no commit)** | 5.57% |
| 1 | `entropy_beta=0, floor=0` | pinned | 5.45% |
| 2 | `advantage_clamp=20` (was 1) | **drops to 0.82**, oscillates | 4.27% |
| 3 | watchdog threshold → 1e9 | pinned | 5.48% |
| 4 | all three off | drops to 0.80, unstable | 4.40% |
| 5 | all three + history_len=64 | initial drop, pi→-2M | 4.71% |

### Findings

1. **The advantage clamp is the primary commitment suppressor.**
   Default `clamp(r−V, ±1)` destroys policy-gradient magnitude on
   states where the reward actually differentiates actions (v3
   mid-flight r−V can be ±2–5 routinely). With the clamp widened
   to ±20, entropy drops from 1.386 to 0.82 — policy finally
   commits. (2) and (4) both show this clearly.
2. **Entropy regularization is NOT the cause.** Removing
   `entropy_beta` and `entropy_floor` (ablation 1) leaves entropy
   pinned. The policy isn't being held at uniform by
   regularization — it's being held by the *absence of a
   gradient signal large enough to move it off uniform*.
3. **The magnitude watchdog is NOT the cause in isolation.**
   Disabling it alone (ablation 3) also leaves entropy pinned. It
   plays a role only when interactions get unstable (ablation 5
   shows `pi → −2M` when watchdog is off with wide clamp +
   history; instability follows).
4. **Commitment ≠ better landing.** When the policy does commit
   (ablations 2, 4, 5), cumulative soft-rate *drops* to 4.3–4.7%
   — BELOW the random-action accident rate of 5.57%. The policy
   converges on something strictly worse at landing than random.

### What the ablations prove about the plateau

The 5.57% baseline is exactly the random-action accident rate —
kindle's default stack wasn't learning *anything* on LunarLander,
it was sampling uniformly and receiving the physics-determined
probability of leg+stopped terminal geometry. This explains why
20+ configurations across Tier 3 + M6 v1/v2/v3 all hit the same
number: they were all measuring random-policy physics.

When we remove the clamp that was blocking policy-gradient
signal, kindle *does* learn — but what it learns is a
**crash-fast policy** that's worse at landing than random.
Whether this is (a) a reward-specification issue where the
*global* optimum is still soft-land but policy gradient falls
into a local crash-fast basin, or (b) a reward-specification
issue where the local AND global optima are both crash-fast, is
not distinguishable from our data alone — but in either case the
answer to the user's original question is:

**Under v3 shaping's reward landscape, the policy gradient from
random initialization has the crash-fast basin as its nearest
local minimum. Not the soft-land one.** Back-of-envelope math
suggesting slow-land has better cumulative return than fast-crash
may still be correct (slow-land as a global optimum), but it's
not *reachable* by local search from kindle's initial policy.

### Revised plateau root-cause

Not one cause, but two, layered:

1. **Optimization-stack layer (blocker):** `advantage_clamp=±1`
   keeps the policy pinned at uniform by suppressing the
   per-step policy-gradient signal. Removes kindle's chance to
   learn anything on LunarLander.
2. **Reward-landscape layer (genuine problem):** even with the
   clamp widened so gradients flow, policy gradient from random
   init finds a local optimum that's worse at landing than
   random. Either the reward's local optimum genuinely prefers
   crash-fast, or the soft-land basin is unreachable from uniform
   policy without curriculum / demonstration / structured
   exploration that kindle doesn't have.

Fixing (1) alone doesn't help; committing faster to the wrong
thing isn't progress. Fixing (2) requires either (a) a reward
whose nearest local optimum from uniform IS soft-land — this is
the reward-tuning hypothesis the user originally raised, and
remains untested but is plausibly narrow — or (b) an exploration
mechanism beyond per-step entropy regularization that helps
escape the fast-crash basin.

### The six-env pivot is no longer about "LunarLander is
### structurally outside kindle's class"

It's a pragmatic pivot. LunarLander is solvable by kindle *in
principle* given the right reward landscape + an escape mechanism
from the fast-crash basin, but both are real project-scope
interventions that don't fit within M6. The six working envs
(CartPole, MountainCar, Acrobot, Pendulum, Taxi, GridWorld) all
have reward landscapes where the fast-termination / do-nothing
option *is* locally optimal for homeostasis, so kindle's default
stack (including the advantage clamp) converges cleanly even
when commitment is weak. They are where kindle's design is
working as intended.

The default `advantage_clamp=1.0` is still a reasonable value for
those envs — raising it uniformly would destabilize training
elsewhere. Keeping the knob configurable lets LunarLander-like
envs ablate it without affecting anything else.

## PPO ceiling test (2026-04-19): the reward really is the problem

Following the "is kindle's reward landing-compatible at all?"
question raised after the ablation sweep, ran stable-baselines3
PPO against a gym wrapper that substitutes kindle's homeo reward
for the native env reward. Same 8-dim obs, 4 discrete actions.
Code in `python/examples/lunar_lander_ppo.py`.

100k steps / seed 42, comparing last-third soft-rate (= landing
rate of the converged policy):

| reward | last-third soft-rate | note |
|---|---|---|
| v3 kindle homeo | **2.15%** | PPO converges AWAY from landing (random was 5.57%) |
| v5 kindle homeo (×50 terminal) | 6.28% | stuck at random |
| v6 tuned (no-altitude, crash×50) | **1.02%** | PPO hovers high to avoid ground |
| v3 + ±100 terminal bonus | 2.13% | terminal signal drowned by −1000 per-step sum |
| v3 + +5000 soft bonus | 3.04% | rare-event × rare-rate = low expected value |
| **native gym reward** | **31.52%** | **climbing, classic PPO-LunarLander behavior** |

### Interpretation

**PPO is not the bottleneck.** Given native LunarLander reward
(+approaching pad, +leg contact, +100/−100 terminal), sb3's PPO
reaches 31% last-third and is still improving at 100k. Classic
benchmark behavior.

**Given kindle's homeo reward, PPO converges on non-landing
policies** — sometimes hover (v3), sometimes crash-fast (v5),
sometimes high-altitude glide (v6). Each is the local optimum
of the corresponding reward. None is landing.

**Terminal bonuses don't bridge the gap**, at least not at
reasonable magnitudes. A +100 terminal bonus is drowned by 100
steps of −10 homeo. A +5000 bonus is theoretically dominant but
its *expected value under exploration* is 5000 × 5% (random
soft-rate) = +250 per rollout, still smaller than the −1000
cumulative homeo per episode. The policy rationally avoids the
risky trajectory that might land.

### What distinguishes landing-compatible rewards

Native gym reward has continuous **approach shaping**: each step
it's closer to the pad with good velocity, it earns positive
reward; each step it's further, negative. The gradient of the
reward function points AT the landing pose at every state along
any trajectory. Local policy gradient can follow this gradient
home.

Kindle's homeo primitives can only PENALIZE deviations from
target. They can't emit *positive* reward for "approaching
target from far away" — the homeo function is
`max(0, |val−target|−tol)`, which is zero at target and positive
everywhere else. There's no mechanism in kindle's homeo formula
that says "this is less bad than the last state, here's a
bonus" — only "this is still out of range, cost accumulates".

The surprise and novelty primitives *can* be positive but are
misaligned on LunarLander: chaos rewards higher surprise, random
trajectories reward higher novelty. Only the `order` primitive
aligns with controlled behavior, and it's weighted at 0.5 vs
homeo's 2.0 by default.

### Final root-cause diagnosis

**Kindle's reward class — weighted sum of per-state homeo
deviations + surprise + novelty + order — cannot express
approach-shaped rewards that local policy gradient needs for
event-ordered tasks like LunarLander.** This is structural, not a
tuning issue:

- No subset of homeo variable weights produces a reward whose
  gradient points at landing, because no such weights exist
  within the function class (homeo can only penalize, not reward
  approach).
- Adding terminal bonuses (±100, ±5000) isn't enough because
  rare-event bonuses have low expected value during exploration
  relative to dense per-step cost.
- Fixing kindle's optimizer stack (as the ablation sweep showed
  we can) doesn't help because PPO — a much better optimizer —
  can't land under kindle's reward either.

The plateau was two things compounded:

1. **Advantage clamp blocked kindle from learning at all** on
   LunarLander. Fixable, single-knob change. Value remains at
   1.0 as default for the 6 other envs where it's fine.
2. **The reward landscape itself has no landing optimum
   reachable by local policy gradient.** This is the one that
   needs a structural change — add approach-shaping as a first-
   class kindle primitive, or concede that event-ordered tasks
   like LunarLander are outside the reward class kindle's design
   can express without external signals.

### Implication for the project

Kindle's ideological claim "homeostasis defines good" is valid
for the class of tasks where success is a steady-state condition
on the agent's own observables — CartPole (stay upright), Taxi
(reach delivery state), MountainCar (reach flag), GridWorld
(forage), Pendulum (hold angle). It is NOT valid for tasks where
success is a specific terminal EVENT that requires
approach-rewarded navigation — LunarLander (land softly at a
specific pose), any pixel-based task with a goal object.

The six working envs are not a pragmatic fallback; they are
**exactly the class the design was always going to succeed on**.
LunarLander is the first clean demonstration of the design's
structural limit. That's a valuable result — the project now
has a sharp characterization of what its principle covers.

Extending kindle to cover LunarLander-class tasks requires
introducing an approach-reward primitive, which means either (a)
a hand-crafted distance-to-goal homeo (external knowledge about
what "goal" means), (b) a learned approach-reward via
goal-discovery (significant new research), or (c) accepting
external supervision for these tasks. None of these fit in M6;
all are project-scope decisions.
