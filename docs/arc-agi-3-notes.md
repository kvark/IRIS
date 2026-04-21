# ARC-AGI-3: first integration and diagnosis

## Pulled

- `arc-agi` python toolkit (0.9.8) and `arcengine` (0.9.3) — both
  available on PyPI without the full agents repo's heavy LLM
  dependencies (langchain / smolagents / langgraph etc.).
- Anonymous API key auto-obtained via `arc_agi.Arcade()` — no
  signup needed for local play.
- Local execution of 25 shipped ARC-AGI-3 environments via
  `arc_agi.LocalEnvironmentWrapper`, gym-style `reset()` /
  `step(action)` interface.

## Adapter

`python/examples/arc_agi3_batch.py` — minimal kindle↔ARC-AGI-3
bridge:

- **Observation**: 64×64 int8 grid (colours 0–15) → mean-pool to
  8×8 (64 cells) → flatten to 64-dim float in `[0, 1]`. Matches
  kindle's `OBS_TOKEN_DIM = 64`.
- **Action**: kindle emits discrete action index 0..N−1; we map
  to the game's current `available_actions` list (filtered to
  ACTION1..ACTION7 "simple" actions only). `N` is set from
  `min(6, len(simple_available))` to match kindle's
  `MAX_ACTION_DIM = 6`.
- **Homeo**: two variables:
  - `levels_completed` delta (spikes `-scale` on level
    completion; otherwise zero — this IS the only reliable
    progress signal on ARC-AGI-3).
  - frame-entropy proxy (1 − unique_colours/16): small negative
    when the agent is in a visually monotonous state.
- **Episode boundary**: `state ∈ {NOT_PLAYED, GAME_OVER, WIN}`
  or `ep_step ≥ max_episode_steps`.

### What the adapter doesn't handle

- **Complex actions** (`ACTION6`, sometimes `ACTION5`) that take
  `(x, y)` coordinates. Several games (m0r0, s5i5, sp80, ...)
  mark these as `available_actions` but stepping them raises
  `KeyError: 'x'` without coordinate data. Our adapter filters
  to simple actions only; games whose simple-action set is
  empty are out of scope.
- **Multi-frame state** (`frame` is a list; only frame[0] used).
  ARC-AGI-3 games can present multiple frames per step; we
  drop all but the first.
- **Visual-pattern reasoning**: the whole point of ARC-AGI-3 is
  that the agent reasons about the 64×64 grid. Our 8×8 mean-pool
  preprocessor destroys most of the visual information that a
  real ARC solver would need.

This is intentional — we're checking kindle's ability to learn
from a sparse-signal interactive env, not ARC-solving ability.

## Results

Tested on `ls20` (keyboard game, 4 simple actions, 7 levels to
win, baseline L1 = 22 expert actions).

| agent | steps | episodes | level events | notes |
|---|---|---|---|---|
| Random | 500 | 3 | **0** | baseline — random never reaches L1 |
| Kindle (advantage_clamp=20, no M6/M7) | 2000 | 15 | **0** | same as random; entropy pinned at 1.38 ≈ ln(4) |

Kindle's policy entropy stays at near-maximum (1.38 / 1.386)
for 2000 steps. No commitment, no level-1 reach. Same failure
mode we characterized extensively on LunarLander: the reward
circuit provides no signal that differentiates useful from
useless action sequences BEFORE the first level-completion
event, and level-completion is astronomically rare under a
random policy (L1 needs 22 expert actions from a 4-way choice,
so random completion probability is roughly 4^(−22) ≈ 10^(−13)
per attempt).

## Why this was expected

Structurally, ARC-AGI-3 is an even more extreme instance of the
task class LunarLander exemplified:

- **Event-ordered terminal success**, not steady-state homeo.
- **No dense shaping**: no approach signal, no distance-to-goal,
  no per-step reward. The *only* signal is `levels_completed`
  rising by 1 at sparse moments.
- **No natural homeostatic variable** to track. Unlike
  LunarLander (altitude, velocity, angle — well-defined
  quantities we can target), ARC-AGI-3 games have arbitrary
  internal logic per game; there is no universal homeostat.
- **Pure discovery**: no instructions, no rules. By design,
  agents must figure out the game. This explicitly tests the
  capability kindle's reward class doesn't cover.

The M6 PPO ceiling test (commit 486c129 in prior PR) showed
that even a best-in-class RL optimizer (sb3 PPO) cannot land
LunarLander under kindle's v3 homeo — because the reward's
optimum isn't landing. ARC-AGI-3 is worse: there isn't even a
hand-designed homeo shaping that approximates a ground-truth
dense reward. The intrinsic reward kindle has access to
(surprise, novelty, homeo, order) contains ~zero information
about level-completion.

## What would be needed to make kindle-class agents competitive
## on ARC-AGI-3

Multiple stacked problems, each genuine:

1. **Visual-pattern encoder**: the 64×64 grid is the agent's
   entire state. A conv-net encoder is the obvious choice;
   kindle's current encoder is a dense MLP on an 8-dim-range
   obs token vector. Redesign of the encoder stack is needed.
2. **Complex-action parameterization**: several games need
   coordinate actions. Kindle's action adapter (discrete +
   continuous vectors, `MAX_ACTION_DIM = 6`) doesn't compose
   with (x, y) coords. A structured action head would be
   needed.
3. **Goal / milestone discovery beyond kindle's current M7**:
   we showed M7's self-supervised prototype ranking fails on
   LunarLander where success is well-defined in obs-space
   features. ARC-AGI-3 success is a position in game-internal
   state space that the agent can only know by completing a
   level once — a bootstrap problem M7 doesn't solve.
4. **Stateful exploration**: ARC-AGI-3 explicitly tests
   exploration efficiency. Kindle's per-step-entropy
   exploration is near-random; meaningful progress would need
   novelty-driven, information-gathering exploration (RIDE,
   NGU, or similar).

Each of those four is a substantial research direction.
None of them fit within M6/M7 shaped patches.

## Status

ARC-AGI-3 integrated mechanically (installable, running,
stepping, recording). No learning on the tested game. The
integration is committed as a standalone example and documents
the gap precisely — useful for future project decisions about
scope.

## 2026-04-20 update: CNN encoder + curiosity tuning

Landed as first-class AgentConfig: `EncoderKind::Cnn { channels,
height, width }` (see commit 241cbec). When set, kindle builds
a small conv-net over the raw visual input (flat NCHW), folds in
the task embedding, and produces the latent for WM. Task
embedding post-fused via a small linear projection. Replay and
drift-measurement paths are guarded off in CNN mode (no visual
frames stored on transitions). Byte-parity on MLP-encoder gym
envs preserved.

### Result on ls20 (2000 steps, same harness):

  encoder  wm@500  wm@1500  steps/s  level events
  MLP      0.074   0.079    370      0
  CNN      0.016   0.010    294      0

**CNN WM loss is 7-8× lower** — the conv successfully models the
64×64 grid where the MLP-on-mean-pool couldn't. Throughput
penalty is ~20% (conv op overhead). But no level events.

### Curiosity sweep (5000 steps each, CNN encoder):

Tested several primitive-weight combinations:

  config                                        nov   surp  hom    ep_ent   events
  surprise=10, novelty=2, homeo=0.1, grid=0.5   0.02  0.34  −0.34  1.38     0
  surprise=1,  novelty=50, homeo=0.1, grid=0.5  0.02  0.34  −0.34  1.38     0
  surprise=1,  novelty=20, homeo=0.1, grid=0.05 0.02  0.34  −0.34  1.38     0

Surprise and novelty primitives are both STUCK at low values
regardless of weights. Once the CNN's WM loss settles (~0.007),
the surprise primitive (L2 norm of prediction error) is
constant at ≈ 0.34 across all actions, so the per-action
differential is ~0 — no gradient direction for the policy.
Novelty is similarly flat: the CNN compresses game states to a
tight latent region where visit counts are quickly uniform.
Reducing `grid_resolution` to 0.05 didn't help because the
problem is latent clustering, not bucket granularity.

Entropy pinned at 1.38 / 1.386 throughout.

### Why neither primitive works here

- **Surprise** stops being informative once the WM predicts well.
  Classic curiosity-death problem — by design, prediction-error
  decays as training converges. Fixable with techniques that
  decouple curiosity from WM quality (Random Network
  Distillation, predictor ensembles, etc.), but none of those
  are in kindle today.
- **Novelty** with latent-space visit counts fails when the
  encoder clusters all game states tightly. Would need raw-obs
  buckets or per-image hash keys — neither is kindle's current
  design.

### Exposed config knobs (useful regardless)

  - `AgentConfig::reward_weights` exposed via Python:
    `reward_surprise`, `reward_novelty`, `reward_homeostatic`,
    `reward_order`. Lets harnesses tune the primitives per env
    without rebuilding kindle.
  - `AgentConfig::grid_resolution` exposed via Python:
    `grid_resolution`. Lets harnesses tune novelty-bucket
    granularity per env.
  - `--encoder {mlp,cnn}` on the ARC-AGI-3 harness.

### Honest next-step options

Concretely for ARC-AGI-3 specifically:

  (a) **Random Network Distillation** curiosity — a fixed random
      target net + trained predictor; prediction error doesn't
      decay as WM converges. Adds a 4th intrinsic primitive and
      a second small session. Solid ~200 LOC.

  (b) **Raw-obs visit counts** — hash frame contents (or a
      spatial digest) into visit buckets instead of using the
      latent. Decouples novelty from encoder convergence.
      Smaller change, but the digest needs some design thought
      to avoid bucketing all game states together.

  (c) **Coordinate action head** — needed for games that use
      complex actions. Kindle's MAX_ACTION_DIM=6 currently only
      supports 6 simple actions; adding `(x, y)` parameters
      requires either a new action shape or a separate
      continuous-coord head.

None of these solves the *reward-class* problem — level
completion is still outside the primitive class. But any of (a)
or (b) would give kindle exploration strong enough to at least
probe the game's reachable state space, which random and
current-curiosity policies don't.

## 2026-04-20 update: Random Network Distillation landed

Added `AgentConfig::rnd_reward_alpha` + supporting knobs
(`rnd_feature_dim`, `rnd_hidden_dim`, `rnd_lr`) and a new CPU
module `kindle/src/rnd.rs` implementing a minimal RND primitive:
2-layer frozen random target MLP + trainable predictor MLP, both
operating on the obs TOKEN (pre-encoder, 64-dim). Predictor
trains every step by MSE against the target; intrinsic reward is
the predictor's current squared error. Unlike kindle's surprise
primitive, RND's target is independent of the WM so the signal
doesn't decay when the WM converges.

4 unit tests verify: positive MSE at init, predictor converges
to target on repeated input, novel states yield higher reward
than familiar ones, target network frozen (never drifts).

Critical design choice: RND reads the **obs token**, not the
encoder latent. An early experiment using `z_row` directly
produced MSE ≈ 0.01 within 1000 steps — the CNN encoder
clusters all game states into a tight latent region where the
predictor can fit the target quickly on any input. Switching to
the pre-encoder 64-dim obs token gave MSE ≈ 0.06 → 0.02 decay
over 4000 steps, matching the classical RND signal shape.

### Result on ls20 (5000 steps, CNN + α=50 + homeo=0.1):

  step  wm    pi      entropy  rnd_mse  level_events
  1000  0.011 +4.95   1.37     0.06     0
  3000  0.007 +1.60   1.35     0.02     0
  4000  0.007 +0.86   1.35     0.01     0

**This is the first configuration where kindle's policy commits
on an ARC-AGI-3 game.** Entropy drops from 1.39 (max for 4
actions = ln(4) = 1.386) down to 1.35 — small but real
departure from uniform. Policy loss trajectory (+4.95 → +0.86)
shows meaningful learning. No level events yet; the specific
22-action expert sequence on ls20 isn't in the direction RND
randomly pushes the policy toward.

### What RND actually delivered

Mechanism: ✅ RND signal behaves as designed (initial high
MSE, decay as predictor fits, meaningful per-step magnitude
when scaled by α ≈ 50). Entropy drops. Policy committed.

Result: entropy-floor-breaking on kindle-with-curiosity was
previously uncrossable; now crossed. But commitment direction
is RND-arbitrary, not task-aligned. Without a reward signal
pointing at level-completion (which kindle's primitives still
don't provide), committing to arbitrary directions doesn't
solve 22-action puzzle games.

### Updated honest list of what kindle needs for ARC-AGI-3

  ✅ Vision encoder (commit 241cbec): WM models grids 7-8×
     better than MLP-on-mean-pool.
  ✅ RND curiosity: breaks entropy-at-max failure mode.
  ❌ Reward signal aligned with level-completion: still missing.
     Structural. Would require external supervision or
     self-supervised milestone discovery beyond M7's prototype
     ranking.
  ❌ Coordinate action head: needed for games with ACTION5/6/7
     (x, y). Most of the 25 shipped envs need these.

RND removes a real blocker, but the remaining blockers
(reward-class, coord actions) are each real additional
research/engineering items.

## 2026-04-20 update: complex action support + first level event

Two small changes to the ARC-AGI-3 harness:

1. **Complex actions enabled** — the adapter no longer filters
   out ACTION6 (the one coordinate-parameterized action kindle
   needs to handle for several games). Random `(x, y) ∈ [0, 63]`
   coords are attached per step via `GameAction.set_data()`.
   Kindle's policy controls the discrete action ID; the coords
   are exploratory noise. Games sb26, ft09, su15, cd82 that
   require ACTION6 now run without errors (they previously
   raised `KeyError: 'x'`).

2. **Homeo signal fixed** — the original implementation
   emitted `value = −delta_levels · scale`, which produced a
   one-step NEGATIVE spike at level completion (penalizing
   progress). Replaced with `value = (win_levels −
   levels_completed) · scale` — a persistent positive-valued
   deviation that decreases in magnitude as levels are
   completed, giving a sustained "distance-to-win" signal
   instead of a one-frame anti-reward at level-ups.

### First level event on cd82

    game=CD82 id=cd82-fb555c5d available_actions=[1, 2, 3, 4, 5, 6] win_levels=6
    config: CNN encoder, RND α=50, homeo weight 0.5, levels-reward-scale 5.0

At step 2000–3000: agent completes level 1 for the first time.
Over 10–50k subsequent steps: **90% of episodes end at level 1**.
Never reaches level 2.

This is the **first level event on any ARC-AGI-3 game** under any
kindle configuration we've tested. Proves the infrastructure end
to end: vision encoder → RND-driven exploration → complex
actions → homeo pointing at distance-to-win → credit propagates
through 55-action episode → policy learns "reach L1".

But the agent then stalls. Possible reasons (each testable
separately):

- RND saturates as predictor fits the post-L1 state
  distribution. Once rnd_mse → 0, curiosity stops driving.
- The policy learned "L0→L1" behaviour generalizes poorly to
  "L1→L2" game dynamics (different actions needed).
- Max_episode_steps = 400 might cap attempts before L2 sequences
  are tried. cd82 baseline L2 = 8 expert actions; random prob
  from L1 start ≈ 6^−8 ≈ 10^−6, so over hundreds of L1-reaching
  episodes we might still miss it.

Other games (ls20, ft09, su15, sb26) still show 0 level events
in 15k steps with the same config. Possible reasons: more
restrictive action sets ([6] alone for ft09), different
specific-sequence requirements, spatial-clicking (complex
action coord specifics that random can't find).

### Updated blocker status for ARC-AGI-3

  ✅ Vision encoder (commit 241cbec): WM 7-8× better.
  ✅ RND curiosity (commit f24996b): entropy drops below max.
  ✅ Complex action plumbing (this commit): games with ACTION6
     now run; sb26/ft09/su15/cd82 playable.
  ✅ Homeo signal corrected: level-ups give positive advantage
     instead of one-step penalty.
  ✅ **First level event achieved on cd82.**
  ⚠ Multi-level progression: agent reaches L1 reliably on cd82
    but never progresses. Structural — RND saturates, policy
    overfits to single-level behaviour.
  ❌ Coordinate action policy: `(x, y)` still random, not
     policy-controlled. Games that need SPECIFIC coords for
     progress are gated by chance.
  ❌ Multi-game generalization: only cd82 shows level events.
     Most ARC-AGI-3 games still require specific action
     sequences that kindle's exploration doesn't find.

## 2026-04-20 update (2): RND reset + coord action head

Two more primitives added, matching the "continue" discussion:

1. **`Agent::reset_rnd_predictor()`** — re-initializes the RND
   predictor weights, keeping the target frozen. Re-activates
   curiosity when the state distribution shifts (e.g. on
   level-up). Exposed to Python as
   `BatchAgent.reset_rnd_predictor()`; ARC harness calls it
   automatically with `--rnd-reset-on-level` whenever
   `delta_levels > 0`. Unit test verifies: reset spikes MSE on
   a previously-familiar state, target weights stay identical.

2. **`coord` module + `CoordHead`** — CPU MLP
   `z → (μ_x, μ_y) ∈ [−1, 1]`, Gaussian-REINFORCE-trained. New
   `Agent::sample_coords(&mut rng)` returns per-lane `(x, y)`
   samples; `Agent::train_coord_head()` runs the REINFORCE
   update using the last step's reward (EMA baseline-centered,
   `α = coord_action_alpha`). ARC harness samples coords
   before `env.step`, scales `[-1, 1]` → `[0, 63]`, attaches
   via `GameAction.set_data()`. Now kindle controls both the
   discrete action ID and the `(x, y)` clickpoint — instead of
   random noise as before. 4 unit tests (forward-range,
   per-lane sample caching, positive-advantage-pulls-toward-
   sample, zero-advantage-noop) all pass.

### Results on cd82 (30k steps each, CNN + all primitives):

  config                                  events   L1 reach rate
  prior M7+RND+homeo-fix                     1      90%
  + RND reset on L1                          1      (unchanged)
  + coord head (α=1)                         1      **100%**

With all three enabled — RND, RND-reset-on-level, coord head,
fixed homeo — **kindle reaches L1 on every completed episode**
(100% reliability, up from 90% with RND alone). The stack is
stable and the L1-reaching policy is fully learned.

But **no L2 events**. Observed mechanism: RND-reset fires on
L1, MSE spikes 0.01 → 0.12, pi_loss jumps +11.55 (policy
disrupted). MSE decays to 0.01 within ~5000 steps as the
predictor re-fits the post-L1 state distribution. The agent
then stabilizes on "reach L1 and stop" because that's the
nearest local optimum in the current reward landscape. L2
requires an 8-action specific sequence FROM L1; random +
coord-policy exploration in the 300–400 step window after L1
doesn't find it.

### sb26 (coord-controlled clickpoints):

Entropy drops to 1.09 (near max for 3 actions = ln(3)=1.099),
policy commits slightly, coord head trains every step. But 0
level events in 20k steps. Without any level events to
bootstrap from, the coord head's REINFORCE advantage signal is
~uniform (all rewards comparable, baseline tracks), so the
coord policy wanders randomly. Consistent with the
"exploration without ground truth" failure mode.

### Updated infrastructure checklist

  ✅ Vision encoder (commit 241cbec)
  ✅ RND curiosity (commit f24996b)
  ✅ Complex action plumbing (commit 019f3df)
  ✅ Fixed homeo signal
  ✅ First level event on cd82
  ✅ RND predictor reset
  ✅ Coord action head (kindle-controlled clicks)
  ✅ 100% L1 reliability on cd82
  ⚠ Multi-level progression: still gated on per-game dynamics
    and requires finding level-completion sequences that our
    current exploration doesn't discover (cd82 never sees L2,
    sb26 never sees L1).

### Remaining honest options for ARC-AGI-3 progression

Each is genuine research-scale work:

  - **Staged curriculum**: train on 2-3 adjacent level
    transitions explicitly. Breaks the "local-optimum at L1"
    problem by giving the agent L2-adjacent initial states
    via save-state replay.
  - **Learned go-explore**: archive novel states, periodically
    reset the agent to archive states for deeper exploration.
    Known technique for sparse-reward envs (Ecoffet et al.
    2019).
  - **Self-supervised sub-goal discovery**: explicitly identify
    state transitions that are POTENTIALLY level-boundaries
    (e.g. cluster "mode shifts" in the CNN latent) and give
    the agent sub-rewards for triggering them. Kindle-flavored
    variant of option discovery.

The current infrastructure supports any of these — we have
clean slots for additional primitives, a working vision
encoder, and a functional RND+coord stack to build on.

### M8 v1 — delta-goal bank (state-delta-triggered), 2026-04-20

First try at the "self-supervised sub-goal discovery" direction:
`DeltaGoalBank` in `kindle/src/delta_goals.rs`. Every per-step
obs-token delta above `delta_goal_threshold` writes `obs_cur` into
a rolling bank (size `delta_goal_bank_size`); `merge_radius` drops
near-duplicates. Reward each step is `-α · min_i ‖obs − g_i‖`
(clamped), pulling the policy toward the nearest bank entry.

Note: like RND, M8 reads the 64-dim obs TOKEN, not the post-encoder
latent. First run with the latent had `dg=0` across 2000 steps on
cd82 — the CNN latent clusters tight enough that near-zero per-step
deltas wash out the threshold. The obs token carries the raw
per-frame variation.

**A/B on cd82 (10k steps each, seed 42, full CNN+RND+coord stack):**

| config           | eps | levels (mean/end) | lvl_events | dg bank |
| ---------------- | --- | ----------------- | ---------- | ------- |
| no M8 (baseline) | 99  | 0.99 / 1          | 1          | 0       |
| M8 α=0.3 th=0.3  | 99  | 0.99 / 1          | 1          | 64 full |

**Task-level outcome identical.** Bank fills to capacity within
~500 steps, then saturates.

**Mechanism**: the bank represents the agent's routine trajectory
points, not genuinely goal-worthy states. Because the agent already
visits these states, the nearest-distance reward is ~uniformly
small everywhere it goes — no gradient toward new behavior.

**Next iteration (M8 v2)**: gate recording on world-model
prediction error, not raw obs-delta. "Record `obs_cur` as a goal
only when the WM failed to predict the transition that led here."
This banks surprising transitions, not routine ones. Still
self-supervised — no task-specific priors.

### M8 v2 — surprise-gated recording, 2026-04-20

Added `delta_goal_surprise_threshold`: a step's goal-candidate is
only banked when `pred_error >= threshold` AND `|Δobs| >= delta`.
`prev_obs` is updated regardless of the gate so the next step's
delta is measured against the immediate previous obs.

**cd82 (10k steps, full CNN+RND+coord stack, A/B/C):**

| config         | dg bank | eps | levels end | lvl_events |
| -------------- | ------- | --- | ---------- | ---------- |
| no M8          | 0       | 99  | 0.99 / 1   | 1          |
| v1 (no gate)   | 64      | 99  | 0.99 / 1   | 1          |
| v2 surp th=0.5 | 62      | 99  | 0.99 / 1   | 1          |
| v2 surp th=1.0 | 0       | 99  | 0.99 / 1   | 1          |

**sb26 (10k, same stack):**

| config         | dg bank | eps | levels end | lvl_events |
| -------------- | ------- | --- | ---------- | ---------- |
| no M8          | 0       | 45  | 0.00 / 0   | 0          |
| v2 surp th=0.5 | 1       | 45  | 0.00 / 0   | 0          |

**All four configs produce identical task-level outcomes across
both games.** The surprise gate correctly filters routine steps
(62 vs 64 on cd82 at th=0.5, 1 vs 64 on sb26) but the resulting
bank still doesn't alter policy progression. At th=1.0 the gate
blocks everything (pred_error ∈ [0.27, 0.56] post-convergence, never
crosses 1.0) — equivalent to M8 off, still identical outcome.

**Interpretation.** The M8 primitive (both variants) is empirically
null on ARC-AGI-3 with the current CNN encoder and 10k-step
budgets. The cd82 L1 and sb26 L0 policies are both nearest
local optima under existing reward primitives; an additional
distance-to-bank reward doesn't change that. This is consistent
with the `project_kindle_structural_cap.md` evidence: the
bottleneck is finding the specific multi-step sequences that
trigger level events, not the reward primitive class.

**Next lever**: cross-episode memory (ARC 2 in the reward-class
roadmap). Each ARC episode currently restarts fresh; if the agent
could retain "I tried action-sequence X → result Y" across attempts
within a game, systematic exploration of untried sequences becomes
possible. This is where "reach L1 every episode but never try
what's past L1" would become "reach L1 AND carry the memory that
nothing worked there last time, so try something different now".

### xeps memory — cross-episode state-action novelty, 2026-04-20

Module `kindle/src/xeps_memory.rs`: persistent (quantized_obs, action)
counter shared across lanes, spanning episodes. Per-step intrinsic
bonus `α / sqrt(1 + count(prev_obs, prev_action))`. Obs-keyed (not
latent-keyed) because the CNN-encoded latent saturates to 1-2 grid
cells on ARC — verified before switching (xeps=6 at default grid,
xeps=12 at fine grid, vs xeps=2200+ with obs-keying).

**cd82 A/B (10k steps, full stack):**

| config                | xeps pairs | eps | levels/end | lvl_events |
| --------------------- | ---------- | --- | ---------- | ---------- |
| no xeps               | 0          | 99  | 0.99 / 1   | 1          |
| xeps α=0.1 grid=0.05  | 2202       | 99  | 0.99 / 1   | 1          |
| xeps α=0.3 grid=0.05  | 2322       | 99  | 0.99 / 1   | 1          |
| xeps α=1.0 grid=0.05  | 2322       | 99  | 0.99 / 1   | 1          |

At α=1.0, entropy held pinned at ln(6)=1.79 — xeps reward dominated
the advantage signal and the policy stayed fully stochastic. At
α=0.1, entropy behaved normally but the signal was too weak to
change task outcomes. No α produced a level event that baseline
didn't also produce.

**sb26 A/B:** xeps=3 at α=0.3 (obs is near-static — the whole 10k
trajectory collapses into one obs cell × 3 actions). Same 45/0/0.

**Interpretation.** Cross-episode state-action novelty, at both
extremes of the saturation spectrum (tight obs → few pairs;
loose obs → many pairs but α required is too high to commit), does
not move cd82's L1→L2 gap or sb26's L0→L1 gap. Reward-primitive
experiments M6, M7, M8 v1/v2, RND, RND-reset, coord-head, and xeps
have all now been tried; **ALL null on ARC-AGI-3 multi-level
progression** under 10k-step budgets.

The L1→L2 gap on cd82 needs 8 specific actions in sequence AT L1.
At the agent's ~10 steps/L1 of random-with-xeps exploration, the
probability of stumbling on a correct 8-step action sequence is
~6^-8 ≈ 6e-7 — essentially zero in 10k steps. This is not a reward
circuit problem; it's an exploration-depth / action-abstraction
problem that no per-step intrinsic reward can solve.

**Next remaining lever**: **action macros** (ARC 3 in the roadmap).
Cluster frequently-occurring action subsequences into atomic
options the policy can commit to in one decision, shortening the
effective episode depth. Or model-based planning: use kindle's
world model to roll out hypothetical action sequences and pick
one that's predicted to reach a novel state. Both are substantial
architecture work beyond reward-primitive tweaks.
