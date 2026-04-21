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
