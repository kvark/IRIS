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
