# Phase M7 — Approach-reward primitive from discovered prototypes

## Motivation (distilled from the M6 closeout)

M6 exhausted the reward-class that can be expressed as
`α · R̂(z_{t−k:t})` added to kindle's existing four primitives.
The PPO ceiling test (commit `9a8dcce`) showed that even a
best-in-class policy-gradient optimizer cannot land LunarLander
under any configuration of kindle's homeo reward because the
reward class itself is missing an **approach-shaping**
signal. Native gym reward has continuous "this step is closer
to the goal than the last one → positive reward"; kindle's
`max(0, |val − target| − tol)` homeo function structurally
cannot emit positive per-step reward for approach.

The six kindle-compatible envs (CartPole, MountainCar, Acrobot,
Taxi, Pendulum, GridWorld) don't need approach-shaping because
their success criterion *is* a steady-state homeo condition.
LunarLander needs it because success is a specific terminal
event reachable only via an extended trajectory.

M7 is kindle's first **learned-goal** primitive. It preserves the
"no external supervision" principle by deriving its target from
the agent's own observed distribution of terminal-state outcomes,
and produces the positive per-step approach signal the reward
class has been missing.

## Design principle

- **Goals emerge from self-observation.** At every
  `env_boundary`, the agent observes the terminal latent and the
  episode's cumulative return. Over many episodes, a distribution
  of (z_end, R_ep) pairs forms. The "good" region of this
  distribution is extracted by the agent, not handed down.
- **Additive, not replacement.** The existing four primitives
  stay frozen. M7 adds a fifth reward component, gated on
  `approach_reward_alpha > 0`. Byte-identical default.
- **Stateful, not per-step.** The prototype is updated on
  episode boundaries, not every step. No per-step graph
  modification; CPU-side prototype with a simple distance read
  in `observe()`.
- **Single prototype, not a cluster set.** v1 keeps it minimal:
  one centroid representing "where high-return episodes end up
  in latent space". Multi-prototype extensions can come later if
  v1 works.

## The primitive

Maintain two state pieces on the agent:

```
terminal_buffer : VecDeque<(Vec<f32>, f32)>  // (z_end, R_ep)
                  capped at approach_buffer_size (default 100)

approach_centroid : Option<Vec<f32>>  // None until seeded
```

At each `env_boundary` for lane `i`:
1. Push `(z_end_i, R_ep_i)` onto the terminal buffer.
2. Every `approach_update_interval` episodes (across all lanes),
   recompute the centroid:
   - Sort the buffer by `R_ep` descending.
   - Take the top `approach_top_frac` fraction.
   - Centroid = mean of their `z_end` vectors.

At each step in `observe()`:
- If `approach_centroid` is `Some`:
  `r_approach = -alpha * ||z_t - centroid||` (L2, optional clamp)
- Added to the per-step reward fed to the policy.

Parameter defaults (written as AgentConfig fields):
- `approach_reward_alpha: f32 = 0.0` (disabled by default)
- `approach_buffer_size: usize = 100`
- `approach_top_frac: f32 = 0.2`
- `approach_update_interval: usize = 10` (episodes, not steps)
- `approach_distance_clamp: f32 = 10.0` (symmetric cap on
  distance before `α` multiplication)
- `approach_warmup_episodes: usize = 20` (don't emit bonus until
  buffer has this many entries)

## How this differs from M6

| aspect | M6 | M7 |
|---|---|---|
| primitive shape | `α · R̂(z_t)` | `α · (−‖z_t − centroid‖)` |
| learned part | MLP (weights) | centroid position |
| training signal | MSE on centered episode return | top-P% selection |
| per-step output | neural-net forward | L2 distance |
| failure mode we saw | intra-episode flat → constant shift, no gradient | TBD |
| addresses | target signal in a learnable primitive | *approach gradient* per-step |

M7 produces a **per-step monotonic approach gradient** toward a
prototype. M6 could not because R̂ was a state-scalar whose
per-step value depends on the learned function; M7's
per-step value is a closed-form distance whose gradient with
respect to z_t always points toward the centroid. That's exactly
what the PPO ceiling test showed the reward class was missing.

## Why this preserves kindle's principles

- **No external reward.** The episode return `R_ep` is summed
  from kindle's own intrinsic reward — no gym `reward` enters
  the system.
- **No hand-crafted goal.** The centroid is `mean(z_end)` over
  high-return terminals. What counts as "high return" is
  determined by the buffer's own distribution, not a target
  specified externally.
- **Frozen primitives remain frozen.** M7 is additive; the four
  existing primitives keep their weights and formulas unchanged.
- **Reducible to zero.** `approach_reward_alpha = 0` gives
  byte-identical pre-M7 behaviour.

## Predicted failure modes

Each has a ready diagnostic:

1. **Prototype is noise.** If the top-P% of terminals spans
   genuinely different outcome classes (e.g., some are lucky
   soft landings, some are long-surviving hovers), the centroid
   averages over both and points at neither. Diagnostic: log
   centroid drift between recomputes — if it jitters
   episode-to-episode, prototypes haven't converged.
2. **Prototype is a local attractor that doesn't include
   landing.** LunarLander's top-20% terminals under kindle homeo
   might all be hover-terminations. Agent converges on hover,
   matching M7's prototype perfectly but never landing.
   Diagnostic: classify each terminal as soft/crash/timeout in
   the Python harness; if centroid is built from mostly-crash
   terminals, the approach signal doesn't help.
3. **Distance metric isn't meaningful.** z-space L2 assumes all
   latent dims contribute equally. If the encoder's WM-trained
   latent embeds altitude at 10x the scale of angle, distance
   is dominated by altitude. Not obviously broken, but worth
   monitoring with a per-dim distance breakdown.
4. **Bootstrap failure.** Early training has no good terminals.
   The warmup threshold skips until we have ≥ 20 episodes, but
   the first valid centroid might be junk and lead the policy
   astray before later updates correct it.

## Two-track evaluation

The PPO ceiling test from M6 closeout gave us the ground-truth
rubric: PPO on v3 homeo gets 2.15%; PPO on native reward gets 31%.
M7 should be tested against both:

1. **Kindle + M7 on LunarLander** — does kindle's own stack
   (with default advantage clamp, no M6, M7 active) lift the
   soft-rate? Primary test of "does the project's optimizer
   path benefit from approach-shaping?"
2. **PPO + M7 on LunarLander** — port the prototype logic to
   the `KindleRewardWrapper` in `lunar_lander_ppo.py` (compute
   prototype in obs space since PPO has no encoder), add
   approach reward. Tests "does approach-shaping itself unlock
   landing under a battle-tested optimizer?"

Pass bar:
- **Strong pass**: PPO + M7 reaches ≥ 15% last-third soft-rate,
  and kindle + M7 shows a measurable lift over baseline (>
  5.57%).
- **Partial pass**: PPO + M7 lifts to 10-15%, suggests approach-
  shaping is necessary but not sufficient.
- **Null**: PPO + M7 stays near 5%, design is wrong; document
  and reconsider.

If PPO + M7 works but kindle + M7 doesn't, the issue is back in
kindle's optimizer stack (likely the advantage clamp, previously
diagnosed).

## What M7 does not do

- **No multi-prototype clustering.** A single centroid is a
  deliberate v1 simplification. If the buffer's top-P% bimodally
  represents two good outcomes (e.g., left-pad landings vs.
  right-pad landings), the centroid may be between them,
  literally nowhere useful. Multi-prototype extension is v2.
- **No prototype-conditional policy.** The agent gets a
  distance-to-prototype bonus; it does not see the prototype as
  an explicit input. No goal-conditioned policy.
- **No adaptive P.** `approach_top_frac = 0.2` is a fixed knob.
  More elaborate self-supervised goal discovery (e.g.,
  percentile by rarity × reward product) is future work.
- **No encoder unfreezing.** The encoder still trains on WM
  loss; M7 does not flow gradient back through z_t into the
  encoder. Distance is computed on detached latents, just like
  M6's R̂ input.

## Rollout order

1. `approach.rs` module with prototype state + update logic.
2. `AgentConfig` fields + `Agent::new` initialization.
3. `observe()` integration — terminal push at `env_boundary`,
   centroid recompute, per-step distance read into reward.
4. Diagnostics: `approach_distance`, `approach_centroid_age`,
   `approach_buffer_fill`.
5. Python binding: knobs + r_approach getter for instrumentation.
6. `lunar_lander_batch.py`: add `--approach-alpha` etc.
7. Kindle + M7 eval (100k, 4 lanes, seed 42, v3 shaping, various
   α).
8. Port M7 to `lunar_lander_ppo.py`'s wrapper (obs-space
   prototype).
9. PPO + M7 eval.
10. Write up findings.

## Results (2026-04-20)

100k / 4 lanes (kindle) or 1 env (PPO) / seed 42 / v3 shaping.

| setup | last-third soft% | note |
|---|---|---|
| kindle default | 5.57% | random (advantage clamp) |
| kindle + M7 α=0.1 | 5.47% | no change — clamp still pinning |
| kindle + M7 α=0.1 + clamp=20 | ~5% | commits but to wrong thing |
| kindle + M7 α=1.0 + clamp=20 | 4.70% | loud M7 hurts |
| PPO + v3 homeo alone (prior) | 2.15% | diverges from landing |
| PPO + v3 + M7 raw α=0.1 self-sup | 1.41% | worse than baseline |
| PPO + v3 + M7 raw α=0.1 native-rank | 0.00% | prototype correct, bonus too weak |
| PPO + v3 + M7 potential α=1.0 native-rank | 0.35% | potential-shaping alone insufficient |
| **PPO + M7 potential α=1.0 native-rank, homeo=0** | **14.84%** | approach alone DOES work |
| PPO + M7 potential α=1.0 kindle-rank, homeo=0 | 0.00% | self-supervised ranking collapses |
| PPO + native gym reward (ceiling) | 31.52% | reference |

### Three findings, ordered by importance

**1. The approach-reward PRINCIPLE works.** The isolated test —
homeo disabled, prototype ranked by native return, potential-
based shaping — reaches 14.84% last-third and is still climbing
at 100k. That's ~half the ceiling (native-gym PPO at 31.52%)
achieved purely from a prototype-discovery approach reward, no
distance-to-pad supervision, no landing detection. The primitive
does what we designed it to do.

**2. Homeo actively blocks M7.** Every config that keeps v3
homeo active gets 0-5% soft-rate regardless of α, ranking, or
shaping formula. The homeo reward's optimum (crash-fast or
hover) is so aligned with the policy-gradient basin from random
init that M7's signal, even at loud α, gets dominated.
Removing homeo (`homeo_weight=0`) is what unlocks the 14.84%
run. This is a fundamental *integration* problem: M7 as an
additive primitive can't override an antagonistic dominant
primitive.

**3. Self-supervised ranking collapses without an external
anchor.** When the prototype is ranked by kindle's own reward
(the only option that preserves kindle's "no external reward"
principle), the ranking signal is self-referential the moment
homeo is zeroed — cumulative return becomes cumulative M7
bonus, prototype drifts wherever the agent happens to cluster,
reward becomes meaningless. 0.00% last-third. The
native-rank version works; the kindle-rank version does not.

### Revised picture

M7's `approach.rs` infrastructure is correct and the principle
is validated — but the kindle-clean version (homeo + M7,
self-supervised ranking) does not unlock LunarLander. Two real
obstacles remain:

- **Homeo-vs-approach antagonism.** Additive combination doesn't
  work because homeo dominates. Multiplicative gating,
  conditional reward-mixing, or homeo-weight annealing are all
  design decisions needed to resolve this. Each breaks a
  different part of kindle's "frozen primitives" principle.
- **Ranking signal for self-supervised prototype discovery.**
  Without external reward, the agent must rank its own
  experiences by some internally-consistent criterion that
  correlates with "good terminal outcome" on the target task.
  Kindle's primitives (homeo/surprise/novelty/order) don't
  provide this on LunarLander — surprise/novelty reward chaos,
  homeo rewards fast-termination, order is weak. There is no
  kindle-intrinsic ranking that aligns with landing under the
  current primitive set.

### Implication for the project

M7 does not close the LunarLander ceiling under kindle's design
principles. The isolated "approach works" result (14.84%) is a
meaningful positive — it shows that the approach-shaping class
kindle's reward was missing CAN be added cleanly, with only
~160 lines of code, and gets us halfway to the native-reward
ceiling. But making that primitive work WITHIN kindle requires
resolving the two obstacles above, and each resolution is a
principled tradeoff.

The M7 code (both Rust and Python sides) stays in as a clean
platform. It composes with the existing M6 infrastructure and
the ablation-sweep knobs (advantage_clamp, entropy config,
watchdog) — so future work on the antagonism problem has a
ready surface to iterate on.

### What LunarLander specifically teaches about kindle's design

The chain of findings from Tier 3 through M6 through the
ablation sweep through M7 now paints a consistent picture:

- **Tier 3 (capacity, goals, credit)** fail because the reward
  signal doesn't reach the policy coherently.
- **M6 (learnable reward)** fails because the reward's training
  target on LunarLander doesn't differentiate soft from crash.
- **Ablation sweep** shows kindle-default never committed — the
  clamp clipped the signal. Fixing the clamp lets kindle learn,
  but learn the wrong thing (crash-fast basin).
- **M6 with v3 shaping** hits a pure reward-signal-content
  problem; with v5 shaping the target carries the signal but
  the bonus is flat per-step.
- **PPO ceiling test** proves the reward is the problem and
  approach-shaping is the missing piece.
- **M7** adds approach-shaping and proves it works in isolation
  (14.84% vs native 31.52%), but homeo antagonism and
  self-supervision failure prevent it from working within the
  kindle-clean design.

The LunarLander plateau is now characterized at four distinct
layers: optimizer-stack (clamp), reward landscape (homeo's
wrong-basin attraction), reward signal content (soft-vs-crash
not in return), and reward primitive class (no approach
shaping). M7 addresses the fourth; the first three remain as
genuine project-scope challenges for LunarLander-class envs.
