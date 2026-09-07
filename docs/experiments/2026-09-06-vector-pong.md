# Vectorized native LeVJEPA Pong

Declared 2026-09-06, before this experiment's control or training runs. This is
a fresh vector-collection protocol, not continuation of the interrupted serial
[mastery experiment](2026-09-06-levjepa-pong.md). Its stronger competence gate is
retained. No vectorized Pong mastery result is available at declaration.

## Fixed protocol

- One native F32 LeVJEPA frontend, one shared Dreamer world model/policy/learner,
  and independent ALE streams. Per-stream visual cache, belief, RNG and replay
  history remain separate; there are no independent learners or swarm services.
- Train three fresh models with seeds **0, 1, 2**, each for **200,000 aggregate
  executed actions**. A vector tick contributes N actions. No restored training
  state, forced-random coverage, demonstrations, intrinsic reward or scripted
  gameplay. Use explicit, unshaped Pong point rewards only.
- Keep the prediction-only 12M recipe: B16, T=BPTT64, row microbatch 16, replay
  ratio 256, replay capacity 100,000, world/behavior LR 4e-5, warmup 1,000 updates,
  AGC 0.3, reconstruction 0, future prediction 0.25. No B32/B64 cadence change.
  All other parameters are recorded and must match across training seeds.
- Select N once, before training, from the fixed-B16 N=2/4/8 throughput matrix
  in `runs/vector-20260906-temporal/`. Among complete, accounting-valid jobs that
  retain the GPU memory guard, choose the fewest streams within 3% of the fastest
  measured final-1,024-action window. This is a throughput choice, not selection
  by game score. Keep the same N for all three training seeds.
- Environment seeds are `(model_seed + stream * 1_000_003) mod 2^32`; live policy
  and posterior RNG seeds follow the recorded native rule, `config.seed + stream`
  modulo 2^64. Independently initialized games may die and reset naturally; no
  cloning/rewinding of a running stream is used.
- Use `ALE/Pong-v5`, ALE 0.12.1, the published wrapper, all 18 actions, repeat 4,
  zero reset no-ops/sticky actions, and a 100,000-frame episode cap. Consume each
  terminal observation before resetting its stream. Count actual emulator frames.
- Save rolling checkpoints every 20,000 aggregate actions for diagnosis/recovery.
  Only the declared 200,000-action final save is scored. A restored model lacks
  exact replay/RNG/environment/live-state continuation and cannot silently
  complete this uninterrupted protocol.
- Evaluate each final model for **75,000 frozen sampled actions in N=1**, with
  fresh visual history/belief and the corresponding environment seed. Exactly
  zero learner updates; no greedy fallback or best-checkpoint selection.
- Run a fresh seed 0, N=1, **20,000-action zero-update control** using the same
  executable/frontend/recipe except replay ratio 0. The earlier serial control
  remains contextual evidence, not this new executable's matched control.

## Unchanged mastery gate

For **every seed's final frozen evaluation**, require at least **20 complete
natural games**, mean return **at least +15**, and **at least 90% natural wins**.
A win has positive return, terminated=true and truncated=false. Mean return
and win fraction include **all completed games**, including timeouts; timeouts
are not wins. Report unfinished tails and timeouts separately. If any seed
fails, report failure and design a new experiment without lowering these gates.

`python/examples/audit_pong.py --vector-envs N` applies these gates on top of
the strict vector action/reward/replay/warmup/update ledger. It verifies fresh
training, frozen evaluation, source/frontend identity, the exact final checkpoint
hashes and finite tensors. The serial audit remains a separate supported mode.
Interrupted jobs stop the launcher; they must not quietly advance to the next
seed or be counted as complete.

## Implementation and measurement

The frontend remains `galilai-group/LeVJEPA-VideoMix-Large`, revision
`e831a0347737fcaa660b39c57d41c109de399845`, weight SHA-256
`da8bd836ce6532e1b0074ee5a6a46c65b67103f96323529ec4195be1538edc7d`, encoding
`levjepa-large-f32-chunk16-letterbox224-jl64-pool2-v1`. Its CC-BY-NC-4.0 weights
are separate from Kindle's license. This is not a DINO run.

Replay encoding and independent prediction/reward/value heads are now batched
across time. RSSM/posterior recurrence and full BPTT remain sequential. Numerical
checks and rejected candidates are recorded in the
[throughput report](2026-09-06-vectorization.md); this change is not bit-identical
to serial-head training. Meganeura remains pinned to 35a410c and Blade to b208f3b.

Artifacts go in `runs/levjepa-vector-pong-20260906/`; checkpoint targets use
`checkpoints/levjepa-vector-pong-{untrained,seed0,seed1,seed2}-20260906`.
Before launch, record selected N, source commit, native/runner/wrapper hashes
and the exact launch command. The launcher checks identities before every job.
Record construction separately, actual aggregate and per-stream simulated/wall
ratios, learner-stage time, debt, and a 1 Hz trace of GPU UUID
`GPU-6869e50d-83aa-bec7-6169-adc413f49b32` (RTX 5080). GPU-heavy jobs are serialized.
High busy percentage or fast frozen inference does not establish accelerated
playing plus training or game competence.

## Selection before launch

Source commit `cea5a93` contains the temporal batching implementation and this
predeclared protocol. The N=2/4/8 matrix completes at 7.0963/7.1422/7.3663 actions/s,
with every accounting audit valid. The declared rule selects **N=8**: N=4 narrowly
misses the 3%-below-best cutoff. Each seed therefore contributes 25,000 actions
per stream and 200,000 total; this is not 200,000 per environment. Peak observed
VRAM is 14,148 MiB, with the 10% reserve intact. No score-based selection occurred.

Native extension SHA-256:
`f663dd9317bd934f173b240041ea68b7e21e0f7d037ebaf22b9549a9ae91bb4e`.
Vector runner SHA-256:
`54ed3e3d91fa098d2257370fdf170d7a2fd26eb407d53a15ee8b63a6a4e11ee6`.
Shared wrapper SHA-256:
`ea05a28053b92bc7d462ecd56b358d0a63e4eeb73b71fcece710c2323a8474fe`.
The launcher also pins both audit implementations. It writes a fresh manifest
with exact source/launcher hashes, runs one GPU job at a time and stops after
an interrupted or invalid job instead of silently continuing the queue.

The final restore smoke check passes: the corrected executable restores the
earlier pixel canary into N=1, performs zero updates and reproduces all 64 frozen
evaluation-prefix transitions. This is compatibility evidence, not competence.

Launch command:

```sh
MEGANEURA_DEVICE_ID=0x2c02 python/.venv/bin/python \
  runs/levjepa-vector-pong-20260906/run_mastery.py --num-envs 8
```

Order: fresh zero-update control, then training and frozen evaluation for
seeds 0, 1 and 2. Passing the launch checks does not pass the mastery gate.

The queue launched **2026-09-07 00:01:17 UTC**, from clean source commit
`9ecc733a82c2d02394a15411b85d23f7b3c27da4`. Launcher SHA-256:
`cc1f917d30bb547b8b4b5e1488108d3fbfc10d929700b5f756b3730ad16f9c12`.
The manifest, GPU trace and launcher log are in the artifact directory above.
The initial control completed at **2026-09-07 00:18:32 UTC**: 20,000 actions,
79,980 actual frames, 22 natural games, mean return −20.4545, no wins/timeouts
and exactly zero updates. Execution took 968.58 s, plus 65.31 s construction.
The complete accounting/protocol audit passes. Its 20 action-count/reward/frame
intervals and all 22 episode records match the prior serial zero-update control;
all 241 model/optimizer tensors match by value. This preserves initial behavior,
not equivalence of the later vectorized learning trajectories.

Seed 0 training started at **2026-09-07 00:18:32 UTC**, with fresh counters and
the declared eight independent environment seeds. The first scheduled update
is at aggregate action 1,528, as required by replay prefill. Its first 3,072
actions match the N=8 throughput canary: all 384 vector transitions and 387
learner model reports are identical, excluding timestamps/timings; config and
implementation identities match. This verifies training startup, **not a Pong
mastery result**. The fixed final budgets and frozen gates remain unchanged.

The read-only control and training-prefix comparisons are saved as
`control-comparison.json` and `training-prefix-comparison.json` in the artifact
directory. No production binary, runner, auditor or learning setting was changed
after launch.

## Intermediate checkpoints

The run-local `watch_checkpoints.py` watches the specific live launcher and
preserves each completed save under `seed{seed}-{step:06d}-checkpoint/` in the
artifact directory. It checks the logged fingerprints and runs the CPU-only
tensor/report inspection. It never stops, restarts or restores training. The
zero-update control is a structural reference for all seeds; parameter deltas
are reported only for its matching seed 0. These model archives are diagnostic
artifacts, not exact recovery of replay, RNGs or live environment state.

All listed seed-0 prefixes pass read-only inspection: finite learner reports
with continuous counters, logged checkpoint identity, all 241 tensor
names/shapes/dtypes, finite parameters and optimizer moments, and nonnegative
second moments. World, actor, value and slow-value parameters have changed
from the matched zero-update control. Every archived file matches the completed
save's fingerprint; the rolling training target remains untouched. All listed
games are natural completions, with **no wins or timeouts** and zero reported
training debt.

| Aggregate actions | Updates | Games | Mean return | Future loss | Policy entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 4,619 | 17 | −20.7647 | 165.98 | 0.2460 |
| 40,000 | 9,619 | 42 | −20.8333 | 82.80 | 0.6299 |
| 60,000 | 14,619 | 53 | −20.5472 | 79.02 | 0.2943 |
| 80,000 | 19,619 | 62 | −20.0161 | 73.60 | 0.2984 |

Loss and entropy are means over each prefix's last 100 updates; the first 100
updates average 36,778.23 and 2.8903. These use changing training batches, not
held-out data: decreasing model loss and a concentrated policy do not establish
better control. Reports are `seed0-{step:06d}-inspection.json`; they do not
replace full-run accounting or final frozen evaluation. The watcher completed
the 40k/60k/80k inspections at 2026-09-07 01:49:13/02:34:59/03:21:30 UTC.

In the 35k–40k window, four completed games average −20.75, with five positive
points and 94 negative points observed. By 55k–60k, four completed games average
−19.25, with 14 positive points and 58 negative points. Sample-weighted replay
reward predictions improve from +0.0313/−0.8795/−0.00295 to
+0.3526/−0.8670/−0.00243 for positive/negative/zero events. This is modest
progress, not reliable gameplay.

The 65k–70k window has no completed games: its episode-return mean is undefined,
not zero. It scores 11 points and concedes 47. In the 75k–80k window, two games
finish with mean −17.5; 13 positive points and 33 negative points are observed.
Its sample-weighted positive/negative/zero replay predictions average
+0.6401/−0.8864/−0.00169. The corresponding `seed0-window-{start}-{end}.json`
reports preserve these diagnostics without treating either window as frozen
evaluation or an accepted endpoint.

The read-only `learning-context-{start}-{end}.json` reports compare exact
5k-action windows with the interrupted serial LeVJEPA seed and historical DINO
seeds. At 55k–60k, serial LeVJEPA averages −13 across two games; DINO seeds 0/1
average +14.33/−6.5 across three/two games, versus −19.25 here. Different
collection protocols and executables make this context, not a matched frontend
comparison. Do not select an earlier checkpoint or change the live recipe in
response to these intermediate scores.

Steady throughput windows each contain 10,000 aggregate actions and 2,500
updates. GPU activity and power are means from the 1 Hz trace, with
1,385/1,379/1,377/1,388 samples respectively:

| Action window | Wall seconds | Aggregate actions/s | GPU activity | Power, W |
| --- | ---: | ---: | ---: | ---: |
| 10k–20k | 1,385.86 | 7.216 | 60.22% | 138.12 |
| 30k–40k | 1,379.31 | 7.250 | 59.62% | 138.82 |
| 50k–60k | 1,377.25 | 7.261 | 60.12% | 138.56 |
| 70k–80k | 1,388.15 | 7.204 | 59.66% | 137.87 |

Peak memory remains 14,148 MiB (13.82 GiB) in all four windows. Each spends
1,069–1,075 s learning, 300–310 s handling observations and only 4.3 s stepping
environments, with zero reported training debt. Aggregate simulation speed is
about 0.48× the game clock, or 0.060× per stream. Memory is stable, but the GPU
is not saturated and super-real-time training remains unachieved. CPU subprocess
environments would not address the measured bottleneck.

## Follow-up diagnosis if final gates fail

Probe the trained recurrent belief and imagined reward/action predictions
before expanding the visual input or adding temporal machinery. The existing
held-out frozen-feature probes already compare LeVJEPA's projected 14×14 grid
with its pooled 7×7 grid: mean position R² is 0.9621 versus 0.9612, and motion
R² is 0.6178 versus 0.6338. These linear probes do not establish the cause of
weak control, but do not justify a fourfold input expansion as the default fix.
Their reports are retained in the original LeVJEPA experiment; per-frame probe
features were not cached. Do not launch another GPU probe alongside the pinned
learning queue just to regenerate those data.
