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

The additional run-local `audit_checkpoint_prefix.py` copies an exact byte
prefix through both the checkpoint and progress records at a chosen intermediate
boundary. It uses the unchanged, hash-pinned vector ledger checker, requiring
all record-level checks to reach its specific `missing run_end` rejection.
Other errors propagate; no final event is synthesized, no checker is patched,
and the result explicitly says `budget_complete: false`. A closing checkpoint
or progress record verifies that the last vector round is settled. This is
prefix-ledger evidence, not full-run or mastery acceptance.

The following prefix ledgers pass with zero debt. Each snapshot is bound to
its archived checkpoint fingerprints; reports are
`seed0-{step:06d}-prefix-accounting.json`. This verifies logged accounting,
not the contents of sampled replay tensors or a completed training budget.

| Aggregate actions | Actual frames | Replay insertions | Retained records | FIFO evictions |
| ---: | ---: | ---: | ---: | ---: |
| 80,000 | 319,946 | 80,070 | 80,070 | 0 |
| 100,000 | 399,942 | 100,076 | 100,000 | 76 |
| 120,000 | 479,933 | 120,080 | 100,000 | 20,080 |
| 140,000 | 559,920 | 140,087 | 100,000 | 40,087 |
| 160,000 | 639,913 | 160,092 | 100,000 | 60,092 |
| 180,000 | 719,904 | 180,097 | 100,000 | 80,097 |

Nineteen CPU tests cover exact copying, checkpoint/progress ordering, corrupted
ledgers, incomplete rounds and FIFO eviction at three small capacities.
Production runners, native code and auditors remain unchanged.

All listed seed-0 prefixes pass read-only inspection: finite learner reports
with continuous counters, logged checkpoint identity, all 241 tensor
names/shapes/dtypes, finite parameters and optimizer moments, and nonnegative
second moments. World, actor, value and slow-value parameters have changed
from the matched zero-update control. Every archived file matches the completed
save's fingerprint; the rolling training target remains untouched. All listed
games are natural completions, with **no timeouts** and zero reported training
debt. There are no wins through 160k, then five by 180k.

| Aggregate actions | Updates | Games | Mean return | Future loss | Policy entropy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 4,619 | 17 | −20.7647 | 165.98 | 0.2460 |
| 40,000 | 9,619 | 42 | −20.8333 | 82.80 | 0.6299 |
| 60,000 | 14,619 | 53 | −20.5472 | 79.02 | 0.2943 |
| 80,000 | 19,619 | 62 | −20.0161 | 73.60 | 0.2984 |
| 100,000 | 24,619 | 68 | −19.3676 | 68.43 | 0.3365 |
| 120,000 | 29,619 | 72 | −19.2083 | 67.34 | 0.3734 |
| 140,000 | 34,619 | 79 | −18.4684 | 66.05 | 0.8925 |
| 160,000 | 39,619 | 84 | −17.9048 | 62.69 | 0.7390 |
| 180,000 | 44,619 | 89 | −16.2472 | 57.32 | 0.6293 |

Loss and entropy are means over each prefix's last 100 updates; the first 100
updates average 36,778.23 and 2.8903. These use changing training batches, not
held-out data: decreasing model loss and a concentrated policy do not establish
better control. Reports are `seed0-{step:06d}-inspection.json`; they do not
replace full-run accounting or final frozen evaluation. The latest 180k
inspection completed at **2026-09-07 07:11:50 UTC**; earlier timestamps remain
in `checkpoint-watcher.log`.

Exact 5k-action training windows show learning progress, not mastery.
Point counts cover all transitions in each window; whole-game returns cover
only games that finish there and may include play before the window. With no
completed games, the return mean is undefined, not zero. Reward predictions
below are sample-weighted replay means for positive/negative/zero events.

| Action window | Games | Mean return | Points scored / conceded | Replay predictions + / − / 0 |
| --- | ---: | ---: | ---: | --- |
| 15k–20k | 2 | −20.50 | 1 / 132 | −0.01408 / −0.29602 / −0.01647 |
| 25k–30k | 8 | −21.00 | 0 / 136 | −0.01162 / −0.67741 / −0.00801 |
| 35k–40k | 4 | −20.75 | 5 / 94 | +0.03132 / −0.87953 / −0.00295 |
| 55k–60k | 4 | −19.25 | 14 / 58 | +0.35263 / −0.86698 / −0.00243 |
| 65k–70k | 0 | undefined | 11 / 47 | +0.49399 / −0.87213 / −0.00210 |
| 75k–80k | 2 | −17.50 | 13 / 33 | +0.64014 / −0.88638 / −0.00169 |
| 95k–100k | 3 | −12.33 | 7 / 35 | +0.75056 / −0.89997 / −0.00126 |
| 115k–120k | 0 | undefined | 12 / 26 | +0.80539 / −0.88602 / −0.00102 |
| 135k–140k | 1 | −9.00 | 19 / 23 | +0.85276 / −0.84386 / −0.00091 |
| 155k–160k | 1 | −8.00 | 41 / 8 | +0.89375 / −0.87223 / −0.00053 |
| 175k–180k | 0 | undefined | 44 / 4 | +0.94217 / −0.89381 / −0.00033 |

These are the vector rows of the saved `learning-context-{start}-{end}.json`
reports and the later `seed0-window-{start}-{end}.json` reports, not frozen
evaluations or accepted endpoints. The 155k–160k point balance is strongly
positive, but its one completed game still loses: its earlier play precedes
this window, while points also include unfinished games. No game has been won
through 160k. Keep the declared final checkpoint and frozen gate unchanged.

All five games completed during 160k–180k are natural wins: returns
**+6, +10, +5, +17 and +20**, mean **+11.6**. The first is at action 160,344.
They come from five collection streams sharing one evolving learner, not five
independent training seeds. The 175k–180k point balance remains positive but
contains no complete games. Neither result passes the final frozen mastery gate.

The read-only `learning-context-{start}-{end}.json` reports compare exact
5k-action windows with the interrupted serial LeVJEPA seed and historical DINO
seeds. At 55k–60k, serial LeVJEPA averages −13 across two games; DINO seeds 0/1
average +14.33/−6.5 across three/two games, versus −19.25 here. Different
collection protocols and executables make this context, not a matched frontend
comparison. Do not select an earlier checkpoint or change the live recipe in
response to these intermediate scores.

At 100k inspection, the run-local summary helper was found to mislabel vector
reports as DINO: it read only top-level perception metadata and silently
defaulted to DINO, ignoring `model_provenance.perception`. This reporting bug
is corrected; all seven then-existing context/window reports take their encoder
label from the original run headers. Their numerical summaries are unchanged
and regenerate exactly. Sixteen CPU tests cover current/legacy schemas, missing
and conflicting identity, the four actual source headers and saved reports.
The training logs, checkpoint identities and experiment configuration have
always recorded LeVJEPA for the vector run; none were altered.

Steady throughput windows each contain 10,000 aggregate actions and 2,500
updates. GPU activity and power are means from the 1 Hz trace, with
1,385/1,379/1,377/1,388/1,380/1,381/1,383/1,386/1,386 samples respectively:

| Action window | Wall seconds | Aggregate actions/s | GPU activity | Power, W |
| --- | ---: | ---: | ---: | ---: |
| 10k–20k | 1,385.86 | 7.216 | 60.22% | 138.12 |
| 30k–40k | 1,379.31 | 7.250 | 59.62% | 138.82 |
| 50k–60k | 1,377.25 | 7.261 | 60.12% | 138.56 |
| 70k–80k | 1,388.15 | 7.204 | 59.66% | 137.87 |
| 90k–100k | 1,380.52 | 7.244 | 59.69% | 138.01 |
| 110k–120k | 1,381.38 | 7.239 | 59.66% | 137.80 |
| 130k–140k | 1,382.93 | 7.231 | 59.45% | 137.60 |
| 150k–160k | 1,386.43 | 7.213 | 59.88% | 137.24 |
| 170k–180k | 1,386.57 | 7.212 | 59.80% | 136.91 |

Peak memory remains 14,148 MiB (13.82 GiB) in all nine windows. Each spends
1,069–1,075 s learning, 300–310 s handling observations and only 4.3 s stepping
environments, with zero reported training debt. Aggregate simulation speed is
about 0.48× the game clock, or 0.060× per stream. Memory is stable, but the GPU
is not saturated and super-real-time training remains unachieved. CPU subprocess
environments would not address the measured bottleneck.

Throughput remains steady after replay reaches capacity. GPU memory is exactly
14,148 MiB throughout each measured trace from 90k–100k through 170k–180k.
Endpoint learner RSS at 100k/120k/140k/160k/180k is
10.22/10.08/10.23/10.23/10.10 GiB, each with zero process swap; these are
snapshots, not a CPU high-water trace. At 180k, the host has 17.25 GiB available.

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
