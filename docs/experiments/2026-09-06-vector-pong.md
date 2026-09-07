# Vectorized native LeVJEPA Pong

Declared 2026-09-06, before this experiment's control or training runs. This is
a fresh vector-collection protocol, not continuation of the interrupted serial
[mastery experiment](2026-09-06-levjepa-pong.md). Its stronger competence gate is
retained. No vectorized Pong mastery result is available at declaration.

Status 2026-09-07: seeds 0 and 1 complete training and final frozen evaluation;
both fail the predeclared mastery gate. Their frozen means/wins are +10.2778
with 18/18 wins and +0.5 with 7/12 wins, respectively. Seed 2 is training; the
three-seed campaign remains unfinished.

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

## Training checkpoints and final handoff

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
debt. There are no wins through 160k, then five by 180k and thirteen at the
declared final 200k checkpoint.

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
| 200,000 | 49,619 | 97 | −13.7423 | 52.33 | 0.6038 |

Loss and entropy are means over each prefix's last 100 updates; the first 100
updates average 36,778.23 and 2.8903. These use changing training batches, not
held-out data: decreasing model loss and a concentrated policy do not establish
better control. Reports are `seed0-{step:06d}-inspection.json`; they do not
replace full-run accounting or final frozen evaluation. The final 200k tensor
inspection completed at **2026-09-07 07:57:37 UTC**; earlier timestamps remain
in `checkpoint-watcher.log`.

Seed 0 completed its declared training budget at **2026-09-07 07:57:32 UTC**:
200,000 actions, 799,884 actual frames and 49,619 updates. The unchanged full
ledger audit passes, including the real `budget_complete` final event; this is
not the intermediate-prefix check. Its 200,105 replay insertions leave 100,000
retained records after 100,105 FIFO evictions, with zero training debt.
Execution took 27,472.16 s (7.63 hours), excluding construction.

The launcher then started the declared N=1, 75,000-action frozen sampled
evaluation. A separate read-only check confirms that its restore matches the
final save event, rolling checkpoint and archived 200k checkpoint, including
all tensor hashes, config, frontend and implementation identities. Reports are
`seed0-train.accounting.json` and `seed0-final-training-verification.json`.
The latter verifies completed training and evaluation startup only; the
completed frozen result below has its own full audit. Intermediate saves are
not evaluated and the training recipe remains unchanged.

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
| 195k–200k | 0 | undefined | 24 / 12 | +0.96728 / −0.91029 / −0.00022 |

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
The eight games completed during 180k–200k are also natural wins, with mean
return +14.125. These remain training outcomes from the same evolving policy.

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
1,385/1,379/1,377/1,388/1,380/1,381/1,383/1,386/1,386/1,376 samples respectively:

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
| 190k–200k | 1,376.46 | 7.265 | 59.06% | 137.59 |

Peak memory remains 14,148 MiB (13.82 GiB) in all ten windows. Each spends
1,069–1,075 s learning, 296–310 s handling observations and only 4.3 s stepping
environments, with zero reported training debt. Aggregate simulation speed is
about 0.48× the game clock, or 0.060× per stream. Memory is stable, but the GPU
is not saturated and super-real-time training remains unachieved. CPU subprocess
environments would not address the measured bottleneck.

Throughput remains steady after replay reaches capacity. GPU memory is exactly
14,148 MiB throughout each measured trace from 90k–100k through 190k–200k.
Endpoint learner RSS at 100k/120k/140k/160k/180k is
10.22/10.08/10.23/10.23/10.10 GiB, each with zero process swap; these are
snapshots, not a CPU high-water trace. At 180k, the host has 17.25 GiB available.

## Seed 0 final frozen result: mastery gate failed

The final evaluation completed at **2026-09-07 08:59:39 UTC**: 75,000 sampled
actions, 299,958 actual emulator frames and exactly zero learner updates.
All 18 completed games are natural wins; there are no timeouts. Mean completed
return is **+10.2778**, compared with −20.4545 for the matched zero-update control.
An unfinished tail has 5,408 actions and return −4; it is reported separately,
not counted as a completed game or win.

| Predeclared criterion | Required | Seed 0 | Result |
| --- | ---: | ---: | --- |
| Natural completed games | ≥20 | 18 | Fail |
| Mean return over all completed games | ≥+15 | +10.2778 | Fail |
| Natural wins / all completed games | ≥90% | 18/18 = 100% | Pass |

The unchanged accounting/protocol/checkpoint checks pass. The restored model
matches the declared final save, archived checkpoint, config and implementation
identities; all checkpoint tensors remain finite and fingerprint-valid.
`seed0-final-frozen-audit.json` retains the complete seed/control audits,
all episode records, criterion results and unfinished tail. This is a completed
single-seed result, not an accepted three-seed campaign. Since every seed must
pass, this recipe already fails the overall gate; the remaining runs measure
seed variation without changing budgets or selecting different checkpoints.

Frozen execution takes 3,660.12 s plus 65.66 s construction: 20.491 actions/s
and 1.3659× the game clock. This is not playing-plus-training throughput.
Seed 1 training started at **2026-09-07 08:59:39 UTC**, with fresh counters,
the declared eight environment seeds, no restore and the unchanged recipe.
Its first scheduled update occurs at action 1,528; the 2,000-action progress
record has 119 updates and zero debt. These are startup checks, not a completed
seed-1 budget or competence result.

## Seed 1 training progress

Every 20k checkpoint through the final 200k passes all 241 tensor checks: expected
names/shapes/dtypes, finite parameters and optimizer moments, and nonnegative
second moments. The seed-0 zero-update checkpoint is a structural reference
only; no cross-seed parameter-delta claim is made. Each preserved intermediate
byte-exact prefix passes the per-stream action/reward/reset, replay-credit and
game-clock checks. The full-run auditor rejects those prefixes for missing
`run_end`; they are not completed budgets or mastery. The final 200k log passes
the complete-budget audit described below, including its real `run_end`.
Reports are `seed1-{step:06d}-{inspection,prefix-accounting}.json`.

The final checkpoint was archived at **2026-09-07 16:40:31 UTC**. At 200k it
has 799,888 actual frames and 200,134 replay insertions. Replay retains its
100,000-record capacity with exactly 100,134 FIFO evictions; the ledger validates
past the capacity boundary and training debt remains zero.

| Aggregate actions | Updates | Natural games | Mean return | Wins | Last-100 future loss | Last-100 entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 4,619 | 22 | −20.8636 | 0 | 169.81 | 0.0985 |
| 40,000 | 9,619 | 47 | −20.9149 | 0 | 75.44 | 0.8758 |
| 60,000 | 14,619 | 68 | −20.8971 | 0 | 60.32 | 0.8431 |
| 80,000 | 19,619 | 90 | −20.8889 | 0 | 53.21 | 0.7462 |
| 100,000 | 24,619 | 101 | −20.7426 | 0 | 57.96 | 0.5572 |
| 120,000 | 29,619 | 107 | −20.6542 | 0 | 56.78 | 0.5670 |
| 140,000 | 34,619 | 112 | −20.4643 | 0 | 60.00 | 0.5777 |
| 160,000 | 39,619 | 118 | −20.0508 | 0 | 59.79 | 0.5191 |
| 180,000 | 44,619 | 121 | −19.7025 | 0 | 61.08 | 0.7437 |
| 200,000 | 49,619 | 126 | −19.0794 | 1 | 56.30 | 0.8618 |

There are no timeouts. The only training win is **+1 at action 193,232**, from
stream 3 after a 5,084-action game. This belongs to the evolving shared policy,
not frozen evaluation or an independent seed. The exact trailing windows are:

| Actions | Completed games | Mean return | Points scored / conceded | Sample-weighted replay prediction: positive / negative / zero |
| --- | ---: | ---: | ---: | --- |
| 15k–20k | 7 | −20.8571 | 0 / 136 | −0.00455 / −0.25571 / −0.01858 |
| 35k–40k | 7 | −20.8571 | 1 / 104 | +0.00265 / −0.88400 / −0.00306 |
| 55k–60k | 4 | −21.0000 | 1 / 122 | +0.15597 / −0.93605 / −0.00159 |
| 75k–80k | 4 | −20.5000 | 2 / 105 | +0.39083 / −0.91065 / −0.00223 |
| 95k–100k | 4 | −18.2500 | 2 / 50 | +0.51914 / −0.90889 / −0.00190 |
| 115k–120k | 2 | −19.0000 | 4 / 38 | +0.67286 / −0.90965 / −0.00159 |
| 135k–140k | 2 | −15.5000 | 20 / 27 | +0.58199 / −0.89366 / −0.00123 |
| 155k–160k | 1 | −13.0000 | 17 / 24 | +0.68514 / −0.87578 / −0.00088 |
| 175k–180k | 2 | −5.5000 | 16 / 18 | +0.79010 / −0.87698 / −0.00043 |
| 195k–200k | 0 | undefined | 17 / 20 | +0.83292 / −0.88707 / −0.00021 |

Whole-game returns may include earlier play; point counts cover only window
transitions, and replay samples are not limited to that interaction window.
The `seed1-window-*.json` reports derive from preserved byte-exact prefixes
and use the same recipe, frontend and executable as seed 0, differing in the
declared model/environment seeds. Lower prediction loss and improved negative-
reward predictions are not evidence of winning gameplay. The fixed final
budget and gate are unchanged.

Every throughput window below contains 2,500 updates with zero debt. GPU memory
stays exactly 14,148 MiB. Game-time ratios are about **0.48× aggregate and 0.060×
per stream**, not super-real-time playing plus training.

| Actions | Wall seconds | Aggregate actions/s | Mean GPU activity | Mean power (W) |
| --- | ---: | ---: | ---: | ---: |
| 10k–20k | 1,382.00 | 7.236 | 60.09% | 137.06 |
| 30k–40k | 1,387.41 | 7.208 | 60.02% | 136.49 |
| 50k–60k | 1,391.40 | 7.187 | 60.02% | 136.00 |
| 70k–80k | 1,390.54 | 7.191 | 59.59% | 135.77 |
| 90k–100k | 1,379.25 | 7.250 | 59.29% | 136.32 |
| 110k–120k | 1,385.31 | 7.219 | 59.83% | 135.86 |
| 130k–140k | 1,386.32 | 7.213 | 59.18% | 135.66 |
| 150k–160k | 1,382.23 | 7.235 | 58.89% | 135.95 |
| 170k–180k | 1,386.13 | 7.214 | 59.43% | 135.53 |
| 190k–200k | 1,391.23 | 7.188 | 59.15% | 135.24 |

The latest learning/observation/environment times are 1,082.57/302.84/4.36 s;
the measured bottleneck remains learning/perception, not environment stepping.
Reports are `seed1-throughput-*.json`, including the CPU-work overlap notes.
Two CPU-only profiler builds overlap 30k–40k; their start-to-observed-completion
intervals are retained in `runs/readback-profile-20260907/builds.json`.
The 50k–60k window overlaps CPU-only accounting checks and the separate 0.45 s
scripted Freeway fixture; the eight-title CPU preflight finished before it.
The 70k–80k window overlaps read-only monitoring/source inspection and
documentation work; independent host activity was not controlled.
The 90k–100k window includes CPU-only canary report-guard development, tests
and historical report checks. The Nsight no-op tooling preflight and its
12:18 UTC verification finished before this window.
The 110k–120k window overlaps isolated batched-DINO/control CPU builds, Clippy,
tests, documentation and historical trace reanalysis, approximately 13:10–13:34
UTC. CPU-only candidate proofs are in `runs/batched-dino-20260907/`;
no DINO candidate GPU execution occurred. The unchanged live native extension
and runners remain pinned to the original LeVJEPA protocol.
The 130k–140k window overlaps CPU-only diagnostic-worker development and tests
at approximately 14:03–14:18 UTC. The separate DINO/release and readback test
binary builds and CPU validation finished around 13:51 UTC, before this window.
The 150k–160k window overlaps CPU-only world-pretraining implementation, builds,
tests, Clippy and source-graph checks at approximately 14:30–14:57 UTC, plus
the subsequent initializer review/implementation beginning around 15:01 UTC.
The 170k–180k window overlaps the isolated initializer's debug tests, Clippy,
release build and release CPU tests through 15:45:32 UTC, plus the short
read-only host-memory sample below. No initializer hardware test ran.
The 190k–200k window contains read-only process/log monitoring and audit-source
inspection. Host-feature candidate builds and CPU tests finished by 16:10:09 UTC,
before the window. Other host activity remained uncontrolled.
These are run-health observations, not quiet-system timing comparisons or
demonstrated speed changes. No extra GPU job ran. The isolated parent/candidate
profiler canaries are built, but hardware parity and profiler overhead remain
untested until the pinned campaign finishes.

A read-only host sample at **15:48:36–15:49:21 UTC** covers 45.06 s, 82
consecutive learner reports and 45 GPU samples (58.58% mean activity). The
trainer uses 30.26 CPU-seconds across its threads and has **zero major faults**,
unchanged 5,322,724 KiB `VmSwap` and negligible system memory-pressure stalls.
The historical swap total is therefore not evidence of active paging in this
interval. It also incurs 3,072,080 minor faults. Investigate recurring host
buffer allocation/copying alongside readback waits; minor faults do not directly
measure allocated bytes, elapsed allocation cost or GPU idle time. This sample
starts after the isolated release build, but other host activity is uncontrolled.
Raw counters and thread deltas are in
`seed1-host-memory-20260907-154836.json`. No live process was attached to or changed.

## Seed 1 completed training and frozen handoff

The launcher verified the completed training budget at **2026-09-07 16:39:51
UTC**: 200,000 actions, 49,619 updates, 126 natural games, one win and no timeouts.
The complete accounting/protocol audit passes, not just the intermediate-prefix
check. Execution took 27,542.47 s (7.65 hours), plus 65.67 s construction.

The declared N=1, 75,000-action frozen sampled evaluation then started from the
final checkpoint. Its first header was written at **16:40:58 UTC**, after 66.80 s
construction. A read-only audit matches the restore to the final logged save,
rolling checkpoint and archived 200k checkpoint, including metadata and tensor
fingerprints, config, encoder, implementation and action identities. Fresh
recurrent histories and the recorded resumed RNG scheme are retained; this is
not exact restoration of the live training trajectory.

Reports are `seed1-train.accounting.json`, `seed1-200000-inspection.json` and
`seed1-final-training-verification.json`. The latter verifies completed training
and frozen startup only: it explicitly says evaluation is incomplete and mastery
has not been evaluated. The recipe, final checkpoint and frozen budget remain
unchanged. The all-seed mastery gate already fails on seed 0; finish the declared
evaluations to measure variation rather than selecting checkpoints or thresholds.

## Seed 1 final frozen result: mastery gate failed

The launcher verified the complete evaluation at **2026-09-07 17:42:04 UTC**:
75,000 sampled actions, 299,984 actual frames and zero learner updates.
All 12 completed games ended naturally, with **7 wins and mean return +0.5**.
There are no timeouts. The unfinished tail has 309 actions and return 0;
it does not count as a completed game or win. The matched zero-update control
has mean return −20.4545.

| Predeclared criterion | Required | Seed 1 | Result |
| --- | ---: | ---: | --- |
| Natural completed games | ≥20 | 12 | Fail |
| Mean return over all completed games | ≥+15 | +0.5 | Fail |
| Natural wins / all completed games | ≥90% | 7/12 = 58.33% | Fail |

The complete accounting/protocol/checkpoint audits pass. The final save,
rolling and archived checkpoints, and frozen restore identities still match,
including the same 241 tensors inspected after training. Source, encoder,
recipe and action identities match the seed-0/control references. All 240
positive and 234 negative point events reconcile with completed returns and
the unfinished tail. `seed1-final-frozen-audit.json` retains these checks and
all episode records; `seed1-eval.accounting.json` is the launcher's budget audit.
The closed evaluation log SHA-256 is
`a41bae482c8e54c97f37c162b3d9ef8de90bdb116875fcb7f3d5e5ba36dd9d66`.

Frozen execution takes 3,664.55 s plus 66.80 s construction: 20.466 actions/s
and 1.3644× the game clock. This is not playing-plus-training throughput.
The poorer final result than seed 0 reinforces the need to diagnose learned
belief and imagined reward/action predictions before scaling the recipe.
No endpoint, budget or mastery threshold changes as a result.

Seed 2 training started at **2026-09-07 17:42:04 UTC**, PID 1906792, with the
unchanged eight-stream recipe. Its header at **17:43:11 UTC** has seed 2, fresh
zero counters, no restore and the declared independent environment seeds.
The first scheduled update is step 1 at action 1,528. All 17 campaign/diagnostic
pins remain unchanged. `seed2-startup-verification.json` binds the header,
first-update log prefix and live launcher-child identity; it is not a completed
training ledger or a competence result.

## Seed 2 training progress

All completed checkpoints pass all 241 tensor checks, with no timeouts:

| Actions | Updates | Actual frames | Natural games | Natural wins | Mean completed return |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 4,619 | 79,983 | 18 | 0 | −20.6667 |
| 40,000 | 9,619 | 159,979 | 43 | 0 | −20.8372 |
| 60,000 | 14,619 | 239,963 | 57 | 0 | −20.7544 |
| 80,000 | 19,619 | 319,953 | 66 | 0 | −20.3485 |
| 100,000 | 24,619 | 399,934 | 75 | 5 | −17.9600 |

The first training win is +1 at aggregate action 86,352 on stream 6
(10,794 stream actions). The five winning returns through 100k are
+1, +1, +17, +18 and +12, in their original order.

The byte-exact prefix ledgers reconcile replay and update credit. The latest
prefix has 100,083 inserted records, 100,000 retained and 83 FIFO evictions,
with zero training debt. Earlier checkpoints through 80k have no evictions.
The unchanged full-run auditor still rejects each unfinished prefix for missing
`run_end`; these are not completed budgets or frozen evaluations. Archives and
reports use `seed2-{step:06d}-{checkpoint,inspection.json,prefix-accounting.json}`.

Each warmed window below contains 2,500 updates. Game/wall ratios use actual
emulator frames; the latest window contains 39,990.

| Actions | Wall seconds | Aggregate actions/s | Aggregate game/wall | GPU activity | Power W | Full learner-call seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k–20k | 1,376.84 | 7.263 | 0.4842× | 59.65% | 136.67 | 1,069.90 |
| 30k–40k | 1,386.38 | 7.213 | 0.4808× | 59.65% | 136.03 | 1,073.30 |
| 50k–60k | 1,382.57 | 7.233 | 0.4821× | 59.56% | 136.48 | 1,072.10 |
| 70k–80k | 1,376.67 | 7.264 | 0.4842× | 59.88% | 137.16 | 1,069.92 |
| 90k–100k | 1,383.93 | 7.226 | 0.4816× | 59.93% | 136.94 | 1,071.66 |

Per-stream game/wall ratios remain about 0.0601×–0.0605×. All measured GPU
windows keep VRAM fixed at 14,148 MiB, with maximum sample gaps of 1.018 s.
The latest window contains 1,383 GPU samples; observation processing takes
306.09 s and environment stepping 4.32 s. No candidate build or CPU-heavy
test overlaps these windows; read-only monitoring, checkpoint saves and
uncontrolled other host activity remain.
These are health windows, not quiet-system speed comparisons or calibrated
idle traces. `seed2-throughput-*.json` binds the action and GPU-log prefixes.

The latest 95k–100k training window has two completed games, both wins with
mean return +15, and 54 positive / 9 negative point events. Learner metrics,
completed-game results and sample-weighted reward predictions for every sampled
window are retained in `seed2-window-*.json`. These are training diagnostics,
not final frozen results or a passed mastery gate.

`seed2-10000-20000-runtime-budget.json` derives a conditional cost model from
the 10k–20k window: 0.42796 s per full learner call, including cleanup outside
the internal `LearnTiming` total, and 30.6938 ms of other work per aggregate
action. Holding those costs fixed projects 0.7918× at R128 and 1.1605× at R64;
these recipes have not been run. At unchanged R256, aggregate 1× needs learner
calls at or below 0.14386 s; aggregate 2× needs 0.01054 s unless other costs
also improve. The model excludes construction and changes in policy, episode
length, replay distribution, scheduling or GPU occupancy. It is an experiment-
selection aid, not a measured speedup, retained-quality result or adopted change.

## Queued diagnostic handoff

The bounded readback worker started at **2026-09-07 14:26:18 UTC**, PID 1852657
(execution session 64105), and has emitted `waiting_for_campaign_and_watcher`.
It binds the original launcher and watcher by PID, start time and script identity;
it runs no child command until both have exited. The full campaign auditor must
then validate all three completed budgets and frozen evaluations into a fresh
report. A valid failed mastery result allows diagnostics to proceed; a missing,
interrupted or inconsistent campaign does not. The original launcher exits 1
after writing a failed mastery decision, so exit status alone is insufficient.

The worker runs the profiler's three relevant ignored hardware tests separately,
then three eight-update parent/candidate canary pairs in AB, BA, AB order:
12M/B16/T64/row16, prediction-only, 48 synthetic updates in total. Each hardware
invocation must execute exactly one passing test. Each canary must finish with
eight complete finite reports, expected candidate profile counters, all 241 checkpoint
tensors with valid schemas and moments, and exact parent/candidate non-timing
reports and named tensor bytes. The untrained LeVJEPA checkpoint supplies only
the tensor schema; its perception identity and parameter values are not a
synthetic-control reference. This schema check also passes on the preserved
eight-update `head-batching-candidate-v2-20260906` checkpoint.

All binaries, auditors, worker source and reference files are pinned in
`runs/readback-hardware-20260907/manifest.json`. A selected-device memory/activity
guard screens for obvious competing use before each GPU child, with one 1 Hz
monitor for the diagnostic sequence. It is not a calibrated GPU-idle detector.
Failures retain their artifacts and stop the worker; cancellation terminates
only its own diagnostic children, never the original training processes.
Do not start another GPU workload until this worker has exited.

CPU validation passes 31 tests, including PID reuse, observed process errors,
missing campaign output, zero-test invocations, changed pins, nonfinite tensors,
negative second moments, timeout cleanup and early monitor failure. The worker
source SHA-256 is `a1789caa6eaf788326635fb7ca341791cf0642a6d15439ab355007b73ef96c05`;
its source and tests are in `runs/readback-profile-20260907/`.
These are scheduling/CPU proofs only. Hardware results remain pending. Discard
the first two updates of each canary for the subsequent warmed timing comparison;
report all three paired ratios and their spread before interpreting overhead.
This worker does not adopt the profiler, run Nsight, validate pixels or establish
gameplay quality. Those require their own serialized follow-up checks.

## Reconstructed gameplay footage

The run-local `replay_first_game.py` replays the first completed N=1 game from
the frozen evaluation and zero-update control on the CPU. It reuses the pinned
wrapper and existing recorder, checks every reward/boundary/actual-frame count
against the original action log, and never constructs or trains an agent.
The videos are **reconstructions of logged actions**, not original capture or
new evaluations. Original pixel bytes were not retained for direct comparison.

- `seed0-eval-first-game-reconstructed.mp4`: +14, natural completion,
  2,816 actions / 11,262 frames / 187.70 s of game time.
- `seed1-eval-first-game-reconstructed.mp4`: −1, natural completion,
  6,859 actions / 27,434 frames / 457.23 s of game time. This first game is
  reconstructed while the fixed evaluation budget is still running.
- `untrained-first-game-reconstructed.mp4`: −21, natural completion,
  792 actions / 3,168 frames / 52.80 s of game time.

All use nominal 60 Hz, not the original acting wall clock. Their `.mp4.json`
manifests bind the exact source-log prefix, original frontend/checkpoint header,
reconstructed raw frames, video, wrapper and recorder identities. Fifteen
run-local tests validate provenance and reject divergent outcomes, counters,
learning logs and incomplete games; the nine existing recorder tests also pass.
The first games are used without score-based selection. A single video does
not replace the declared complete frozen budget or three-seed mastery gate.

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
