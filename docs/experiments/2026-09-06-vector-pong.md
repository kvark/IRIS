# Vectorized native LeVJEPA Pong

Declared 2026-09-06, before this experiment's control or training runs. This is
a fresh vector-collection protocol, not continuation of the interrupted serial
[mastery experiment](2026-09-06-levjepa-pong.md). Its stronger competence gate is
retained. No vectorized Pong mastery result is available at declaration.

Status 2026-09-07: seed 0 completes training and final frozen evaluation, but
fails the predeclared mastery gate. It wins 18/18 completed frozen games with
mean return +10.2778. Seed 1 has started; remaining-seed results are unfinished.

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

The 20k checkpoint was archived and inspected at **2026-09-07 09:44:07 UTC**.
All 241 tensors have the expected names/shapes/dtypes, finite parameters and
optimizer moments, and nonnegative second moments. The seed-0 zero-update
checkpoint is a structural reference only; no cross-seed parameter-delta claim
is made. The exact-byte prefix ledger also passes, with 79,991 actual frames,
20,030 retained replay records, no eviction and zero debt. Reports are
`seed1-020000-inspection.json` and `seed1-020000-prefix-accounting.json`.
The latter deliberately remains rejected as a full run because `run_end` is
absent; no completed budget or mastery result is inferred.

The 40k checkpoint also passes all 241 tensor checks and the exact-byte prefix
ledger: 159,986 actual frames, 40,055 replay records, no eviction and zero debt.
Its reports are `seed1-040000-inspection.json` and
`seed1-040000-prefix-accounting.json`; the full-run auditor still correctly
rejects the unfinished prefix for missing `run_end`.

| Aggregate actions | Updates | Natural games | Mean return | Wins | Last-100 future loss | Last-100 entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20,000 | 4,619 | 22 | −20.8636 | 0 | 169.81 | 0.0985 |
| 40,000 | 9,619 | 47 | −20.9149 | 0 | 75.44 | 0.8758 |

There are no timeouts. In the exact 15k–20k training window, seven games finish
with mean −20.8571; all transitions in that window contain 0 points scored and
136 conceded. Sample-weighted replay reward predictions for positive/negative/
zero events are −0.00455/−0.25571/−0.01858. Whole-game returns may include earlier
play, and replay samples are not limited to the current interaction window.
`seed1-window-15000-20000.json` uses the same recipe, frontend and executable
as seed 0, differing in the declared model/environment seeds. It is early
training evidence, not a causal explanation or frozen competence result.

In the 35k–40k window, seven games finish with mean −20.8571 and no wins;
window transitions contain 1 point scored and 104 conceded. Sample-weighted
positive/negative/zero replay predictions are +0.00265/−0.88400/−0.00306.
`seed1-window-35000-40000.json` is derived from the preserved byte-exact 40k
prefix. Lower prediction loss and improved negative-reward predictions are not
yet evidence of winning gameplay. The fixed final budget and gate are unchanged.

The 10k–20k steady window contains 2,500 updates in 1,382.00 s: **7.236 aggregate
actions/s**, 0.4823× aggregate game time and about 0.0603× per stream. Across
1,382 GPU samples, activity averages 60.09%, power 137.06 W, and memory stays
exactly 14,148 MiB. Learning takes 1,072.72 s, observation handling 301.36 s and
environment stepping 4.31 s; debt is zero. This is consistent with seed 0's
7.216 actions/s and 60.22% activity at the same window, not a new speedup.
The report is `seed1-throughput-10000-20000.json`; the measured bottleneck remains
learning/perception. A post-checkpoint process snapshot is 8.35 GiB RSS with zero swap.

The 30k–40k window takes 1,387.41 s for 2,500 updates: 7.208 aggregate actions/s,
0.4805× aggregate game time and about 0.0601× per stream. Its 1,387 GPU samples
average 60.02% activity and 136.49 W, with exactly 14,148 MiB throughout.
Learning/observation/environment times are 1,078.33/301.03/4.22 s and debt is zero.
Two CPU-only profiler canary builds overlap this window, each within a 70 s
start-to-observed-completion interval. They are explicitly retained in
`seed1-throughput-30000-40000.json` and
`runs/readback-profile-20260907/builds.json`; this is not a quiet-system timing
comparison or a demonstrated speed change. No extra GPU job ran. The isolated
parent/candidate canaries are built, but hardware parity and profiler overhead
remain untested until the pinned campaign finishes.

## Reconstructed gameplay footage

The run-local `replay_first_game.py` replays the first completed N=1 game from
the frozen evaluation and zero-update control on the CPU. It reuses the pinned
wrapper and existing recorder, checks every reward/boundary/actual-frame count
against the original action log, and never constructs or trains an agent.
The videos are **reconstructions of logged actions**, not original capture or
new evaluations. Original pixel bytes were not retained for direct comparison.

- `seed0-eval-first-game-reconstructed.mp4`: +14, natural completion,
  2,816 actions / 11,262 frames / 187.70 s of game time.
- `untrained-first-game-reconstructed.mp4`: −21, natural completion,
  792 actions / 3,168 frames / 52.80 s of game time.

Both use nominal 60 Hz, not the original acting wall clock. Their `.mp4.json`
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
