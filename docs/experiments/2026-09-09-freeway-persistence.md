# Freeway persistence: hold64 passes; matched comparison in progress

Predeclared 2026-09-09 in `runs/freeway-persistence-learning-20260909.C0GoqT`.
The 82-pin manifest SHA is
`465c5fedcab95c859dddcd57b63de065532dd8df2034d973b9c057b381cb10ac`.
The launcher started at 00:32 UTC, PID 2120848 at the last check. It waited for
the actual validation follower to finish, reverified the completed runtime gate,
and ran hold64 training from 07:31:02 to 13:59:12 UTC, completing its fixed
200,004 actions and 49,651 updates. The final checkpoint passes complete-state
checks and matches the frozen restore exactly. Unassisted evaluation completed
at 14:38:30 UTC and passes: 36/36 natural rounds reach 25 crossings, mean 31.0556,
with no cutoffs or updates. CPU replay and scoring completed at 14:39:08 UTC.
The matched hold1 arm started at 14:39:09 UTC, PID 2206517, process start ticks
103690355. Its result and the restored untrained control remain pending.

## Question and fixed comparison

Can persistent exploration produce a learned, unassisted Freeway policy?
The [CPU discovery study](2026-09-08-freeway-discovery.md) found no rewards in
three independent-action arms, but hundreds with held random actions. Those
are exploration results, not learned skill. This experiment now compares the
actual learner with only random-action persistence changed.

| Order / arm | Random-block probability | Hold decisions | Training actions | Frozen actions |
| --- | ---: | ---: | ---: | ---: |
| 1 / persistent | .5 | 64 | 200,004 | 75,000 |
| 2 / independent | .5 | 1 | 200,004 | 75,000 |
| 3 / untrained | None | None | 0 | 75,000 |

Both trained arms are fresh model seed 0, N6/R256/B16/T64/full BPTT64/micro16/F32,
12M Dreamer with native LeVJEPA. Keep LR 4e-5, warmup 1000, AGC .3,
reconstruction 0, future .25, extrinsic 1 and intrinsic 0. Both use the same
isolated 9cf1316b native package and `exp/persistent-exploration` source at
`9fc8108`; the old plain-policy pilot is context, not this matched hold1 arm.

Every decision still executes up to four raw Atari frames and then observes
and learns normally. Holding an action does not skip observations or updates.
Both policy and random blocks have the same duration, independent per stream;
resetting one stream does not rewind RNG or reset other histories. There are
no game-specific controls, masks, demonstrations, shaping, privileged policy
inputs, clones or rewinds. The executed-action/override ledger is mandatory.

Evaluate only the final model, sampled with fresh recurrent state and environment
seed 100,000, six streams, zero updates and no exploration overrides. The
untrained control is a fresh same-seed model saved after six frozen actions,
then independently restored for its full evaluation. Both trained arms complete
their fixed budgets even if one fails its task gate; there is no checkpoint
selection, selective extension or changed action mode. The selected order gives
the promising arm's result first; it is not a timing counterbalance.

## Acceptance and evidence

The Freeway bar is unchanged: at least 20 natural completed timed rounds,
mean at least 25 crossings, at least 90% of rounds with 25 crossings, no cutoffs.
Partial tails are ungraded. More rewards during overridden training are not
enough: success requires the unassisted frozen evaluation and complete final
checkpoint, native optimizer, config, encoder, action/reward/frame/update and
independent CPU replay checks. Each evaluation gets an unselected whole-stream-0
video, not a highlights reel. Original run pixels are not archived.

This is one paired pilot seed, not reliability. All five game criteria and the
fresh replication roots 1009/2017/3019 remain required. No success here would
repair the old failed three-seed Pong result or establish cross-game transfer.

## Completed unassisted hold64 result

The final trained policy passes the unchanged Freeway gate across all 75,000
sampled actions: 36/36 natural rounds exceed 25 crossings, mean 31.0556, no
cutoffs and zero updates. Exploration overrides are disabled and recurrent
state is freshly initialized. No checkpoint or episode selection is involved.

- [Full stream-0 video](../../runs/freeway-persistence-learning-20260909.C0GoqT/hold64-evaluation.mp4):
  all six rounds and the ungraded partial tail, 49,999 raw frames at 60 fps
  (13:53.32). This is reconstructed from the complete action ledger, not
  archived original pixels or selected highlights.
- [Task score](../../runs/freeway-persistence-learning-20260909.C0GoqT/hold64-evaluation.score.json)
  and [full CPU replay](../../runs/freeway-persistence-learning-20260909.C0GoqT/hold64-evaluation.replay.json):
  all six streams' actions, rewards, boundaries, resets and actual frame counts.
- [Independent artifact verification](../../runs/freeway-hold64-frozen-20260909.EX9X2o/result.json):
  fresh candidate-package scoring reproduces the result, all 241 final saved
  entries match the completed-training inspection, all 82 input pins are
  unchanged, and all 49,999 video frames decode with matching identity/duration.
  Result SHA-256 is
  `a3cf319bf35fd65d5b0f9b23d512f4848b723fb874ce19225e3e7327fe16953e`.

The frozen loop takes 2,288.49 seconds: 32.773 actions/s, 2.1848× aggregate
real time and 0.36413× per stream. Its separate GPU phase has 9,407 samples,
maximum gap 0.268 seconds, at least 3,413 MiB directly free and 87.42% mean
activity. Coverage and the free-memory gate pass. This is evaluation-only
throughput, not playing-plus-training speed or a matched runtime gain.

This establishes one trained policy's unassisted task pass. The matched hold1
and untrained results are still required before attributing the improvement
to persistence or selecting a recipe. Fresh seeds 1009/2017/3019 remain required
for reliability; no replication or five-game completion is claimed.

## Completed hold64 training and frozen handoff

All 96 natural training rounds receive rewards, with mean 14.1146 crossings
and no cutoffs. The completed rounds total 1,355 rewards; ungraded partial
tails add 36. These are assisted training results, not unassisted competence.
Of 49,651 updates, only two report zero absolute advantage. All reported
world/behavior learner scalars are finite; this does not establish held-out
reward calibration or that the policy learned to cross unaided.

A CPU-only [completed-training verification](../../runs/freeway-hold64-training-20260909.BK74w3/result.json)
reproduces the full action/reward/frame/update and exploration ledger, checks
the successful process exit and output identities, and verifies all 241 final
saved tensor entries. Names, shapes and dtypes are complete, values are finite,
Adam second moments are nonnegative, return normalizers are valid, and optimizer
counters are 49651/49651/0. The final save, disk checkpoint and frozen restore
identities agree exactly. All 82 experiment pins are unchanged. A fresh CPU
recheck reproduces the saved state and source/header/prefix identities.
Result SHA-256 is
`f1a84fe9e894334eebbb4fdb9956149dd28869fb9f0063e0376437de9886a1ce`.
This is not atomic live-state recovery or an independent ALE training replay.

The training loop takes 23,221.04 seconds: 8.613 actions/s, 0.5742× aggregate
real time and 0.09570× per stream, counting actual emulator-frame increments.
Full-training GPU coverage passes with 93,013 samples, maximum gap 0.268 seconds,
at least 3,302 MiB directly free, and 68.63% mean activity. This includes the
training process's construction/warmup; it is not a matched speedup, SM occupancy
measurement or the completed pilot's all-phase memory gate.

The completed frozen result above uses only this final model. Hold1 and the
untrained control retain their declared budgets. No recipe is adopted and no
fresh-seed replication is started.

## First training signal, not competence

A post-hoc fixed prefix through the 5,010-action progress record contains
11 crossing rewards, distributed 2/1/3/2/1/2 across streams. The first arrives
at action 978 on stream 5 under an explicit override. There are no completed
rounds yet. Of 902 learner updates, 900 report nonzero absolute advantage and
nonzero imagined mean reward; the inspected learner scalars are finite.
The original plain-policy pilot found no rewards in its entire 200,004 actions;
the new matched hold1 result remains pending.

This establishes reward discovery and a nonzero learning signal, not useful
reward discrimination or unassisted behavior. At the prefix endpoint the
imagined policy entropy is 2.89012 nats, near log(18), and positive/zero replay
reward prediction means are almost identical (0.00107050/0.00107025).
The final frozen evaluation remains mandatory; neither arm's budget changes.

The source is `hold64-train.jsonl` in the declared run directory. Its first
1,838,307 bytes end at that settled progress record and hash to
`c8d8d9ea81f0bf99d36a8132a7a345adba593d2f1c9f6410e5727dd3f259c474`.
A second read reproduces the prefix hash. This small CPU inspection checks
recorded action/update counters, reward-channel agreement and progress totals;
it is not a full exploration-ledger, checkpoint or independent ALE replay audit.

The 13,026-action prefix includes the first six natural rounds, ending together
at action 12,288: returns 7/3/5/5/4/7, each 2,048 decisions, with no cutoffs.
Their mean is 5.17 crossings, all below 25 and still assisted. Including later
partial tails, 34 rewards have been recorded. Of 2,906 updates, 2,904 have
nonzero reported absolute advantage; inspected learner scalars remain finite.
At this endpoint imagined-policy entropy has fallen to 1.581 nats, while the
single inspected replay batch's positive/zero prediction means are still only
0.002910/0.002822 (three positive targets). This is not held-out calibration,
live conditional-policy entropy, evidence of collapse, or an unassisted win.
Keep the world-model and behavior checks separate as learning continues.

This later fixed prefix is 5,698,571 bytes with SHA-256
`5be012232a7f8cc9ecf0ddadbef65597acdba600e601371da12812bd3d4591f8`,
again reproduced by a second read. It retains all six rounds and partial tails;
no final score, checkpoint selection or experiment setting changes.

## First periodic checkpoint health

The scheduled save at 20,004 actions / 4,651 updates completed at 08:09 UTC.
A CPU-only [inspection and archive](../../runs/freeway-first-checkpoint-20260909.mK2sXv/result.json)
checks all 241 tensor entries against the declared schema: complete names,
shapes and dtypes, finite values, nonnegative optimizer second moments, valid
return normalizers and correct world/behavior/slow-value counters (4651/4651/0).
Metadata, backend/perception identity and the saved action/update counters match
the training record. All 82 experiment pins remain unchanged.

The archived checkpoint and 9,062,133-byte log prefix match their source hashes
before and after capture. A fresh CPU process independently reproduces the
saved-state, archive and prefix checks. Result SHA-256 is
`4e30662962f0f65b5661742ffe0fff8e5bc6eee05ec974051f9db95ee7b8cf4b`.
No agent was constructed or evaluated. This is not complete-run accounting,
proof against other learning failures, or atomic live-state recovery. Keep this
first-save diagnostic; do not repeat it for every periodic checkpoint without
a new anomaly or decision it can inform. Only the declared final checkpoint is
eligible for the fixed frozen evaluation.

## Verified handoff; fixed arms in progress

The original Freeway pilot, common-recording world diagnostic and
[candidate GPU gate](2026-09-08-persistent-exploration.md) have all completed.
This separate learning worker verified the completed queue and terminal parent,
then rechecked actual native test outputs, four default-path trials,
complete checkpoints and non-timing learner reports, the override integration,
and raw GPU samples. All 13 gate phases need adequate coverage and at least
2,048 MiB directly free; both timing orders must pass the 95% non-regression bar.
All checks pass for the actual 9cf1316b bytes: exact default-path parity,
candidate/control throughput 0.999197 and 0.999774, 13 covered GPU phases and
at least 3,303 MiB directly free. The saved
[runtime proof](../../runs/freeway-persistence-learning-20260909.C0GoqT/runtime-proof.json)
has SHA-256 `ff18757852d52840c5654febad3a63c533d5edb9431ca36d6019e5988914628e`.
A fresh independent CPU recheck reproduces it exactly. Hold64 has completed
training, frozen evaluation and replay; hold1 is now training and the restored
untrained control follows without changing their budgets.

The 47 CPU launcher/proof tests pass. They include real old v2 ledger parsing
through the candidate package and explicitly fabricated gate fixtures. They
are distinct from the now-verified GPU outputs and do not count as learning
results. The actual child package resolves to native 9cf1316b; orchestration's
old-package measurement helpers never execute the new agent or audit v3 data.

Keep all pinned source, package, declaration and test artifacts unchanged.
The process-bound wait is capped at 36 hours, each training arm at 12 hours,
each frozen phase at 2 hours, and each CPU replay at 10 minutes. Expect roughly
6.5 training hours per arm at the measured control rate, plus evaluation; this
is not a throughput claim. GPU work remains serialized with 4 Hz direct-memory
monitoring. Failures preserve artifacts and stop only this runner's child;
there is no parent interruption, automatic retry, recipe adoption or replication
launch. A task-score failure is valid data and does not stop the remaining arms.
