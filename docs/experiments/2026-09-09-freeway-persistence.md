# Freeway exploration: completed paired pilot

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
The matched hold1 arm also completed 200,004 actions / 49,651 updates, at
21:07:19 UTC. Its unassisted evaluation finished at 21:46:39 UTC and also passes:
36/36 qualifying natural rounds, mean 29.0278, no cutoffs or updates. Replay and
scoring completed at 21:47:18 UTC. The separately restored untrained evaluation
finished at 22:27:45 UTC with zero rewards in all 36 rounds. Its replay and all
six GPU-phase checks pass. The launcher exited normally at 22:28:24 UTC.
Preserve this completed queue; do not restart it.

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

This establishes one trained policy's unassisted task pass. The completed hold1
comparison below also passes, so held exploration is not necessary for success
on this seed. Both outperform the separately restored untrained control.
Fresh seeds 1009/2017/3019 remain required for reliability; no replication or
five-game completion is claimed.

## Completed matched hold1 result

Both arms finish the same training and frozen budgets with the same package.
Only their training-time exploration block duration differs; neither frozen policy
uses exploration overrides.

| Training exploration | Rewarded training rounds / 96 | Assisted training mean | Frozen mean | Frozen qualifying rounds / 36 |
| --- | ---: | ---: | ---: | ---: |
| Hold64, probability .5 | 96 | 14.1146 | 31.0556 | 36 |
| Hold1, probability .5 | 67 | 6.6771 | 29.0278 | 36 |
| Untrained | Not trained | Not trained | 0 | 0 |

Hold1's frozen returns range from 27 to 32 crossings. All rounds terminate
naturally, with zero learner updates and fresh recurrent state. Its
[task score](../../runs/freeway-persistence-learning-20260909.C0GoqT/hold1-evaluation.score.json)
and [complete CPU replay](../../runs/freeway-persistence-learning-20260909.C0GoqT/hold1-evaluation.replay.json)
pass. A separate CPU recheck reproduces the score through a fresh candidate-
package process, validates both full ledgers and all 82 pins, and rechecks all
241 complete finite saved entries, optimizer counters and final-save/restore
identities. The four training/evaluation/replay/scoring processes exit normally
and their output hashes agree with the launcher records. The score SHA-256 is
`ef81f3dbca21e8821a086d6162a953906a28836a311a1614eb190adfce63a303`.

The [full hold1 stream-0 video](../../runs/freeway-persistence-learning-20260909.C0GoqT/hold1-evaluation.mp4)
contains all six rounds and the ungraded tail: 49,999 reconstructed raw frames
at 60 fps, 13:53.32. All frames decode successfully; video SHA-256 is
`65ea74c0b9fef836640fb48a4736eb40fe2fa09ddd21bba95281e8600b18546c`.
Original pixels were not archived; reconstruction uses the complete executed-
action ledger, not selected highlights.

Hold1 trains at 8.6126 actions/s (0.57416× aggregate real time); its frozen
evaluation reaches 32.7557 actions/s (2.18367× aggregate, 0.36394× per stream).
These fixed-order measurements are not a runtime counterbalance or speedup.
Whole-pilot GPU coverage and memory validation now pass, as detailed below.

Hold64 has a 2.0278-crossing mean advantage in this paired seed, but both arms
meet the task gate. Do not turn that difference into a seed-reliability claim
or claim held actions are required for learning. The earlier plain-policy
failure and random-discovery controls are context, not extra matched training
arms. For fresh Freeway confirmation, provisionally select hold64 for its larger
frozen score margin and broader rewarded training coverage, retaining hold1 as
the simpler successful control. This is an explicit post-pilot choice, not
automatic adoption by the launcher or proof of reliability. Do not apply held
exploration to other games without evidence; keep all fresh replication gates
and seeds unchanged.

## Completed control and whole-pilot validation

The independently restored untrained model receives zero rewards in every one
of its 36 natural rounds and all partial tails, with zero cutoffs or updates.
Its [score](../../runs/freeway-persistence-learning-20260909.C0GoqT/untrained-evaluation.score.json),
[full CPU replay](../../runs/freeway-persistence-learning-20260909.C0GoqT/untrained-evaluation.replay.json)
and [whole stream-0 video](../../runs/freeway-persistence-learning-20260909.C0GoqT/untrained-evaluation.mp4)
are preserved. All 49,999 video frames decode, at 160×210 and 60 fps. Its initial
checkpoint was saved after six frozen actions and zero updates, then restored
with fresh recurrent state; it is not an already-trained baseline.

The [completed pilot](../../runs/freeway-persistence-learning-20260909.C0GoqT/completed.json)
has SHA-256 `9e9827102f01544ea17ef359ba45d137a3c121e7ab5f25974d5b5153682f1de9`.
A separate CPU recheck confirms the terminal launcher, all 11 successful command
exits in order and their output hashes, both complete training/exploration ledgers,
all frozen ledgers and task scores, the 241 complete finite saved entries per
model, native optimizer state, full replay bindings, video identities, and all
82 unchanged pins. Recomputing every GPU window from the closed raw CSV exactly
reproduces the launcher's six coverage/memory results:

| Phase | Samples | Minimum directly free MiB | Mean GPU activity |
| --- | ---: | ---: | ---: |
| Hold64 training | 93,013 | 3,302 | 68.63% |
| Hold64 frozen | 9,407 | 3,413 | 87.42% |
| Hold1 training | 93,020 | 3,303 | 68.65% |
| Hold1 frozen | 9,418 | 3,413 | 87.20% |
| Untrained initial save | 271 | 3,415 | 0.38% |
| Untrained frozen | 9,414 | 3,413 | 87.19% |

Every phase passes the 2 GiB reserve and coverage rules; maximum sample gap is
0.269 seconds. These windows include construction. Activity is not SM occupancy,
and frozen speed is not training throughput. Both training arms remain near
0.574× aggregate real time. No fresh replication has started.

## Action ordering versus a simple UP bias

The completed policy selects an UP-labelled action variant on 68,596/75,000
decisions (91.46%). A separate, post-hoc
[CPU diagnostic](../../runs/freeway-open-loop-20260909.Mm13RB/completed.json)
compares its recorded action order with constant UP and three independently
shuffled orders. Each shuffle preserves every stream's exact action counts
across all 12,500 decisions, including the partial tail. Controls use the same
ROM, wrapper, six environment seeds and 75,000-action budget; their actions
are precomputed and do not use observations.

| Action sequence | Mean crossings | Rounds reaching 25 / 36 |
| --- | ---: | ---: |
| Recorded policy order | 31.0556 | 36 |
| Constant UP | 21.3333 | 0 |
| Shuffle 70271 | 19.8056 | 0 |
| Shuffle 81271 | 20.7778 | 0 |
| Shuffle 92271 | 20.0000 | 0 |

All rounds terminate naturally. Every arm repeats exactly in fresh environments
with reversed stream execution order, including observation and transition
hashes. The recorded-order arm also reproduces every original transition and
the archived reconstruction's stream-0 raw-frame hash. A separate CPU recheck
verifies all 17 pins, saved action bytes, shuffle counts/RNG rules, episode/tail
accounting and task scores. Five fabricated action-list checks pass; these are
not learning evidence. The completed-result SHA-256 is
`14512d415abe69e758210255a0539a15110c12e24d22ef52f108295361df6da8`.

The learned action frequencies alone do not reproduce its score under these
controls. Ordering matters, but shuffling also changes run lengths and timing:
this does not isolate visual feedback from useful open-loop temporal structure
or prove planning. The shuffles use the evaluated policy's own action marginals,
so they are not held-out policy benchmarks or additional Kindle wins. This
CPU-only work loaded no native model extension and changed no live inputs.
It does not replace hold1, the untrained control or fresh-seed replication.

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

The completed frozen result above uses only this final model. Both matched
arms and the untrained control have now finished without any budget extension
or checkpoint selection.

## Preserved early diagnostics

Hold64 first discovers a crossing at action 978 under an explicit override.
Its first six training rounds score 7/3/5/5/4/7, still assisted and below the
task threshold. Those early signals did not change the declared final budgets
or establish held-out reward calibration. The complete training log retains
all prefix observations; the final evaluations above determine competence.

The [first scheduled checkpoint inspection and archive](../../runs/freeway-first-checkpoint-20260909.mK2sXv/result.json)
at 20,004 actions / 4,651 updates preserves all 241 entries and its log prefix.
Its SHA-256 is `4e30662962f0f65b5661742ffe0fff8e5bc6eee05ec974051f9db95ee7b8cf4b`.
This is saved-state health evidence, not atomic live-state recovery or a
selected winning checkpoint. Keep the artifacts without repeating every
periodic-save inspection absent a new anomaly or decision.

## Verified prerequisites and preserved queue

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
A fresh independent CPU recheck reproduces it exactly. Both trained arms and
the restored untrained control have completed all declared phases.

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
