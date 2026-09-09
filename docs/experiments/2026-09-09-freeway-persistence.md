# Freeway persistence: learning pilot running

Predeclared 2026-09-09 in `runs/freeway-persistence-learning-20260909.C0GoqT`.
The 82-pin manifest SHA is
`465c5fedcab95c859dddcd57b63de065532dd8df2034d973b9c057b381cb10ac`.
The launcher started at 00:32 UTC, PID 2120848 at the last check. It waited for
the actual validation follower to finish, reverified the completed runtime gate,
and started hold64 training at 07:31:02 UTC. The training child is PID 2164617,
process start ticks 101121662. No final learning or frozen-policy result exists yet.

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
A fresh independent CPU recheck reproduces it exactly. Hold64 is now training;
hold1 and the restored untrained control follow without changing their budgets.

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
