# Freeway persistence: conditional learning pilot

Predeclared 2026-09-09 in `runs/freeway-persistence-learning-20260909.C0GoqT`.
The 82-pin manifest SHA is
`465c5fedcab95c859dddcd57b63de065532dd8df2034d973b9c057b381cb10ac`.
The launcher started at 00:32 UTC, PID 2120848 at the last check, and is waiting
on the actual validation follower's PID/start identity. No new training or
evaluation has started, and there is no candidate learning result yet.

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

## Conditional handoff

The existing Freeway pilot finishes first, followed by the common-recording
world diagnostic and the [candidate GPU gate](2026-09-08-persistent-exploration.md).
This separate learning worker cannot bypass them. It requires the completed
queue, then rechecks actual native test outputs, four default-path trials,
complete checkpoints and non-timing learner reports, the override integration,
and raw GPU samples. All 13 gate phases need adequate coverage and at least
2,048 MiB directly free; both timing orders must pass the 95% non-regression bar.
Only then can the conditional learning declaration become executable.

The 47 CPU launcher/proof tests pass. They include real old v2 ledger parsing
through the candidate package and explicitly fabricated gate fixtures. They
do not validate the pending GPU outputs or count as learning results. The
actual prospective child package resolves to native 9cf1316b; orchestration's
old-package measurement helpers never execute the new agent or audit v3 data.

Keep all pinned source, package, declaration and test artifacts unchanged.
The process-bound wait is capped at 36 hours, each training arm at 12 hours,
each frozen phase at 2 hours, and each CPU replay at 10 minutes. Expect roughly
6.5 training hours per arm at the measured control rate, plus evaluation; this
is not a throughput claim. GPU work remains serialized with 4 Hz direct-memory
monitoring. Failures preserve artifacts and stop only this runner's child;
there is no parent interruption, automatic retry, recipe adoption or replication
launch. A task-score failure is valid data and does not stop the remaining arms.
