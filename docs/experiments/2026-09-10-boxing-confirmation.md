# Fresh three-seed Boxing confirmation

Started September 10 at 01:10 UTC. No confirmation result is available yet.

The R256 pilot's 162/162 frozen wins and mean +92.4877 justify testing its
stability, not assuming it. The current Meganeura package has completed exact
learning and runtime qualification. Use it now for learning; the optional
world-sync optimization affects only about 16 ms of a 345 ms learner update
and must not indefinitely postpone independent seeds or new games.

## Fixed protocol

The immutable declaration and worker are in
`runs/boxing-confirmation-20260910.hTEDcu`. Its 61 CPU checks pass and 429
content pins bind the inputs. Before starting the actual training child, the
worker independently reverified all 374 pins and raw evidence of the completed
current-backend pixel gate: exact full state, reports and traces, all ten GPU
phases and at least 3,302 MiB directly free. This is reuse of completed evidence,
not a repeated GPU gate or a measured speedup.

- Fresh roots **1009, 2017, 3019**, in that order. Each trains for **200,004
  actual actions**, then evaluates only its final checkpoint for **75,000
  sampled, unassisted frozen actions**. Six streams share one learner/policy;
  their histories and beliefs remain independent. Live RNG ranges are disjoint
  across model roots. Frozen environment base seed is 100,000 for all policies.
- LeVJEPA, 12M, N6/R256/B16/T64, full 64-step BPTT, 16-row world microbatch,
  F32, learning rate .00004, warmup 1,000, AGC .3, reconstruction 0, causal
  prediction .25. Extrinsic rewards only; no random overrides or shaping.
- Qualified native **f6a2b6ad**, source **90b4763**, Meganeura **4d45ba3a**
  (upstream **e59bd32d** plus required cache fixes). Keep the matching isolated
  Python package and runner; do not substitute main's historical auditor/API.
- Published ALE 0.12.1 wrapper: full 18 actions, repeat 4, zero sticky actions
  and reset no-ops, 100,000-frame episode cap. Keep actual emulator-frame clocks.
- Each root also gets a separately initialized untrained control, saved after
  six frozen actions/zero updates and restored for the same 75,000-action
  evaluation. Every evaluation receives a full CPU ALE replay and whole
  stream-0 movie, including losses and the partial tail.
- Serialize all GPU work. Monitor direct free/reserved memory at 4 Hz and
  require ≥2,048 MiB free with complete coverage for every native phase. Stop
  on integrity, process or safety failure; preserve partial work and declare a
  continuation explicitly. Do not restart this queue.

## Acceptance and interpretation

Every trained seed must meet the unchanged Boxing gate: **≥20 natural games,
mean ≥+50, ≥90% natural wins and no cutoffs**. Require its paired untrained
control to fail this gate and have a lower mean. Continue collecting the other
seeds if a competence gate fails: no checkpoint selection, extra budget for a
weak seed, new assistance or changed thresholds after seeing results.

| Training root | Training | Final frozen gate | Untrained control |
| --- | --- | --- | --- |
| 1009 | Running | Pending | Pending |
| 2017 | Queued | Pending | Pending |
| 3019 | Queued | Pending | Pending |

The early learning-enabled 2,004→4,008-action window executes 501 updates with
no training debt at either endpoint: 8.569 actions/s, 0.5713× aggregate real
time and 0.09521× per stream. It spends 73.97% of wall time in learning and
25.49% in observation. The corresponding 934 GPU samples average 69.26%
activity, with a 0.267 s maximum gap and at least 3,303 MiB directly free.
This agrees with the qualified runtime scale; it is neither a matched speedup
comparison nor a complete-run safety or learning-quality result. The readout
reverifies all 429 declaration pins and records a hash of the exact log prefix
in `initial-runtime-readout.json`. The earlier warmup's 32.9 actions/s has zero
updates and must not be reported as learning throughput.

The first completed seed-1009 save at **20,004 actions / 4,651 updates** is
archived as `seed1009-020004-checkpoint`. Its recorded save identity, all 241
logical/optimizer tensor entries, shapes, dtypes and finite values verify;
optimizer second moments are nonnegative and native optimizer counters match.
The 9,124 GPU samples through that completed save retain at least 3,302 MiB
directly free, with a 0.268 s maximum gap. The CPU-only inspection is
`seed1009-020004-inspection.json`, SHA-256
`2259fddca1a5523e3f94b9ab00b7bdd76182061f1672ff892d4e21759a1d2d79`.
This is an early state-health snapshot, not stable-training proof or frozen
competence. Training and final-checkpoint acceptance are unchanged; do not
substitute this archive for the declared final model.

The worker reuses the unchanged match scorer, strict complete-checkpoint auditor
and campaign checker’s match-replay binding. It does **not** invoke or bypass
the old replication-v2 runtime checker: that checker deliberately binds an older
runtime protocol. This is a separately declared Boxing confirmation, not an
all-five-game declaration or goal-completion certificate. A later campaign-wide
audit must bind all game/seed records and controls to their actual qualified
packages and protocols.

## Artifacts

Read `events.jsonl`, `seed1009-train.stdout` and `gpu.csv` for live progress.
Actual launcher PID at start was 2303115, with training child 2303477; check
the process command and start identity, not just these recorded numbers.
Per-seed `*-evaluation.score.json`, `*-result.json` and complete replays will
appear only after their corresponding phases finish. Movies will be
`seed{seed}-evaluation.mp4` and `seed{seed}-untrained-evaluation.mp4`.
`completed.json` is written only after all three roots and controls finish.

Do not call the queued evaluations successful videos. Existing successful
pilot movies remain linked from the [five-game campaign](2026-09-08-atari-five.md)
and [Freeway persistence report](2026-09-09-freeway-persistence.md).
