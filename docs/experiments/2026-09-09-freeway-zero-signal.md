# Freeway: predictive training without reward-driven policy signal

Measured 2026-09-09, first from a fixed 72,000-action prefix and then from the
completed 200,004-action plain-policy Freeway training run. These are post-hoc
training diagnostics; the subsequent frozen evaluation is recorded separately
below. Neither establishes a candidate result or five-game success. The original
experiment and its full budget remain unchanged.

The complete pilot finished at 06:06:48 UTC. Trained and untrained frozen
policies both return zero in all 36 natural rounds. Complete checkpoint, replay
and all-phase GPU memory checks pass; the declared competence gate fails.

## Original fixed-prefix diagnostic

Artifacts: `runs/freeway-prefix-diagnostic-20260909.kRl6ho/`. The script reads
only the already-complete prefix, checks it again byte-for-byte, and constructs
no agent or GPU context. Its 32,383,130-byte source-prefix SHA is
`bc783d358ab1316dd1706e22c75d68e18283bb868bc98999471c73cc0615619c`;
result SHA is `c6957676a5b3a082773356e8f592caf7c9eb3011100f1a34b59439c5bd59568d`.
The later training records do not change this prefix.

## Observations

- 72,000 real actions, 17,650 learner updates, 30 natural completed rounds.
  Every completed round returns zero; all six streams have zero total reward.
- No sampled replay target has nonzero reward. Every logged replay reward MAE,
  imagined batch-mean reward, return mean and mean absolute advantage is zero.
  The zero-reward baseline therefore matches the reward model on this data;
  accuracy on positive rewards is entirely untested.
- Every numeric scalar in the recorded learner reports is finite. This does
  not inspect parameters or rule out every learning pathology.
- Future-prediction training loss falls from a mean 11,750.6 in actions
  0–12,000 to 60.55 in 60,000–72,000. This is training loss on the collected
  distribution, not held-out or control-relevant world-model competence.
- Imagined-state policy entropy settles at 2.8903708 nats, near log(18) =
  2.8903718. The executed-action marginal entropy is 2.8902058; within-episode
  adjacent actions repeat 5.586% of the time, near 1/18 = 5.556%.
  The longest identical-action run is four decisions; the longest sequence of
  UP-labelled controls is eleven.

Imagined-policy entropy is not measured live conditional policy entropy. A
nearly uniform action histogram does not prove independence or randomness.
UP labels alone do not prove movement or a necessary duration for crossing.
An imagined batch mean of zero does not bound every individual prediction.

## Interpretation and next action

The source at `exp/atari-five` commit `da2080d` constructs policy-score targets
as discount weight × normalized advantage × the selected one-hot action.
The reported mean absolute advantage comes from those same per-row advantages.
The actor loss adds entropy regularization separately. Thus the observed zero
advantages identify missing reward-driven score-function signal in these
batches, while entropy favors diffuse actions. This is a more specific stall
than an unsupported claim of numerical collapse.

The world can fit the unrewarded observations while the actor has no useful
reward signal to favor successful behavior. This supports the already
[predeclared persistence comparison](2026-09-09-freeway-persistence.md): first
test whether new rewarded experience becomes unassisted learned skill. Do not
change the live control, its learning rate, clipping, architecture or budget
based on this prefix. No claim here establishes the cause of the separate Pong
seed variation; its common-recording world diagnostic remains pending.

## Completed training and saved-state check

Training exited successfully at 04:45:50 UTC after 200,004 actions and 49,651
updates. The original launcher accepted its declared accounting and started
the 75,000-action frozen evaluation, now complete as recorded below. The CPU
training-result artifact retains its original 04:59 UTC scope and pending flags;
it is not rewritten to include later phases. Whole-pilot results follow below.

The separate CPU-only [completed-training result](../../runs/freeway-training-complete-20260909.MnYwlJ/result.json)
checks the closed log, all 34 declaration pins and the exact final checkpoint.
It does not construct an agent or replay ALE. Its
[saved-state evidence](../../runs/freeway-training-complete-20260909.MnYwlJ/saved-state.json)
requires every schema tensor name, shape and dtype, including all optimizer
moments, rather than accepting a finite but incomplete checkpoint.

- All 96 completed rounds are natural, with zero return and no cutoffs. Every
  stream's total reward is zero, including its unfinished tail.
- Every one of the 49,651 learner reports is finite. Sampled positive/negative
  reward counts, replay reward MAE, imagined batch-mean reward, return mean and
  mean absolute/weighted absolute advantage remain exactly zero throughout.
- All 241 saved tensor entries are present and finite, and every saved second
  moment is nonnegative. World and behavior optimizer counters both equal
  49,651; the slow-value copy has no optimizer steps. Metadata and recorded
  final-checkpoint hashes agree.
- Future-prediction training loss averages 31.68 over the final 12,000 actions,
  versus 11,750.62 over the first 12,000. This remains training fit, not
  held-out predictive quality or useful behavior.

The completed run reinforces reward starvation as the actionable failure:
there was no reported reward-driven policy advantage at any update. Finite
saved state does not rule out other learning pathologies or establish stability
across seeds. Preserve the queued matched hold64/hold1 discovery experiment and
its strictly unassisted frozen evaluations; do not silently extend this control.

The training loop took 6.41 hours at 8.663 actions/s, 0.5775× aggregate real time
and 0.09625× per stream. Learning consumed 74.0% of loop time, observation 25.4%
and emulator stepping 0.5%. The full training command has 92,481 GPU samples,
a maximum 0.272-second gap, 69.15% mean activity and at least 3,303 MiB directly
free; this phase passes the declared coverage and memory gates. These are
whole-pilot observations, not matched optimization timings: CPU preparation
overlapped parts of training. Activity is not occupancy or measured idle time;
the completed all-phase audit is recorded below.

Closed training-log SHA-256:
`d18454bda31630d859401d047ba078c16d61fac5df3ded83ea841793ee803302`.
Result SHA-256:
`5e4928c496da2d7e36370e1d71296c9b0789bff1137732b8117ce30499856d42`.
The result also records its script/helper hashes and the final checkpoint's
metadata and three tensor-file hashes. The original prefix evidence is retained.

## Final frozen policy: no learned crossing skill

The declared final checkpoint completed all 75,000 sampled evaluation actions
at 05:25:07 UTC, with zero learner updates. All 36 completed rounds are natural;
every round and unfinished tail returns zero. There are no cutoffs or task
successes. It fails the unchanged Freeway gate: mean ≥25 and at least 25
crossings in ≥90% of ≥20 natural rounds. This is a learned-policy failure, not
merely an evaluation too short to count enough rounds.

The launcher completed full CPU replay and scoring before starting its untrained
control. A separate read-only rerun of the task auditor reproduces the saved
[score](../../runs/freeway-pilot-20260908.WWxHEM/evaluation.score.json) exactly,
with all 34 pins unchanged, the final checkpoint complete and finite, and
training/evaluation identities and budgets matching. The
[replay](../../runs/freeway-pilot-20260908.WWxHEM/evaluation.replay.json) checks
every reward, boundary and emulator-frame count while executing the recorded
actions across all six streams; original RGB frames were not archived.

The [whole stream-0 video](../../runs/freeway-pilot-20260908.WWxHEM/evaluation.mp4)
contains all six completed rounds and its unfinished tail: 49,999 frames,
833.317 seconds, 160×210 H.264. It is reconstructed from the verified action
ledger, not a selected success. Video SHA-256:
`c01f05f97531b82d75ece11efea6d72b2bfd58f137bbc722ff3aac98c1316f78`.
Score SHA-256:
`0bfed44777555f1c5a710baa3a581a893fe1ed6366af95f9282e40d25ba48b7d`.

Frozen collection runs at 32.788 actions/s, 2.186× aggregate and 0.3643×
per-stream real time. These are evaluation rates, not accelerated learning.

## Completed untrained comparison and handoff

The fresh zero-update model saves and restores successfully. Its complete
75,000-action frozen evaluation also returns zero in all 36 natural rounds,
with no cutoffs or updates. Its [score](../../runs/freeway-pilot-20260908.WWxHEM/untrained-evaluation.score.json)
and [full replay](../../runs/freeway-pilot-20260908.WWxHEM/untrained-evaluation.replay.json)
pass independent read-only revalidation against the raw log, complete initial
checkpoint and unchanged declaration. The
[untrained full-stream video](../../runs/freeway-pilot-20260908.WWxHEM/untrained-evaluation.mp4)
also contains 49,999 frames over 833.317 seconds, including every stream-0 round
and its unfinished tail. Neither policy demonstrates any crossing skill.

A CPU comparison of the trained and fresh same-seed behavior files finds all
11 actor parameter tensors differ (792,594 parameters). This is not itself
skill: entropy regularization can change the actor even when reward-driven
advantages remain zero. The evidence supports missing rewarded discovery, not
an inference that no parameters were updated.

The [whole-pilot result](../../runs/freeway-pilot-20260908.WWxHEM/completed.json)
was written at 06:06:48 UTC. A separate CPU rerun reproduces the baseline audit,
all four memory windows and command exit/output hashes, with all 34 pins intact.
Minimum directly free memory is 3,303 / 3,413 / 3,415 / 3,413 MiB for training,
trained evaluation, initial save and untrained evaluation. Every phase passes
the sample-density and coverage checks; the largest sample gap is 0.272 seconds.
This completes the pilot data, not Freeway competence or training-seed reliability.

Result SHA-256:
`797d53f1cd02874685f3e3b670cd62d5d3e8f6df31f378e2a0a11583d5e5902c`.
Untrained video SHA-256:
`6388d2511ff843af2ff13c64f8c0a75d09fb94c26bd6a70408c0a88ec3257ac8`.

The original parent exited, and its bound follower started common-world GPU
diagnostics at 06:07 UTC. Keep the declared runtime-validation and matched
persistent-exploration comparison next; do not restart the plain control.
No recipe is adopted and no fresh reliability replication has started.
