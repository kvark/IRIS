# Freeway: predictive training without reward-driven policy signal

Measured 2026-09-09, first from a fixed 72,000-action prefix and then from the
completed 200,004-action plain-policy Freeway training run. These are post-hoc
training diagnostics, not frozen evaluations, candidate results or five-game
success. The original experiment and its full budget remain unchanged.

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
the 75,000-action frozen evaluation. That evaluation and the separately restored
untrained comparison are still pending; the whole pilot is not complete.

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
the whole pilot's GPU audit must still cover its remaining phases.

Closed training-log SHA-256:
`d18454bda31630d859401d047ba078c16d61fac5df3ded83ea841793ee803302`.
Result SHA-256:
`5e4928c496da2d7e36370e1d71296c9b0789bff1137732b8117ce30499856d42`.
The result also records its script/helper hashes and the final checkpoint's
metadata and three tensor-file hashes. The original prefix evidence is retained.
