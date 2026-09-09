# Freeway prefix: predictive training without reward-driven policy signal

Measured 2026-09-09 from the fixed first 72,000 actions of the still-running
plain-policy Freeway pilot. This is a post-hoc training diagnostic, not a
complete-run audit, frozen evaluation, candidate result or five-game success.
The original run and its full budget remain unchanged.

Artifacts: `runs/freeway-prefix-diagnostic-20260909.kRl6ho/`. The script reads
only the already-complete prefix, checks it again byte-for-byte, and constructs
no agent or GPU context. Its 32,383,130-byte source-prefix SHA is
`bc783d358ab1316dd1706e22c75d68e18283bb868bc98999471c73cc0615619c`;
result SHA is `c6957676a5b3a082773356e8f592caf7c9eb3011100f1a34b59439c5bd59568d`.
The source log can continue growing without changing this prefix.

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
