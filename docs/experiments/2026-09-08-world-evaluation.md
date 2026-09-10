# Frozen world forecasts versus the recorded Pong matches

Completed 2026-09-08. This evaluates the world model separately from gameplay,
without retraining or changing the failed three-seed mastery decision.

## Inspect the result

Start with the [common-input video report](../../runs/common-world-report-20260909.O7nqqe/report.html):
all three models predict the same three recorded matches. The earlier
[own-policy report](../../runs/world-evaluation-20260908.Xzx3pN/report.html)
also includes five- and fifteen-action forecasts.
It plots real point rewards against one-step forecasts and after-frame estimates,
lists every scored/conceded point, and seeks the existing video from each point's
time. Full per-target JSONL traces and all horizon metrics are linked alongside.
These git-ignored artifacts are not a hosted dashboard.

All three final 200k-action checkpoints replay their archived first completed
frozen game. Sampled actions, rewards, boundaries and actual emulator frames
match exactly; none is forced to follow an unexpected policy action. The games
return +14 / −1 / +20 for seeds 0 / 1 / 2, over 2,816 / 6,859 / 1,713 actions.
There are zero learner updates. These are the same first-game videos already
published locally, not newly selected successes or new mastery evaluations.

## Protocol and implementation

The pre-hardware declaration is
`runs/meganeura-refresh-20260908.ltOGRe/declaration.md`; execution, pins,
source-prefix hashes, summaries and report generation are in
`runs/world-evaluation-20260908.Xzx3pN/`. The main implementation is
[`probe_atari_dynamics.py`](../../python/examples/probe_atari_dynamics.py),
introduced in `0198b0a`. It extends the existing forced-random probe with exact
frozen-match replay, baselines, reward-event statistics and prediction traces.
The Python suite passes 253 tests, including fail-fast recording validation,
causal forecast ordering, unchanged learning counters and action-mismatch refusal.

Use the recorded native extension (`f663dd93…`), encoder (`da8bd836…`) and final
checkpoint hashes. The default historical extension remains intact. This
diagnostic deliberately does **not** use the backend-refresh candidate: otherwise
new backend arithmetic could confound the trained-model comparison. Restore still
validates backend identity; do not edit checkpoint metadata to bypass it.

Forecast the next feature, reward and continuation at every action. Every fifth
origin additionally rolls forward up to 15 actions without consuming intermediate
observations. The future controls are the recorded actions, known retrospectively;
they are not a plan chosen online by the model. A second rollout uses independent
unrelated controls. The diagnostic uses a reproducible latent sample at each
step, not a predictive ensemble, and leaves the live action RNG untouched.

Targets are the frozen LeVJEPA features and actual rewards. They are not future
RGB reconstructions. Persistence repeats the origin feature. The zero-reward
baseline exposes errors hidden by sparse events. Keep forecasts separate from
posterior reward inference, which has already seen the target frame. Continuation
targets are zero on termination, otherwise f32(1−1/333); truncation is not a
termination target. Reset targets are scored before pending rollouts are cleared.

## Feature prediction

Lower ratios are better. Each seed sees its **own** policy's trajectory, not a
common evaluation distribution; the rows cannot establish a causal cross-seed
ranking.

| Seed | H1 feature MSE | H1 / persistence | H1 / unrelated controls | H5 / unrelated controls | H15 / unrelated controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.01635 | 0.0337 | 0.775 | 0.307 | 0.273 |
| 1 | 0.02266 | 0.0469 | 0.817 | 0.408 | 0.427 |
| 2 | 0.01105 | 0.0224 | 0.638 | 0.203 | 0.143 |

All models use action-conditioned dynamics usefully on these matches. This is
not proof that every counterfactual action is modeled correctly: unrelated
controls are compared with the outcome of the actual controls, not their own
unobserved outcomes.

A post-hoc sanity check separates one-step targets at 16-arrival visual-cache
boundaries from other targets. Persistence is particularly poor at these resets.
Excluding them, model/persistence ratios remain 0.094 / 0.133 / 0.062 and
model/unrelated-control ratios remain 0.774 / 0.815 / 0.634. The useful result
does not disappear, but the headline persistence improvement overstates ordinary
within-chunk prediction. RSSM belief does not reset with the visual cache.

## Sparse reward reliability

These are event-conditioned absolute errors in the game's ±1 reward units,
not probabilities or a distributional calibration test.

| Seed | Positive / negative points | H1 positive MAE | H1 negative MAE | After-frame positive / negative MAE | Overall H1 MAE / zero baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 21 / 7 | 0.068 | 0.254 | 0.043 / 0.281 | 0.00171 / 0.00994 |
| 1 | 20 / 21 | 0.388 | 0.504 | 0.351 / 0.430 | 0.00457 / 0.00598 |
| 2 | 21 / 1 | 0.051 | 0.011 | 0.049 / 0.039 | 0.00142 / 0.01284 |

Seed 1 has useful event ranking (one-step event-vs-zero ROC-AUC 0.9973), but
weaker reward magnitude estimates. Its mean positive prediction is +0.612 and
mean negative prediction −0.496; even after seeing the frame they reach only
+0.649 and −0.570. That makes a purely missing-temporal-information explanation
less compelling, without establishing that the reward head causes weak play.
Seed 2's single negative event is insufficient to establish reliable loss
prediction across games.

Long-horizon results need their class counts:

- H5/H15 retain 563/561, 1,371/1,369 and 342/340 targets respectively.
- Seed 0 has four negative and **no positive** points at both horizons. Its H5
  overall reward MAE is 0.00779, slightly worse than zero's 0.00710. Do not
  report its absent positive class as perfect.
- Seed 1 has seven positive and three negative points. H5/H15 positive MAE is
  0.437/0.558 and negative MAE 0.982/0.978: conceded rewards are barely
  anticipated in magnitude on this small subset.
- Seed 2 has four positive and **no negative** points. Positive MAE is
  0.00223/0.00144; this small, phase-sampled subset cannot stand for all points.

Fixed stride can miss entire sparse classes. Preserve this declared diagnostic;
use predeclared complete or phase-balanced origins and multiple held-out games
for a stronger follow-up, not a favorable stride chosen after inspecting scores.
AUC is a ranking measure, not evidence of accurate reward magnitude.

Continuation is not established: each H1 match contains one true terminal,
and H5/H15 contain none. H1 mean squared errors are approximately
0.000353 / 0.000142 / 0.000628, against always-continue errors
0.000353 / 0.000145 / 0.000580. Tiny nonterminal error is not terminal competence.

## Decision

Keep separate world-model evaluation in the workflow. The model is not simply
blind to Pong motion or actions; the weak seed's sparse reward reliability is a
more specific follow-up than expanding perception speculatively. Next compare
reward/value predictions and policy action use on a fixed common held-out set,
including meaningful event counts and the complete observed returns. Keep
policy quality and model quality as distinct gates. Longer all-seed training or
changed loss/replay settings remain new experiments, not repairs to the failed
200k-action mastery result. No swarm or actor/learner separation is required.

## Common-recording diagnostic: complete

The isolated `exp/common-world-probe` candidate at `a425b29` adds explicit
recorded-action conditioning while retaining strict own-policy replay as a
separate mode. All 282 Python CPU tests pass. The declaration in
`runs/common-world-20260908.7gWHsJ/manifest.json` pins 35 inputs; CPU reconstruction
verifies all 11,388 transitions, including 62 positive and 29 negative points.

All three final models see the same three first matches. Every one-step target
is scored, with exact reproduction of the original three H1 diagonals before
cross-model comparison and identical initial/target RGB and feature hashes.
Unmasked action probabilities/value are recorded before forcing controls,
without calling the forced trajectory the evaluated model's own play.
Cross-policy logged returns are not unbiased critic targets. These recordings
were held out from training, but have already been inspected; this is a
diagnostic dataset, not untouched confirmation.

The CPU result-binding tests also pass using explicitly fabricated new outputs
against the real source prefixes and original traces. They do not validate GPU
predictions. The bound follower ran all nine serialized GPU combinations from
06:07 through 06:47 UTC on September 9, after the entire Freeway pilot completed
and its parent exited. All three same-model diagonals reproduce their original
one-step forecasts exactly; all six cross-model runs pass common RGB/feature
input checks. Each model sees the same 11,388 transitions, for 34,164 forecasts
in total, with zero learner updates. All nine GPU phases pass coverage and
memory checks, with at least 7,469 MiB directly free and a maximum 0.268-second
sample gap. These are diagnostic trajectories, not new policy rollouts or wins.

The [completed data manifest](../../runs/common-world-20260908.7gWHsJ/completed.json)
records every result/trace hash and memory window. Its SHA-256 is
`8a3f25ad6c92be9bb563e1bcc741ceaa21aa73d1a6500e15784acd2078eca48a`.
The controller exited successfully, and the bound follower started the declared
exploration runtime gate at 06:47:48 UTC. Preserve the original executable,
35 input pins and all results; do not restart this completed diagnostic.

The [common-input video report](../../runs/common-world-report-20260909.O7nqqe/report.html)
and [complete metrics/provenance](../../runs/common-world-report-20260909.O7nqqe/summary.json)
are now generated, after the following runtime gate completed at 07:30 UTC.
The CPU-only builder revalidated all nine comparisons, original pins and video
identities in a fresh process using the historical native package; it constructed
no agent. Its 13 fabricated-fixture tests remain separate implementation evidence.
Every scored/conceded point links to its existing recording.

### Shared outcomes change the diagnosis

All models retain an action-conditioned feature advantage, including within
visual chunks. Lower ratios are better. Pooling weights all 11,388 transitions
equally; recording 1 supplies about 60% of them, so inspect the per-match rows too.

| Model | Feature MSE | / unrelated controls | Within-chunk / persistence |
| --- | ---: | ---: | ---: |
| 0 | 0.02426 | 0.841 | 0.141 |
| 1 | 0.02519 | 0.838 | 0.145 |
| 2 | 0.03055 | 0.865 | 0.174 |

The strongest player, model 2, does not have the lowest common-pool feature or
reward error. Every model has its lowest feature error on its own recording.
Positive-point forecasts show the same strong dependence on the recording:

| Recording / positive events | Model 0 MAE | Model 1 MAE | Model 2 MAE |
| --- | ---: | ---: | ---: |
| 0 / 21 | 0.068 | 0.877 | 0.956 |
| 1 / 20 | 0.758 | 0.388 | 0.877 |
| 2 / 21 | 0.618 | 0.981 | 0.051 |

The pooled reward errors below use the same 62 positive and 29 negative events
for every model. These are magnitude errors in reward units, not probabilities.

| Model | Prior positive / negative MAE | After-frame positive / negative MAE | All-frame prior MAE |
| --- | ---: | ---: | ---: |
| 0 | 0.477 / 0.499 | 0.474 / 0.429 | 0.00558 |
| 1 | 0.755 / 0.453 | 0.703 / 0.382 | 0.00752 |
| 2 | 0.624 / 0.788 | 0.612 / 0.707 | 0.00657 |

Always-zero reward has event MAE 1 and pooled MAE 0.00799. Model 1 is worst on
positive magnitude, but best on pooled negative magnitude; a blanket claim that
its whole world model is worst is unsupported. Model 2's one negative point in
its own match concealed poor negative prediction on the other recordings.
Seeing the target frame does not remove the large cross-recording reward errors.
Four of six cross-model pairs have all-frame prior reward error worse than zero.

This is evidence of limited generalization across these recorded trajectories,
not proof of a particular failure mechanism or an unbiased estimate over Pong.
It strengthens the case for examining rewarded experience coverage and robust
reward learning before enlarging perception. It does not establish that changing
coverage will fix policy reliability. Preserve all three models and all nine rows.

Only three terminal targets are available. Their MSE is 0.993 / 0.985 / 0.971,
versus always-continue 0.994: none provides evidence of reliable terminal prediction.
Common-pool policy entropy and value are descriptive only. Logged-action
agreement is not action quality, and another policy's return is not a critic target.

### Completed training coverage, not a causal explanation

The CPU-only [full training summary](../../runs/pong-training-coverage-20260909.LEJXHN/summary.json)
reads all three closed 200k-action logs, rather than selected trailing windows.
Their SHA-256 identities match the original final-training verifications.
It reconciles every reward with the final per-stream totals, all completed returns
and partial tails, 200,000 actions and 49,619 consecutive updates per seed.
All ten 20k-action windows remain in the result. No agent was constructed.

| Seed | Positive / negative points, first 80k | First-80k replay batches without positive targets | Positive / negative points, full 200k | Positive replay samples, full 200k |
| --- | ---: | ---: | ---: | ---: |
| 0 | 87 / 1,380 | 12,540 / 19,619 (63.9%) | 596 / 1,907 | 87,013 |
| 1 | 12 / 1,961 | 16,972 / 19,619 (86.5%) | 296 / 2,703 | 37,440 |
| 2 | 76 / 1,494 | 12,923 / 19,619 (65.9%) | 1,406 / 1,642 | 217,920 |

Replay sample counts include repeated use of the same event; they are not new
or independent experience. All three seeds have exactly two initial updates
with zero reported absolute advantage, then nonzero values in every remaining
update. Sparse *positive* coverage in Pong is different from the original
Freeway control's absence of any reward-driven signal.

The final 180k–200k windows have sample-weighted posterior positive-reward
estimates 0.959 / 0.821 / 0.994 for the actual +1 targets. These training-side
estimates are not held-out prior forecasts. Imagined-policy entropy is likewise
not live conditional entropy or an independent quality measure.
The unchanged historical `compare_early_learning.py` independently agrees on
each final window's updates, points, games, wins, replay counts/means and entropy.
Two fabricated arithmetic tests check weighted aggregation, not learning.

This confirms a prolonged positive-experience deficit in seed 1, but does not
identify its cause. Seed 0 has *more* positive points than seed 2 through 80k,
yet wins much later. Policy and coverage co-evolve; neither total event count
nor training reward fitting explains the complete ranking. Do not infer a
broken sampler, adopt reward-balanced replay, or reinterpret the completed Freeway arms
from these correlations. The old adjacent live-RNG overlap still applies.

The summary hash is
`b7912afaa450d2db29523b6f865d2a59721d1218a5a6f49ed4b517dcb1e206e1`;
it records the script and all three source-log hashes. The script refuses to
overwrite its result. Preserve the original logs and completed frozen failures.

### Next bounded decision

The completed Freeway hold64/hold1 comparison tests reward discovery and
subsequent unassisted learning, not this Pong diagnosis; preserve both arms.
For a Pong follow-up,
declare training-side reward-coverage measurements and fixed all-seed learning
budgets before changing exploration, replay or reward losses. Require own-policy
frozen gameplay and a newly held-out multi-match forecast set; do not select a
recipe using these already inspected three recordings. A longer budget is a new
experiment, never an extension that relabels the failed 200k-action mastery gate.
No result here warrants actor/learner separation or a larger video encoder.

The report builder's SHA-256 is
`ec71dad0ea9fac89ff716e19bd153323cd5953839a1f5fb2a8621b5dbd93ec2c`;
report SHA-256 is `fe84e0c66f9eb5a2f9333fa39e2781eb47c93120e4fbb65f95e96eaf74d67595`;
summary SHA-256 is `0b156990a2c7725d828aad365edf917154fe9408030a30f37ddaa1ef0ed0eb78`.
The builder refuses existing outputs. Preserve the completed report and data;
do not rerun GPU forecasts or overwrite these artifacts.
