# Frozen world forecasts versus the recorded Pong matches

Completed 2026-09-08. This evaluates the world model separately from gameplay,
without retraining or changing the failed three-seed mastery decision.

## Inspect the result

Open the local [video-linked report](../../runs/world-evaluation-20260908.Xzx3pN/report.html).
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

## Common-recording follow-up declared, GPU checks pending

The isolated `exp/common-world-probe` candidate at `a425b29` adds explicit
recorded-action conditioning while retaining strict own-policy replay as a
separate mode. All 282 Python CPU tests pass. The declaration in
`runs/common-world-20260908.7gWHsJ/manifest.json` pins 35 inputs; CPU reconstruction
verifies all 11,388 transitions, including 62 positive and 29 negative points.

All three final models will see the same three first matches. Score every
one-step target, require exact reproduction of the original three H1 diagonals
before cross-model comparison, and verify identical initial/target RGB and
feature hashes. Record unmasked action probabilities/value before forcing
controls, without calling the forced trajectory the evaluated model's own play.
Cross-policy logged returns are not unbiased critic targets. These recordings
were held out from training, but have already been inspected; this is a
diagnostic dataset, not untouched confirmation.

The CPU result-binding tests also pass using explicitly fabricated new outputs
against the real source prefixes and original traces. They do not validate GPU
predictions. The diagnostic is not running and has no new model-quality result;
its nine serialized GPU combinations wait for the entire Freeway pilot queue.
The original executable and all historical results remain unchanged.
