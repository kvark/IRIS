# Multi-match world forecasts from vectorized evaluations

The isolated `exp/multimatch-world-probe` candidate (`b2f0ddd`) extends the diagnostic,
not the learning recipe. It starts at current-episode source `24b2968`, carries
the common-recording probe from `a425b29`, and retains the unchanged f6a2b6ad
native package. It is not GPU-qualified or adopted. Boxing and every declared
successor keep their original inputs and priority.

## Recording contract

The existing first-game N1 reader remains the default. Explicit
`--recorded-episodes N` selects **stream 0's first N complete episodes** from
a complete, unassisted sampled v2/v4 evaluation. Selection never searches for
wins or skips cutoffs. The full source ledger must pass, including every other
stream and the tail after the selected prefix; an incomplete episode budget
cannot supply a valid recording. Keep the original vector header and aggregate
action indices. Bind the complete file and exact selected prefix separately.
This is a diagnostic subset, not a replacement for all-stream policy evaluation.

Strict mode requires the exact native, model, checkpoint counters/files and
sampled actions. Explicit recorded-action conditioning remains separate: it
reports the evaluated model's unmasked policy/value before forcing the controls
and makes no own-policy or critic-accuracy claim. Both modes require the same
actual training-action budget and compatible config/representation/backend.
Conditioned vector comparisons permit different learner counts at that budget,
because episode resets can advance replay warmup. Both actual counts are
retained; strict replay and the historical first-game contract remain exact.

Every forecast precedes its target observation. Multi-match horizons stop at
episode boundaries; episode resets initialize perception and belief, while
ordinary 16-arrival visual chunks retain belief. One-step forecasts cover every
action. Longer horizons still require a declared complete or phase-balanced
origin schedule; a sparse favorable stride is not representative evaluation.
The v4 result records initial/reset RGB and feature identities, every target's
identities and episode/source action indices. Features are not imagined RGB.
Persistence, unrelated actions and zero rewards remain separate baselines;
report reward-sign and terminal/cutoff counts at every horizon.

## CPU evidence

The 633 Python tests pass with the source-matched current package. The 74 focused
probe tests include unchanged historical behavior, full-recording rejection,
missing/extra matches, retained negative/cutoff outcomes, causal ordering,
reset identities, horizon boundaries, changed source content and strict versus
forced controls. Their synthetic agents and frames are not GPU evidence.

Real-recording checks in `runs/multimatch-world-cpu-20260910.5SQRCl` extract and
independently ALE-replay 25,136 actions: four stream-zero Freeway rounds from
each completed hold64, hold1 and untrained evaluation, plus one Boxing match
from the completed historical v4 capture. Rewards, boundaries and actual frame
counters match. These reproduce existing behavior, not new policy rollouts.
All three original Pong first-match selections also retain their exact prefix
hashes, byte counts and episode records. No native agent or forecast is produced.
The initial extraction was an uncommitted source snapshot with content pins.
The completed `cpu-evidence.json` binds committed source `b2f0ddd` with 111 pins,
reruns the full suite and real replays, verifies both actual ROM identities, and
reproduces that snapshot exactly. Preserve both; neither is a native-forecast gate.

Native/build inputs, packaged Python modules, training runner and auditors are
unchanged from `24b2968`. Use its matching bundle explicitly; do not install over
the historical extension. Rust formatting checks pass. This Python-only change
does not need another native build or compiler load alongside training.

## Next evidence, before use

After the existing serialized queues release the GPU, declare a diagnostic
gate against actual current-native final recordings. First require strict
serial replay to reproduce the selected vector stream's sampled actions and
transitions over multiple episodes. CPU fixtures do not establish that parity.
Require unchanged complete model state, zero updates and full GPU coverage with
at least 2,048 MiB directly free. Compare the same model's strict and forced
one-step forecasts, including every initial/reset/target identity, before using
cross-model forecasts. Keep an untouched historical first-match control.

The separately declared [Pong confirmation](2026-09-10-pong-confirmation.md)
now reserves the first four complete stream-zero evaluation matches from every
final model, not whichever recordings look informative afterward. Its fresh
roots 1009/2017/3019 each receive 400,008 training actions; it is not running.
Run all three same-model checks before the six cross-model combinations after
native diagnostic qualification. Keep own-policy competence, event coverage
and common forecast error separate. Neither this candidate nor the new learning
worker launches native world forecasts or an automatic follower. The failed
200k-action mastery decision is unchanged.
