# Common-match world-model diagnostic

Staged on 2026-09-08 in `exp/common-world-probe`. All 282 Python CPU tests
pass, including 50 focused probe tests. GPU validation and the cross-model
comparison have not run. The active Freeway pilot retains exclusive GPU use.

## Question and fixed inputs

The earlier [world-model report](2026-09-08-world-evaluation.md) compares each
Pong model on its own first frozen match. Its seed differences can reflect both
model quality and the different trajectories. Score every final model on the
same three matches to separate those effects.

Use the final 200k-action models for training seeds 0, 1 and 2, and the first
complete recorded frozen match from each. All nine model/recording combinations
are required. These are previously evaluated trajectories, held out from
parameter training but not an untouched confirmatory dataset. Do not choose
games or checkpoints based on the new forecast scores.

One-step forecasts cover every action, without stride selection. The shared
pool contains 11,388 transitions, 62 positive points, 29 negative points and
three terminals. Three terminals still cannot establish terminal reliability.
Retain per-recording rows as well as transition-weighted pooled results; these
three matches are not a broad-game generalization test.

## Implementation and safeguards

`probe_atari_dynamics.py --condition-on-recorded-actions` explicitly follows
recorded controls even when the evaluated model would choose something else.
It requires `--recorded-run`, the same actual native executable, full frontend/
backend provenance, matched configuration except model seed, and matched
training counters. The evaluated checkpoint gets its own full content identity;
the producer's checkpoint remains separately identified.

Ordinary recorded replay still requires the original checkpoint and exact
unforced sampled actions. Its mismatch refusal is not relaxed. The new mode
reports `recorded_action_conditioning`, `actions_forced_to_source: true` and
no assertion that the evaluated policy selected those actions. This is offline
model evaluation, not gameplay, extra training experience or a task-gate win.

Before each real transition, record unmasked policy probabilities, entropy and
posterior value, then execute only the recorded control. Forecast features,
reward and continuation before consuming the next frame. Hash each observed
RGB frame and exact F32 target feature vector; all models must receive the same
inputs for each recording. Verify actual rewards, boundaries and emulator-frame
counts against the complete source prefix, with zero learner updates.

The declared one-step analysis retains persistence, unrelated-control,
zero-reward and always-continue baselines, signed-event counts and visual-cache
boundary strata. Predicted features are not RGB reconstructions. The controls
for the unrelated-action baseline do not have observed counterfactual outcomes.

Value needs a different interpretation: another policy's logged return is not
ground truth for the evaluated policy's expected return. Record value and action
likelihood descriptively; do not turn cross-policy value residuals or imitation
of a recorded action into an unbiased policy-quality/causality claim. On the
same-model diagonal, observed returns remain noisy own-policy samples.

## GPU gate and run order

Artifacts will be declared in `runs/common-world-20260908.7gWHsJ` after CPU
reconstruction and identity checks. Use the original `f663dd93…` executable,
not the current backend package: the old checkpoints retain strict backend
identity. Do not modify them or rebuild over their historical extension.

After the Freeway queue completes and releases the device:

1. Run the three same-model diagonals first, using one-step forecasts and
   explicitly recorded actions. Require exact agreement with every original
   H1 forecast/reward/continuation/error trace; the added read-only policy/value
   calls and forcing must not perturb the world model.
2. Only after all diagonals pass, run the other six combinations. Validate
   every RGB and feature hash against the shared recording/diagonal, along with
   the complete action/reward/boundary ledger and unchanged learner counters.
3. Report all nine rows and all failures. Use matched per-event reward errors
   and common-input feature errors to assess the reward-reliability lead.
   Do not infer that the reward head causes policy failure merely from correlation.

Serialize GPU runs, keep directly measured free memory ≥2,048 MiB with coverage
checks, and use a 40-minute limit per combination. This is not a speed benchmark
or another training-seed replication. It neither repairs Pong's failed mastery
gate nor satisfies the five-game goal.
