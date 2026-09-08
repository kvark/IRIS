# Persistent exploration candidate: CPU-ready, not adopted

Staged 2026-09-08 in `exp/persistent-exploration`, isolated from the live
Freeway source and package. There is no GPU correctness, speed, learned-score
or five-game reliability result for this candidate.

## Why this experiment

The [CPU Freeway discovery study](2026-09-08-freeway-discovery.md) found no
crossing rewards from independent random actions in three 200,004-action arms.
Holding uniformly sampled actions for 16 or 64 decisions found hundreds in
every seed, still far below the task gate. The unchanged native Freeway pilot
continues to its fixed budget. These results motivate a bounded exploration
ablation, not a longer unchanged reward-starved run or a claim of learned skill.

## Small implementation

The native vector agent can override selected actions explicitly. It still
computes and samples the ordinary policy, preserving every per-stream policy
RNG draw. The actual returned action becomes the pending action consumed by the
RSSM and replay. All override lengths/ranges are checked before native mutation.
No model, loss, optimizer, precision, reward or observation-cadence change is made.

The Python runner optionally chooses, independently for each stream, whether
each fixed-length block follows the policy or holds one uniform random action.
Both kinds of block have the same duration, so probability describes the
approximate fraction of exploratory actions, not a per-decision trigger whose
holds inflate that fraction. Natural resets cancel only that stream's hold;
they do not rewind RNG or counters. The independent exploration RNG and its
seed rule are versioned. There are no game-specific controls, privileged inputs,
demonstrations, reward shaping or emulator rewinds.

`--exploration-probability` defaults to zero; the default path makes the original
native call and emits the unchanged v2 fields. Positive probability requires the
new native API and a fresh run, and selects protocol `kindle-vector-v3`.
`--exploration-hold` counts agent decisions, not raw frames; Atari still executes
up to four raw frames and observes after every decision. These knobs are not
an adopted learning recipe. Frozen evaluation refuses exploration and remains v2.

Every v3 transition records the override and actual action. The CPU auditor
reconstructs the complete random-block/reset sequence, checks executed actions,
counts and helper identity, and retains ordinary reward/update/frame accounting.
Match and task audits allow v3 training followed by unassisted v2 evaluation
without relaxing model/checkpoint/game identity. Their training accounting
retains exploration provenance. Existing campaign declarations intentionally
reject this new protocol; passing a task audit is not a replication certificate.

## Completed CPU checks

Artifacts: `runs/persistent-exploration-20260908.KpQGgs/`.

- 95 Rust CPU tests pass; 22 GPU tests remain ignored. Rust formatting and both
  workspace/Python-binding all-target Clippy checks pass.
- 547 Python CPU tests pass against the actual isolated `package-review`.
  They include independent RNG/block references, partial resets, mixed policy
  blocks, tampered ledgers, old-native fail-fast handling, and runner checks that
  the returned override reaches the environment and observation ledger.
- The release native SHA is
  `9cf1316bf08567eb0ddc2b2e87ceb2ebcacb6e239d3312d1a500ba3ce2161652`.
  Both the live 9cd176c1 package and historical f663dd93 extension are unchanged.
- Builds used cached targets and two jobs, with no GPU execution. The first
  wheel install selected Python 3.13 and rejected the CPython 3.14 wheel; the
  corrected isolated install explicitly selects the existing 3.14 interpreter.
  It did not replace the default environment or either control package.

Tests and logs are implementation evidence, not learning evidence.

## Required next gates

Keep GPU work serialized after the entire Freeway pilot queue and the declared
common-recording world diagnostic. Preserve their pinned inputs.

1. Run the new tiny native override/serial-belief/replay test plus the existing
   default vector/serial and vector-one learning/checkpoint tests.
2. Compare the candidate's default path with the unchanged 9cd176c1 package at
   N6/R256/B16/T64/full BPTT64/F32: fixed 3,840 fresh training actions and
   768 frozen actions, AB/BA order. Require exact action/reset/episode histories,
   every non-timing learner report, full logical tensors/optimizer state and
   checkpoint metadata. Exclude only declared executable/runner identities.
3. Measure directly reported free VRAM throughout construction, training and
   restore: at least 2,048 MiB, adequate 4 Hz monitoring coverage, no competing
   GPU worker. Report warmed matched-update throughput, not only GPU activity.
   The old package's successful N6 memory gate is not proof for these new bytes.
4. Run a short, separately declared actual-pixel override integration check.
   Then declare a fixed-budget learning comparison, including unassisted frozen
   evaluation and the untrained control. Choose probability, hold and all budgets
   before launching; do not tune them against the final evaluation.
5. Only a successful learning comparison can select this behavior for a new
   fresh-seed replication declaration. Keep all five game gates and seeds
   1009, 2017 and 3019. No result here changes the failed Pong all-seeds decision.

The detailed GPU manifest and launcher are not yet declared. Do not run root
binaries by accident: cached build outputs may belong to another isolated
candidate. Select or rebuild the explicitly identified package/test executable.
