# Persistent exploration candidate: runtime gate passed

Staged 2026-09-08 in `exp/persistent-exploration`, isolated from the original
Freeway source and package. Full runtime validation passed on September 9;
the fixed learning comparison is now running. The recipe is not adopted.

## Why this experiment

The [CPU Freeway discovery study](2026-09-08-freeway-discovery.md) found no
crossing rewards from independent random actions in three 200,004-action arms.
Holding uniformly sampled actions for 16 or 64 decisions found hundreds in
every seed, still far below the task gate. The completed native Freeway pilot
finds no training rewards, and its frozen policy and untrained control both score
zero across 36 natural rounds. These results motivate a bounded exploration
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

## Validation protocol

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

## Completed serialized runtime gate

The gate declaration is
`runs/persistent-exploration-gate-20260908.OVSB5q/manifest.json`, with 66 input
pins. It adds an explicit default-path throughput non-regression bar: candidate
throughput must be at least 95% of its paired control in both orders. The short
Freeway integration uses probability .25 and hold 64; these are integration
settings, not an adopted long-training recipe. Its three Rust hardware tests
use an archived test executable, not a mutable root build output.

The 31 CPU gate/handoff checks pass. They bind a real completed old N6 control,
validate its complete checkpoint and reject changed identities and learner reports; candidate rows
and completed queue children in the unit fixtures are explicitly fabricated.
These tests do not establish candidate GPU parity or learned competence.

`follow_queue.py` started at 23:58 UTC on September 8, bound to the actual live
Freeway launcher PID 2105757 and its process start identity. Its 107-pin
declaration waits for the entire pilot to complete and the process to finish,
then runs the common-world diagnostic and this gate in separate processes.
The follower was PID 2119271. It polled every 30 seconds,
had a bounded wait, and neither restarted nor stopped the Freeway parent. Failed
prerequisites or child failures stop the follow-on without automatic retry.

The follower ran this gate from 06:47:48 through 07:30:36 UTC on September 9,
after Freeway and common-world completed. All three focused hardware tests pass,
each executing exactly one test with zero failures or skips. All four default
12M pixel-learning trials match exactly: all 241 saved tensor entries, optimizer
state, normalizers, comparable metadata, non-timing learner reports, training
actions and restored frozen action/reset/episode traces.

| Order / role | Warmed actions/s | Candidate / paired control |
| --- | ---: | ---: |
| 0 / control | 8.57559 | — |
| 1 / candidate | 8.56870 | 0.999197 |
| 2 / candidate | 8.58989 | 0.999774 |
| 3 / control | 8.59184 | — |

Both predeclared non-regression gates pass; this is unchanged throughput, not a
speedup. The candidate reaches 0.5712–0.5727× aggregate real time, about 0.095×
per stream. Learning takes about 132.5 seconds and observation 45–46 seconds of
each approximately 179-second warmed window. Neither vector collection nor
these action overrides removes the dominant learner/perception costs.

The Freeway integration completes 3,840 training actions, 610 updates and
768 restored unassisted frozen actions. Its mixed override ledger validates;
there are no completed rounds in this short integration, hence no competence
claim. All 13 GPU phases pass coverage and memory checks, with a maximum
0.279-second sample gap and at least 3,303 MiB directly free. Train/restore
minimum free memory is 3,303/3,413 MiB. Short whole-process activity includes
construction and warmup; do not compare it directly with warmed/long-run activity.

The [completed gate](../../runs/persistent-exploration-gate-20260908.OVSB5q/completed.json)
has SHA-256 `529caf0f16696867f62c8f592897984cacf125800f052c945315db0db3ce0f44`.
The queue completed and its controller exited. Preserve all 66 gate pins,
107 queue pins, archived test binaries and raw outputs; do not restart it.

The separate [conditional learning declaration](2026-09-09-freeway-persistence.md)
reverified the actual raw tests, complete state/trace/report comparisons and
GPU windows before starting hold64 training at 07:31:02 UTC. A separate fresh
CPU recheck reproduces its runtime proof exactly. The gate itself launches no
long learning run and cannot adopt exploration. Learning quality and fresh-seed
reliability remain pending; keep all original artifacts and failed results.

Do not run root binaries by accident: cached build outputs may belong to another
isolated candidate. Select the explicitly identified package/test executable.
