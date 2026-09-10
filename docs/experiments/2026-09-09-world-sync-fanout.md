# World-parameter readback fan-out candidate

Prepared 2026-09-09 at `63d3df7` on `exp/world-sync-fanout`, based on the isolated persistent-
exploration candidate `9fc8108`. Not adopted; no GPU correctness, memory or
throughput result exists yet. No active package, runner or queue is changed.

## Bounded change

Each learner update currently reads overlapping world parameters separately for
six core inference sessions: batched/live observation, batched/live transition,
and batched/live heads. The candidate reads the union once and distributes those
values to the existing targets. An optional diagnostic predictor joins that same
read. Initialization uses the same helper.

Each target still receives its original source-parameter order through
Meganeura's `set_parameter`, including its waits and derived-weight/cache refresh.
No direct GPU-buffer overwrite bypasses backend bookkeeping. Matching still
requires the prefix, name and element count. Behavior synchronization, critic
EMA and the vector collector's separate final synchronization are unchanged.
There is no changed arithmetic, sampling, loss, optimizer, replay ratio, BPTT,
frontend or action/exploration setting.

A CPU-only inspection of the actual six 12M inference graph definitions finds
98,914,816 bytes of repeated logical F32 parameters versus a 29,200,640-byte union
(94.33 versus 27.85 MiB). This omits compiler-derived caches and is **not** measured
PCIe traffic, free VRAM or a timing win. World synchronization costs only about
15 ms of the roughly 343-ms learner update; this bounded change cannot by itself
make R256 training faster than real time.

## Validation status

The workspace has 96 passing CPU tests; 24 hardware tests remain ignored.
Workspace and Python-binding Clippy checks and formatting pass. Python source
is unchanged. A separate release extension is now built, but no existing package
or active input has been substituted.
CPU logs are in `runs/world-sync-fanout-20260909.zxMPa9/`.
The archived `kindle-tests` binary has SHA-256
`84d13e8a2539ef4b51da7f03b25c601b818cbd202fd85cc8dde3ffb319ce3be8`.
Shared debug build outputs now contain this candidate; they are not controls.

The isolated package is `runs/world-sync-fanout-20260909.zxMPa9/package`, built
from the same `63d3df7` source with CPython 3.14.4, Rust 1.98.0, locked/offline
dependencies, one low-priority CPU job and a fresh Cargo target directory.
All 547 Python CPU tests pass with this native module imported in the test
process and its identity/hash checked again after the suite. All six packaged
Python files match source. The compiled library, wheel payload and imported
extension are byte-identical, with native SHA-256
`c2f5f3d8e87907e8da8071e7e65aad5becbe612e9f64f67a7ca75630762d44f3`.
The historical `f663dd93`, active `9cd176c1` and queued `9cf1316b` packages retain
their original hashes. `cpu-evidence.json` records the build and evidence hashes;
`package-identity.json` and `python-tests-bound.xml` retain the checks.

This CPU build overlapped Freeway training, not a matched timing window. It
constructed no GPU agent and provides no new timing or learning result. The
source branch remains pinned to its pre-build commit; these are later local
artifacts. Do not use the new package in any existing queue.

Two new ignored GPU tests compare fan-out with the independent serial helper:

- `dreamer::runtime::tests::parameter_fanout_matches_serial_subsets_and_updates`
  checks overlapping/disjoint targets, changed source weights, exact copies,
  unchanged source, mismatched sizes, missing names and prefixes, and empty targets.
- `dreamer::runtime::tests::parameter_fanout_preserves_derived_cache_refresh`
  checks exact nonzero convolution outputs after two weight updates, including
  tied Winograd caches and a genuine parameter name ending in `:winograd`.

The logical-overlap CPU test is not a test of GPU transfer correctness. Before
adoption, require both new hardware tests, existing act/learn/restore and vector
parity checks, and complete matched-learning evidence against the unchanged
parent: all logical parameters and optimizer moments, normalizers, non-timing
reports, action/reward/reset traces and learner counts. Use N6/B16/T64/full BPTT,
F32/R256, at least 2,048 MiB directly reported free, and untraced warmed AB/BA
pixel timing. Require a repeatable end-to-end gain in both orders; otherwise
retain the parent. Short correctness tests are not a learning-quality result.

Do not insert this candidate into the existing queue. Freeway, common-world,
the persistent-exploration runtime gate and its learning pilot, then the already
declared frozen episode-budget gate retain precedence. This branch stages code
for a later explicitly pinned comparison; it starts no follower or training run.

At the later September 9 check those predecessor queues are complete (the
episode gate through its separately declared continuation). Upstream has since
moved, so the [requested backend update](2026-09-09-meganeura-update.md) now takes
precedence. Keep this original candidate and draft unchanged and unrun; do not
combine backend arithmetic changes with its fan-out comparison.
