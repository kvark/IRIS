# Current upstream refresh: compatibility candidate

The September 9 remote check resolves Meganeura main to
[`e59bd32d`](https://github.com/kvark/meganeura/commit/e59bd32d2aeb200eb00b1a15698a777d7df8db4f).
Unlike the earlier metadata-only `970da8e3` review, this includes runtime changes:
softplus negative-tail values/gradients, new multimodal/cache operations, native
capture support and pipeline-label reuse, and weighted/small-tile matmul epilogues.
These fixes do not by themselves explain the old Pong seed failures.

## Preserve the encoding contract

Upstream's new cached block attention masks queries by token position. LeVJEPA
needs every patch in the current frame to attend to the entire current frame,
plus earlier frames. Substituting the new operator would silently change the
representation. Main also lacks the carried early cache-write alias correction:
a view of the write result can otherwise point at a separate unused buffer.

The new downstream revision
[`4d45ba3a`](https://github.com/kvark/meganeura/commit/4d45ba3a1830107769ae07fabcf3b95d0762973c)
retains the validated tiled F32 cached-query generator with a distinct shader
identity, alongside upstream's token-causal operator. It resolves both ordinary
and prefix cache-write aliases before allocating views. The existing 1/3/33/196
query CPU-reference/reset hardware fixture is retained unchanged; a new small
CPU regression checks views of both write forms. No PR was merged and the
user's `megakernel-probe` worktree remains untouched.

Kindle's isolated candidate is `90b4763` on `exp/meganeura-refresh-20260909`,
starting at the completed exploration package's `9fc8108`. Its only source changes are the
dependency pin, both lockfiles and reported backend identity. Both locks retain
the same remaining dependencies, including Blade 0.9.0. The world-sync fan-out
candidate is not included. Historical binaries and restore identities remain
strict and unchanged.

## Gates and current status

Preparation and logs are in `runs/meganeura-refresh-20260909.xfF3AZ`.
Workspace/Python formatting and all-target Clippy pass. The Kindle workspace
has 95 passing CPU tests (22 hardware tests ignored); all 547 Python tests pass
with the actual isolated extension imported and rechecked after the suite.
The backend's 21 code-generation tests and small cache-alias CPU test pass.
The first locked build failed because the fresh upstream worktree has no tracked
Cargo.lock; the separate retry generated an offline lock and completed. Preserve
both logs. A separate 16-test fabricated-state suite checks the full-learning
comparator, including missing moments, incorrect identities, torn saves, shape
changes, nonfinite state and any changed non-timing report.

The isolated package is `runs/meganeura-refresh-20260909.xfF3AZ/package`, native
SHA-256 `f6a2b6ad6256fde2ad5209f90d5e94533537421a2adf603d0834d7c082fada74`.
The wheel, built library and imported bytes agree; all six Python files match
source. `cpu-evidence.json` binds 61 inputs. The historical/control extensions
remain unchanged; this is not an instruction to replace them.

All 18 declared GPU tests completed at 23:41 UTC. An independent CPU audit
rechecks all 261 pins, command exits/output hashes and complete direct-memory
coverage. The production B16/T64 loss/all-gradient test passes at its unchanged
tolerance (worst relative gradient L2 0.00074572). LeVJEPA passes its pinned
reference/reset and asymmetric-stream tests; N4/N6/N8 each have zero measured
dense-feature error against serial encoding. Ordinary and zero-update checkpoint
roundtrips, device transfer, vector learning and executed-action overrides pass.
The hardware group retains at least 6,545 MiB directly free; this is not combined
learner memory or a throughput measurement.

[`hardware-result.json`](../../runs/meganeura-refresh-20260909.xfF3AZ/hardware-result.json)
has SHA-256 `549f96ea00a70bb83130ff43553057222b627e9caabb7450b612b7ef63e92f09`.
Both fresh eight-update 12M/B16/T64 canary pairs completed at 23:47 UTC in
parent/candidate then candidate/parent order. All 241 tensors, complete optimizer
state, normalizers and eight non-timing reports match exactly in both pairs;
only the declared backend identity and report timing differ. The independent
audit rechecks all 309 pins, ten successful commands and four complete GPU
windows, with at least 9,559 MiB directly free. This synthetic run excludes the
visual frontend, so it does not establish combined learner memory or speed.
`canary-result.json` has SHA-256
`35e69e2be0690a228b296c77d3793673f96a3c99a10b6d16cb63b6e04b26ef34`.
The first preflight found a missing standalone candidate example before creating
any declaration or GPU job; the unchanged example was then built explicitly.
That preparation failure is retained, not a numerical failure or partial run.

The separately declared N6 pixel comparison started at 23:56 UTC, with 374 pins
and 22 passing CPU comparison tests. Four fresh Boxing trials run in
parent/candidate/candidate/parent order, each 3,840 training and 768 restored
frozen actions. Require exact complete state, all non-timing reports and
action/reward/reset traces. The fixed 2,304–3,840 window contains 384 updates;
retain at least 98% of parent throughput in both orders. A speedup requires both
ratios above 1.005. A short candidate Freeway override/frozen integration follows.
All ten GPU phases need complete direct-memory coverage and at least 2,048 MiB
free. `pixel-declaration.md`, `pixel-manifest.json` and `pixel-events.jsonl` bind
the active gate; do not modify or restart it. No pixel result is claimed yet.

Before adoption, complete the pixel comparisons and directly measured
combined-learner GPU headroom. A corrected backend may change arithmetic: diagnose any mismatch
instead of editing old models or silently relaxing the comparison. Retain all
logical weights and optimizer moments; only identified derived caches may differ.
Declare matched timing before running it, and distinguish correctness adoption
from an actual speedup. No new long learning run is started by this preparation.

This user-requested update now precedes the still-unrun world-sync comparison.
Do not rerun the completed Freeway, common-world or episode-evaluation queues.
The [world-model reports](2026-09-08-world-evaluation.md) deliberately continue
to use each historical model's original executable, not this new backend.

## New facilities are not automatic optimizations

Kindle already constructs its shared context through `GpuOptions::from_env`, so
the new `MEGANEURA_GPU_CAPTURE=1` can enable native-tool names/debug information
in a separately declared capture. It does not require enabling per-dispatch
timing or changing the production schedule. It is disabled throughout this gate.
Usable imported coverage and numerical qualification remain necessary; labels
alone do not resolve the old queue-only/idle-gap limitations.

Upstream also exposes `Session::share_parameter_from`. Do not replace world
synchronization with it blindly: equal physical byte counts are weaker than
matching logical shape/format, and sharing an allocation does not refresh a
target session's derived parameters or Winograd caches. Any such optimization
needs its own mutation/restore/cache checks and serialized learning comparison.
The current update uses neither sharing nor the staged read-once fan-out.
