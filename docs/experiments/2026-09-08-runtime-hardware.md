# Serialized runtime hardware checks

Declared before candidate GPU execution, while the three-seed LeVJEPA campaign
was finishing its last frozen evaluation. The complete campaign now records
two failed seeds and one passed seed, with valid accounting throughout. Keep
its inputs and the queued readback worker unchanged, and wait for both the
campaign/watcher and worker to exit.
This is a runtime diagnostic and parity experiment, not another mastery test.

## Prepared inputs

The uninstrumented Rust parent is `73273df`; its Rust sources and lockfiles
still match the main branch. The active Python extension remains
`f663dd9317bd934f173b240041ea68b7e21e0f7d037ebaf22b9549a9ae91bb4e`.
The isolated candidates are:

| Candidate | Revision | Isolated Python extension SHA-256 |
| --- | --- | --- |
| Host/readback timers | `ec074a5` | `cb8579e7e108d35131cb6c5581a4d3b45918745818a8ec41068c64fcd0a95f56` |
| Reuse imagined-feature capacity | `6ef7ba6` | `21e2244f91a89b519fbe8bb86a2c2cb282d967cc633367220444270351f465e4` |

Both release extensions pass all 229 current-main Python CPU tests. The timer
branch's own 150 Python tests also pass. Builds use separate target directories
and packages; neither the active environment nor prebuilt Rust test/canary
binaries are replaced. All 17 queued-worker pins still match after compilation.
These CPU checks do not establish GPU parity, overhead or a speedup.

Build manifests and packages are under
`runs/readback-python-20260908.XZoN91/` and
`runs/host-features-python-20260908.us04TR/`. Their `cpu-validation.json` files
record source, lockfile and artifact identities. CPU compilation/testing overlaps
seed-2 frozen evaluation at approximately 01:43–01:49 UTC on September 8,
not the completed training health windows. Frozen run timing therefore remains
a health observation, not a quiet-system comparison.

## Order and safeguards

1. Inspect the original processes and complete campaign artifacts. Finish the
   queued worker's three exact hardware tests and three eight-update synthetic
   parent/timer pairs. Keep its recorded AB, BA, AB ordering. Analyze updates
   3–8 separately from warmup, including every paired ratio and its spread.
2. Attempt the separate parent-only
   [Vulkan capture](2026-09-07-vulkan-capture.md). Require usable imported GPU
   workload records, not just a successful CLI exit or host API trace. If usable,
   compare three fresh eight-update traced/untraced pairs in TU, UT, TU order;
   the initial successful trace is T in the first pair. Require complete reports
   and exact non-timing/tensor parity. Retain failed artifacts and do not change
   system permissions to make collection work.
3. Validate the timer extension with the short pixel protocol below, independently
   of external capture. A failed external tracing tool does not invalidate native
   host timers, but prevents calibrated GPU-gap claims.
4. Run the buffer-reuse candidate's two exact hardware tests and three synthetic
   AB, BA, AB pairs from its
   [decision gate](https://github.com/kvark/kindle/blob/6ef7ba6/docs/experiments/2026-09-07-host-feature-reuse.md).
   Only after they pass, run its own short pixel comparison against the
   uninstrumented parent, not against the timer candidate.

Use adapter `0x2c02`, UUID
`GPU-6869e50d-83aa-bec7-6169-adc413f49b32`, with one GPU workload at a time.
Verify binary/source hashes before each sequence, use fresh outputs/checkpoints,
retain failures, and stop on missing budgets or parity failures. Screen for
obvious competing use; never stop unrelated processes. No replay-ratio change,
new training recipe, encoder switch or candidate combination is part of this
comparison.

## Short pixel protocol

For each candidate independently, run two fresh parent/candidate pairs in
AB, BA order. Use the unchanged pinned `atari_vector.py` and `atari.py`, the
same LeVJEPA weights, seed 0, N=8, 12M/B16/T64/full BPTT/row16, R256,
AGC 0.3, LR 4e−5 with 1,000-update warmup, reconstruction 0 and future loss 0.25.
Each process starts from zero counters with no restore, runs **3,072 aggregate
actions**, and saves its final checkpoint. Report every 512 actions.

Require the complete action/reward/frame/reset/replay/credit audit, 387 learner
updates, zero final debt, finite reports and all 241 expected checkpoint
tensors. Compare every non-timing learner report and named tensor/optimizer
value, plus the exact action/reward/boundary sequence. Native-library identity
and timing fields are expected to differ; model, encoder, recipe and game-clock
ledgers must not. This short run need not complete a natural game; it is not a
learning-quality or comprehensive reset-coverage claim.

Measure the final **1,024 actions** (2,048–3,072), after replay eligibility and
initial learner warmup, with exactly 256 updates. Report aggregate and
per-stream actual game/wall ratios, full runner learner-call time, native stage
and contained substage times, construction/prefill separately, GPU activity,
power, VRAM and host-memory/fault counters. Require selected-device 1 Hz samples
to cover at least 95% of this window. Retain both paired ratios and their spread;
do not select the faster pair. No compilation, CPU-heavy tests or other owned
GPU work overlaps these windows.

`LearnTiming.total_seconds` excludes local destruction. Adoption of buffer reuse
requires a repeatable warmed full-call/end-to-end improvement with exact parity
and an acceptable memory tradeoff, not merely a faster inner timer. It retains
150 MiB of host capacity at this shape; it does not save GPU memory.
Readback wait time includes unfinished producer work and transfers. Neither
native host timings nor coarse GPU activity samples establish calibrated GPU
idle time. Preserve unclassified time and quantify instrumentation overhead.

Hardware results are pending. No candidate is adopted by this declaration.
