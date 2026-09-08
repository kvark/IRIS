# Serialized runtime hardware checks

Declared before candidate GPU execution, while the three-seed LeVJEPA campaign
was finishing its last frozen evaluation. The complete campaign now records
two failed seeds and one passed seed, with valid accounting throughout. Keep
its inputs and the queued readback worker unchanged, and wait for both the
campaign/watcher and worker to exit.
This is a runtime diagnostic and parity experiment, not another mastery test.

## Prepared inputs

The uninstrumented Rust parent is `73273df`; its Rust sources and lockfiles
match the main branch at declaration. The active Python extension remains
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

## Readback hardware and synthetic timing result

The queued worker completes at **2026-09-08 02:31:24 UTC**. All three exact
hardware tests pass, and all three eight-update parent/timer pairs pass complete
report validation, the measured transfer-count checks and exact non-timing
reports plus all 241 named parameter/optimizer tensors. Its fresh campaign audit
also preserves the failed all-seeds mastery decision. No original process or
input was replaced.

Warmed means use updates 3–8 from every process, as declared:

| Pair order | Parent full-call ms | Timed full-call ms | Timed / parent |
| --- | ---: | ---: | ---: |
| AB | 440.841 | 429.509 | 0.974294 |
| BA | 439.386 | 439.465 | 1.000178 |
| AB | 441.447 | 441.638 | 1.000434 |

The observed ratio range is 0.974294–1.000434. These short comparisons show no
material timer overhead at this workload; the faster first pair is not evidence
that instrumentation accelerates the learner. Preserve the spread rather than
selecting one result. The canary's full-call timer includes local destruction
and report serialization/output; it is not the pixel runner's timer. The parent
internal totals are 429.368/428.011/429.316 ms, about 11 ms shorter per call.

Across the candidate's 18 warmed updates:

| Measured section | Mean ms/update | Boundary |
| --- | ---: | --- |
| World training | 157.173 | Whole stage |
| Posterior readback wait | 41.516 | Subset of 57.764 ms posterior stage |
| Imagination readback wait | 37.777 | Subset of 167.390 ms imagination stage |
| Imagination input writes | 32.051 | Host calls, not measured bus traffic |
| Imagination target assembly | 30.181 | Includes the flat feature buffer |
| Imagination feature preparation | 19.799 | Initial flattening and per-state joins |
| Imagination readback copy | 14.170 | Completed CPU-visible output copies |

Measured posterior input/readback payloads are 32.390625/10 MiB in 448/64 calls;
imagination payloads are 631.0546875/199 MiB in 109/31 calls. The remaining host
sections, stage totals and all per-process samples are in
`runs/readback-hardware-20260907/timing-analysis.json`, generated by the retained
`analyze_timings.py`. That analysis revalidates command-output fingerprints,
complete reports, checkpoint health, pins and actual worker termination.

Readback waits include unfinished producer computation and transfers; these
numbers do not identify GPU idle gaps. No compilation or CPU-heavy tests overlap
the canaries; read-only monitoring/documentation continues and other host activity
is uncontrolled. The subsequent external capture and timer pixel results are
recorded below. This synthetic result alone does not adopt either candidate.

## Vulkan capture result: GPU-workload gate failed

The installed 2023.4.4 tool completes and imports two captures of the unchanged
eight-update parent: the declared shell-wrapper launch and one explicitly
separate direct-executable retry to isolate a launch-wrapper issue. Both target
runs pass complete report checks, checkpoint health and exact non-timing report
and 241-tensor parity against the preserved uninstrumented parent.

Both automatic imports fail to find the importer. Explicit use of the matching
installed importer and SQLite export succeeds, with `quick_check=ok`:

| Launch | Vulkan API rows | GPU-workload records | Warmed full-call ms |
| --- | ---: | ---: | ---: |
| Shell wrapper | 3,377,306 | 0 | 547.712 |
| Direct canary | 3,377,290 | 0 | 545.503 |

The direct capture attributes its API records to the expected canary executable;
neither capture contains GPU workload/device/queue tables. The canaries themselves
report the expected RTX 5080 / 595.71.05 device. Direct invocation does not fix
the missing workload records; their root cause is not established. Do not infer
an empty GPU from absent trace data or use API-only records as a calibrated
GPU timeline.

The prior untraced parent means are 439.386–441.447 ms. These are descriptive
comparisons with earlier controls, not the conditional alternating trace-overhead
experiment. Because the workload gate fails, its remaining traced/untraced pairs
are not run. No driver, package or security setting changes. No original or
candidate binary changes.

Raw captures, imports, exact report extraction for the direct launch and
`capture-validation.json` are retained under
`runs/nsight-hardware-20260908.wLROAj/`. Raw traces include host/environment
metadata and remain ignored, not published. Continue native host timing and
the independently gated pixel/buffer-reuse comparisons; GPU idle attribution
remains unmeasured.

## Timer pixel result

The seven CPU tests for the bounded pixel collector pass, covering complete and
partial log tails, startup without a header, malformed completed JSON, invalid
job selection and cleanup limited to owned children. The collector and tests are
in `runs/pixel-runtime-20260908.Y2y0ZQ/`; source SHA-256 is
`34cf06fa59ffae1234091cdd1f60b8f37d2f0f010125472167817dec325d240d`.
The four jobs run from approximately **02:50 to 03:20 UTC** in AB, BA order.
All finish their 3,072 aggregate actions and 387 updates with zero debt. Both
pairs have exactly equal headers apart from timing/library identity, every
action/reward/reset event, all non-timing learner reports and all 241 named
parameter/optimizer tensors plus metadata. None completes a natural game;
this validates runtime parity, not competence or comprehensive reset coverage.

Each row below is the declared final 1,024-action, 256-update window:

| Pair / arm | Actions/s | Aggregate / per-stream game/wall | Full learner ms/update | GPU activity |
| --- | ---: | ---: | ---: | ---: |
| AB / parent | 7.310883 | 0.487392 / 0.060924 | 429.284 | 58.307% |
| AB / timer | 7.400747 | 0.493383 / 0.061673 | 424.261 | 60.871% |
| BA / timer | 7.326749 | 0.488450 / 0.061056 | 428.067 | 58.121% |
| BA / parent | 7.349580 | 0.489972 / 0.061246 | 426.384 | 57.164% |

Candidate/parent wall-time ratios are **0.987857 and 1.003116**; full learner-call
ratios are **0.988300 and 1.003949**. Keep both signs and the spread. No material
instrumentation overhead is visible in this bounded check, and the faster first
pair is not an instrumentation speedup. These are exact-recipe runtime controls,
not proof of super-real-time training. Mean power is 138.0–140.1 W; every job
peaks at 14,148 MiB VRAM, with at least 99% 1 Hz window coverage.

The two timer windows measure 160.318/160.713 ms per world update,
60.453/60.824 ms posterior stages and 154.008/155.493 ms imagination stages.
Contained within those stages are 78.747/79.178 ms of total readback waits,
29.705/30.280 ms of imagination input calls, 29.510/31.340 ms of target assembly,
9.730/9.859 ms of feature preparation and 9.481/8.929 ms of imagination readback
copying. All 387 reports in each timed job match the declared 64/31 readback
and 448/109 input-call counts and their payload sizes. Waits still include
producer work. Full learner calls account for approximately 79% of the window;
observation/perception takes about 21%, and environment stepping under 1%.

Construction takes 65.571/66.009/66.502/67.694 s in execution order. The first
learner report is at action 1,528 after 45.338/44.786/45.405/45.386 s from the
run header; that interval includes the first update, not pure prefill. Complete
execution including prefill takes 256.724/253.475/256.278/255.537 s, separately
from construction. Its approximately 12 actions/s must not replace the warmed
training figure: almost half of these short runs precede replay eligibility.

Every job retains approximately 1 Hz host process/fault/memory snapshots with
the latest complete observed action, in addition to the selected-device GPU
trace and unchanged runner's native timings. Host samples are not exact
action-boundary snapshots. The nearby host windows incur approximately
9.87–9.91 million minor faults each, zero major faults and zero sampled swap.
Their observed resident ranges span 7.74–8.14 GiB across processes; these are
host snapshots, not exact-boundary resource counters or allocation timings.
All collector tests and trace import/validation work finish before the first
declared measured window. No owned build, CPU-heavy test or other GPU workload
overlaps any measured window.

`analyze_pixel.py` revalidates complete accounting, tensor health, native timing
counts and device/clock coverage. `compare_pair.py` compares all events, reports
and named tensors. Per-job artifacts and both complete comparisons remain in
the same pixel directory. Comparison SHA-256 values are
`728ec30d4974e46ccc7a911347090ce86d93205c1999b495f0e37e88743a7dfe`
and `c2af2e0b99fef7b2c23ba4ce5a89a7bf4b7e61ee6bda988c3269935876ac3af7`.
The timer remains an isolated diagnostic; its validation does not adopt buffer
reuse or establish calibrated GPU idle gaps.

## Buffer-reuse hardware gate in progress

After both timer pixel pairs pass, the bounded reuse worker starts at
**03:21 UTC** under `runs/host-features-hardware-20260908.e41jkw/`.
Both exact hardware tests pass. The three fresh eight-update parent/candidate
pairs run next in the declared AB, BA, AB order, with the uninstrumented parent.
The worker retains all command-output hashes and whole-child CPU/fault counters;
those counters include construction, prefill and checkpointing. Warmed full-call
analysis and the separate reuse pixel gate are required before adoption.
