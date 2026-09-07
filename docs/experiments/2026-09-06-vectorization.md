# Vectorized LeVJEPA collection and GPU throughput

Started 2026-09-06 after an explicit request to prioritize vectorized environments
and batched inference, even at the cost of a day of setup. This is one shared
policy/learner, not swarm learning or asynchronous actor/learner services.

## Baseline and decision

The original Atari runner used one environment and one-frame live inference.
Only replay training was batched (B16×T64). Its last inspected 30-minute GPU
trace averaged 58.64% activity, 8,217 MiB and 117.90 W on the RTX 5080. Activity
is a sampled busy fraction, not achieved FLOPs or occupancy; power below the
360 W limit alone does not prove inefficiency. Wall-time stages do establish a
large learner bottleneck: about 0.54 s/update and 5.5 executed actions/s.

The serial mastery queue was stopped with SIGINT at 21:01 UTC. Its last logged
learner record is action 64,390 / update 15,827; its verified 60k checkpoint and
logs are preserved. Unlogged in-flight work is not an audited endpoint. The
single-stream LeVJEPA mastery gate remains unachieved; details are in the
[original experiment](2026-09-06-levjepa-pong.md).

## Implemented collection contract

- One frozen LeVJEPA weight set. Dense layers process all stream rows together;
  attention currently dispatches independent stream blocks in the same session.
  Each stream has its own 16-arrival KV history and position. Missing arrivals
  do not advance it. A real reset affects only that stream; chunk boundaries
  still do not reset the RSSM.
- Batched live RSSM posterior and policy sessions. Each stream keeps its own
  belief, pending action and policy/posterior RNG. Stream seeds are model seed
  plus stream index (wrapping u64); environment seeds are separately recorded.
- One learner, optimizer, return normalizer and shared-capacity replay. Replay
  retains per-stream order, evicts globally by arrival age, drains fresh starts
  before uniformly sampling eligible starts across streams, and refreshes the
  correct stream's cached posterior context.
- Every executed action advances the global counter and, once warm, adds
  R/(B×T) credit. Initial/episode-reset frames do neither. Warmup counts eligible
  sequence starts across streams. The first eligible vector round gives one
  update; later rounds drain all due updates, without silently discarding debt.
- Python steps the independent ALE instances synchronously, submits their
  observations together, consumes terminal frames before resetting affected
  streams, and logs all actions/rewards/boundaries. CPU environment time is
  below 1% of measured wall time; subprocess environments are not needed yet.
- Checkpoints additionally record `collection_streams`. Model state can be
  restored into a fresh vector collector or a single-stream frozen evaluator;
  replay, exact RNGs, environments, live beliefs and visual caches are not saved.

Maintained entry points: `VectorDreamerAgent`, `kindle.VectorAgent`,
`python/examples/atari_vector.py`, `python/examples/profile_atari_vector.py`,
and the CPU-only `kindle._vector_audit` ledger checker. The original `Agent` and
Atari runner remain the serial/DINO control. The new vector frontend is LeVJEPA.

## Correctness before throughput

Initial native checks on the selected RTX 5080:

- Batched LeVJEPA matches serial dense and pooled features **bit-for-bit** over
  36 vector rounds with reversed input order, unequal reset times, missing
  arrivals and chunk boundaries. The existing pinned PyTorch Pong reference
  also passes (worst dense absolute error 0.00026131, relative L2 <1e-4).
- Three batched live beliefs and sampled policies match independent serial
  streams under different terminal/timeout/reset schedules.
- The N=1 vector learner matches serial scheduled losses and final world
  parameters exactly, including reset handling and checkpoint restore. The
  initial full-size N=1 benchmark's learner metrics also match the interrupted
  serial seed-0 prefix (world/behavior metrics, not wall-time fields).
- Replay tests cover uneven streams, global eviction and context refresh;
  scheduler tests count actions, not vector ticks or reset records. Python
  audits reject missing updates, wrong resets/rewards, and null/nonfinite losses.

## First matrix: same learner, batched collection

`runs/vector-20260906-first/`: N=1,2,4 executed serially, seed 0, 3,072 aggregate
actions each. Published Atari wrapper, full 12M recipe, B16×T64, full BPTT64,
row microbatch16, R256, learning rate4e-5, warmup1000, AGC0.3, reconstruction0,
future prediction0.25, extrinsic-only own sampled actions. All start fresh.
Timing uses the final 1,024 actions, after warmup: exactly 256 updates in each
window. Construction and prefill are excluded from that window and reported
separately. Brief CPU builds/checks also ran during parts of the exploratory
matrix; small differences need a quiet repeat, not an overprecise speed claim.

| Environments | Aggregate actions/s | Mean update, s | GPU activity | Mean W | Window peak MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.395 | 0.54395 | 58.38% | 117.43 | 8,249 |
| 2 | 5.853 | 0.54632 | 53.45% | 120.87 | 9,145 |
| 4 | 6.029 | 0.53499 | 55.81% | 123.30 | 10,708 |

Cold construction is 91.02 / 92.38 / 91.87 s. Full runs have 497 / 481 / 450
updates because each additional stream needs its own sequence context during
warmup. All three full accounting audits pass. The N=4 prefix completes no
games; this matrix is not a gameplay result or an amended mastery endpoint.

Native SHA-256:
`e99944abd4953bcf3ab6e41cd95c9e6bc95a12fbc49eef2e6bd9488f1b34c4d9`.
Runner SHA-256:
`340ed16d8a2d03653cf1afe47d5ef0d1abd7c0c0238834431df01a610bd61a1b`.
Meganeura remains `35a410ce1262c396e137db5bfab1d58e35cee50a`, Blade
`b208f3b1f97196c2971436b5726e61e71b149c37`. The native executable is archived
beside the matrix. Each run records the pinned encoder identity and wrapper hash.

Vectorization alone yields about 12% more aggregate actions/s. At N=2 the
1,024-action window spends 141.13 s in learning, 33.15 s in observation handling,
0.47 s stepping environments and 0.13 s selecting actions. The learner remains
about 81% of wall time. Aggregate speed must not be called per-stream speed:
at N=4, 6.03 total actions/s is roughly 1.51 actions/s per environment.

## Same-math learner cleanup

`runs/vector-20260906-cpu/` repeats N=2, B16 and the same 3,072-action protocol
after caching softmax exponentials, reusing categorical/decoder scratch and
submitting independent imagination heads before one combined readback/wait.
All **481 world and behavior reports match the original N=2 run bit-for-bit**;
the complete interaction/credit audit passes. Timing fields are excluded from
the numerical comparison.

The final 1,024 actions / 256 updates take 165.16 s: **6.200 actions/s**, mean
update 0.51189 s, GPU activity 58.58%, mean power 125.34 W and peak 9,145 MiB.
That is about 6% faster than the initial N=2 path, not a solution to the remaining
GPU bubbles. Cold construction takes 91.38 s. Native SHA-256 is
`8184d4db5a47c2ddd7ad9956a17ddf2463311834af70b8023d4a46d53fc7f761`;
the runner and dependency pins are unchanged. This executable is also archived.

The numerical/model GPU tests pass. In one combined-process run, the last two
readback tests failed while creating their GPU contexts (`NoSupportedDeviceFound`);
both passed in a fresh process. The cause is not diagnosed, so this is not an
unqualified clean full-suite result. Long training uses one shared context.

### Bounded CPU row parallelism

`runs/vector-20260906-parallel/` adds one local pool of at most eight CPU workers
for large independent categorical and scalar-decoder rows. Uniform draws are
generated in the original serial RNG order; each row retains its arithmetic
order. Small/live batches stay serial. This is not actor/learner separation.

All 481 world/behavior reports and every transition, episode and reset agree
exactly with the preceding N=2 run. The accounting audit passes. The final
1,024-action / 256-update window takes **157.54 s, 6.500 actions/s**, with a
0.47898 s mean update, 61.18% GPU activity, 129.87 W and 9,145 MiB peak VRAM.
Cold construction takes 91.60 s. This is about 20% faster than the initial N=1
path; at N=2 it is 3.25 actions/s per environment, still below real-time learning.

Native SHA-256:
`9d6290a3c4b39f826771e1020976d2be81a5b9e6e3a1a80577ee5a285efac38b`.
Runner SHA-256 (now records CPU worker count):
`ee636253c5c90eb8c7902752660c14ee7b077e025e27a17c982b71e97be9c373`.
The executable and runner are archived; dependency pins and scientific recipe
are unchanged.

## Larger replay batch: separate optimizer-cadence experiment

`runs/vector-20260906-b32/` keeps N=2, T64/full BPTT, R256 and the same learning
rate recipe, but doubles B and its row microbatch to 32. It runs 4,096 aggregate
actions so the measured final 1,024 actions are all after the larger warmup.
The window contains exactly **128 updates**, not 256: equal total replay samples
do not imply equal optimizer trajectories, warmup in interactions or learning
quality. This is not a numerics-preserving systems comparison.

The window takes 126.10 s: **8.121 actions/s**, 0.69326 s/update, GPU activity
53.82%, 135.17 W and 10,416 MiB peak. Cold construction is 93.93 s. The full
accounting audit passes. Native/runner hashes match the CPU-parallel B16 run.
The lower busy fraction despite higher useful throughput is another reason not
to optimize the utilization percentage alone. Keep B16 as the learning control
until a predeclared quality comparison validates a larger batch.

`runs/vector-20260906-b64/` tests B64/full row64 at N=2 and N=4, with 6,144
aggregate actions planned per job. N=2 completes and passes the accounting
audit (121 updates over the full run). Its final 1,024 actions / 64 updates
take **129.50 s, 7.907 actions/s**, mean update 1.45014 s, GPU activity 54.95%,
126.66 W and 12,999 MiB peak. This does not beat B32. Synchronization grows to
roughly 0.12 s world / 0.09 s behavior per update; that observation is not yet a
diagnosis of the cause.

N=4/B64 is **rejected during construction**, before a run header or any training,
by Meganeura's memory-budget guard: 14.91 GB already allocated, another 0.07 GB
requested, and 10% of the 16.57 GB reported device budget reserved for the driver
and pipelines. Do not disable that reserve or report this job as a timing result.
The raw failure is retained in `n4.log`; the original matrix command exits
nonzero and has no complete matrix summary. The profiler now retains explicit
failed-job summaries before returning a nonzero exit status.

B64 native SHA-256 (metadata-getter/API cleanup only since the B32 build):
`86fe65da46c99698a20f471e925b712f0e0689ced6a462085ce0d0b902484c73`.
Runner SHA-256 (rejects zero row-microbatch arguments):
`54ed3e3d91fa098d2257370fdf170d7a2fd26eb407d53a15ee8b63a6a4e11ee6`.

## Compensated tensor-core candidate: not adopted

The production pin remains Meganeura `35a410c`; production LeVJEPA remains F32.
An isolated backend candidate at `bc3a1e41303c046a1126c56cc9048d02eeade6b0`
adds an explicit, opt-in `CompensatedF16` policy. It splits bounded inference
operands into high and scaled-low components, uses three f16-input/f32-accumulator
matrix products, and leaves the sensitive-gradient F32 guard and default `Auto`
policy intact. This is not full-f32 exponent range and not permission to use
raw f16 gradients.

The first cast-roundtrip split produced **exactly the raw f16 result**, with
no residual correction; scaling that residual did not help. Explicit f32
mantissa masking makes the correction effective. Normal/transposed matrix
products and fused tanh epilogues then pass: worst tested plain-matrix absolute
error drops from 0.00155342 to 0.00000715. Both hardware tests and all 218 backend
CPU tests pass, including the tiny-gradient guard. No system profiling/security
settings were changed when `perf` access was denied.

Kindle's isolated experiment is preserved at `70a7ce76bd2be8d27e3b671acdaec84ca735bce1`
on `exp/levjepa-compensated`, not integrated into the production branch. It uses
an explicitly different encoding identity and actual dependency pins. The full
37-observation pinned Pong reference passes, including automatic chunk boundaries,
explicit resets and projected/pooled features: worst dense absolute error
0.00038910 and relative L2 0.00001054 (gates 0.005 and 0.0001).

A matched two-stream diagnostic processes 64 synthetic 160×210 RGB frames per
stream, with unequal resets, excluding the first chunk from timing. Both paths
use the same unoptimized Rust test build and GPU; dense diagnostic readback is
outside the timed encoding call. F32 averages **96.459 ms/vector call** and the
candidate **95.689 ms**, with 97 genuinely compensated dispatches. Maximum dense
disagreement is 0.00019646, relative L2 0.00000562. The <1% mean improvement does
not demonstrate a material frontend win and is not a production throughput claim.
Do not change the production encoding or rerun a long learning suite for this
candidate; keep the smaller, measured F32 control. A different kernel would need
a new latency/accuracy comparison.

## Pixel checkpoint/restore canary

`runs/vector-20260906-canary/` starts a fresh seed-99, N=2/B16 agent with the
production F32 frontend. It completes 2,000 aggregate own-action transitions,
213 scheduled updates and two natural games (both −21). Its 166.46 s execution
time includes replay prefill, so **12.02 aggregate actions/s is not a steady
learning speed claim**. The complete accounting and real-frame clocks pass.

`checkpoints/levjepa-vector-canary-20260906` records format 3, the exact encoder
hash, `collection_streams: 2`, action 2,000 and update 213. Restoring this shared
model into N=1 frozen sampled evaluation completes 1,000 further actions with
exactly zero updates; its one completed game scores −19, not mastery. The frozen
run is bound to the final vector save by equal metadata/tensor fingerprints,
and its full audit passes. It takes 48.43 s excluding construction (20.65
actions/s, 1.376× the game clock), which establishes frozen play only. Replay,
live belief, visual history and RNG continuation remain fresh on restore.

These are integration checks, not a continuation or replacement endpoint for
the interrupted single-stream LeVJEPA mastery experiment. No long learning
queue was restarted by these canaries.

## Temporal replay encoding and heads

The next candidate batches the trainable observation encoder and independent
prediction/reconstruction, reward, continuation and frozen replay-value heads
over replay time. The RSSM and posterior stay sequential, with unchanged hard
posterior samples, reset masks and full BPTT. Balanced concatenation/split trees
keep time-major rows separate and avoid quadratic copying. Per-row RMS norms
do not introduce future information. Parameter names, shapes, initialization,
optimizer cadence and loss coefficients are unchanged; no checkpoint/head or
frontend identity is reinterpreted.

`runs/head-batching-20260906/` preserves the old full-size executable, an initial
rejected candidate, the corrected executable and all logs/checkpoints. The
synthetic 12M B16/T64/R256 prediction-only control runs eight updates. The first
candidate exposes a backend BCE limitation: above 256 rows its nominal scalar
contains workgroup partials. Continuation loss incorrectly reads 0.1641 instead
of 0.6328. It is **rejected**, not a speed/learning result. Explicit weighted
scalar-sized BCE reductions now cover every row. CPU-reference GPU tests at
513 and 1,024 rows verify both the composed loss and all logits gradients.

Corrected synthetic means over updates 3–8: world training **0.20331 → 0.15685 s**
(22.9% less), total measured learner **0.48735 → 0.42870 s** (12.0% less).
Allocated core device memory is **6.265 → 6.416 GB**. Changes in unchanged
posterior/imagination stages also contribute to the total timing difference;
the following real-game matrix is the stronger end-to-end measurement.

This is not bitwise-identical math. The largest loss differences across eight
updates are 0.00284 total, 0.0099 prediction, 0.000384 raw KL, and 0.00003024
continuation. All checkpoint tensors are finite with equal names/shapes. World
parameter max absolute difference is 7.69e-7 (relative L2 1.11e-7); behavior is
6.41e-7 (1.16e-7). Normalized LaProp momentum is more sensitive: world/behavior
aggregate relative L2 differences are 0.002135/0.003003. Preserve both results;
later categorical trajectories can diverge.

Direct no-optimizer gradient comparisons cover asymmetric resets, non-power-of-
two T=3, T=4, prediction-only and reconstruction, free-nats floors and frozen
replay-value input-gradient settings. Worst per-parameter relative L2 is 1.44e-6.
The production B16/T64/12M comparison also passes: worst 0.00074572, within its
pre-run 0.003 per-parameter tolerance plus 1e-7 absolute L2 (loss tolerance
0.0003 × max(1, |reference|)). Larger forward shapes may select different
kernels; sensitive derivative operands remain F32. Run the large check with
`KINDLE_FULL_WORLD_PARITY=1` and the ignored
`temporal_batching_matches_serial_losses_and_gradients` test.
Existing causal-gradient/reset, reconstruction/predictive row-microbatch, and
N=1 vector-learning/checkpoint GPU regressions pass as well.

Corrected native SHA-256:
`f663dd9317bd934f173b240041ea68b7e21e0f7d037ebaf22b9549a9ae91bb4e`.
The runner and wrapper remain unchanged. `runs/vector-20260906-temporal/`
measures N=2/4/8 at the fixed B16 recipe, including complete GPU traces and
accounting. A [new vectorized Pong protocol](2026-09-06-vector-pong.md) preserves
the three-seed mastery gate; it is not completion of the interrupted serial run.

The real-game matrix completes with all action/reward/update ledgers valid:

| Streams | Actions/s | Update seconds | GPU activity | Power W | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 7.0963 | 0.42606 | 55.63% | 134.18 | 9,295 |
| 4 | 7.1422 | 0.42802 | 59.58% | 134.81 | 10,954 |
| 8 | 7.3663 | 0.42274 | 58.51% | 139.54 | 14,148 |

Each final window contains 1,024 actions and 256 updates. Cold construction
takes 65.31/66.35/65.90 s, down from roughly 93 s with the serial-head graph.
The complete 3,072-action jobs perform 481/450/387 updates because independent
streams have different replay-prefill costs. N=2 completes two −21 games;
N=4/8 complete none. These are throughput runs, not competence results.

N=2 improves 9.2% over the preceding CPU-parallel control at unchanged B16/R256.
N=8 improves 36.5% over the original N=1 run (5.3948 actions/s); that combined
comparison includes collection and systems changes, not identical trajectories.
More streams are a small marginal win: N=8 is only 3.14% faster than N=4.
The predeclared rule chooses the fewest streams within 3% of the measured best;
N=4 narrowly misses that cutoff, so **N=8** is selected. This single-window
selection is not a claim of statistically established superiority.

GPU activity remains about 56–60%, not full utilization. At N=2, the measured
window spends 109.93 s in learning, 33.61 s observing and only 0.54 s stepping
environments. CPU environment parallelism is not the present priority. Aggregate
coupled game-clock speed is about 0.49× at N=8, and each stream advances about
0.061×: **super-real-time training is still not achieved**. The memory reserve
remains enabled. The common native binary, runner/wrapper and all raw reports
are archived.

The corrected native executable also restores the earlier 2,000-action pixel
canary checkpoint into N=1. A 64-action frozen smoke check has zero updates,
an unchanged restored checkpoint identity and **all 64 transitions identical**
to the previous executable's evaluation prefix (excluding timestamps). Its
complete accounting passes. This checks checkpoint compatibility, not new
training trajectory equivalence or Pong mastery.

## Remaining handoffs: read-only inventory

The live N=8/B16/T64/12M run's 90k–100k window contains 2,500 updates.
Mean measured update time is 0.42489 s: posterior sampling 0.06027 s,
imagination/targets 0.15578 s, world training 0.16182 s, behavior training
0.02593 s, and world/behavior synchronization 0.01598/0.00335 s.
The full loop remains 7.244 actions/s; the environment adapter takes only
4.33 s of the 1,380.52 s window. These timings do not distinguish GPU compute,
host work and exposed wait latency inside each stage.

`DreamerCore::sample_posterior_batch` and `imagine_and_target` give the following
**source-derived payload counts per update**, not measured bus traffic:

| Forward stage | Explicit readback waits | Readback payload, MiB | `set_input` payload, MiB |
| --- | ---: | ---: | ---: |
| Replay posterior, 64 × 16 rows | 64 | 10.00 | 32.39 |
| Imagination, 1,024 starts × 15 transitions | 31 | 199.00 | 631.05 |

The 12M state has 2,048 deterministic and 512 stochastic values. Each of the
16 imagination states sends its 2,560-value feature separately to online and
slow behavior heads, plus equivalent state inputs to world heads. Fifteen
transition calls then send the state again. Each state reads action logits,
three 255-bin predictions and continuation; each transition reads deterministic
state and prior logits. The 31 waits are 16 combined-head reads plus 15
transition reads; they are not 31 measured idle gaps. Posterior sampling has
one read per replay time step.

Inputs use Meganeura's host-visible writes; readback uses a transfer and wait.
This inventory excludes parameter synchronization, training inputs/outputs,
live collection/vision and additional CPU allocations/copies. It must not be
turned into a claim about PCIe bandwidth or time saved. Exact shapes, source
hashes, formulas' results and measured stage means are retained in
`runs/levjepa-vector-pong-20260906/seed0-transfer-inventory-100000.json`.

Even deleting both entire measured posterior and imagination stages at zero
replacement cost would yield only 11.90 aggregate actions/s if other wall time
stayed fixed. That deliberately optimistic bound is below the 15 actions/s
needed for even aggregate 1× Pong at repeat four. Removing these round trips
alone therefore cannot justify promising super-real-time learning.

After the pinned queue, instrument packing, host-visible writes, readback waits,
CPU sampling/decoding and GPU compute separately. The first copy-elimination
candidate should keep deterministic state/features on the GPU while preserving
CPU draw order and categorical/return arithmetic. Moving sampling or two-hot
decoding onto the GPU is a separate numerical change requiring distribution,
gradient and end-to-end checks, not a free optimization. Preserve full recurrence,
replay credit, independent streams and the memory reserve. No live experiment
binary or setting was changed for this inspection.

The host/readback timer candidate is staged separately at `ec074a5`
on `exp/learner-readback-profile`, based on `73273df`. It counts nonempty
readbacks and requested bytes and separates preparation, submission, completion
wait and CPU output copying inside the existing posterior/imagination totals.
It also counts successful host input writes and their f32 payload and times
the `set_input` calls, preserving their order relative to producer submission.
At B16/T64/12M the expected input counts are 448 posterior and 109 imagination
writes per update; measured counters must confirm the source inventory.
Neither host write time nor requested bytes measures PCIe utilization.
Separate host-section timers cover posterior sampling, imagination feature
assembly, two-hot decoding, action/prior sampling and post-rollout targets.
They include the 150 MiB imagined-feature buffer's assembly at this shape;
its size is source-derived, not a measured allocation cost. All substage timers
remain contained in the existing parent totals.
The candidate adds no GPU work or waits. CPU tests (78 Kindle, five gym), workspace/Python
Clippy and formatting pass; hardware tests, numerical parity and instrumentation
overhead remain unmeasured until the active queue exits. It is not part of the
current experiment's executable. Its bounded validation plan is
[`2026-09-07-readback-profile.md`](https://github.com/kvark/kindle/blob/ec074a5/docs/experiments/2026-09-07-readback-profile.md)
on that branch. A blocked host is not necessarily an idle GPU.

The timestamp contract was checked against pinned Meganeura `35a410c` and
Blade `b208f3b`. Blade resolves pass durations when a command buffer is reused;
the public timing data does not retain absolute GPU timestamps.
Meganeura's `profiler::record_gpu_passes` lays these durations consecutively
from a CPU submission offset. Its exported GPU track is therefore **synthetic
placement, not a calibrated queue timeline**. Do not interpret drawn gaps or
overlaps as measured GPU idleness or concurrency. Normal pass durations include
barrier/dispatch cost; per-dispatch instrumentation changes that cost and needs
its own overhead comparison. `DreamerCore::profile_sessions` also excludes
optimizer/clipping/accumulation and profiles only the final row microbatch's
fixed inputs, not the complete learning update. Keep unexplained wall time
unattributed until measured; actual cross-submission GPU gaps require retained
GPU start/end timestamps or a suitable external timeline capture.

Read-only seed-1 host checks on September 7 found about 117 MiB in swap despite
roughly 19 GiB available RAM. Over 30.07 s, the trainer had zero major faults,
unchanged swap residency and no increase in memory-pressure stall totals.
A separate five-second process sample averaged 69,122 minor faults/s and 4.8%
system CPU relative to one logical CPU. Minor faults do not require disk reads;
neither sample identifies the responsible allocation or establishes hot paging.
The source-derived feature-buffer size is a profiling lead, not causal proof.
No memory policy or live learner setting was changed. The observations are in
`runs/levjepa-vector-pong-20260906/seed1-paging-20260907-1006.json` and
`seed1-host-cpu-20260907-1013.json` beside it.

## Next measured changes

Current CPU checks: 75 Kindle and five gym tests pass, with 17 hardware tests
explicitly skipped by the ordinary command; 150 Python tests pass. Rust/Python
Clippy and formatting checks pass. The vector auditor also verifies reported
aggregate/per-stream game clocks and rejects invalid throughput accounting.
Later API-only cleanup exposes learner metadata directly instead of returning
an unused single-stream diagnostic core; the measured binaries remain archived.

1. Keep B16 as the learning control. B32 has a throughput benefit but needs a
   predeclared learning-quality comparison at matched aggregate interactions,
   with its changed optimizer cadence and warmup reported. B64 is not the next
   default: it did not beat B32 at N=2 and failed the memory guard at N=4.
2. Temporal-head timing and numerical checks are complete. Later systems work
   can target grouped RSSM/layout and host/GPU transfers. The compensated frontend did
   not demonstrate a material speedup. Never re-enable reduced-exponent-range
   kernels for sensitive learner gradients.
3. Only after throughput and accounting checks, predeclare fresh vectorized
   Pong training seeds, aggregate interaction budgets and final frozen gates.
   Preserve failed/interrupted runs and do not select winning checkpoints.
