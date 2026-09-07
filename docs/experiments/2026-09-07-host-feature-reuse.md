# Reuse the imagined-feature host buffer

Isolated candidate `exp/host-feature-reuse`, based on `9322291`. Not adopted;
no candidate GPU test or gameplay run has executed. The active LeVJEPA campaign
and queued diagnostic worker keep their original inputs. The base Rust sources
and lockfile match the prebuilt uninstrumented canary parent at `73273df`.

## Small change and its cost

The original learner allocates a flat imagined-feature vector every update:
15 × 16 × 64 × 2,560 F32 values, or **150 MiB** at the current 12M recipe.
It copies the existing per-state feature vectors into that allocation, uploads
the result for behavior training, then drops it at the end of `learn`.

Five net production lines retain that vector in `DreamerCore`. Target assembly
takes the scratch vector, clears its length, reserves the required capacity and
fills it with the same copies at the same points in the loop. Completed learning
returns ownership to the core. There is no new unsafe code, API, GPU session,
graph, RNG draw, action, training credit or checkpoint field. Fresh construction
and restore start with an empty scratch vector. World-pretraining initialization
is a separate branch and is not part of this comparison.

This keeps **150 MiB of host capacity** between warmed updates. It does not
eliminate the separate per-state feature vectors or their copies, reduce GPU
memory, or establish a lower peak resident set. Measure the host-memory tradeoff.

## Evidence so far

The unchanged live seed-1 process incurred 3,072,080 minor faults in a 45.06 s
sample containing 82 learner reports. It had zero major faults and negligible
system memory pressure. Those counters suggest allocation work, not disk paging;
they do not locate it in this function or measure its elapsed cost.

A small CPU-only reproduction uses the same 150 MiB packing geometry, two
warmups and eight measured packs per fresh process, in three alternating pairs
(parent/reuse, reuse/parent, parent/reuse). All parent processes incur 307,211
minor faults; all reuse processes incur three. Parent packing/copy/drop takes
0.2215–0.2760 s per eight packs, versus 0.0725–0.1146 s with reuse. The latter
also verifies stable allocation address and all retained values. It reproduces
the allocation mechanism, not the native learner, GPU input upload or pixel loop.
Training continued, other host activity was uncontrolled, and release compilation
overlapped parts of the last three samples. Do not call these quiet-system timings
or extrapolate them into a gameplay speedup. Exact results, source and executable
fingerprints are in `runs/host-features-20260907/packing-cpu-results.json`.

Debug tests pass: 75 Kindle and five gym CPU tests. Release Kindle tests also
pass all 75 CPU cases; all 17 GPU tests remain ignored. Workspace/Python-library
Clippy and formatting pass. The existing ignored act/learn/restore hardware test
now additionally checks fresh scratch state, expected packed length and stable
allocation/capacity across successive updates after restore. It is not run yet.

## Hardware decision gate

1. Wait for the original campaign, checkpoint watcher and queued readback worker
   to finish; inspect their actual processes and artifacts, not just a lockfile.
   Interpret the queued host-stage instrumentation before attributing the live
   fault sample to a particular operation. No overlapping GPU jobs.
2. On idle adapter `0x2c02`, run this branch's exact ignored
   `dreamer::agent::tests::tiny_agent_completes_an_act_and_learn_cycle` and
   `dreamer::agent::vector::tests::vector_one_matches_serial_learning_and_checkpoint`
   in separate fresh processes. Require one passed test per invocation.
3. Run three fresh eight-update 12M/T64/B16 prediction-only parent/candidate
   canary pairs in AB, BA, AB order. Use the preserved uninstrumented parent,
   not the timing-instrumented candidate. Require complete reports with the
   existing canary auditor, unchanged non-timing reports, and exact named model/
   optimizer tensors plus metadata. Retain failures; do not silently retry.
4. Record process CPU/fault counters and warmed stage timings. The current
   `LearnTiming.total_seconds` is sampled before local batch destructors run;
   it is not full-call wall time. The CPU reproduction includes parent vector
   destruction, so its savings cannot simply be subtracted from that timer.
5. Before adoption, compare warmed **full-call/end-to-end** playing plus training
   under the unchanged N=8 pixel protocol and replay ratio. Include all updates,
   memory and construction/prefill separately, and verify transition/reward/reset
   ledgers and exact model reports. Faster standalone packing is insufficient.

No real-time, GPU-utilization, learning-quality or native speedup claim is made.
