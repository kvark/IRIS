# Pong reliability and device-resident imagination

Follow-up to the immutable
[protocol declaration](2026-09-08-stability-and-device-residency.md).
The original three-seed mastery decision remains **one pass, two failures**.
Pong is our demonstrated Atari learning result, not evidence of Atari breadth.

## What the seed differences actually show

Every model trained for 200k aggregate actions with N8/shared learner, LeVJEPA,
B16/T64/full BPTT, row batch 16 and replay ratio 256. Every final model received
the same 75k-action frozen sampled evaluation, with no learner updates.

| Seed | Frozen wins / games | Frozen mean | First training win at action | Positive points in first 80k actions |
| --- | ---: | ---: | ---: | ---: |
| 0 | 18 / 18 | +10.2778 | 160,344 | 87 |
| 1 | 7 / 12 | +0.5 | 193,232 | 12 |
| 2 | 43 / 43 | +20.4651 | 86,352 | 76 |

Seed 0 misses the mastery count/score bar despite winning every completed game.
Seed 1 first wins just before its training budget ends; this is not evidence
that a previously mastered policy collapsed. The fixed 40k-action training
windows show very different learning speeds. In the last window, sample-weighted
positive reward predictions average +0.945 / +0.795 / +0.990. These are replay
predictions, not held-out reward calibration or explanations of causation.
Sparse positive experience is a concrete lead, not a diagnosed optimizer fault.

The retrospective analysis preserves all logs and five fixed windows in
`runs/seed-stability-20260908.XbiwtA/training-summary.json`. Windows group games
by completion; their scores can include earlier play. Reward-event counts cover
only interactions in the window; replay sample counts include repeated history.

### Completed held-out motion probes

All three frozen models consumed the same forced-random trajectories, every
observation arrival, and 512 labeled displacements in each of four environment
seeds. Two environments fit the ridge probes, one selects the penalty per
target, and the fourth tests it. No labels enter the agent; no agent learns.
All four visual-control results are exactly identical across model seeds.

| Representation | Ball horizontal motion R² | Ball vertical motion R² | Player motion R² | Opponent motion R² | Mean R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pooled LeVJEPA | 0.0984 | 0.7767 | 0.4218 | 0.6851 | 0.4955 |
| Pooled causal history | 0.0959 | 0.8497 | 0.5964 | 0.7577 | 0.5749 |
| Trained RSSM / seed 0 | 0.7470 | 0.8208 | 0.8033 | 0.6551 | 0.7566 |
| Trained RSSM / seed 1 | 0.7780 | 0.8290 | 0.8485 | 0.6626 | 0.7795 |
| Trained RSSM / seed 2 | 0.4327 | 0.8102 | 0.8198 | 0.6365 | 0.6748 |

Every belief contains useful motion information under this probe. The best
player has the weakest mean linear probe, so this does not explain the gameplay
ranking. Forced-random coverage differs from a successful policy's state
distribution, and linear decodability is not control use or dynamics accuracy.
Do not infer a need for a larger grid or longer visual history from these data.

The runs completed 04:08–04:38 UTC on the selected RTX 5080. Source, encoder,
native extension and checkpoint hashes, per-target errors, all results and the
zero-update checks are in `runs/seed-stability-20260908.XbiwtA/`; the compact
cross-seed audit is `motion-summary.json`. CPU builds overlapped these diagnostic
runs; no timing or new gameplay claim comes from them.

Next diagnose held-out reward/value predictions and how actions use the learned
belief. Keep positive, negative and zero reward errors separate, with a
zero-predictor control; average sparse-reward error alone can be misleading.
Choose one bounded intervention from that evidence. If testing a longer learning
budget, declare it for every seed as a new experiment; do not extend only the
failed seeds or rewrite the original mastery decision.

## Runtime implementation

Candidate `4feebdf7c6af13be17502d5b9652d6c872edaead` is isolated in
`/x/Code/.kindle-device-imagination`, from the adopted buffer-reuse source
`b14c32b`. Meganeura, Blade, categorical draws, value decoding, lambda returns,
recurrence, update credit and all scientific settings are unchanged.

The imagination head exposes its concatenated state as a pinned graph output.
Same-context GPU copies feed actor/value sessions, pass deterministic state to
the next transition, and pack each time slice directly into the existing behavior
training input. CPU sampling still consumes the same logits and RNG draws in
the same order. Only the first CPU feature is retained for replay targets.
The old 150 MiB host scratch buffer and per-time CPU feature list are removed.

The private copy helper checks context identity, slot bounds, offset overflow,
alignment and cross-session use. Its command encoder is recycled only after
completion. Copies precede consumers on the shared queue; existing readbacks
complete them before subsequent CPU input writes. It adds no unsafe host access,
OS-handle interop, sampler kernel, new GPU feature-history allocation or backend
worktree edit. Parameter synchronization remains unchanged.

Logical tensor payloads at the declared B16/T64/H15/12M shape:

| Work per learner update | Reuse parent | Device candidate |
| --- | ---: | ---: |
| Imagination CPU input writes | 631.05 MiB / 109 calls | 41.05 MiB / 32 calls |
| Imagination output readbacks | 199 MiB / 31 calls | 79 MiB / 31 calls |
| Behavior imagined-feature CPU upload | 150 MiB | 0 |
| New same-device copies | 0 | 740 MiB / 31 submissions |

These are source-derived payload counts, not hardware PCIe counters. The
remaining 64 posterior and 31 imagination readbacks still synchronize the host.
A readback wait includes producer computation, not just GPU idle time.

## Completed CPU and synthetic gates

- 80 Rust workspace tests and 229 Python tests pass, using the actual isolated
  native extension. Workspace/Python Clippy and formatting pass.
- Three focused GPU tests pass: offset packing/invalid-range rejection;
  act/learn/checkpoint restore; vector-one versus serial learning/restore.
- Three complete eight-update pairs in AB, BA, AB order match every non-timing
  report, all 241 named parameter/optimizer tensors, optimizer metadata and
  checkpoint metadata. Warmed full-call candidate/parent ratios are
  **0.805073 / 0.795658 / 0.798597**: 19.5–20.4% less time.

These synthetic calls include report serialization/output, unlike the pixel
runner's outer learning timer. They do not establish Atari speed or quality.
Reports and hashes are in `runs/device-imagination-hardware-20260908.wtHyHF/`.
The earlier `...Ni3yG7/` attempt passed two hardware tests, then stopped on its
immediate post-test GPU-idle guard. It is retained intact. The fresh controller
requires three quiet samples within a bounded 30-second cooldown and never
stops unrelated processes. The pixel collector and shared helpers pass 11 CPU
tests, including busy-device, identity, incomplete-gate and cleanup checks.

## Completed pixel comparison and adoption

All four jobs completed 04:50–05:11 UTC, serialized without another GPU job or
CPU-heavy build. Each uses a fresh seed-0 model and exactly 3,072 real actions,
N8/12M/B16/T64/full BPTT/row16/R256, with 387 updates and zero debt. The warmed
window is the final 1,024 actions / 256 updates. The parent is buffer reuse;
the only change is device-resident imagination. Both AB and BA pairs match
every action/reward/reset event, every non-timing report, all 241 named tensors
and optimizer/checkpoint metadata. No natural games finish: this is runtime
parity, not another learning-quality result.

| Pair / arm | Aggregate actions/s | Aggregate real time | Full learner ms/update | GPU activity | Peak VRAM MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| AB / reuse | 7.5994 | 0.5066× | 408.304 | 63.21% | 14,148 |
| AB / device | 8.6084 | 0.5739× | 348.120 | 68.74% | 14,212 |
| BA / device | 8.5799 | 0.5720× | 348.005 | 68.03% | 14,212 |
| BA / reuse | 7.5999 | 0.5067× | 408.296 | 62.30% | 14,148 |

Throughput improves **12.9–13.3%**, and full learner time falls **14.74–14.77%**.
Per-stream real time is only 0.0715–0.0717×. Selected GPU memory is 16,303 MiB;
the additional 64 MiB leaves 2,091 MiB, preserving the 2 GiB reserve. Mean power
rises from 143.2–144.0 W to 155.9–156.2 W. Cold construction takes 65.9–66.4 s
and is recorded separately from warmed throughput. All warmed windows have
zero major faults and zero process swap. Host RSS varies; source-level removal
of feature vectors is not evidence of an equivalent whole-process RSS saving.

The code is adopted as `f017830c5eed240fe34c5c4023e484ed33698cb1`, with source
and dependency locks identical to the tested candidate. The original 17 control
pins and every candidate pin remain intact. Complete reports, telemetry,
comparisons and the post-gate `source-adoption.json` are in
`runs/device-imagination-pixel-20260908.SsprHW/`. Earlier stage manifests retain
their original predecision `adopted: false`; the adoption record does not rewrite
their history or change the failed Pong mastery decision.

### Remaining cost and next target

Mean native stages in the pixel windows:

| Stage | Reuse ms/update | Device ms/update |
| --- | ---: | ---: |
| World training | 162.8–162.9 | 163.5–163.7 |
| Imagination | 138.4 | 84.7 |
| Posterior inference | 60.5 | 60.3–60.6 |
| Behavior training | 26.2 | 18.7 |
| World parameter sync | 15.3–15.4 | 15.6–15.7 |
| Behavior parameter sync | 3.3 | 3.3–3.4 |

Other work costs 29.1–29.6 ms per aggregate action, chiefly observation/perception;
emulation is under 1% of wall time. At R256 and B16×T64, every four real actions
require a learner update. Holding other cost fixed, aggregate 1× needs a full
update no longer than 148–150 ms. World training alone already exceeds that.
This is a high-ratio recurrent learning workload on top of a large frozen video
encoder, not an Atari-rendering bottleneck.

Next measure and reduce world-training kernel/layout cost and recurrent
handoffs, then perception. Keep full BPTT, sampling, update credit and memory
headroom unchanged in systems comparisons. Parameter synchronization also
refreshes backend-derived weights; a raw parameter-buffer copy is not sufficient.
Lower replay ratios or larger learner batches require separate learning-quality
comparisons. No concurrent actor/learner service is introduced.

### External profiler: recovered queue coverage, not kernel attribution

The installed Nsight Systems 2023.4.4 captures lacked GPU workloads. A fresh
official 2026.4.1.191 CLI was downloaded and extracted locally, without system
installation, driver updates or security changes. Import and SQLite export pass.
Both eight-update batch-mode captures preserve exact reports/tensors but each
contains only **one** GPU workload: insufficient against 2,111 parent and 2,359
candidate queue submissions. The preliminary collector incorrectly accepted
nonempty workload output; `coverage-validation.json` rejects that decision
without rewriting the raw captures or their original result.

One bounded follow-up changes only `--vulkan-gpu-workload=batch` to `individual`;
the candidate follows after the parent recovers coverage. These captures contain
2,111 / 2,359 GPU records, respectively, with exactly one correlated record per
queue submission across the full run. Both still match all eight untraced
reports and all 241 tensors. No other heavy workload overlaps capture.

Despite the option's name, all returned GPU records are `vkQueueSubmit`, not
individual dispatches. The [NVIDIA manual](https://docs.nvidia.com/nsight-systems/UserGuide/)
also warns that compute/transfer queue command-buffer endpoints may appear
early. Thus queue coverage is recovered, but exact idle gaps, per-kernel
attribution and occupancy are not established. No traced timing is used for
the speedup claim. This is a useful capture path for further diagnosis, not
evidence that the entire 31–32% coarse activity deficit has been classified.

The profiler, download hash, batch captures and overriding coverage audit are
in `runs/nsight-current-20260908.YLoIKA/`; individual-mode parent/candidate
captures are in `runs/nsight-individual-20260908.8xD1MA/` and
`runs/nsight-device-individual-20260908.ND6z7m/`. The official
[download page](https://developer.nvidia.com/nsight-systems/get-started), exact
download URL, CLI/native hashes and capture commands are retained in manifests.
All captures completed; no profiler or diagnostic worker remains queued.

### Use the tested build without replacing controls

The tested CPython 3.14 package is
`runs/device-imagination-python-20260908.UvQQjL/package`; its native extension
has SHA-256 `2b5bc9d1c26630896de8ecde5ed70cb8fd87fb024f59e46e415fbead6691d08a`.
It was built with `maturin build --release --locked`, separate Cargo output,
two build jobs and an isolated `uv pip install --no-deps --target` destination.
The wheel hash is `76b941da6dafb6ea64f850533a99dc904777bfb6ae81e3f1ad8a68e5921fb076`.

For new local experiments, explicitly select that package and verify its import:

```bash
PYTHONPATH=/x/Code/kindle/runs/device-imagination-python-20260908.UvQQjL/package \
  python/.venv/bin/python -c 'import kindle._native as native; print(native.__file__)'
```

Use the same `PYTHONPATH` with a fresh runner output/checkpoint. Alternatively,
build current source into another fresh package. The default editable extension
remains the original `f663dd93…` binary; do not silently replace it or any
historical runner, auditor, checkpoint or isolated control executable.
