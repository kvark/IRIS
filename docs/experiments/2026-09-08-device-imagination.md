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

Pixel comparison and current-profiler capture results are pending. The runtime
candidate is not adopted on synthetic evidence alone.
