# Native LeVJEPA and Pong mastery

Started 2026-09-06. This records the native implementation and interrupted
serial experiment, not the latest gameplay status. The subsequent
[vectorized Pong campaign](2026-09-06-vector-pong.md) has one passing seed and
two failures under its stronger mastery gate. The completed
[Boxing pilot](2026-09-08-atari-five.md) also wins with LeVJEPA, but has not yet
established three-seed reliability. Earlier DINOv3 results remain separate
controls; none of these follow-ups completes the interrupted serial protocol.

## Serial experiment interrupted for throughput work

At 2026-09-06 21:01 UTC the user explicitly prioritized vectorized environments
and batched inference. The serial launcher was stopped with SIGINT during seed 0;
its last complete logged learner record is action 64,390 / update 15,827. The
verified 60,000-action checkpoint and all logs remain intact. Unlogged in-flight
work is not counted as audited experience. No final frozen evaluation or other
training seeds ran; the predeclared mastery gate is **not achieved**. A model
restore is not an exact replay/RNG/environment continuation.

The 1 Hz GPU trace's final inspected 30-minute window contained 1,800 samples:
mean GPU activity 58.64% (range 4–99%), memory 8,217 MiB, mean power 117.90 W
on the RTX 5080 (360 W limit). These are sampled activity/power, not a measurement
of achieved FLOP efficiency. Serial coupled throughput is about 5.5 actions/s;
updates take about 0.54 s, dominated by imagination and world training.
The zero-update control remains useful. Vectorized collection will be a new,
explicitly identified experiment, not completion of this single-stream protocol.

## Frozen frontend contract

The first target implementation uses
[LeVJEPA-VideoMix-Large](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large),
snapshot `e831a0347737fcaa660b39c57d41c109de399845`, weights SHA-256
`da8bd836ce6532e1b0074ee5a6a46c65b67103f96323529ec4195be1538edc7d`.
The weights are separately licensed CC-BY-NC-4.0, not covered by Kindle's MIT
license. Downloaded reference code and weights stay outside the source tree.

The complete 303.1M-parameter, 24-layer ViT-L/16 is transcribed into Meganeura.
The production runtime does not import PyTorch or execute remote Python.
Every incoming RGB observation is letterboxed to 224 square and normalized
with ImageNet statistics. In Atari this follows the unchanged 64-square,
repeat-four/max-pool-two wrapper: one encoder slot per agent observation,
nominally 15 Hz, not the pretraining card's approximately 7.5 Hz clip sampling.

Perception uses nonoverlapping 16-arrival chunks, with an initially growing
prefix and GPU-resident per-layer keys and values. Attention is bidirectional
within a frame and causal across frames. Patches never attend the CLS sink,
so omitting its computation leaves patch features unchanged. The release's
specific three-axis RoPE and erf-GELU are preserved. The current frame's patch
features use the same fixed 64-channel projection and 2×2 spatial pooling as
the DINO control, giving a 7×7×64 observation.

Only perception resets at a chunk boundary; the RSSM, replay and learner keep
running. A real episode reset clears both visual history and live belief.
This is exact causal-prefix execution of bounded clips, **not** a sliding
window whose cached higher-layer features retain undeclared older context.
Recorded replay features are never re-encoded under a different history.

The precision diagnostic found that the original tanh-GELU helper caused
measurable reference drift. Correcting erf-GELU reduced scalar-F32 relative
L2 error to approximately 3.5e-6 or less across two synthetic 16-frame clips
and explicit resets. Reduced-input-precision cooperative matmuls exceeded
the initial 0.05 absolute-error bound, so the production LeVJEPA frontend
uses F32 operands. This changes neither the DINO frontend nor learner math.
Acceptance requires relative L2 <1e-4 and maximum absolute error <0.005
against the pinned F32 reference, including late-context frames and resets.

The first cache implementation required two small backend changes: multiple
queries may attend one valid KV prefix, and CacheWrite aliases must be
established before reshaped views receive buffers. A CPU-reference GPU test
covers multiple queries, grouped heads, non-tile-sized prefixes and stale-cache
exclusion after reset. Backend revision and all executable hashes must be
pinned before measured training.

## Pong acceptance, declared before training

This is stronger than the earlier initial-learning gate. Initial bounded
diagnostics and representation probes do not count as solving Pong.

- Train three fresh LeVJEPA agents, seeds 0, 1 and 2, for **200,000 executed
  actions each**, with explicit extrinsic rewards and no scripted/random-action
  coverage phase, demonstrations, intrinsic bonus or checkpoint selection.
- Use the existing prediction-only 12M recipe: B=16, T=BPTT=64, row microbatch
  16, train ratio 256, world and behavior learning rate 4e-5, 1,000-update
  warmup, AGC 0.3, reconstruction coefficient 0 and causal prediction 0.25.
  Any change after a failed bounded diagnostic is recorded before a fresh run;
  a changed recipe does not reinterpret earlier runs.
- Evaluate each declared final checkpoint for **75,000 frozen sampled actions**,
  with a fresh recurrent state, the matching evaluation environment seed and
  exactly zero learner updates. No greedy fallback or winning-checkpoint search.
- For **every seed**, require at least 20 complete natural games, mean return
  **at least +15**, and **at least 90% natural wins**. A win requires positive
  game return, terminated=true and truncated=false. Compute mean return and
  win fraction over **all completed games**, including timeouts. Report natural
  completions, unfinished tails and timeouts separately; timeouts are not wins.
- Keep the published Atari wrapper profile: full 18-action vocabulary, repeat
  four, no initial random no-ops, no sticky actions and 100,000-frame episode
  cap. Count actual emulator frames separately from nominal frame budgets.
- Retain a zero-update LeVJEPA policy control. Validate checkpoint tensors,
  fingerprints, update schedules, reward/action ledgers and all failures.
  Record construction and execution time, peak VRAM, and end-to-end simulated
  time/wall time with learning enabled.

If these gates fail, report the failure and design the next explicit experiment.
Do not lower the thresholds or relabel DINO success as LeVJEPA mastery.

## Diagnostics before the long run

1. Numerical parity on synthetic and real Pong clips, including native projected
   features, automatic chunk boundaries and explicit resets; perturb future
   reference frames to confirm earlier features remain unchanged.
2. Matched held-out position/motion probes against DINO and two-frame DINO
   history. Consume all arrivals, even unlabeled ones. Existing trained-RSSM
   results remain the stronger contextual baseline, not a matched new run.
3. A short coupled training/checkpoint/frozen-restore canary, measuring learner
   stages and perception latency/VRAM. Optimize measured bottlenecks before
   committing the full interaction budget.

Artifacts live under `runs/levjepa-pong-20260906/`. The reference fixture generator
is `python/examples/levjepa_reference.py`; it verifies all four downloaded
file hashes before importing the inspected local reference implementation.

## Initial diagnostic results

The native F32 port passes dense-token, projected-feature and pooled-feature
parity on both synthetic and actual Pong clips: two full chunks followed by
explicit resets, 37 observations per fixture. Independent RGB preprocessing
matches exactly. Perturbing future reference frames changes earlier features
by exactly zero. These are encoder checks, not policy-learning results.

Matched motion probes use 512 labeled samples per seed: seeds 10–11 train,
12 selects ridge penalties, and 13 is the held-out test. Both encoders consume
every arrival, including unlabeled frames, and reset at actual episode boundaries.
Test R² averaged over four displacement targets:

| Representation | DINOv3 | LeVJEPA |
| --- | ---: | ---: |
| Projected 14×14 patches | 0.270 | 0.618 |
| Pooled 7×7 observation | 0.247 | 0.634 |
| Two successive pooled observations | 0.676 | 0.728 |

The shared single-frame RGB baseline is −0.266. LeVJEPA carries useful temporal
information, but two-frame DINO remains competitive. The older trained-DINO
RSSM probe is a contextual baseline with different provenance, not part of this
matched comparison. Raw reports: `probe-{dino,levjepa}-motion.json`.

The corresponding held-out position probes favor DINO: projected/pooled mean
R² is 0.9915/0.9871 for DINO versus 0.9621/0.9612 for LeVJEPA. In the production
pooled grid, ball-x R² is 0.9642 versus 0.9100 and player-y R² is 0.9898 versus
0.9694. Thus the video frontend improves motion information at a cost in precise
current-frame localization; it is not uniformly better. Both retain strong
position information, which supports proceeding to the actual learning test.
Raw reports: `probe-{dino,levjepa}-position.json`, with the same split roles and
512 samples per seed as above. No probe labels enter agent training.

A 2,000-action seed-99 training canary on backend `0ff33a4` completed 229 updates,
with finite tensors and decreasing prediction/reward losses. Two completed
games averaged −20: no competence is claimed. Construction took 90.79 s,
execution 232.16 s, and learner updates averaged 0.5412 s. Observed VRAM was
8,217 MiB. Warmup inflates the aggregate 8.61 actions/s; steady coupled learning
was approximately 5.3 actions/s (about 0.35× the emulator's 60 Hz).

Its format-3 checkpoint restored the correct frontend automatically. A fresh
1,000-action sampled evaluation made zero updates, completed one −21 game and
ran at 18.39 actions/s (1.23× real time). Fast frozen play is therefore available;
super-real-time playing **plus training** is not established at this ratio.
The canary and frozen restore are preserved as `canary-{train,eval}.jsonl`.

Backend `35a410ce1262c396e137db5bfab1d58e35cee50a` reuses the existing tiled F32
attention generator for cached query blocks. Both reference fixtures still pass;
ordinary attention forward/gradient and cache regressions pass too. Mean encoder
latency across 32 Pong observations decreased from 52.10 to 46.91 ms, with the
slowest late-context frame decreasing from 79.96 to 69.98 ms. This bounded check
is a roughly 10% encoder speedup, not a claim of a 10% learner speedup.

The matched seed-99 2,000-action canary took 220.56 s instead of 232.16 s
(5.0% less execution time), with the same 229 updates, 7,994 actual emulator
frames and reward/action totals. Average learner time was 0.5382 s; construction
took 92.13 s. The frontend-only arithmetic change is not bitwise-identical
training: retain both checkpoints rather than implying exact tensor equality.
Raw artifacts use the `block-canary-` prefix. All nine native GPU learner tests,
including full-recurrence row-batch and causal-gradient checks, pass on this pin.
The optimized checkpoint also restores successfully: its 1,000-action frozen
sampled check performs zero updates at 20.56 actions/s (1.37× real time), with
one natural −21 game. This remains an integration check, not a mastery result.

`python/examples/audit_pong.py` audits this experiment's final-checkpoint gates,
not the historical scorer's 350k–400k-frame training window. The runner records
exact metadata/tensor hashes on save and restore. The audit binds frozen runs
to their final saves, checks uninterrupted seeds, implementation/config identity,
finite tensors and reports, complete reward/action/update ledgers, and the
ratio-256 warmup/cadence. Its zero-update control is a fresh seed-0 agent with
the identical recipe except train ratio zero, acting for 20,000 steps.

## Declared experiment launch

Source commit `60bc30b` was committed and pushed before launch. Backend pin:
`35a410ce1262c396e137db5bfab1d58e35cee50a`; Blade remains
`b208f3b1f97196c2971436b5726e61e71b149c37`. The actual CPython-3.14 extension
SHA-256 is `bd07b277bd379e6b5f24e8c111501903995aff9eaa0b85ee11ccf0c406216476`;
the Atari runner is `ea05a28053b92bc7d462ecd56b358d0a63e4eeb73b71fcece710c2323a8474fe`.
The launcher checks both identities before every job and refuses existing output
or checkpoint targets. DINO reference parity also passes on this backend.

The serial launcher is `runs/levjepa-pong-20260906/run-mastery.sh`; it started
at 2026-09-06 17:27 UTC with the 20k zero-update control, followed by training
and frozen evaluation for seeds 0, 1 and 2. Each 200k training run saves a rolling
recovery checkpoint every 20k actions, but only its declared 200k final state is
accepted. Recovery would start a different segment and fail the uninterrupted
gate; no old checkpoint is selected for scoring. `gpu.csv` samples utilization,
memory, power and clocks once per second. Report observed peak VRAM, not a
guarantee that sampling catches every transient allocation.

No full-budget learning result is available at launch. Passing preflight does
not complete the Pong mastery goal.

The zero-update control completed all 20,000 actions in 974.46 s: 79,980 actual
emulator frames, 22 natural games, mean return −20.4545, no wins or timeouts,
and exactly zero updates. It scored 12 positive points and conceded 464, with
a −2 unfinished tail. The reward/action ledgers and all three fingerprinted,
finite checkpoint files validate (`untrained-audit.json`). Frozen execution
was 1.3679× real time. Seed 0 training began at 2026-09-06 17:45:39 UTC.

At 20k actions, seed 0's rolling checkpoint has the expected 4,729 updates and
all 241 stored tensors are finite with matching save-event hashes. The full
metric prefix is finite and follows the exact update cadence. Gameplay has not
improved: 23 completed games average −20.7391, with no wins. The 15k–20k window
contains six −21 games and no positive points. Mean policy entropy is 0.5946;
the policy strongly favors one movement direction. Reward prediction now
distinguishes many negative events, but positive-event predictions remain poor.

For context, the older DINO causal seeds also struggled in the 15k–20k window:
seed 0 averaged −20.7143 and seed 1 −21, with policy entropy 1.0048/1.2213.
Those are historical trajectories, not new matched LeVJEPA controls, and do
not guarantee recovery. Continue the declared budget; do not reinterpret the
20k save as an accepted endpoint. Window reports and the checkpoint audit are
under the run directory, without changing the authoritative roadmap.

The 40k checkpoint also validates (9,729 updates, 241 finite stored tensors).
Through that exact boundary there are 43 completed games, mean −20.5814 and
no wins. The 35k–40k window averages −20.25 across four games. Positive-event
reward predictions now average +0.1312, versus −0.8186 on negative events and
−0.0039 on zero events. This is improving reward discrimination, not competent
control. Mean update time remains 0.5378 s. The original 200k/frozen-75k gate
and all training settings remain unchanged.

At 60k, the rolling checkpoint validates with 14,729 updates. Its 54 completed
games average −19.6667, with no wins or timeouts. The 55k–60k window contains
two −13 games, 20 positive points and 50 negative points; this is gradual
online improvement, not frozen-policy mastery. Positive/negative/zero reward
predictions average +0.4639/−0.8351/−0.0026 over their corresponding replay
samples. Mean update time is 0.5388 s. The small run-local
`inspect_checkpoint.py` checks rolling-save fingerprints, complete metric
prefixes, update cadence and exact recent windows without running another GPU
agent. The full-budget serial launcher was still active at this checkpoint;
it was later interrupted for vectorization as recorded above.
