# Seed stability and device-resident imagination

Declared after the completed Pong and buffer-reuse experiments, before these
new GPU runs. This does not extend the failed three-seed mastery campaign.

## Questions

1. Does each final seed's recurrent belief retain control-relevant motion cues,
   relative to the identical frozen visual features and a two-frame history?
2. Can device-to-device state/feature handoffs remove measured imagination
   transfers without changing draws, arithmetic, recurrence or training credit?

Pong is our only demonstrated Atari learning result, not Atari breadth. Seed 0
wins all 18 frozen games but misses the count/score bar; seed 1 wins 7/12; seed 2
wins 43/43. These are different failure severities, not two totally nonlearning
seeds. Their root causes remain unestablished. Keep all final endpoints.

## Frozen motion diagnostic

Use the unchanged `probe_atari_perception.py`, LeVJEPA and each archived final
200k-action checkpoint. Run all three seeds in order 0, 1, 2, with zero learner
updates. For each model use four independently initialized environments with
seeds 100/101/102/103 and 512 labeled displacement samples per environment.
The first two environments fit ridge probes, the third chooses penalties and
the fourth is held out. Use the published Pong wrapper, real executed random
actions and every observation arrival. Object labels never enter the agent.

Compare raw RGB, projected 14×14, pooled 7×7, causal pooled history and trained
RSSM policy inputs. This is linear decodability under forced-action coverage,
not policy competence, a dynamics intervention or proof of causation. Retain
per-target errors/R² and all seeds, not only their averages. Preserve input
hashes, outcomes and failed runs in `runs/seed-stability-20260908.XbiwtA/`.
The isolated, hardware-validated buffer-reuse extension is the executable;
it has the same frozen math as the campaign. These probes are not timing
benchmarks: CPU-only implementation/build work may overlap them.

## Runtime candidate and gate

Stage the candidate from `b14c32b` separately from every pinned control. The
existing Meganeura input/output buffer handles permit same-context GPU copies;
do not add OS-handle interop or change the user's backend worktree. Keep CPU
categorical draws, action order, value decoding and return calculations intact.
First target deterministic imagined states and the packed behavior features,
not a new sampling algorithm. Device copies require explicit shape, context,
lifetime and submission-order checks. Preserve the memory reserve.

Before adoption, require CPU/format/Clippy checks, focused GPU copy/packing and
act/learn/restore tests, complete eight-update synthetic report/tensor checks,
and fixed-recipe pixel comparisons against the buffer-reuse parent. Use the
same 3,072-action N8/B16/T64/R256 protocol and final 1,024-action/256-update
window, with two fresh pairs in AB, BA order. Keep the actual package/runner
hashes and all failures. No opaque parameter or shader-tolerance relaxation.
If exact parity fails, diagnose it before any performance adoption.

Record full learner calls, native stages, host transfers/readbacks, construction,
GPU activity/power/VRAM and actual game/wall ratios. GPU busy percentage is not
SM occupancy, and readback waits include producer work. The installed external
profiler failed its GPU-workload gate; do not repeat its unchanged captures or
claim calibrated idle attribution. Report the remaining unclassified time.

No lower replay ratio, shorter BPTT, encoder switch, concurrent learner service
or new learning campaign is part of this same-recipe runtime comparison.
