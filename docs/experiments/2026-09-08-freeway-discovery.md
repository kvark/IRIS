# Freeway reward discovery: temporal persistence matters

Completed 2026-09-08, 22:53–23:06 UTC. This is a CPU exploration diagnostic,
not learned gameplay, a native-package test or a replacement for the running
Freeway pilot and its untrained control.

## Protocol and evidence

`runs/freeway-discovery-20260908.Zig71a/manifest.json` was declared before any
arm. Three random seeds (70271, 81271, 92271) each run three fresh arms: choose
a uniform random action every 1, 16 or 64 decisions. Each arm executes exactly
200,004 actual actions across six environments. All use 18 actions, repeat 4,
sticky 0, reset no-ops 0 and the 100,000-raw-frame cutoff. No Kindle is constructed,
and no reward shaping, demonstrations, privileged policy inputs or game-state
rewinds are used. The live GPU learner is unchanged.

Each stream has its own Python action RNG and initial environment seed
`seed + stream`; after natural termination, reset normally and sample a new
held action. This is not the native policy RNG or Kindle vector runner's
environment-seed rule. Comparisons between hold lengths share this declared
CPU protocol, not identical trajectories or a matched untrained neural policy.
The 16/64 holds span about 1.07/4.27 simulated seconds, but observations and
action accounting still occur every repeat-4 decision.

All nine arms complete **96 natural rounds each**, with no cutoffs. Unfinished
tails remain separate. Rewards below include complete rounds and partial tails;
round means use only the natural completed rounds.

| Decisions per held random action | Crossing rewards, three seeds | Mean crossings per complete round, three seeds | First reward at aggregate action |
| ---: | --- | --- | --- |
| 1 | 0 / 0 / 0 | 0 / 0 / 0 | None in any 200,004-action arm |
| 16 | 293 / 324 / 328 | 2.990 / 3.333 / 3.365 | 351 / 614 / 258 |
| 64 | 706 / 716 / 721 | 7.188 / 7.365 / 7.406 | 256 / 253 / 253 |

The largest round return in any arm is 14, well below the unchanged ≥25
Freeway competence bar. These are **reward-discovery results, not wins**.
Independent random actions find no rewards across 600,012 actions in these
three runs; this does not prove their probability of discovery is zero.

The action-generation and episode-accounting audit validates all 1,800,036
stored action bytes against the declared per-stream RNG/hold/reset rule,
complete-round and partial-tail accounting, summary means and original source/
ALE/ROM pins. It does not independently replay ALE or verify archived RGB;
the original diagnostic itself executed the ROM and recorded the outcomes.
Every action ledger and per-arm result is retained.

Manifest SHA-256: `3623f946ec60eea3101702d658eea34f322a52c01e64496086ff115173c45463`.
Completed summary: `45a1fa22732605e2ca3b2da0ec91e200bc1fe1fa77fb3247779eacc11ae59649`.
Accounting audit: `5604d08540a8d53f050a1dcdc733ada6b92ce5a43fc7ecee049059687a09297d`.

## Direction

If the native control remains reward-starved, test a small, explicitly declared
temporally persistent exploration component before merely extending the same
uniform-like behavior. It can preserve observations, executed-action labels,
the reward function and the world/actor objectives. Keep the current fixed
pilot unchanged; the CPU controls are not its completed learning result.

This is related to
[temporally extended epsilon-greedy exploration](https://arxiv.org/abs/2006.01782),
which samples action-repeat durations. Our fixed-hold, fully random CPU arms
are not a replication of that algorithm or evidence of its benefit to Kindle.

The serial agent supports masked actions, but the current native vector API
does not expose per-stream overrides. A minimal follow-up needs an explicit
executed-action seam, per-stream exploration state/RNG, boundary handling and
provenance in the runner/auditor. Preserve default actions/RNG and the full
reward/update/checkpoint ledger; never change the environment's input without
also giving the actual action to the RSSM and replay. Measure the actual
fraction of exploratory actions, not just a block-start probability.

Then compare equal-budget learning with and without this component, with a
strictly unassisted final frozen evaluation and the same task bar. This changes
collection behavior, not GPU throughput or the intrinsic-reward objective.
Intrinsic motivation remains a separate later ablation; no architecture
expansion or swarm is needed to test this simpler discovery lead.
