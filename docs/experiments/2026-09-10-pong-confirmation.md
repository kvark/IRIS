# Fresh Pong confirmation at a larger fixed budget

Declared September 10. **Not running.** The historical 200k-action LeVJEPA
campaign still fails its all-seeds mastery gate. This follow-up tests reliability
at a larger exposure budget; it does not extend or relabel that campaign.

The immutable declaration and worker are in
`runs/pong-confirmation-20260910.zFks3A`: **735 content pins and 95 passing CPU
tests**. An actual CLI launch while the bound Boxing controller was live refused
before GPU queries, native construction or run outputs. No follower was started.
Passing launcher tests is not gameplay, GPU qualification or reliable learning.

## Fixed protocol

Each fresh root **1009, 2017, 3019** receives **400,008 actual training actions**,
twice the current N6 pilot budget. Historical training coverage motivates this
bounded test: first wins arrived late, especially for seed 1, and positive-reward
coverage differed strongly. All three had nonzero advantage after their first
two updates. Neither delayed wins nor this follow-up establishes numerical
collapse or its cause. Give every root the same budget; never extend only a
weak seed or select an intermediate checkpoint.

Use unchanged current-episode source **24b2968** and its matching **f6a2b6ad**
native package. LeVJEPA, N6/R256/12M/B16/T64/full-BPTT64/F32, microbatch 16,
learning rate .00004, warmup 1,000, AGC .3, reconstruction 0, causal prediction
.25, extrinsic scale 1 and intrinsic scale 0 remain fixed. There are no action
overrides, held exploration, reward shaping, pretraining or cross-game weights.
Count actual interactions, not vector ticks or reset observations. Derive the
exact update count from the complete reset-dependent replay warmup/action-credit
ledger; do not substitute another game's fixed count.

Evaluate only each declared final checkpoint, sampled and unassisted with zero
updates. The v4 stopping rule requires **four completed episodes per stream**,
with a **600,000-action hard cap**. Retain all extra completed episodes and partial
tails; reaching the cap without the episode target is incomplete. Environment
base seed is 100000; model/policy seed remains the training root. Independently
initialize same-seed untrained weights, save after six frozen actions/zero
updates, then restore for the identical evaluation protocol. All evaluations
receive complete CPU ALE replay and whole stream-zero movies, including failures.

Every trained root must pass the unchanged Pong gate: **at least 20 natural
games, mean return at least +15, at least 90% wins and no cutoffs**. Its paired
untrained control must fail the gate and have a lower mean. Verify complete
parameters, optimizer moments, counters and normalizers, with distinct full
initial and trained parameter fingerprints across roots. Continue after valid
competence failures; stop on integrity or runtime-safety failure.

This is a new confirmation protocol, not an isolated budget-effect experiment
against the historical N8/old-backend results. Frozen stopping, backend and
collection configuration also differ from those historical runs. A success
would establish this declared recipe's three-seed result, not that doubling
exposure alone caused an improvement.

## Separate world-model evidence

Before seeing any scores, select **the first four complete stream-zero matches
from every final evaluation**, including failures and cutoffs. The CPU-qualified
[multi-match extractor](2026-09-10-multimatch-world-probe.md) at `b2f0ddd` audits
the entire source ledger, including other streams and tails, and binds both its
whole-file identity and the exact selected prefix. Selection constructs no
native agent and makes no forecasts.

The intended comparison is all nine model/recording combinations, with one-step
forecasts at every action. Require all three same-model diagonals before the six
cross-model cases. Forecast before consuming the target; separate features,
reward sign/magnitude and continuation, with persistence/unrelated-action/zero
baselines. Report event counts and visual-cache reset strata. Own-policy returns
remain separate from common-input forecast quality.

Native serial/vector replay, strict/forced forecast parity and GPU safety still
need their own declared diagnostic gate. This worker launches no world-model
GPU work. A complete recording set is not complete forecasts, and no long-horizon
diagnostic is silently added to this budget.

## Serialized handoff

Keep the existing order: Boxing confirmation → current episode runtime gate →
corrected Breakout/Qbert pilots → corrected Freeway confirmation → this Pong
follow-up. The read-only Freeway handoff verifies all 18 declared commands,
twelve native GPU windows, complete trained/untrained checkpoints and replays,
and the preceding runtime/pilot evidence. Valid task failures remain failures
but do not masquerade as missing data. Positive verification must wait for those
experiments to finish; synthetic fixtures do not release the GPU.

Require at least **2,048 MiB directly free**, 4 Hz monitoring, at least two
samples/second and no gap over 1.5 seconds across every native phase. Timeouts
are eighteen hours for training, eight hours for frozen evaluation and one hour
for CPU replay. Training alone is roughly thirteen hours per root at the current
8.6 actions/s; record actual wall time, aggregate/per-stream game clocks and
learner updates. This is not a super-real-time claim.

The [manifest](../../runs/pong-confirmation-20260910.zFks3A/manifest.json) SHA-256
is `94b7699e713c90c6ce217913cc32b989ebe76b4c81b4bde7413b3ac61986093b`.
`live-parent-refusal.json` records the actual negative launch check and reverified
pins. Preserve every input. Launch the declared `run_confirmation.py` only after
the complete predecessor evidence passes; it starts no automatic follow-up.
All three Pong roots and controls must pass for Pong confirmation. The other
four games and broader completion audit remain required for the five-game goal.
