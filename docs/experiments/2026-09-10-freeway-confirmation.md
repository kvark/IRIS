# Fresh three-seed Freeway confirmation

Declared September 10. **Not running.** This tests the provisional hold64
recipe on fresh training roots; the successful seed-0 pilot is not replication.

The worker and immutable declaration are in
`runs/freeway-confirmation-20260910.megagv`, with 98 passing CPU tests and
617 content pins. Before declaration, a read-only audit reproduced both
completed pilot scores, their complete checkpoints and CPU replays/videos,
all eleven command completions and six GPU memory windows. Hold64 and hold1
remain successes on seed 0, with frozen means 31.0556 and 29.0278; their
untrained control remains zero. Minimum directly free memory was 3,302 MiB.
This confirms the existing evidence, not new gameplay or persistence necessity.

## Fixed protocol

Each fresh root **1009, 2017, 3019**, in order, gets **200,004 training actions**
and the unchanged schedule of 49,651 updates. Evaluate only its final checkpoint
for **75,000 sampled, unassisted frozen actions**. Each root also gets separately
initialized same-seed weights, saved after six frozen actions/zero updates,
then restored for the same 75,000-action evaluation.

Use the qualified current **f6a2b6ad native / 90b4763 source** and its matching
Atari Python bundle. This is the package already running Boxing, not the
episode-count candidate. The full current-backend runtime evidence is rechecked
independently before launch, including its action-override integration.
LeVJEPA, N6/R256/12M/B16/T64/full-BPTT64/F32, world microbatch 16, learning
rate .00004, warmup 1,000, AGC .3, reconstruction 0 and causal prediction .25
remain fixed. Only external game rewards enter learning.

Training alone uses v3 `persistent-uniform-v1` exploration, probability **.5**,
**hold64**, with independent per-stream RNG tied to the model root. Verify the
action actually executed and the complete override ledger. Frozen runs use
ordinary v2, no overrides, no episode-count stopping and zero updates.
Environment base seed is 100000; model/policy seed remains its training root.
Keep independent per-stream visual caches, beliefs and causal replay histories.

The published ALE wrapper remains unchanged: full 18 actions, repeat4, sticky0,
reset-noops0 and the 100,000-emulator-frame cap. All evaluations receive full
CPU ALE replay and whole stream-0 movies, including failures and partial tails.
Require complete parameters, optimizer moments/counters and normalizers.
Initial and trained parameter fingerprints must differ across the three roots;
different seed labels alone are insufficient.

## Acceptance and handoff

Every trained root must pass the unchanged Freeway gate: **at least 20 natural
rounds, at least 90% reaching 25 crossings, mean at least 25 and no cutoffs**.
Its paired untrained control must fail the task gate and have a lower mean.
Continue the other roots after a valid competence failure; never extend only
a weak root, select an intermediate checkpoint or add evaluation assistance.

Preserve the existing order: Boxing confirmation → current episode runtime
gate → first Breakout/Qbert pilots → Freeway confirmation. The new read-only
predecessor checker binds all 14 Breakout/Qbert commands, their complete
training/control state, v4 evaluations, replays/videos and eight GPU windows.
It preserves competence failures as failures, while allowing complete valid
data to release the next experiment. Its positive checks cannot run until
those pilots finish; passing fabricated CPU fixtures is not predecessor
completion or new GPU evidence.

An actual launch attempt while the bound Boxing controller was live refused
before GPU queries, native construction or run outputs. No follower or worker
is active for Freeway. Preserve every pinned input and do not restart any
completed or incomplete queue. After prerequisites pass, launch the declared
`run_confirmation.py`; it starts no automatic follow-up.

GPU work stays serialized. Each native phase requires at least 2,048 MiB
directly reported free memory, 4 Hz monitoring and complete sample coverage.
Timeouts remain twelve hours for training, two hours for frozen evaluation
and fifteen minutes for CPU replay. Process, integrity or memory failure stops
the queue and preserves artifacts for a separately declared continuation.

The [manifest](../../runs/freeway-confirmation-20260910.megagv/manifest.json)
SHA-256 is `633a3cadc16e00b61c1395df679f6f6c7b2d9ed927d1f366026a93b581b0ab8a`.
`pilot-preflight.json`, `runtime-preflight.json` and `live-parent-refusal.json`
record the completed CPU evidence. Final per-seed scores, replays and videos
do not exist yet. Even a complete successful Freeway confirmation would not
satisfy the other four games or establish cross-game policy transfer.
