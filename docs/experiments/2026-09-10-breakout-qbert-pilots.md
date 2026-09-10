# First Breakout and Qbert learning pilots

Declared September 10, before either game is trained. **Not running.** These
are fixed-budget seed-0 pilots, not fresh-seed confirmation or learned wins.

The worker in `runs/breakout-qbert-pilots-20260910.h0l2PM` has 78 passing CPU
tests and 512 content pins. It requires the unchanged Boxing confirmation to
finish and the separately declared current-package episode runtime gate to
pass. Before learning, it independently recomputes the gate's complete state,
reports, traces, frozen prefixes, twelve declared commands and eight GPU
memory/coverage windows. A completion flag alone is insufficient.

The actual pilot CLI was tested while the bound Boxing controller was live.
It refused before any GPU query, native construction or run outputs. No worker
or follower is active for these pilots. Preserve all inputs and do not run
them alongside Boxing or the runtime gate.

## Fixed comparison

Order: Breakout, then Qbert. Each game receives a **fresh seed-0 model and
200,004 actual training actions**, expected 49,651 learner updates. There is
no cross-game initialization, restored replay, exploration override, shaping
or intrinsic reward. The existing CPU random controls discover rewards in
both titles; Freeway's successful held exploration does not establish that
these games need it.

Both use the qualified f6a2b6ad native with current-episode Python source
24b2968 and its source-matched import bundle: LeVJEPA, N6/R256/12M/B16/T64,
full 64-step BPTT, F32, world microbatch 16, learning rate .00004, warmup 1,000,
AGC .3, reconstruction 0 and causal prediction .25. Only game rewards enter
learning. Actual actions, independent recurrent histories and emulator clocks
retain the published ALE wrapper contract.

Evaluate only each **declared final checkpoint**. The v4 frozen stopping rule
requires **four completed episodes per stream**, with a **600,000-action hard
cap**, sampled unassisted actions and zero updates. Environment seed is
100000; model/policy seed stays 0. Keep every completed episode, including
extras from faster streams, and report partial tails separately. Reaching
the cap without the episode target is incomplete. This target guarantees at
least 24 completed episodes under the 25,000-decision wrapper cap, not task
success or natural termination.

Each game also gets separately initialized seed-0 weights, saved after six
frozen actions and zero updates, then restored for the same evaluation.
Complete checkpoint checks and full CPU ALE replays follow both evaluations.
Each movie contains the whole of stream 0, including failures and tails.
Task observers stay outside policy inputs and training rewards.

## Acceptance and safety

The existing thresholds remain unchanged:

- Breakout: both walls/864 points in at least 90% of at least 20 completed
  episodes.
- Qbert: first-pyramid completion at that rate **and mean final score at
  least 15,000**.

A task reached before a later cutoff still counts without calling that episode
natural. A cutoff without task completion fails. Require the paired untrained
control to fail the task gate and have a lower mean before calling the pilot a
learning success. Continue Qbert after a valid Breakout competence failure;
do not extend a weak arm or change assistance or thresholds after seeing scores.
Neither pilot establishes reliability: a selected recipe still needs the
fresh roots 1009/2017/3019 and all five game-specific gates.

Serialize GPU work and record direct free/reserved memory at 4 Hz. Every native
phase requires at least 2,048 MiB directly free and complete sample coverage.
Timeouts are 12 hours for training, eight hours for frozen evaluation and one
hour for CPU replay. The larger frozen cap needs more than Boxing's historical
two-hour timeout. Integrity, process or memory failure stops the queue and
preserves partial artifacts; restarting requires a separate continuation.

## Artifacts and handoff

The immutable declaration is
[`manifest.json`](../../runs/breakout-qbert-pilots-20260910.h0l2PM/manifest.json),
SHA-256 `3edb9092c5e88542078886549c3b307870041139856278200e9b86db6643d950`.
Per-game replay declarations bind back to that manifest and its content pins.
`cpu-tests.xml` contains implementation checks, not learning results;
`live-parent-refusal.json` records the actual negative launch check.

After all prerequisites complete, the declared worker is `run_pilots.py`.
It starts no automatic follow-up. Per-game training, evaluation, replay/video,
untrained-control and score artifacts appear only when their phases finish.
`completed.json` requires both complete pilots and controls, including valid
competence failures. It can never certify training-seed reliability or completion
of the five-game goal. There are no Breakout/Qbert rollout results to view yet.
