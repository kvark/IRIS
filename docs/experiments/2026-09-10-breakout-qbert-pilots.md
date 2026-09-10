# First Breakout and Qbert learning pilots

Declared September 10, before either game is trained. **Not running.** These
are fixed-budget seed-0 pilots, not fresh-seed confirmation or learned wins.

The corrected v2 worker in `runs/breakout-qbert-pilots-v2-20260910.9zf9T3` has
92 passing CPU tests and 521 content pins. It requires the unchanged Boxing
confirmation to finish and the separately declared current-package episode
runtime gate to pass. Before learning, it independently recomputes the gate's complete state,
reports, traces, frozen prefixes, twelve declared commands and eight GPU
memory/coverage windows. A completion flag alone is insufficient.

The actual pilot CLI was tested while the bound Boxing controller was live.
It refused before any GPU query, native construction or run outputs. No pilot
worker is active. The separate [serial follower](2026-09-10-atari-serial-handoff.md)
now waits on Boxing and retains the full runtime prerequisite before these
pilots. Preserve all inputs and do not manually launch a duplicate worker.

## Fixed comparison

Order: Breakout, then Qbert. Each game receives a **fresh seed-0 model and
200,004 actual training actions**. The complete replay ledger determines the
exact update count, including warmup and actual-action credit. There is
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

## Prelaunch schedule correction

The unstarted v1 declaration in `runs/breakout-qbert-pilots-20260910.h0l2PM`
is withdrawn before launch, not a failed learning run. Its 512 pins remain
unchanged. It incorrectly imposed Boxing's 49,651 updates on both games.
Initial and episode-reset observations add replay frames without adding action
credit; early resets can make replay ready sooner. The native scheduler and
existing full ledger auditor already implement this correctly.

The CPU diagnostic in `runs/atari-schedule-check-20260910.VJO46i` collected
1,404 random actions on six streams for each of the five games and independently
replayed every stream through ALE. Breakout's six early resets permit the first
scheduled update at action 1,398, versus 1,404 in the other four sampled cases.
Under the declared schedule, those prefixes imply 49,652 versus 49,651 updates
at 200,004 actions. These are scheduler simulations from random CPU experience,
not actual learner updates or predictions of the future native policy's count.

The v2 validator obtains the count from the complete source-matched ledger audit
and binds both the frozen restore and checkpoint counters to it. It does not
merely accept any positive update count: missing/extra reports, wrong credit and
reset order still fail the auditor. Fourteen added CPU cases cover these checks
using explicitly synthetic learner reports and real CPU arrival histories.
Budgets, native arithmetic, evaluation and task thresholds are unchanged.

## Artifacts and handoff

The immutable declaration is
[`manifest.json`](../../runs/breakout-qbert-pilots-v2-20260910.9zf9T3/manifest.json),
SHA-256 `ff4d4b529aac58cd06345da3246c647a36f6beedc60a88c60e545645b71f71c9`.
Per-game replay declarations bind back to that manifest and its content pins.
`cpu-tests.xml` contains implementation checks, not learning results;
`live-parent-refusal.json` records the actual negative launch check.

After all prerequisites complete, the declared worker is `run_pilots.py`.
It starts no automatic follow-up. Per-game training, evaluation, replay/video,
untrained-control and score artifacts appear only when their phases finish.
`completed.json` requires both complete pilots and controls, including valid
competence failures. It can never certify training-seed reliability or completion
of the five-game goal. There are no Breakout/Qbert rollout results to view yet.
