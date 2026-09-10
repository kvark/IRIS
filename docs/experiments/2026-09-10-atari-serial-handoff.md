# Serialized execution of the declared Atari follow-ups

Started September 10 at **04:40 UTC**, waiting on the actual live Boxing
confirmation controller. This supplies scheduling, not a new learning recipe
or relaxed acceptance. The five-game objective remains incomplete.

The separate follower is
`runs/atari-serial-handoff-20260910.zF8Hfh/follow_queue.py`, with **756 content
pins and 52 passing CPU scheduling tests**. It binds Boxing PID 2303115 and
start ticks 107474767. Its own launch records PID 2318785/start ticks 108736692;
verify those live identities before relying on them. The initial real-process
check confirms it is sleeping on the bound live parent with no child. The only
reported GPU compute process remains Boxing's native worker 2303477.

## Fixed order

After all three Boxing training/frozen/untrained sequences finish normally:

1. Run the current-package episode-count runtime gate, retaining its full
   default-learning, frozen-prefix, state and memory checks.
2. Run the corrected seed-0 [Breakout/Qbert pilots](2026-09-10-breakout-qbert-pilots.md).
3. Run the corrected fresh three-seed [Freeway confirmation](2026-09-10-freeway-confirmation.md).
4. Run the fresh larger-budget [Pong confirmation](2026-09-10-pong-confirmation.md).

The follower invokes each existing entrypoint without arguments or source
changes. All game thresholds, budgets, roots, packages, exploration choices,
final-checkpoint evaluations, controls and video requirements remain pinned.
Each entrypoint independently rechecks its complete raw predecessor/runtime
evidence before GPU work. A follower completion flag never replaces those checks.
Existing child declarations retain `automatic_followup: false`; this separately
declared outer follower supplies the handoff.

It waits for the actual child exit before advancing and requires the matching
declared data/runtime completion record. Valid competence failures remain
failures and allow other games/seeds to proceed. A failed process, incomplete
result, changed input, runtime-safety failure or timeout stops the handoff.
There are no retries, restarts or replacement outputs. No future worker has
started yet; do not manually launch one alongside this follower.

## Safety and evidence

The waiting loop polls the bound process every 30 seconds and emits a heartbeat
every five minutes. A process-read error is not treated as completion. The
72-hour wait ceiling does not stop or restart Boxing. The follower constructs
no native agent and makes no GPU queries while waiting.

Every future stage's original top-level input set is pinned. Any new non-cache
entry refuses launch, including another worker starting that stage. Exclusive
follower output creation prevents duplicate launch. Before each stage, require
at least 32 GiB free disk; native workers retain their existing quiet-device and
2,048 MiB directly-free GPU guards and full monitoring. The outer timeout ceilings
are six hours for the runtime gate, eighty for Breakout/Qbert, sixty for Freeway
and 140 for Pong, above their unchanged per-command limits. They do not enlarge
training budgets. Cancellation owns only the follower's active child, never Boxing.

CPU fixtures cover declaration mutation, parent identity/observation failures,
wait expiry, changed stage entries/pins, duplicate launch, insufficient disk,
failed or incomplete children, exact order and retained competence failures.
They are not native qualification or learned results. The actual launched
follower's `launch.json`, `events.jsonl` and process identity provide waiting
evidence separately; inspect `controller.stderr` on an unexpected exit.

The [manifest](../../runs/atari-serial-handoff-20260910.zF8Hfh/manifest.json) SHA-256
is `11a411572a0adc82754e6543b28c17aa4f5b5bf2316bb4b97da81e3a1d8dab50`.
Preserve all 756 inputs throughout the wait and subsequent stages. Do not
restart a partial or completed handoff. Its final result explicitly does not
certify five-game mastery, native world forecasts or transfer. Breakout/Qbert
still need independent fresh-seed confirmation after their pilot recipe is
assessed; every game and control needs the broader completion audit.
