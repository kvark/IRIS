# Bounded episode-budgeted frozen evaluation

Staged on 2026-09-09 in `exp/episode-budget-evaluation`. This is a Python-only
candidate, not an adopted evaluation protocol, GPU speedup or learned result.
All 580 Python CPU tests pass, including fabricated unequal-length vector
episodes and stopping-rule failures. Four actual old N6 train/frozen ledgers
retain exactly their saved v2 accounting. No native source or live input changed.

## Why change the evaluation budget?

The CPU check in `runs/atari-eval-budget-20260909.QW90eT` freshly replays the
complete existing 8,192-action random adapter logs for Breakout and Qbert.
Every action, RGB hash, reward, boundary and frame count matches. There are
66 and 122 positive reward events respectively. These are random discovery
controls, not untrained native-policy evaluations or learned task wins.
They do not justify applying Freeway's persistent exploration to both games.

A new, fixed NOOP Breakout fixture reaches the wrapper's 100,000-frame cutoff
at exactly 25,000 decisions, with zero reward and no natural termination.
It constructs no agent and uses no GPU, cloning, RAM writes or training.
The 11 CPU arithmetic tests also pass. This establishes a real worst-case
episode length, not a successful rollout.

For six balanced streams, repeat four and a 100,000-frame episode cap:

| Aggregate decisions | Guaranteed completed episodes |
| --- | ---: |
| 75,000 | 0 |
| 150,000 | 6 |
| 300,000 | 12 |
| 600,000 | 24 |

The bound counts cutoffs too; it guarantees neither natural games nor task
success. In general it is
`streams * floor((actions / streams) / ceil(frame_cap / repeat))`.
Partial tails are not completed episodes.

Blindly running 600k frozen actions would cost about 5.1 hours per policy at
the completed short N6 control's 32.52–32.73 actions/s. That is an extrapolation
from two 768-action Boxing restore loops, excluding construction, not a long-run
benchmark or playing-plus-training speed claim. See
`runs/vector-memory-runtime-20260908.CcWv0d/order{0,1}-n6-restore.jsonl`.

## Candidate contract

`atari_vector.py --evaluate --restore CHECKPOINT --episodes-per-env 4`
uses `kindle-vector-v4`; `--steps` is an explicit hard cap. The proposed
N6 Breakout/Qbert evaluation uses four completed episodes per stream and a
600,000-action cap. The [new pilot declaration](2026-09-10-breakout-qbert-pilots.md)
now pins these settings, but remains conditional on complete current-package
runtime qualification. Existing fixed-action evaluations remain unchanged.

- Stop at the first fully accounted vector tick when **every** stream has
  reached its episode target. Do not stop on reward, task success, or only the
  fastest stream. The native batched action/observe/reset path is unchanged.
- Keep every completed episode, including extra episodes in faster streams,
  losses and cutoffs. Score them all; partial tails remain separate. Four per
  stream yields at least 24 completed episodes, not necessarily exactly 24.
- Frozen restore only: no learning, exploration overrides or checkpoint writes.
  Retain sampled/greedy mode explicitly; campaign acceptance still requires
  sampled evaluation and the declared final model.
- Record the requested episode target, maximum actions, actual actions,
  per-stream counts and an explicit stop reason. Reaching the cap without the
  episode target is incomplete, not a completed evaluation.
- The CPU auditor reconstructs every count and rejects actions after the first
  eligible stopping tick, hidden episode budgets in older protocols, early
  success claims, training, or checkpoint writes. Final-model identity and the
  complete replay/task checks remain required.
- Old campaign declarations reject v4. Add a separately declared campaign
  contract/checker before replication; do not reinterpret old 75k-action
  results or treat this change as satisfying any game gate.

All five task criteria and training seeds 1009/2017/3019 remain unchanged.
In particular, Breakout still needs both walls in at least 90% of at least
20 completed episodes; Qbert still needs the first pyramid at that rate and
mean final score at least 15,000. The first seed-0 pilots now declare 200,004
training actions per game; fresh-seed replication remains separate. Stopping
by episode count does not solve learning.

## Evidence and remaining gates

The candidate changes only the vector runner, its CPU accounting and the
final train/evaluation protocol pairing, with tests. Its test environment
uses the source Python package plus a symlink to the unchanged archived
`9cf1316b…` extension; it constructs no native agent. This does not adopt
the separately staged persistent-exploration package.

CPU artifacts are in `runs/episode-evaluation-20260909.gv9pgd`:
`python-tests.xml`, `check_old_ledgers.py`, and `old-ledger-binding.json`.
The last contains hashes and exact comparisons for both orders' N6 3,840-action
training and 768-action frozen ledgers. It is backward reader evidence, not
GPU execution of this runner.

Before use, require a serialized GPU comparison against fixed-action frozen
execution with the same actual native package, checkpoint, frontend, seeds
and controls: exact common-prefix action/reward/reset traces, zero updates,
complete unchanged checkpoint state, correct earliest stop and direct-memory
coverage/reserve checks. Validate the native path rather than relying only
on fabricated CPU streams. Do not insert work into the current pinned queues;
Freeway, common-world, the exploration gate and its declared learning pilot
completed in that order. The following frozen-only GPU gate is now complete
through its separately declared continuation; no new long learning run has
been launched.

The cap-check manifest SHA-256 is
`7107388290448246c77e46db85fcc133e472d80ec31a628cd195b2ce207d657f`;
its result SHA-256 is
`f4669872d97524ea2e409762c78ca99cb81e3e993c5704a871ab3b964dd787cb`.

## GPU check interrupted; continuation complete

The frozen-only check is now declared in
`runs/episode-evaluation-gate-20260909.8f1yKj`, with 55 input pins and manifest
SHA-256 `84e82739d5ddcdc38e7bd06c0d0dbcf3dbb95caf1d32b8b964fedff2edce5c9b`.
It binds the actual persistence-learning launcher, PID 2120848/start tick
98610096. There is no new follower or automatic launch. The real readiness
check refuses execution while that predecessor is live, before GPU queries
or creation of run-event outputs.

The Freeway learning pilot completed normally at 22:28:24 UTC on September 9.
An independent CPU recheck verifies all trained/control results, complete saved
state, replay bindings, 82 pins and all six raw GPU windows. Only then was this
gate launched manually, with its first control starting at 22:30:32 UTC.
Its first two phases complete with full native-state and ledger checks: the
fixed control executes 18,000 actions; the episode-budget candidate stops at
3,618, with one natural episode per stream and zero updates. The latter also
matches the original fixed-control prefix in the continuation's CPU recheck.

The launcher records an interruption signal (`SystemExit(130)`) at 22:50:10 UTC.
The third phase stops at 9,150/18,000 actions with `reason=interrupted`; the
controller and native worker are terminal. No numerical failure is reported,
but the four-phase gate is incomplete. Preserve every original artifact,
including this interrupted trajectory. Do not restart the original queue.

Using the completed Boxing R256 model, the fixed order is control v2 for
18,000 actions; candidate v4 until every stream completes one episode, capped
at 18,000; candidate default v2 for 18,000; then candidate v4 with only six
allowed actions, which must remain incomplete. Both roles use the same archived
9cf1316b extension, N6, sampled controls and evaluation seed 100000. This tests
the stopping implementation, not the future four-per-stream competence sample.

The diagnostic driver passes the actual native agent to the runner and captures
fresh before/after state outside its interaction loop. Require exact full
parameters, optimizer moments/counters and normalizers against the source,
plus exact complete/default and shorter-prefix action/reward/reset traces.
Only verified environment counters and collection stream count may change in
state metadata. Every phase receives an independent fresh-process CPU audit.
Keep direct free memory at least 2 GiB with 4 Hz coverage checks; each native
phase is bounded at 30 minutes. Captures and source hashing preclude a speed
claim from these timings.

All 30 CPU gate tests pass. They include reading the real completed checkpoint's
241 saved tensors and explicitly fabricated binding/capture/stopping fixtures,
not execution of this GPU comparison. Preserve the pinned candidate inputs.
The original command, already interrupted and not to be rerun in place, was:

```sh
python/.venv/bin/python runs/episode-evaluation-gate-20260909.8f1yKj/run_gate.py
```

Failures preserve artifacts and stop only this check's child. A pass still
does not adopt an evaluation protocol, launch training or establish mastery.

### Separately declared continuation

`runs/episode-evaluation-continuation-20260909.6YKbjp` pins 100 inputs and
original artifacts; manifest SHA-256 is
`ed640b77b2d3769bcf121f39bece604afa4ca283986c0d03b1ec6cda9478f8fe`.
Its 51 CPU checks include the existing 30 gate tests and 21 new continuation
binding fixtures, not new GPU results. Before declaration and launch, a fresh
CPU audit rechecks the two completed original ledgers, complete saved state,
exact shorter prefix and both original GPU memory/coverage windows.

Only the unfinished phases run again: all 18,000 candidate fixed actions from
a fresh restore of the original Boxing checkpoint, then the six-action negative
cap case. It never appends to the partial trajectory or restores its checkpoint.
The unchanged original capture implementation writes to fresh continuation
paths. All settings, full-state/trace requirements and memory gates are retained.

The new fixed phase started at 22:58:58 UTC and completed all 18,000 actions.
The six-action cap check then completed normally with `action_cap_reached`,
zero episodes and an explicitly incomplete episode budget. The continuation
finished at 23:10:56 UTC; its controller and children are terminal. The original
interrupted queue remains incomplete and unchanged.

An independent CPU recheck reproduces all four ledgers, full frozen native
state, the complete default trace and both shorter prefixes. It verifies every
command exit/output identity, all 100 pins and all four raw GPU windows. The
minimum directly free memory is 3,413 MiB; the largest sample gap is 0.268 s.
The completed result is
[`completed.json`](../../runs/episode-evaluation-continuation-20260909.6YKbjp/completed.json),
SHA-256 `96e4cdc1800afcb65df204e226578f6d705d78f2681c77561f640077d1203f10`.

This validates the frozen stopping implementation with the unchanged native
package. It does not adopt a campaign protocol, establish learning reliability
or provide a speed benchmark. Preserve both artifact roots; neither queue
should restart. The newly requested backend refresh now precedes the unrun
world-sync comparison, without altering that candidate's pinned inputs.

## Current-backend carry: CPU-qualified, GPU work not started

The isolated `exp/current-episode-evaluation` candidate at **24b2968** carries
exactly the four Python implementation/test files from `4281242` onto the
qualified current source `90b4763`. All Rust/backend/build inputs remain
unchanged, and the fresh import bundle uses the actual qualified **f6a2b6ad**
native binary, without rebuilding or overwriting any historical package.

All **580 Python CPU tests** pass with that native and the six source-matched
Python modules. `runs/current-episode-package-20260910.etyDN4/cpu-evidence.json`
records 61 pins, the clean source commit, exact earlier Python-file identities,
unchanged native inputs/bytes, actual test exits and the complete test XML.
Its SHA-256 is `9ee5b4eac750fa802016bc5a2a25db9c0641e2c82d99e17d00f7f237739132cb`.

No new GPU gate has started, and this bundle is not runtime-qualified.
After the active Boxing confirmation releases the device, require a separately
declared current-package default-training state/report/trace check against the
retained pixel control, frozen default/v4 prefix checks, a negative cap case,
complete frozen state and direct-memory coverage. Keep all current queues fixed.
The separate Breakout/Qbert pilot declaration now supplies fixed learning budgets
and stopping targets; none of these implementation checks is a learned result.

### Declared current-package runtime gate

`runs/current-episode-runtime-20260910.uRF9VK` now contains the executable gate,
498 content pins and 47 passing CPU tests. Its manifest SHA-256 is
`80f8b02a8c4f1fe880ee716b9405954be75c4ea78f03ddd557a79f04e041ad9d`.
It binds the actual Boxing confirmation controller, PID 2303115/start tick
107474767. The real negative launch check returns the expected live-parent
refusal before a GPU query, event file or new GPU process; all 498 pins reverify.
This is a readiness check, not a failed GPU experiment or a started queue.

After Boxing finishes, the new read-only checker must reconstruct all three
trained/untrained comparisons, exact declared commands, complete checkpoints,
six replay/video bindings and all twelve native memory windows. It preserves
the competence outcome: completed data can be valid even if a seed fails its
gate, but an interrupted or incomplete queue cannot release this gate.

The eight serialized native phases then compare a fresh default-learning pair
on control/candidate Python packages against the retained current-backend pixel
anchor, followed by the four frozen default/episode/negative-cap captures.
Native bytes stay f6a2b6ad. Full states, all learning reports, action/episode/reset
traces and frozen prefixes must match; all phases require direct-memory coverage
and ≥2,048 MiB free. The single timed pair guards a >2% regression, not an AB/BA
speedup claim. The capture implementation is byte-identical to the earlier gate.

The GPU worker has not started. A separate
[serial follower](2026-09-10-atari-serial-handoff.md) now waits on the actual live
Boxing controller before invoking this unchanged gate. Keep its inputs fixed;
do not manually launch a duplicate. Its complete predecessor proof must still
pass before GPU work. The separately declared
Breakout/Qbert pilots remain conditional on this gate; it starts no long run.
