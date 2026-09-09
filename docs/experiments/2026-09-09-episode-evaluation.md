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
future N6 Breakout/Qbert evaluation uses four completed episodes per stream
and a 600,000-action cap. These settings still need a content-pinned pilot
declaration before use. Existing fixed-action evaluations remain unchanged.

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
mean final score at least 15,000. Training budgets and recipes remain to be
declared for these games. Stopping by episode count does not solve learning.

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
retain their order. No new GPU worker or long learning run has been launched.

The cap-check manifest SHA-256 is
`7107388290448246c77e46db85fcc133e472d80ec31a628cd195b2ce207d657f`;
its result SHA-256 is
`f4669872d97524ea2e409762c78ca99cb81e3e993c5704a871ab3b964dd787cb`.
