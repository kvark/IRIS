# Atari panel: CPU adapter preflight

Declared 2026-09-07 while the pinned LeVJEPA Pong queue is still running.
This is forced-random environment coverage with **no Kindle agent, encoder,
GPU work or learning**. It does not advance the gameplay/mastery gates or
replace the planned matched frontend comparison.

## Fixed scope

Run `python/examples/check_atari_adapter.py` on Pong, Breakout, Boxing, Freeway,
Seaquest, Frostbite, Qbert and Private Eye. Each title has two independently
initialized ALE instances, interleaved synchronously. Each stream executes
4,096 uniformly random actions, then a fresh instance replays its exact
declared action sequence. The replay is a diagnostic, not additional training
or continuation of a live learning trajectory; no environment state is cloned.

- Environment seeds: 8,001 and 1,008,004. Separate Python action RNG seeds are
  each environment seed XOR `0xA7A21000`; verify this sequence independently
  of image/reward equality, since actions can be aliases or visually hidden.
- Existing `published` Atari wrapper: full legal action vocabulary, repeat four,
  no sticky actions, no reset no-ops and 100,000-frame episode limit.
- Preserve wrapper/checker/native-ALE hashes, ROM identity, package versions,
  every action, RGB8 hash, raw summed reward, boundary and actual frame count.
- Check each wrapper step's frame delta against ALE's own frame counter.
  Record terminal observations before ordinary resets; preserve incomplete tails.
- Retain all failures and return nonzero if any title fails. Fresh output paths
  are required. A failed replay must not produce an accepted `run_end`.

The panel executes 65,536 collection actions and 65,536 serial replay actions.
All are CPU adapter diagnostics, not agent-collected experience or a random
competence baseline with an adequate evaluation budget. No cold/steady GPU
throughput claim is drawn from this run. Its CPU load overlaps Pong training
and must be disclosed if that time interval is used for systems comparisons.

## Gates and boundaries

Require exact fresh-serial agreement for both streams' observations, actions,
rewards, terminal/truncation flags, resets and clocks; finite rewards; RGB8
64×64 observations; and balanced episode/action/reward ledgers. Report actual
action coverage and observed natural/truncated episodes. Passing this prefix
does not prove unobserved reward or boundary behavior, or all game mechanics.

The existing vector runner's `natural_wins` field uses positive terminal
episode returns, which has the intended meaning for Pong, not arbitrary Atari.
Do not change the active pinned runner/auditor. Before other vectorized learning
runs, separate generic episode statistics from game-specific competence/win
criteria. This checker reports returns and boundary counts, not wins.

Twenty CPU tests cover fresh-instance replay, unequal resets, unfinished tails,
RGB8 validation, corrupted observations/actions/rewards/boundaries/clocks,
nonfinite rewards, output preservation and explicit failed summaries. They
forbid construction of either Kindle agent class. The complete CPU suite passes
221 tests, including run-local Pong audits.

```sh
python/.venv/bin/python python/examples/check_atari_adapter.py \
  runs/atari-adapters-20260907
```

## Real-ROM result

All eight titles pass. The run was observed from **10:51:16 to 10:52:08 UTC**,
September 7; collection plus serial-replay work totals 52.43 s. It executes the
declared 65,536 collection and 65,536 replay actions, with **261,923 actual frames
in each phase**, zero learner updates and no agent construction. Every stream
exercises all 18 actions. All games produce natural episode boundaries; no
wrapper timeouts occur in these prefixes. Separate timeout coverage remains in
the synthetic adapter tests, not this real-ROM run.

| Game | Natural episodes | Diagnostic mean return | Positive / negative reward events |
| --- | ---: | ---: | ---: |
| Pong | 8 | −20.25 | 6 / 191 |
| Breakout | 43 | 1.53 | 66 / 0 |
| Boxing | 4 | +1.75 | 122 / 118 |
| Freeway | 4 | 0 | 0 / 0 |
| Seaquest | 16 | 55 | 44 / 0 |
| Frostbite | 18 | 78.89 | 153 / 0 |
| Qbert | 23 | 163.04 | 122 / 0 |
| Private Eye | 2 | 100 | 3 / 0 |

These are short forced-random adapter returns, not competence results. All 18
Frostbite episodes have positive return, illustrating why Pong's positive-return
win rule must not be generalized. Freeway's reward channel has not been exercised
by this prefix. Boxing's repeated-action rewards include ±2, and Qbert's include
325; do not apply Pong-only ±1 reward validation to this panel.

Source is `e8c26533f386ad5adb4fdf8bd86a877106dc1837`; checker SHA-256 is
`6bebb325845bd0035e5bf19bb15ab0e455a7f9a6decb141dc071f1a6f93ed9ba`.
ALE 0.12.1 native SHA-256 is
`b9c810858e2d791eaf51d75cf72fc2057a2191a2e87e19f81657e492896f330d`.
Gymnasium/NumPy/Pillow are 1.3.0/2.5.2/12.3.0. Every game log records its actual
ROM path and SHA-256 and the unchanged wrapper hash. Full logs, summaries and
independent saved-ledger/hash checks are in `runs/atari-adapters-20260907/`,
including `verification.json`. Incomplete episode tails remain in the logs.

The active Pong native extension, wrapper, vector runner and auditors are
unchanged. Native video/learner quality and GPU throughput remain separate gates.

## Separate Freeway reward fixture

Declared before execution: one fresh environment at seed 9,001, exactly 1,024
constant `UP` actions, and the same published wrapper. This is a **scripted
reward-wiring diagnostic**, not a random control, learned behavior, demonstration
training or a competence result. It must log every actual action, observation
hash, reward and frame count, and reports whether any positive game reward was
observed. Do not extend the budget or include it in the panel's random returns.
