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
forbid construction of either Kindle agent class. Real-ROM results are pending.

```sh
python/.venv/bin/python python/examples/check_atari_adapter.py \
  runs/atari-adapters-20260907
```
