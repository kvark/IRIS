# Generic Atari episode accounting candidate

Staged on `exp/atari-episode-accounting`, based on `9a2f174`. Not adopted into
the active Pong queue. Its original native extension, runner and auditors remain
unchanged. This patch changes reporting semantics, not observations, actions,
rewards, collection, learner scheduling, model parameters or checkpoints.

The CPU adapter panel demonstrates the problem: all 18 random Frostbite episodes
have positive return. Calling them wins would imply competence that the test
does not establish. Pong's positive-return win rule belongs in its game-specific
scorer, not the generic vector ledger.

## Versioned log boundary

New runner output uses `kindle-vector-v2`:

| Legacy v1 summary | New summary |
| --- | --- |
| `completed_games` | `completed_episodes` |
| `natural_wins` | `positive_return_natural_episodes` |
| No explicit boundary split | `natural_episodes`, `truncated_episodes` |
| `mean_completed_return` | Unchanged: all completed episodes, including truncations; null when empty |

The generic reader accepts unchanged v1 files, verifies their original fields
against the episode ledger, and returns the descriptive names for both versions.
It does not modify old files or introduce a generic win claim. Mixed-version
fields, incorrect/noninteger counts and nonfinite or nonscalar returns fail.
An episode with both boundary flags is truncated, not natural.

The Pong-only readers explicitly reject other environment names and derive wins
from Pong episode records. The current three-seed mastery protocol is still
locked to v1; a test ensures it cannot silently accept v2. Future experiments
need their own declared protocol, not retroactive relabeling of the current gate.

## Validation before adoption

All 198 CPU tests pass. They cover both schemas, misleading positive scores, empty summaries,
timeouts and simultaneous flags, malformed data, and a full mocked vector
runner producing a valid v2 ledger without constructing a GPU agent. The staged
checkout uses a byte-identical copy of the active native library for CPU API
tests; no installed extension is replaced and no native code changes.

The candidate reader also checks the original complete v1 zero-update control,
seed-0 training and seed-0 frozen evaluation. All action/update, episode,
positive-return and score counts agree with their preserved original audits.
Pong's final result remains 18 wins in 18 games, mean +10.2778, which fails the
declared mastery gate. The unfinished seed-1 40k prefix is still rejected for
missing `run_end`. These are read-only compatibility checks, not new evaluations.

After the pinned queue completes, verify a short native v2 pixel run before
adoption. CPU mock integration is not a claim of native runtime validation.
This candidate is independent of the isolated learner profiler branch.

## Native adoption gate, 2026-09-08

The follow-on `exp/atari-five` branch carries this patch onto the validated
Meganeura refresh. Its isolated package passes 281 Python tests before launch.
A fresh Boxing run completes 3,072 N8 training actions and 387 updates, then
restores into 8,192 frozen N1 actions: four natural matches, no timeouts and
zero learning updates. Both v2 ledgers pass; the final checkpoint has 241
finite tensors and matching saved/restored identities. This clears native
accounting integration, not a Boxing mastery gate. The five-game pilot uses
the v2 runner while the historical main-checkout Pong controls remain intact.
The new match/declaration scorer brings the candidate Python suite to 313 tests.
Artifacts: `/x/Code/kindle/runs/atari-five-20260908.db0XSW`.
