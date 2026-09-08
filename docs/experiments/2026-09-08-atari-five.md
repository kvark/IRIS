# Five-game learning campaign: Boxing replay-ratio pilot

Declared and launched 2026-09-08. **In progress, not a competence result.**
The active objective is reliable learning and wins/task completion on five
Atari games. The completed backend refresh and world-model diagnostics are
enabling work, not satisfaction of that objective.

## Panel and success definitions

Use Pong, Boxing, Freeway, Breakout and Qbert. These test interception,
opponent interaction, sparse crossings, paddle/ball control and navigation.
Keep the other titles from the eight-game adapter panel as later breadth tests.
Training separately on five games establishes algorithm breadth, not one
policy that transfers between them.

Require every seed of a fresh three-seed replication to pass its declared
final evaluation. Pilot seeds used for recipe selection do not count as an
independent replication. Preserve the failed original three-seed Pong gate.

| Game | Task meaning | Status of acceptance protocol |
| --- | --- | --- |
| Pong | Win the match; mean return ≥15 and ≥90% natural wins over ≥20 games | Existing 200k-action recipe fails two of three seeds; new recipe must be separately declared |
| Boxing | Win the match; mean score difference ≥50 and ≥90% natural wins over ≥20 games, no timeouts | Implemented for the pilot; frozen scores pending |
| Freeway | ≥25 crossings per complete timed round; mean ≥25 and ≥90% qualifying rounds over ≥20 natural rounds, no timeouts | Observer verified against the actual ROM; holding UP scores 21 |
| Breakout | Clear both walls: 864 points is the original one-player win, not merely a positive score | Validate completion and cutoff handling before its training/evaluation protocol |
| Qbert | Sustained progression: first-pyramid completion in ≥90% of ≥20 complete episodes, plus mean final score ≥15,000 | First-pyramid observer verified; one cleared pyramid alone is not mastery |

The score differences and natural termination used for Boxing are implemented
in the [pinned ALE 0.12.1 game source](https://raw.githubusercontent.com/Farama-Foundation/Arcade-Learning-Environment/v0.12.1/src/ale/games/supported/Boxing.cpp).
Freeway awards a point for each crossing, rather than an opponent-match win;
see the [official game description](https://ale.farama.org/environments/freeway/).
Breakout's two-wall/864-point goal comes from the
[original Atari manual](https://www.atariage.com/2600/manuals_old/breakout.html).
Qbert's [official task description](https://ale.farama.org/environments/qbert/)
requires changing all cubes to the target color. Any privileged completion
observer stays outside policy inputs and does not add reward shaping.

For Breakout, require both walls in ≥90% of ≥20 completed episodes for every
replication seed. For Breakout and Qbert, retain task completion observed before
a subsequent time cutoff without relabeling that cutoff as a natural terminal;
a cutoff without the task is a failure. Partial final episodes remain separate.
These bars are fixed before Kindle trains on these titles. Declare sufficient
fixed evaluation budgets and the common three-seed training budget before their
campaign; passing a wiring fixture or only the Qbert first-pyramid milestone
does not meet the five-game goal.

## Bounded first comparison

Boxing provides frequent signed feedback, making it a useful next test of
reward learning while Pong's sparse-event reliability is unresolved. Compare
replay ratios **64 and 256** at equal interactions. A lower replay ratio changes
learning compute and is not an identical-recipe runtime optimization.

- One independent model per arm, seed 0, fresh weights and replay; no restore
  from Pong, the integration check, or the other ratio.
- LeVJEPA, 12M, eight collection streams sharing one learner, B16/T64,
  full 64-step recurrence, 16-row world microbatch, learning rate 0.00004,
  1,000-update warmup, AGC 0.3, reconstruction 0 and causal prediction 0.25.
- Each arm executes 200,000 aggregate actions. Evaluate the final checkpoint
  for 75,000 sampled actions across eight independent streams with zero updates.
  Eight frozen streams are deliberate for this new experiment; the historical
  Pong gate used one. Do not compare these as identical evaluation protocols.
- Both frozen arms use environment base seed 100,000 and the same stream rule.
  The untrained control restores seed-0 initial weights, saved after eight
  frozen actions and zero updates, then uses that same evaluation protocol.
- Published wrapper: all 18 actions, action repeat 4, zero sticky actions and
  reset no-ops, 100,000-frame episode cap. Pin the actual ROM, ALE native,
  wrapper, runner, encoder, package and backend identities in the manifest.
- Run order: native v2 integration, R64 training/evaluation, untrained control,
  R256 training/evaluation. Serialize GPU work. CPU development/analysis may
  overlap; these are useful end-to-end learning costs, not calibrated AB/BA
  timing windows. Preserve all failures and partial final episodes.

Do not select a ratio automatically from one seed. Compare final frozen
quality against the untrained policy as well as actions, updates, wall time,
aggregate/per-stream game clocks and GPU memory. If both fail, that is a
learning result requiring diagnosis, not permission to call positive rewards
mastery. A selected recipe needs a new, predeclared replication on three fresh
seeds (for example 3, 4, 5) before a reliability claim. Any longer budget applies
to every replication seed and is fixed before starting them.

## Implementation and artifacts

Candidate worktree: `/x/Code/.kindle-atari-five`, branch `exp/atari-five`.
The pinned runner source is `1e94c194c343b2de6a562176aabfc720fc5ddba4`,
carrying the staged version-2 accounting onto the adopted Meganeura refresh.
Main's historical runner/native controls remain unchanged.

All artifacts are in
[`runs/atari-five-20260908.db0XSW`](../../runs/atari-five-20260908.db0XSW/):

- `manifest.json`, `run_boxing.py`, `events.jsonl`: declaration, content pins,
  owned child PIDs, command outcomes and accounting gates. No automatic resume.
- `package/`: actual isolated wheel installation; native SHA-256
  `831c631dbfe43832dda0945fdc88b781ebc21ac281da815157a037af44a49022`.
  Rust/backend code is unchanged from the validated refresh. The different
  extension hash is a new build from the candidate source/worktree.
- `integration-{train,eval}.*`: bounded native accounting/restore check, not
  a gameplay benchmark. Training completed 3,072 actions and 387 updates;
  the 8,192-action N1 frozen check completed four natural matches, zero updates
  and no timeouts. Both ledgers pass. All 241 checkpoint tensors are finite
  and match the validated model schema; final saved/restored identities agree.
- `boxing-r{64,256}-seed0-{train,eval}.*`, final checkpoint directories,
  `boxing-untrained.*`, `gpu.csv`: pilot outputs as they are produced.

The candidate's `python/examples/audit_atari.py` reuses the vector ledger and
checks final saved/restored identities, logical tensor names/shapes/dtypes,
finite weights and optimizer moments. Match scoring is intentionally limited
to Pong and Boxing; other games fail until their rules exist. It includes
stream-resampled mean-return intervals and descriptive Wilson win intervals,
neither of which measures variation across independent training seeds. Its
optional `--declaration` additionally checks the pilot's budgets, seeds,
recipe, frontend and content pins; a generic match score alone does not
certify that a declared campaign was followed.
The original Pong mastery auditor and its decision are not reinterpreted.

Validation so far: 281 Python tests on the built package before launch; 313
including the new scorer/declaration tests. Native v2 integration is complete;
the fresh R64 learning arm started at 08:27:26 UTC. The first post-warmup
2k–3k-action interval runs at 19.65 actions/s, about 1.31× aggregate and 0.164×
per stream; this is an early learning-cost observation, not a final timing or
quality result. Complete pilot scores remain pending. No other learned Atari
game is certified by this declaration or by the CPU adapter results.

The [task-observer follow-up](2026-09-08-atari-task-observers.md) adds actual
Qbert/Freeway ROM checks and CPU-only full-trajectory reconstruction. Its pinned
follower will populate the [frozen Boxing report](../../runs/atari-five-20260908.db0XSW/analysis/report.html)
with both final ratio arms, the zero-update control and complete stream-0
videos. The candidate suite now passes 338 tests. The retained first 20k-action
R64 checkpoint also has all 241 expected tensors, finite values and nonnegative
optimizer second moments; this checks numerical health, not seed reliability.

### Longer early runtime window

The fixed 40k–50k R64 interval executes 10,000 actions and 625 updates in
507.845 seconds: **19.691 actions/s, 1.313× aggregate real time, 0.164× per
stream**. Its 508 GPU samples average 82.12% activity and peak at 14,212 MiB.
Python stage totals are 284.987 s observing (56.1%), 216.736 s learning (42.7%),
5.394 s emulating (1.1%), and less than 1 s combined acting/resetting. Observation
time includes perception and posterior inference; it is not a pure encoder
measurement or a calibrated GPU idle-gap measurement.

This is a descriptive window with disclosed CPU analysis overlap, not an
AB/BA benchmark or a learning-quality result. It changes which stage deserves
the next profile **if** R64 preserves useful learning. The first four complete
training episode cohorts have means −0.25, +1.5, −0.125 and −3.375. They do not
yet demonstrate useful improvement, and do not replace the final frozen gate.
The prefix-bound measurement and raw interval endpoints are in
`runs/atari-task-observers-20260908.TzCy2B/boxing-r64-40k-50k.json`.
