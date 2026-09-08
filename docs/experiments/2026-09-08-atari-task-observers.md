# Atari task observers and CPU rollout reconstruction

Implemented on `exp/atari-five` at `865456e`. No native learning code or active
runner/package was changed. The candidate Python suite passes 338 tests.

## What is measured

`python/examples/atari_tasks.py` has small episode observers for the five
pinned ROMs. Pong and Boxing require a positive **natural final match** result,
not an early lead. Freeway requires 25 crossings in a complete natural round.
Breakout records the first attainment of 864 points: both walls, according to
the [original Atari manual](https://www.atariage.com/2600/manuals_old/breakout.html).
Qbert records the first simultaneous conversion of all 21 initial cubes.

Only Qbert needs RAM to identify task completion. This is an evaluation-only
observer; RAM is never passed to Kindle or used to modify the game. The cube
display-color addresses come from the
[pinned OCAtari extractor](https://raw.githubusercontent.com/k4ntz/OC_Atari/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram/qbert.py)
and are checked on the actual ROM. An initial board of color 148 must be seen
before all 21 cubes have target color 26. The maximum is simultaneous target
cubes, not a sum of visits: enemies can revert previously converted cubes.
The observer latches the first completion before the subsequent color animation.

These are task events, not a five-game mastery certificate. A Qbert first
pyramid is an initial progress milestone; sustained competence needs the
additional final-score/progression gate in the five-game plan. Breakout and
Qbert preserve completion observed before a later time cutoff, without calling
that cutoff a natural win. Partial final episodes remain ungraded, with their
observed milestones retained separately. Freeway cutoffs do not count as
complete timed-round success.

## Actual ROM checks

Artifacts are in
[`runs/atari-task-observers-20260908.TzCy2B`](../../runs/atari-task-observers-20260908.TzCy2B/).
`observer-fixtures.json` records policies, actions, frame counts and outcomes.
No agent was constructed; there was no learning, state cloning, RAM writing,
reward shaping or ingestion of these fixtures into Kindle's replay.

- A privileged **scripted fixture**, not Kindle, completes Qbert's first
  pyramid at actual frame 1,084. All recorded actions, rewards, boundaries and
  128 RAM bytes replay exactly. Continuing for 240 NOOP frames observes 3,100
  bonus points; the first-completion marker remains 1,084. The fixture video
  contains all 1,324 frames, including the completion animation.
- Qbert idle and repeated-fall fixtures do not trigger completion. Unit tests
  reject initialization, large incidental scores, and separate visits to all
  cubes when they are never simultaneously converted.
- Three simple paddle-controller fixtures at action repeat 4 earn positive
  Breakout returns 25/21/15 and are correctly **not** marked complete. Unit
  checks reject 432 and 863, and accept a genuine 864-point milestone. A real
  positive two-wall fixture remains unverified; do not claim it from those
  synthetic boundary checks. Exploratory one-frame controllers also failed
  to clear both walls and are not Kindle results.
- Holding UP in Freeway earns 21 in each complete 8,192-frame round at
  environment seeds 9001/9002/9003. The fixed deterministic controller has
  identical outcomes; these are not independent learned policies. The planned
  25-crossing bar requires more than this simple control.

## Replay and videos without more GPU time

`python/examples/replay_atari.py` audits the complete frozen vector log and
replays every recorded action in fresh environments. It verifies actual rewards,
terminal/cutoff flags, resets and emulator frame counts, while collecting task
outcomes per episode. ROM, ALE native and wrapper identities are checked against
the source manifest. No policy is constructed or updated.

An optional movie records **every episode of the chosen stream**, including
losses, with one frame per actual emulator step at nominal 60 Hz. It is not a
new evaluation or original screen recording. Original raw pixels are unavailable
for direct comparison; reconstruction verifies the recorded trajectory fields.

The completed Boxing integration replay checks all 8,192 actions and four
natural matches: two wins, mean return −1. The movie has exactly 32,756 frames,
160×210 at 60 fps. This nearly untrained integration checkpoint is not a
Boxing competence result or the campaign's untrained control.

## Automatic frozen-result follow-up

The live pilot has a separate **CPU-only** follower:
`runs/atari-five-20260908.db0XSW/analyze_when_ready.py`.
It is bound to the verified live launcher process, waits for each completed and
accounted frozen run, then checks final checkpoint/declaration identities and
reconstructs all eight evaluation streams plus a complete stream-0 movie.
It never starts GPU work, stops the trainer, selects a recipe, or treats one
seed as reliability. Its source, dependencies and schema reference are pinned;
do not modify them while it is live.

Results populate the local
[frozen Boxing report](../../runs/atari-five-20260908.db0XSW/analysis/report.html).
Rows are explicitly pending until their frozen data and CPU audits complete.
Training continues under the original pinned launcher; a stopped/missing
launcher is an error, not a reason to restart or silently resume training.
