# Atari task observers and CPU rollout reconstruction

Implemented on `exp/atari-five` at `865456e`. No native learning code or active
runner/package was changed. The candidate Python suite passes 338 tests.

## What is measured

`python/examples/atari_tasks.py` has small episode observers for the five
pinned ROMs. Pong and Boxing require a positive **natural final match** result,
not an early lead. Freeway requires 25 crossings in a complete natural round.
Breakout records the first attainment of 864 points: both walls, according to
the [original Atari manual](https://atari.com/pages/breakout).
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
  checks reject 432 and 863. The subsequent actual two-wall fixture below
  verifies the positive milestone independently of those synthetic checks.
- Holding UP in Freeway earns 21 in each complete 8,192-frame round at
  environment seeds 9001/9002/9003. The fixed deterministic controller has
  identical outcomes; these are not independent learned policies. The planned
  25-crossing bar requires more than this simple control.

### Verified Breakout two-wall fixture

A privileged one-frame paddle controller with varied return angles produces
three natural game results: **432 / 827 / 864**. These are scripted observer
fixtures, not Kindle, learned policies or demonstration data. All three replay
exactly in fresh CPU environments, checking every action, reward, boundary,
emulator frame number and BCD score. Only the 864-point game passes the observer.
Its first wall clears at frame 11,395, the second at 21,186, and a natural
terminal follows at 21,223. The partial episode at the milestone remains ungraded.

An independent RAM bitmap check counts 108 initial bricks and zero at each
wall clear, using masks derived from the
[pinned OCAtari source](https://raw.githubusercontent.com/k4ntz/OC_Atari/99c874675df6b76a33a80b57776c123fbcd051af/ocatari/ram/breakout.py).
The first verifier mistakenly required every brick-storage byte to be zero:
unused low bits remain set. Its failed assertion, source and movie are retained;
the separately named v2 verifier corrects that auxiliary check. The production
score-based observer and acceptance threshold never changed.

Evidence in the artifact directory:

- `breakout-vary-angle-fixtures.json` and all three raw action logs.
- `breakout-scripted-verification-v2.json`: complete fresh-replay verification.
- [Verified fixture video](../../runs/atari-task-observers-20260908.TzCy2B/breakout-scripted-observer-fixture-verified.mp4):
  all 21,223 raw frames, 160×210 at 60 fps. This is the selected **scripted**
  positive fixture, not a Kindle rollout or an unbiased policy evaluation.
- `breakout-verification-v1-failure.md`: preserved first-verifier failure.

Earlier constant-angle raw-frame fixtures are also retained. Three reach
65,536 frames, where ball motion stops while paddle inputs still work; ALE
reports neither terminal nor truncation. This is not an emulator-wide halt,
and its cause is unresolved. They remain failed/partial fixtures, bounded by
their declared budgets. Do not invent a natural terminal or task completion
from that state. The production wrapper's existing 100,000-frame cutoff bounds
such episodes; the current short-round Boxing pilot is unaffected.

## Frozen task scoring

`exp/atari-five` commit `8a04c85` adds `python/examples/audit_atari_tasks.py`
without editing any dependency pinned by the live Boxing follower. Together
with the match auditor, all five games now have explicit scoring paths. The
new scorer requires complete fresh-training and frozen-evaluation ledgers,
the declared final saved/restored checkpoint and a matching full CPU replay.
It checks replay source/content identities, every completed episode's original
ledger fields, contiguous per-stream frame ranges and the partial-tail ledgers.

Freeway needs the declared crossing mean/fraction and no timeouts. Breakout
needs the two-wall completion fraction. Qbert needs both the first-pyramid
completion fraction and the ≥15,000 mean final score; the first pyramid alone
still fails. A later cutoff preserves Breakout/Qbert milestones without
becoming a natural win, and partial tails never enter the completed-episode
mean or success fraction. Duplicate episodes and inconsistent task evidence
are rejected.

The output's `task_gate_passed` describes one frozen policy. It explicitly leaves
`campaign_declaration_verified` and `reliability_assessed` false: fixed campaign
budgets, seed independence and all-seed acceptance must also be verified.
No learned final checkpoint on these three games has been scored yet.

For a future completed experiment, select the isolated package and run:

```bash
PYTHONPATH=/x/Code/kindle/runs/atari-five-20260908.db0XSW/package \
  /x/Code/kindle/python/.venv/bin/python \
  /x/Code/.kindle-atari-five/python/examples/audit_atari_tasks.py \
  --train TRAIN.jsonl --evaluation EVAL.jsonl --checkpoint FINAL_CHECKPOINT \
  --schema VALIDATED_SAME_MODEL_CHECKPOINT --replay EVAL.replay.json \
  --output FRESH.score.json
```

All **401 Python tests pass**, including 63 new threshold, replay-binding and
learning-lifecycle tests. The new scorer also accepts all nine actual recorded
Freeway/Breakout/Qbert fixture outcomes, grading each complete fixture
individually below the 20-episode minimum. These checks are not agent evaluation.
The test report and content-pinned fixture validation are in
[`runs/atari-task-scoring-20260908.eOahCe`](../../runs/atari-task-scoring-20260908.eOahCe/).

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
