# Five-game learning campaign: Boxing replay-ratio pilot

Declared and launched 2026-09-08. **R64 seed 0 passes its frozen Boxing gate;
the pilot and three-seed reliability are incomplete.**
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
| Boxing | Win the match; mean score difference ≥50 and ≥90% natural wins over ≥20 games, no timeouts | R64 seed 0 passes: 40/40 frozen wins, mean +51.55 versus untrained +0.125; R256 and replication pending |
| Freeway | ≥25 crossings per complete timed round; mean ≥25 and ≥90% qualifying rounds over ≥20 natural rounds, no timeouts | Observer verified against the actual ROM; holding UP scores 21 |
| Breakout | Clear both walls: 864 points is the original one-player win, not merely a positive score | Positive and negative actual-ROM fixtures verified; declare its training/evaluation budgets next |
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
seeds **1009, 2017 and 3019** before a reliability claim. Any longer budget applies
to every replication seed and is fixed before starting them.

These seed numbers are selected before replication or final pilot results.
The earlier adjacent-seed example is deliberately replaced: each N8 model
seeds live policy/posterior streams with `config.seed + stream`, followed by
the corresponding RNG-domain XOR. Adjacent roots therefore reuse seven of
eight seeded live streams. Distinct model initialization and replay RNGs mean
the trajectories need not match, but this is unwanted shared randomness in
the replication. The selected live-base ranges are 1009–1016, 2017–2024 and
3019–3026, mutually disjoint and separate from the pilot/integration ranges.
This is a protocol choice, not a runtime change or an explanation of Pong's
variation. Preserve the historical results and failed all-seeds gate.
The source-bound check is `runs/learning-review-20260908.W6fAHO/replication-seeds.json`.
Training/evaluation budgets and the chosen recipe remain to be declared before
those fresh runs; changing seed spacing alone does not establish reliability.

### Replication acceptance checker

The isolated candidate now includes `python/examples/audit_atari_campaign.py`.
It requires all 15 game/seed records, unchanged task criteria, one complete
declared config, per-game fixed action budgets, fresh training, sampled frozen
evaluation, final-checkpoint identity, complete tensors and replay binding.
Inputs include content-pinned source, backend, encoder, ROMs and checkpoint
schema. Reused artifact paths or trained tensor states are rejected. Every
seed must pass; two good seeds cannot hide a third failure.

The declaration uses `kindle-atari-five-replication-v1`, seeds 1009/2017/3019
and N8 training/evaluation. Its complete `config` omits `seed`; each run supplies
that declared seed. `header` binds the remaining common runtime/model identity;
`games` fixes budgets and criteria, and `runs` fixes the 15 train/evaluation/
checkpoint/replay paths. Declare and pin this manifest before launching its
first run; the checker also rejects recorded starts before declaration time.
No replication recipe or budgets have been selected by adding this checker.

Its 76 new CPU tests bring the candidate suite to 477 passes. The real retained
R64 and untrained 75k-action/40-match replay bindings pass; altered scores fail,
and the real seed-0 pilot cannot be reused as fresh seed 1009. Evidence is in
`runs/atari-campaign-audit-20260908.yhmold/`, including `pytest-final.xml` and
`completed-boxing-check.json`. These are checker validations, not 15 trained
models. `replication_passed` covers the declared final task gates; untrained
controls and the broader goal-completion audit remain separate.

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
to Pong and Boxing; other games are rejected by that match-specific entry point. It includes
stream-resampled mean-return intervals and descriptive Wilson win intervals,
neither of which measures variation across independent training seeds. Its
optional `--declaration` additionally checks the pilot's budgets, seeds,
recipe, frontend and content pins; a generic match score alone does not
certify that a declared campaign was followed.
The original Pong mastery auditor and its decision are not reinterpreted.
The separate [task scorer](2026-09-08-atari-task-observers.md#frozen-task-scoring)
now handles the declared Freeway, Breakout and Qbert gates, including complete
replay binding and final-checkpoint checks. Its 63 new tests bring the candidate
suite to 401 passes. It does not certify campaign budgets or independent seeds;
those remain required before declaring reliable five-game results.

Validation: 281 Python tests passed on the built package before launch; the
separate match/task scoring and replay additions bring the candidate suite to
401 passing tests. Native v2 integration is complete. Complete pilot scores
remain pending. No other learned Atari game is certified by this declaration
or by the CPU adapter results.

The [task-observer follow-up](2026-09-08-atari-task-observers.md) adds actual
Qbert/Freeway ROM checks and CPU-only full-trajectory reconstruction. Its pinned
follower published the R64 result in the [frozen Boxing report](../../runs/atari-five-20260908.db0XSW/analysis/report.html),
including its complete stream-0 video. The other rows are still pending.
The retained first 20k-action R64 checkpoint also has all 241 expected
tensors, finite values and nonnegative optimizer second moments; this checks
numerical health, not seed reliability.

### Completed R64 training and frozen gate

The fresh R64 arm ran from 08:27:26 to 11:17:16 UTC and exited successfully at
exactly 200,000 actions and 12,405 updates. Its complete version-2 ledger passes:
104 natural training episodes, no cutoffs, 75 positive returns and mean return
13.5. This mean includes early exploration and learning; neither it nor later
training wins is a final-policy competence score. The remaining half-update
credit is the expected fractional remainder, not an accumulating backlog.

The action loop took 10,122.083 s: 19.759 actions/s, **1.317× aggregate real
time and 0.1646× per stream**. Including process startup and shutdown, the
launcher measured 10,189.482 s, 19.628 actions/s and 1.308× aggregate real time.
The complete loop spent 56.3% observing (perception plus posterior inference),
42.5% learning and 1.1% in the emulator. These are end-to-end pilot costs with
disclosed CPU overlap, not calibrated GPU idle gaps or an equal-recipe speedup.

The final-checkpoint 75k-action frozen evaluation ran from 11:17:19 to
11:55:22 UTC. It finished with **40/40 natural wins, mean +51.55, no draws,
losses or cutoffs, and zero updates**. Eight partial final episodes are not
graded. This passes the unchanged ≥20-game, ≥50-mean, ≥90%-win gate for this
one policy. The stream-bootstrap 95% mean interval is [47.25, 55.325], and
the descriptive Wilson win interval is [0.9124, 1]. These describe the frozen
policy, not variation across independently trained models; the score margin
is modest and three-seed reliability is still untested.

The independent scorer verifies the original declaration, final 200,000/12,405
counters, saved/restored identity and all 241 complete finite tensors. CPU
reconstruction exactly checks every action, reward, boundary, reset and actual
frame count across all eight streams. Original raw pixels were not recorded.
The [full stream-0 movie](../../runs/atari-five-20260908.db0XSW/analysis/boxing-r64-seed0-eval.mp4)
contains 37,485 frames at 60 Hz (10m25s): all five matches, with returns
+54/+56/+60/+60/+67, then the ungraded tail. Stream 0 was specified before
results, not selected for its wins. H.264 metadata and a decoded first-match
frame were also inspected; its 82–28 score agrees with the +54 ledger.

The authoritative records in the declared root are the complete train/eval
JSONL files, `analysis/boxing-r64-seed0-eval.score.json` and
`analysis/boxing-r64-seed0-eval.replay.json`. The evaluation log SHA-256 is
`f17c7cfee0fecc3abb34e338519e3a026789a36a67e1ecd3a4231981c5db77c6`;
the movie SHA-256 is
`8f5f471c7e2530677cf68bccd4f7402233747f7133e9e5d15c7aa50151ae1247`.

### Preserved zero-update restore failure

The untrained initialization completed its declared eight actions and zero
updates, but its frozen restore exited at 11:57:47 UTC before any evaluation
actions. Meganeura lazily omits unallocated optimizer moments when saving;
Kindle's strict training restore rejected missing
`adam_m.world.dynamics.core.dynin0.weight`. The launcher and CPU follower both
stopped. Preserve their failure records and completed R64 evidence; the
original queue must not be restarted.

The save-only repair and hardware/continuation gates are declared in
`runs/zero-update-checkpoint-20260908.yxaalA/declaration.md`. The repair is
implemented in main `1acaca7` / candidate `9d9cec4`: fresh saves materialize the
initial zero moments without taking an optimizer step. Already allocated state
and all restore requirements remain unchanged; old incomplete zero-update
checkpoints are not silently accepted.

Validation passes: workspace formatting/Clippy, 92 Rust CPU tests, four GPU
regressions (fresh/acted zero-update saves, trained restore, vector-one/serial
learning and logical-cache restore), and 401 Python tests against the isolated
actual package. The real N8/12M/B16/T64 initialization matches every original
action and all 95 original parameter tensors exactly, adding only 146 complete
F32-zero optimizer tensors. Restoring/resaving the trained R64 model retains all
241 tensor payloads and every saved counter exactly. The two pixel checks peak
at 14,094 MiB, leaving 2,209 MiB. Their manifests, raw logs and comparisons are
in `runs/zero-update-checkpoint-20260908.yxaalA/`.

### Live continuation

The remaining stages started at 12:29:21 UTC in
[`runs/atari-five-continue-20260908.JrdVto`](../../runs/atari-five-continue-20260908.JrdVto/).
Its [current report](../../runs/atari-five-continue-20260908.JrdVto/report.html)
links both completed score audits and full-stream movies; R256 remains pending.
The original report and failure artifacts stay unchanged.

The control reused the verified eight-action initialization and repaired native
package `9cd176c1…`. Its full 75k-action sampled N8 evaluation finished at
13:07:22 UTC with **21 wins, six draws and 13 losses over 40 natural matches,
mean +0.125, no cutoffs and zero updates**. Its stream-bootstrap 95% mean
interval is [−2.35, 2.6]; the descriptive Wilson win interval is [0.3750, 0.6706].
All 241 checkpoint tensors and the complete replay pass their checks. Eight
partial final episodes remain ungraded. The H.264 movie contains all five
stream-0 matches (−1/+2/+5/−27/−3) and the tail: 37,485 frames at 60 Hz.

| Frozen policy | Natural wins / matches | Mean score difference | Declared gate |
| --- | ---: | ---: | --- |
| Initial seed-0 weights, zero updates | 21/40 | +0.125 | Fail |
| R64, seed-0 final 200k-action checkpoint | 40/40 | +51.550 | Pass |
| R256, seed-0 final 200k-action checkpoint | Pending | Pending | Pending |

The observed mean gap is +51.425 points. This supports substantial learned
improvement for this Boxing seed, not reliability across training seeds or a
selected replay ratio. Both frozen runs use the declared 75k-action/N8 sampled
protocol. Preserve the disclosed save-only package difference for the control;
its initial parameters/actions match exactly, as verified above.

The control's source log SHA-256 is
`78800eb98d36f073bdd0b98250522d0fdc810ce477414d1f34bab4a6aec2d1c2`;
the movie SHA-256 is
`80673b91da8c7d695f945dc114ac777cfcf81bcafee60d6352993728b15ba4f7`.
The complete evidence is `boxing-untrained.{score.json,replay.json,mp4}` in
the continuation root.

Fresh R256 training started at 13:07:57 UTC with the original `831c631d…`
package, matching the completed R64 arm. Its header verifies seed 0, no restore,
zero starting counters, N8 and the declared 200k-action/R256 LeVJEPA recipe.
The first 2k actions produce the expected 119 updates with zero debt; early
throughput includes prefill and is not a steady-state result. Its final 75k
frozen evaluation is still pending. The continuation pins both packages,
initial state, repair evidence, old results, runner and auditors. It scores and
reconstructs each complete frozen run without automatic recipe selection or
restart. Finish R256 before choosing a ratio; fresh three-seed replication and
the other games still remain.

The scheduled 20k-action R256 checkpoint is retained in
`runs/boxing-r256-health-20260908.YeSTeX/checkpoint`. Its CPU check binds the
recorded save and log prefix, verifies all 241 complete finite tensors against
the validated logical schema, and checks nonnegative optimizer second moments.
World/behavior native counters and Kindle's learner counter all equal 4,619;
the slow-value copy has zero optimizer steps, and training debt is zero.
`summary.json` records this numerical-health check, not a final-policy or
seed-reliability result. No GPU restore/evaluation or live recipe change occurred.

### Fixed 40k–50k runtime comparison

Both intervals are complete. The R256 window was fixed before reaching 40k;
the comparator verifies matching native/frontend/protocol/config identity
apart from the replay ratio. Each interval executes 10,000 actions and 39,976
actual emulator frames across eight streams. This is a descriptive pilot cost
with permitted CPU overlap, not a calibrated AB/BA benchmark or a quality gate.

| Replay ratio | Learner updates | Elapsed seconds | Actions/s | Aggregate / per-stream real time | Mean GPU activity |
| --- | ---: | ---: | ---: | ---: | ---: |
| R64 | 625 | 507.845 | 19.691 | 1.31195× / 0.16399× | 82.12% |
| R256 | 2,500 | 1,150.461 | 8.692 | 0.57913× / 0.07239× | 68.59% |

Observation costs are nearly unchanged: 284.987 versus 288.246 seconds.
Learning costs 216.736 versus 855.981 seconds, or 346.8 versus 342.4 ms/update.
Thus learning occupies 74.4% of R256's interval, while observing occupies 56.1%
of R64's. Emulation costs 5.394 seconds in either arm. Observation includes
perception and posterior inference; these host stages are not pure encoder or
kernel timings. Reprofile the dominant stage of the eventual replicated recipe,
not GPU activity alone. R64's lower cost also buys four times fewer updates;
finish the frozen comparison before selecting a ratio.

The 508/1,150 GPU samples cover both windows, with maximum gaps of 1.001/1.018
seconds. Peak memory is 14,212 MiB in both; mean sampled power is 185.75/153.64 W.
Coarse activity and power do not identify occupancy or calibrated idle gaps.
Source-log and GPU-monitor prefixes, endpoints, coverage checks and the
comparison are in `runs/boxing-ratio-window-20260908.ybLT7Z/summary.json`.

Preserve the earlier R64 artifact in
`runs/atari-task-observers-20260908.TzCy2B/boxing-r64-40k-50k.json`. Its nominal
four-frames/action conversion gave 1.31274× aggregate real time. The new result
uses the recorded 39,976 frames, correcting that small difference without
changing any raw timing, action or score evidence.

### Bounded learning-equation review

A read-only comparison with the local pinned upstream DreamerV3 source and
the actual adopted Meganeura checkout finds no new defect in replay context,
reward/action alignment, reset masking, return indexing, continuation weights,
online-value targets, slow-value regularization or the AGC → RMS → momentum
optimizer sequence. Upstream's configured `slowtar: False` matters; inspecting
only the helper's default would give the wrong comparison. This source review
is not a new numerical equivalence test for the complete architecture.

The source-review scope and original R64 prefix-bound summary are preserved in
[`runs/learning-review-20260908.W6fAHO`](../../runs/learning-review-20260908.W6fAHO/).
The [observer follow-up](2026-09-08-atari-task-observers.md#verified-breakout-two-wall-fixture)
now also verifies a real scripted 864-point Breakout fixture, not a Kindle win.

### First-80k reward fitting, not held-out forecasting

A post-hoc CPU comparison now covers the same two 40k-action windows in both
arms. It verifies matching header/config identity except replay ratio, all
transition/update counts, extrinsic-only reward storage, and exact reproduction
of the original R64 prefix and summary. Both arms encounter positive and
negative rewards well before learning strong reward-sign separation.

| Ratio | Action window | Updates | Actual positive / negative reward events | Mean prediction on positive / negative replay rows |
| --- | --- | ---: | ---: | ---: |
| R64 | 1–40k | 2,405 | 614 / 582 | +0.00048 / +0.00043 |
| R64 | 40k–80k | 2,500 | 742 / 732 | +0.00728 / −0.01132 |
| R256 | 1–40k | 9,619 | 656 / 651 | +0.05241 / −0.10211 |
| R256 | 40k–80k | 10,000 | 1,499 / 1,091 | +0.69681 / −1.08658 |

Actual nonzero rewards have magnitude one or two. Predictions are weighted by
the number of replay rows of each sign, including repeated experience. They
come from the replay posterior before the learner update: the model has already
consumed the target frame. These are conditional prediction means, not per-event
errors or prior forecasts on a common held-out distribution.

R256 separates reward signs earlier by interaction count while spending roughly
four times the updates. Its own training trajectories also differ. This does
not select a ratio, explain Pong's seed variation, or establish final-policy
quality. In particular, R64's weak early fitting did not prevent its later
frozen pass. Finish the unchanged fixed-budget comparison before choosing an
intervention. Prefix hashes, counts, behavior statistics and the reproducible
script are in `runs/boxing-reward-fit-20260908.cDT725/summary.json` and
`compare_prefixes.py`; the original R64 evidence remains unchanged.
