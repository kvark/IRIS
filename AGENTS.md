# Kindle working direction

Kindle is a Rust agent that learns while acting. Each environment contributes
its own continuing stream of experience; vector collection shares one learner
and policy without joining causal histories. Games are the first testbed.
Intrinsic motivation and experience sharing between independent Kindles are
long-term goals; game rewards and human guidance are allowed while establishing
reliable learning.

- Favor minimalism, expressiveness, safety, and speed. Keep the native learning
  and inference path on Meganeura and Blade. Python is for adapters, controls,
  and analysis.
- Maintain one authoritative research plan at
  `docs/kindle_single_life_dreamer_plan.md`. Keep its claims tied to code and
  measured results. Experiment logs belong in `runs/`, not in an ever-growing
  chronological plan.
- Prioritize one learning actor: Atari breadth, video/world pretraining and fast
  accelerated playing plus training, then mind-games (vkQuake2/TMNF), GOG/Wine games,
  cross-game adaptation and retention. Pong's initial-learning gate is achieved,
  not consistent mastery. Require strong single-actor GOG and transfer results
  before swarm learning. Prioritize vectorized environments and batched live
  inference for one shared learner/policy, as explicitly requested. This is a
  collection/throughput protocol, not swarm learning or separate learner services.
  Keep each environment's visual cache, recurrent belief, RNG and replay sequence
  independent. Count actual interactions across all environments, not vector ticks;
  preserve train-ratio credit and report aggregate and per-environment throughput.
- Preserve a measured Dreamer control. Add JEPA-style prediction as a causal,
  action-conditioned objective that predicts an observation before consuming
  it. Predicting the current frozen DINO features from the posterior is already
  the existing feature-reconstruction control.
  Retain the accepted DINOv3 plus causal-prediction Pong controls. Native
  batched DINO is an explicit matched-control candidate, not a default frontend
  switch; require GPU stream parity and a newly declared comparison before use.
  All three LeVJEPA vectorized seeds have completed frozen evaluation. Seed 2
  passes the predeclared mastery gate; seeds 0 and 1 fail, so the recipe does
  not pass the all-seeds gate.
  Do not equate training wins or one frozen seed with three-seed mastery,
  or describe the DINO stepping stone as the full video pivot.
  Native LeVJEPA work and the stronger, predeclared three-seed Pong mastery gate
  are tracked in `docs/experiments/2026-09-06-levjepa-pong.md`. Its 16-arrival
  causal chunks reset only perception; episode boundaries also reset belief.
  Do not confuse chunked prefixes with a sliding window or reset the RSSM every
  chunk. Checkpoint format 3 records the actual frontend and encoding semantics;
  historical format-2 runs require their original executable.
  The fresh vectorized Pong protocol is in
  `docs/experiments/2026-09-06-vector-pong.md`. Do not replace the binaries,
  runners or auditors of an active pinned experiment. Stage follow-on candidates
  separately and keep GPU-heavy checks serialized with measured training.
  The readback worker, timer and host-buffer-reuse hardware/canary/pixel gates
  have completed; preserve their controls and do not restart their queues.
  Original pinned inputs are in `runs/readback-hardware-20260907/manifest.json`;
  earlier results are in `docs/experiments/2026-09-08-runtime-hardware.md`.
  Device-resident imagination is adopted after exact hardware, synthetic and
  pixel checks. It removes redundant host feature/state transfers and scratch
  without changing learning arithmetic. The Meganeura refresh is now adopted
  at a7e2efd9 (main df11bb0c plus the two required LeVJEPA cache patches), with
  shared registry Blade 0.9.0 and Rust 1.92 minimum. Do not lose cached query
  blocks or cache aliases when updating again. Require logical weights and all
  optimizer moments on restore, excluding only plan-identified Winograd caches.
  Keep backend identity checks and historical executables intact.
  Current exact pixel pairs reach 8.64–8.70 actions/s, only 0.576–0.580× aggregate
  real time and 0.0720–0.0725× per stream. GPU activity spans 66–70%; 14,212 MiB
  peak usage left only 1,631 MiB of directly reported free memory in the completed
  pilot. The old total-minus-used check omitted driver reservations and does
  not establish the 2 GiB safety gate. Preserve those raw results but withdraw
  the reserve-pass claim. Record memory.free and memory.reserved directly;
  require at least 2,048 MiB measured free before new long-run replication or
  a larger batch. The unchanged pilot has finished; do not restart its queue.
  Keep GPU-heavy work serialized. This backend update is not a major
  speedup; see `docs/experiments/2026-09-08-meganeura-refresh.md` for the tested
  current package and the preserved audit-only failure plus completed continuation.
  A CPU graph check verifies 588 MiB of F32 visual KV cache per stream: N6/N4
  would remove 1,176/2,352 MiB versus N8, without changing B16/T64 or full BPTT.
  These are logical bytes, not measured free VRAM or throughput. The LeVJEPA
  `memory_candidate_streams_match_serial` GPU test now passes for N4/N6/N8,
  with zero measured dense-feature error against serial encoding and unchanged
  pooled/dense tolerances. This does not establish combined learner memory.
  The fixed-R256 N8/N6/N4 then N4/N6/N8 comparison completed in
  `runs/vector-memory-runtime-20260908.CcWv0d`: every same-N full checkpoint and
  action/episode/reset trace repeats exactly. Select N6 for the unchanged
  repaired native package: 8.55–8.57 actions/s, at least 3,302 MiB directly free.
  N4 retains 4,889 MiB but is slower; N8 retains only 1,630 MiB and fails the
  2 GiB gate. N6 is about 1.2% slower than N8, not a throughput improvement.
  These short runtime repeats do not test training-seed reliability. Preserve
  the completed evidence and pinned inputs. The isolated replication-v2 checker
  at `exp/atari-replication-v2` binds a declared N4/N6/N8 to matching complete
  runtime evidence; real N4/N6 pass and N8 is rejected. Its 504 passing CPU
  tests are not a replication result. Keep all five game gates and fresh seeds.
  A grouped-RSSM gate candidate passed focused output/gradient checks but failed
  exact full-learning parity from update 3. It is not adopted; preserve branch
  `exp/rssm-gate-batching` and `runs/grouped-rssm-20260908.Vodj6w`. Main source
  is restored, but its old release binaries still contain that candidate;
  use the documented isolated package or rebuild before running root binaries.
  Do not repeat large CPU graph
  compilation alongside training: the first memory-plan probe caused host
  pressure, and its two capped follow-ups failed without yielding smaller-row
  world estimates. Preserve those failures; CPU-only does not mean low impact.
  Track remaining world-training/recurrent/perception costs and profiler
  coverage in `docs/experiments/2026-09-08-device-imagination.md`.
  The current profiler's alternate mode recovers queue-submission coverage,
  not per-dispatch kernel detail or verified idle gaps. Completed captures are
  diagnostic artifacts, not another pending queue or traced speed benchmark.
  The tested Python package is isolated; the default editable extension remains
  the pinned historical control. Select the documented package or build current
  source into a fresh package for new experiments. Do not overwrite controls.
- Change one scientific variable per comparison. Report real interactions,
  learner updates, wall time, model/data provenance, all seeds, and failures.
  A short integration test or an historical score is not a matched benchmark.
  Match head structure and initialization when comparing objectives, and version
  changed heads or intrinsic hash schemes instead of reinterpreting old state.
  Verify the actual encoder file on restore; matching shapes are not identity.
  Require complete checkpoint tensors; a detected torn save is not atomic recovery.
- Profile learner stages, synchronization, and GPU idle time before committing
  days of compute. Check existing branches and local run artifacts before
  repeating old experiments. Preserve corrected full-precision gradients and
  full-recurrence row microbatching when integrating backend work.
  Judge useful throughput at the declared replay ratio, not GPU busy percentage
  alone. Retain the GPU memory safety margin; a larger batch needs both a timing
  win and a learning-quality comparison before becoming the new control.
  Lower replay ratios are separate learning-throughput ablations, not identical-
  recipe speedups; retain the original-ratio control and test learning quality.
  The completed Boxing 40k–50k windows attribute 74.4% of R256 time to learning
  and 56.1% of R64 time to observation. Use actual emulator-frame increments
  for game clocks, and reprofile the dominant stage after recipe selection.
  Host readback waits include unfinished producer computation and transfers;
  do not relabel them GPU idle time. Substage timings are contained in their
  parent stage totals, not additional elapsed time.
  GPU traces synthesized from host submission times are not calibrated
  GPU idle-gap measurements; distinguish pass durations from timeline placement.
  External captures require usable imported output and expected GPU workload
  coverage across the run; a successful CLI exit or one GPU row is insufficient.
  Preserve raw results when a later coverage audit rejects a preliminary gate.
  Batch row-independent replay encoding and heads across time without batching
  away recurrence or introducing future inputs. Check production-sized losses,
  all parameter gradients and reset causality; composed losses need complete
  scalar reductions, not backend workgroup partials.
- Test dense Atari, sparse Atari, and a small native persistent environment.
  Positive terminal return is a Pong win rule, not a general Atari competence
  criterion. Keep game-specific wins separate from generic episode accounting.
  The active five-game objective targets Pong, Boxing, Freeway, Breakout and
  Qbert; see `docs/experiments/2026-09-08-atari-five.md`. The isolated
  `exp/atari-five` v2 runner completed the fixed 200k-action Boxing R64/R256
  pilot and 75k-action N8 frozen evaluations. R64 seed 0 passes its frozen gate:
  40/40 wins, mean +51.55, with complete checkpoint/declaration/replay checks.
  The original queue and follower stopped on a zero-update control restore
  failure. The save-only repair now passes CPU/GPU checks, exact 12M initial
  actions/parameters and complete trained-state preservation, retaining strict
  restore checks. Preserve the original artifacts. The completed continuation in
  `runs/atari-five-continue-20260908.JrdVto` completed the repaired zero-update
  control: 21/40 wins, mean +0.125, with full checkpoint/replay checks. R256
  finished 200k actions and 49,619 updates with the original native package.
  Its 75k frozen evaluation passes: 162/162 natural wins, mean +92.4877,
  no cutoffs or updates, with complete checkpoint/declaration/replay checks.
  Use R256 provisionally for its larger score margin, retaining R64 as the faster
  ablation; this costs roughly 2.29 times the training-loop wall time. Preserve
  the completed queue, repair evidence, packages and shared auditors. One pilot seed does not establish
  reliability: require a separately declared fresh three-seed replication
  using 1009, 2017 and 3019.
  Adjacent roots reuse live policy/posterior RNG streams under `seed + stream`;
  keep the declared live-seed ranges disjoint, without reinterpreting old
  results or rewriting the completed pilot. Verify actual child
  processes before waiting, and do not treat completed tooling as five-game wins.
  Its CPU-only frozen-result follower pins the candidate match auditor,
  `replay_atari.py`, `atari_tasks.py` and their dependencies too; keep them
  unchanged while live. Task observers are post-hoc evaluation, never policy
  inputs or added training rewards. Scripted observer fixtures are not Kindle
  wins. Qbert's first pyramid alone is not the sustained-competence gate.
  Breakout's 864-point rule has a verified actual-ROM scripted fixture and
  negative controls; preserve its distinction from Kindle's learned results.
  The separate candidate `audit_atari_tasks.py` covers Freeway/Breakout/Qbert
  final checkpoint and replay scoring. A task-gate pass alone does not verify
  campaign budgets or independent training seeds. The separate candidate
  `audit_atari_campaign.py` checks all 15 declared game/seed records, fixed
  budgets/config, fresh models, final checkpoints and replays. Its CPU checks
  are complete, not a replication result. Keep untrained controls and the
  broader goal-completion audit; do not infer them from `replication_passed`.
  The declared N6 sparse Freeway pilot started at 22:19 UTC in
  `runs/freeway-pilot-20260908.WWxHEM`, with the unchanged repaired native package,
  R256 and 200,004 fresh seed-0 training actions. It evaluates only the final
  model for 75,000 sampled actions, then runs a separately restored zero-update
  control under that evaluation protocol. Full CPU replay and complete-stream
  videos follow both. Keep its 34 pins unchanged and GPU phases serialized;
  do not run large CPU graph builds alongside it. This is a pilot, not fresh
  replication, a Freeway success claim or five-game completion.
  The completed CPU-only Freeway discovery check in
  `runs/freeway-discovery-20260908.Zig71a` compares hold lengths 1/16/64 at
  200,004 actions each on three seeds. Independent random actions find no
  crossing rewards; hold16 finds 293/324/328 and hold64 finds 706/716/721.
  These are random exploration controls, not learned task wins or matched
  native-policy evaluations. All action-generation/episode accounting passes;
  independent ALE replay was not performed. If the native control remains
  reward-starved, prioritize a separately declared persistent-exploration
  ablation over simply extending it. The isolated `exp/persistent-exploration`
  candidate adds explicit native action overrides and versioned per-stream random
  holds, with 95 Rust and 547 Python CPU tests passing. It is not adopted and has
  no GPU or learning result. Preserve actual executed actions in RSSM/replay,
  independent RNG, default-path parity and strictly unassisted frozen evaluation.
  See `docs/experiments/2026-09-08-persistent-exploration.md` for the required gates;
  run only after Freeway and the declared common-world diagnostic release the GPU.
  Existing campaign declarations reject the changed exploration protocol. A new
  package requires its own matching runtime/memory evidence before long training.
  That gate is declared in `runs/persistent-exploration-gate-20260908.OVSB5q`
  with 66 pins and 31 passing CPU gate/handoff tests. Its 107-pin follower is
  bound to the actual Freeway launcher; it waits for full pilot completion, then
  runs common-world and the candidate gate once, stopping on failure. Preserve
  these inputs. It starts no long learning run and does not adopt exploration.
  A separate conditional learning declaration now waits on that actual follower:
  `runs/freeway-persistence-learning-20260909.C0GoqT`, 82 pins and 47 passing
  CPU launcher/proof tests. It reverifies raw runtime evidence before any GPU
  use, then compares .5 exploration probability with hold64 versus hold1, each
  fresh seed 0, 200,004 training and 75,000 unassisted frozen actions, followed
  by a separately restored untrained control. Both arms use the same new package;
  the old plain-policy pilot is context, not the matched hold1 arm. Keep its
  inputs fixed; it has no learning result and cannot adopt a recipe or launch
  replication. See `docs/experiments/2026-09-09-freeway-persistence.md`.
  The fixed 72k-action Freeway prefix has zero rewards across 30 natural rounds
  and zero reported absolute advantage in all 17,650 updates, despite declining
  prediction training loss and finite learner scalars. Imagined-policy entropy
  is near its maximum. Prioritize rewarded discovery, not an unsupported claim
  of numerical collapse; this is not frozen competence or a Pong diagnosis.
  Details: `docs/experiments/2026-09-09-freeway-zero-signal.md`.
  The CPU episode-budget check freshly replays Breakout/Qbert random rewards
  and verifies the 25,000-decision wrapper cap. Their rewards are discoverable
  without Freeway's random holds. The separate `exp/episode-budget-evaluation`
  candidate at `4281242` adds frozen-only v4 stopping when every stream reaches
  a predeclared episode count, retaining every completed episode and a hard cap.
  Its 580 CPU tests and exact old-ledger accounting are not GPU validation or
  adoption. See `docs/experiments/2026-09-09-episode-evaluation.md`; preserve all
  current fixed-action queues, game criteria and fresh-seed gates. Require GPU
  prefix/state checks and a new declaration before using this stopping rule.
  Its frozen-only GPU check is now declared, not running, in
  `runs/episode-evaluation-gate-20260909.8f1yKj`: 55 pins, 30 passing CPU tests,
  bound to the actual persistence-learning launcher. It refuses live-parent
  execution and starts no follower. Preserve candidate inputs; run only after
  the entire existing queue completes, checking full frozen state and prefixes.
  Keep external reward and intrinsic reward separate. Retain an extrinsic-only
  control for every intrinsic-reward experiment.
  Record human guidance and the action actually executed; distinguish assisted
  behavior from unguided evaluation and game rewards from human feedback.
  Distinguish agent-collected online learning from forced-random coverage tests;
  verify learned behavior under frozen evaluation against untrained controls.
  Evaluate the declared final checkpoint and confirm independent training seeds;
  do not select a winning checkpoint or weaken acceptance after observing results.
  Record sampled/greedy action mode and recurrent-state initialization; isolate
  their effects when diagnosing a frozen-policy failure.
  Visual novelty is not task competence. Evaluate intrinsic exploration through
  held-out dynamics and later guided adaptation, with explicit reward provenance.
- Gate video encoders and pretraining on evidence: a usable pinned checkpoint,
  causal streaming semantics, native numerical parity, latency, and improved
  held-out control-relevant probes. A paper alone is not an implementation plan.
  Probe the trained recurrent belief before inferring a need for more temporal
  input from single-frame feature probes.
  The completed three-seed motion diagnostic finds useful motion information
  in every final belief, without matching the gameplay ranking. Prioritize
  sparse-positive discovery, reward/value calibration and action-use diagnosis
  over speculative visual expansion. Delayed first wins are not proof of
  numerical training collapse. Declare longer budgets for all seeds as a new
  experiment; never relabel the failed 200k-action mastery campaign.
- Evaluate the world model separately from its policy. The completed frozen
  first-match replays in `docs/experiments/2026-09-08-world-evaluation.md` match
  every recorded action and transition without learning. All three use action
  information in feature prediction; seed 1 has weaker point-reward magnitude
  estimates even after seeing the frame. Prioritize reward/value reliability
  and policy action use on a common held-out distribution, not speculative
  perception expansion. These own-policy trajectories do not establish causation.
  Forecast before consuming the target; separate prior from posterior reward
  estimates, include persistence/unrelated-action/zero-reward baselines, and
  report positive/negative/terminal counts. LeVJEPA cache resets can inflate
  persistence error, and a fixed stride can miss sparse classes entirely.
  Feature error is not imagined RGB, AUC is not magnitude calibration, and a
  strong model score is not policy competence. Preserve the original executable
  for historical model diagnostics; do not rewrite backend metadata to restore.
  The isolated `exp/common-world-probe` candidate at `a425b29` has 282 passing
  Python CPU tests. The declared follow-up in `runs/common-world-20260908.7gWHsJ`
  CPU-reconstructs all 11,388 first-match transitions and pins 35 inputs. It
  explicitly conditions every old model on the same three recordings, preserving
  strict unforced replay separately. Require three exact same-model H1 diagonals
  before the six cross-model runs, with common initial/target RGB and feature
  hashes. GPU work has not started; run only after the entire Freeway queue
  completes and releases the device. Use historical native f663dd93, not the new
  package. Keep source and declaration pins intact. Forced controls are offline
  diagnostics, not that model's policy rollout; another policy's logged return
  is not unbiased ground truth for the evaluated critic.
  The CPU common-input report builder in `runs/common-world-report-20260909.O7nqqe`
  has 13 passing fabricated-fixture tests, not new GPU results. Generate its
  report only after all nine comparisons and the following matched runtime
  benchmark finish, preserving both per-recording and pooled event-class errors.
- Distinguish video-encoder initialization, action-conditioned world pretraining
  and policy-skill transfer. Missing action/reward labels are not NOOP/zero.
  The isolated `exp/world-pretraining` candidate includes world-only updates and
  strict fresh-runtime dynamics initialization, not supported dataset training or
  transfer. Require content-verified ingestion and GPU/adaptation gates before
  adoption. Initialized checkpoints require format-4 offline source lineage;
  do not silently reinterpret them as ordinary format-3 checkpoints.
  Hold target titles out of source data and tuning; measure adaptation and
  forgetting. Retain the source policy when testing full-policy transfer, while
  declaring head, optimizer, normalizer, replay and recurrent-state resets.
- Reuse mind-games' launch, time-control, capture and input infrastructure.
  Verify its Kindle revision/API before integration; legacy BatchAgent adapters
  are not the current Dreamer path. Keep privileged reward/task observers outside
  policy inputs, and do not inherit unreported shaping or scripted gameplay.
- Respect each stream of external consequences. Natural deaths and respawns
  are allowed; cloning or rewinding a live game for training is not. Independently
  initialized vector environments are allowed under a declared new protocol;
  do not relabel their experience as a continuation of a single-life experiment.
  Distinguish uncapped stepping, super-real-time playing plus training, and a
  free-running game without time control. Measure simulated/wall time with
  learning enabled; fast frozen inference is not training throughput. Preserve
  arrival order, actual action durations and observation gaps, and bound training
  debt. Try measured serial scheduling before any actor/learner separation.
- Delete superseded code and redundant documentation when they have no current
  purpose. Git retains history. Prefer small concrete modules over speculative
  frameworks, broad configuration surfaces, or premature swarm infrastructure.
- Work autonomously on authorized implementation, diagnostics, and experiments.
  Follow `/mnt/data/GUIDELINES.md` when available: attention is expensive, keep
  code self-describing, and only the user merges pull requests. Commit and push
  are permitted; do not send messages to other people without authorization.
- Preserve unrelated working-tree changes. Serialize GPU-heavy tests on a
  shared device and record the selected adapter. Run relevant formatting,
  Clippy, Rust/Python tests, and numerical checks for learning/backend changes.
