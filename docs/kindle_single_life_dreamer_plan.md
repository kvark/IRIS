# Kindle: one actor learning to play

Updated 2026-09-07. This is the single project roadmap. Detailed measurements
and commands live in the [kickoff report](experiments/2026-09-05-kickoff.md) and
[self-learning report](experiments/2026-09-05-self-learning.md), not a second plan.
Working rules are in [AGENTS.md](../AGENTS.md).

## Direction

Build one agent that improves through its own interactions, initially with
explicit game rewards and human guidance. Prove broad gameplay, useful
pretraining, cross-game transfer and retention before distributing experience
between Kindles. Keep Rust, Meganeura and Blade as the native computation stack;
reuse mind-games for game execution instead of building another harness.

The order is:

1. Keep Pong as an achieved initial-learning milestone and regression test.
2. Establish the causal video frontend and broaden learning across Atari.
3. Make single-actor playing plus training fast; add measured video/world
   pretraining without making it a hidden source of target-game experience.
4. Use mind-games for vkQuake2 and TMNF, then a small GOG/Wine portfolio.
5. Demonstrate a reusable gameplay actor, including rapid adaptation to an
   unseen shooter and retention of previous games.
6. Only after strong single-actor GOG and transfer results, investigate swarms.

Vectorized collection and batched live inference for one shared learner are
implemented, explicitly requested after inspecting GPU utilization. Reducing
measured learner/perception costs remains the runtime priority. Independent
environments are collection streams, not independent
learners or a swarm. Preserve each stream's causal history, belief, RNG and
replay continuity; count all executed interactions and honor the training ratio.
Separate learner services and swarm infrastructure remain deferred.
A cheap evaluation constructor can be a local optimization, not a prerequisite
for gameplay progress or a reason to split the agent.

## Architecture: measured stepping stone versus target

The original frozen competence controls use DINOv3 ViT-S/16. Native LeVJEPA
now also wins under frozen evaluation: seed 0's final 200k-action model wins
18/18 completed games with mean return +10.2778 in 75k sampled actions and
zero updates. It nevertheless **fails the declared mastery gate**, which
requires at least 20 natural games and mean return ≥+15 as well as ≥90% wins.
The remaining seeds in the [vector experiment](experiments/2026-09-06-vector-pong.md)
are unfinished; this recipe cannot pass the all-seeds gate after seed 0's failure.
Numerical parity and useful learned behavior are demonstrated, not consistent
mastery or a pretrained action-conditioned world model.

| Part | Running implementation | Target and missing work |
| --- | --- | --- |
| Perception | DINOv3 control; native LeVJEPA with one completed frozen seed, both projected/pooled to 7×7×64 | Finish seed-variation measurement, resolve the failed mastery gate and test Atari breadth; reduce runtime cost |
| World model | Categorical Dreamer RSSM; causal feature prediction, reward, continuation, balanced KL and replay value | Retain this learning/control baseline; bootstrap compatible dynamics from other games |
| Behavior | Imagined categorical actor and two-hot critic, trained from the agent's own rewarded actions | Retain behavior across compatible games; measure adaptation and forgetting |
| Runtime | Serial control and native vectorized LeVJEPA/RSSM/policy inference; one shared learner | Reduce learner/perception cost and attribute GPU underutilization; retain trustworthy per-stream clocks and recovery |

The JEPA-inspired change that actually landed is prediction before observation:
the predictive head reads the deterministic prior state, not the posterior that
already contains the target. Prediction-only removes the reconstruction decoder.
This validates one part of the target architecture, not the complete video pivot.

~~~text
h[t]     = recurrent(previous belief, previous action)
u_hat[t] = predictor(h[t])
u[t]     = frozen_encoder(RGB history through t)
z[t]     = posterior(h[t], u[t])

(h[t], z[t]) --> reward / continuation / imagination --> actor / critic
~~~

For DINO, E uses only RGB[t]. LeVJEPA uses the causal prefix of the current
16-arrival chunk, never future frames. Chunk boundaries reset only perception,
not RSSM state or replay. During imagination only the RSSM prior
runs; the video encoder does not. Its short visual history and the RSSM's longer
action-conditioned belief have different jobs.

Keep the measured objective choices: reconstruction control 0.25/0, predictive
auxiliary 0.25/0.25, prediction-only 0/0.25 (reconstruction/future coefficients).
Keep head structure, initialization, loss normalization and behavior settings
matched when comparing objectives. Reset observations are not predictable
transitions. Preserve gradients through earlier observations and full BPTT;
same-shaped features from different encoders are not compatible coordinates.

### What to call the coupled loop

Use **adaptive execution** as project shorthand for a loop that perceives,
acts, receives consequences and updates the same agent. This is a descriptive
name, not a claim to have invented a new learning algorithm. The established
umbrella is online reinforcement learning; sustained adaptation and retention
lead toward [continual reinforcement learning](https://arxiv.org/abs/2307.11046).

In the code, the actor is still the policy network; the agent includes perception,
world model, replay, actor and critics. Acting and learning share parameters and
state, but act does not secretly train: observe and learn_scheduled remain
explicit. Keep that distinction useful for frozen evaluation and diagnostics.

## Pong: achieved, with an honest boundary

Pong is already an Atari game. The next challenge is Atari breadth, not another
proof that learning can happen in Pong.

The historical DINO rows below use the declared final 100k-action checkpoint,
followed by 50k frozen sampled actions with zero updates. There is no checkpoint
selection.

| Objective / seed | Final training mean / games | Frozen mean / natural wins |
| --- | ---: | ---: |
| Prediction only / 0 | +18.1429 / 7 | +19.0714 / 28 of 28 |
| Prediction only / 1 | −0.2500 / 4 | −1.0714 / 3 of 14 |
| Reconstruction / 0 | +17.6667 / 6 | +19.2143 / 28 of 28 |

Both causal seeds pass the predeclared initial-learning gates. The zero-update
baseline is −20.3455. Seed 1 nevertheless loses most frozen games and first wins
at action 91,684, versus 44,075 for seed 0. Thus learning and winning are achieved;
consistent mastery, superiority to reconstruction and broad reliability are not.
Preserve this variation while moving on; Pong remains a cheap regression anchor.

The new native LeVJEPA vector recipe has a stronger, separately declared gate.
Its first final frozen seed wins all 18 completed games but misses both the
20-game minimum and +15 mean-return threshold. Keep the +10.2778 result and
its unfinished −4 tail visible; do not extend the evaluation or lower the bar
after seeing it. Complete the other two seeds to measure reliability before
choosing a bounded follow-up. This is learned gameplay, not a passed mastery claim.

Persistent native GridWorld also passes on three independent causal seeds:
2,495/2,498/2,373 food in 10k frozen greedy actions. The matched reconstruction
control has strong late training but only 2 food / 98 deaths when frozen.
Diagnose sampled-versus-greedy behavior and fresh recurrent state separately;
do not replace its failed endpoint. All reports, tensors, identities, schedules
and reward/action ledgers validate. These results establish the single-agent
learning loop, not the eventual game portfolio.

## Atari: from one success to broad competence

Beating most of a declared Atari suite is a reasonable research ambition, not
something demonstrated by Pong or guaranteed by the Dreamer name.
[DreamerV3](https://danijar.com/project/dreamerv3/) establishes broad control as
a credible direction; our small frozen-encoder implementation does not inherit
the published model's results. A 100k-action experiment measures sample
efficiency, not whether a game is learnable with a larger budget.

Start with a diagnostic panel: Pong, Breakout, Boxing, Freeway, Seaquest,
Frostbite, Qbert and Private Eye. It covers different control, memory and reward
structures without pretending that eight games establish generality.
Private Eye's earlier one-episode reward discovery is not sustained competence.

The [CPU adapter preflight](experiments/2026-09-07-atari-adapters.md) checks
this panel's independent streams, fresh serial replay, RGB observations,
rewards and actual clocks without constructing an agent. Before using the vector
runner beyond Pong, separate its positive-return `natural_wins` summary from
game-specific competence criteria; a positive Atari score is not generally a win.
The active pinned Pong runner and auditor stay unchanged.

Use cheap adapter/reward and numerical checks first, then fixed-budget learning
runs. Choose the video-encoder candidate through a bounded comparison before
launching a full target-architecture suite. DINO remains the labeled control;
do not present another all-DINO sweep as completion of the LeVJEPA pivot.

Expand to the established Atari-26 suite, then Atari-57 if the additional
coverage is useful. Record the exact game list and exclusions before results.
Independent per-game training tests algorithm breadth; it is not yet one
general Atari policy. Sequential transfer is a separate experiment.

For each recipe:

- Pin ROM, ALE/wrapper, sticky actions, action vocabulary/repeat, reset/no-op
  policy, time limits, encoder, rewards and model/data provenance. Keep the
  completed Pong protocol intact; label any new protocol separately.
- Record the 100k-action result. Declare any longer budget, for example 1M
  actions, before that run; do not silently extend only failing seeds or select
  their best checkpoints. Longer runs need their own final evaluation windows,
  not the scorer's hard-coded 350k–400k-frame training window.
- Use random and untrained controls, at least three independent training seeds
  for broad claims, and a fixed frozen evaluation policy/budget with confidence
  intervals. Retain all failures and time-limit cutoffs.
- Report per-game raw score, normalized score, completion/win rate where meaningful,
  learning-curve area, interaction count and wall time. Separate game/seed
  aggregation; one high-score title must not hide a failing majority.

Make “most games” concrete: the proposed broad-competence target is at least
14 of Atari-26 reaching a predeclared game-competence threshold at the final
budget, using the median across seeds. Use human-reference performance where
the protocol supports a valid comparison, or an explicit task-completion bar;
mere improvement over random is the lower learning gate, not “beating” a game.
Report the full seed distribution and which threshold each title passed.
Not every Atari game has a final ending.

Do not spend weeks on an unprofiled full sweep. At the present 3.8 hours per
100k aggregate actions, 26 games × 3 seeds would cost roughly 300 GPU hours before controls.
Use the panel to resolve failure modes and improve serial throughput first.

## Runtime: uncapped, accelerated, and eventually free-running

These are different capabilities:

| Mode | Meaning | Current status |
| --- | --- | --- |
| Uncapped step-driven learning | Advance the game, then compute as long as needed; no wall-clock pacing | Supported by Atari/native runners |
| Super-real-time playing plus training | More than one simulated game second per wall second, including learner work | Not achieved by the tested 12M Pong recipe |
| Free-running learning without time control | The game cannot be frozen while the agent computes | Required eventually; not validated by step-driven Atari |
| Frozen evaluation | Act without parameter updates | Supported; does not establish training throughput |

The tested DINO full-BPTT, B16×T64, row-batch-16, ratio-256 Pong learner runs at
7.26–7.28 actions/s. With action repeat four and a 60 Hz game clock this is
about 0.48× real time, not super-real-time. Frozen evaluation is about 10.9×.
GridWorld has no declared real-time clock, so its actions/s alone is not a
speedup ratio.

The current native LeVJEPA run uses eight environments sharing one learner,
batched live inference, temporal replay encoding and batched non-recurrent
heads. Full RSSM recurrence and BPTT are preserved and production-sized losses
and gradients are checked. Seed 0 completes 200k aggregate actions and 49,619
updates in 7.63 hours of execution, with zero training debt. Its steady windows
measure 7.2–7.3 aggregate actions/s, about 0.48× the game clock in aggregate
and 0.060× per stream. GPU activity is roughly 60%, with stable 13.82 GiB usage
and the memory reserve intact. Environments take under 1% of wall time; the
learner and perception remain the bottleneck, not CPU environment stepping.

This replaces the interrupted serial LeVJEPA campaign with a fresh, declared
vector protocol, not a resumed single-stream trajectory. The fixed-B16
N=2/4/8 throughput rule selected N=8 before observing game scores. A separate
B32 experiment is faster but changes optimizer cadence; learning quality is
not yet validated. See the [vectorization record](experiments/2026-09-06-vectorization.md)
for implementation checks and rejected candidates, and the
[live experiment](experiments/2026-09-06-vector-pong.md) for the fixed recipe,
complete training accounting and final frozen gates. Neither batched inference
nor GPU busy percentage establishes super-real-time training or mastery.

Measure acceleration as simulated game seconds / wall seconds. Report cold
construction separately and also include end-to-end run cost. Track actual
emulator frames, policy decisions, replay samples, imagined transitions and
optimizer updates; none can substitute for another clock.

For B×T replay samples per update, train ratio R and update duration U:

    learner updates/s required = action rate × R / (B×T)

At 15 actions/s and R256, B16×T64 requires 3.75 updates/s. Updates alone at
about 0.43 s each exceed the wall-time budget. Removing a sleep or accelerating
only game rendering cannot fix that.

The next performance target is sustained >1× playing plus training on simple
games, with 2× as a useful stretch target and retained learning quality.
Profile the complete vectorized loop against N=1: environment/capture, preprocessing/encoder,
posterior/action, replay, world update, imagination/behavior and synchronization.
Prioritize grouped RSSM/layout work, batched non-recurrent heads, fewer host/GPU
round trips and measured feature/replay packing. Preserve recovered F32 gradient
safeguards and full-recurrence row batching. Earlier cached-read/materialization
work already reduced a 12M canary from 7.42 to 1.05 s/update; full rows reach 0.54 s.
Those measurements are in the kickoff report; do not repeat that investigation.

The current [handoff inventory](experiments/2026-09-06-vectorization.md#remaining-handoffs-read-only-inventory)
counts 95 explicit posterior/imagination readbacks per update. It is not a
wait-time profile: instrument packing, transfers, sampling and GPU compute
separately after the pinned queue. Prefer eliminating deterministic-state and
feature copies before changing sampling/return arithmetic. Even removing both
whole stages at zero cost would not reach aggregate real time at the current
recipe; world training and perception also need measured improvements.
The backend's GPU trace places pass durations on a synthetic host-submission
timeline, not a calibrated GPU clock. Use those durations for workload cost,
not the drawn gaps as evidence of GPU idleness.
The [bounded hardware/parity handoff](experiments/2026-09-06-vector-pong.md#queued-diagnostic-handoff)
is queued behind the campaign and its checkpoint watcher. It checks three
hardware tests and three alternating parent/candidate synthetic canary pairs;
profiling overhead, calibrated queue gaps and pixel-loop validation remain
separate gates. Do not launch overlapping GPU work when the campaign exits.

Use one learner with batched live inference, not one full GPU model per game.
Keep independent visual caches, beliefs, RNG streams and sequence replay. Sample
uniformly over eligible per-stream sequence starts after the fresh-item queue;
never concatenate unrelated environments into a temporal sequence. Preserve
terminal observations before resetting only the affected streams. Report total
actions, actions per stream, updates and training debt. A vector tick is not one
interaction, and aggregate acceleration is not per-stream real-time play.

Before another multi-day learning queue: finish fixed-ratio throughput controls,
remove measured CPU/readback bubbles, and test larger replay batches if memory
permits. Treat a changed learner batch as an algorithmic ablation, not a free
systems speedup. Require independent reset/cache parity and validated action,
reward, update and checkpoint ledgers before long vectorized training. Declare
its aggregate budget and final frozen evaluations as a new experiment; do not
reinterpret the interrupted single-stream mastery protocol.

Smaller models or lower train ratios are valid experiments, not free speedups.
Change one variable, report actual updates per real interaction, and retest
learning. Do not claim acceleration by dropping owed training, weakening the
reward task or changing the simulated control interval unnoticed.

Use mind-games' time control for accelerated development. Separately test a
single actor in free-running mode later: timestamp observations and executed
inputs, account for game time passing during learning, bound update work/debt
and mark missing observations. Variable-duration transitions need explicit
semantics; never fabricate adjacent frames. Try measured serial scheduling
before considering actor/learner concurrency. Pausing a game does not satisfy
the no-time-control requirement.

## LeVJEPA and video/world pretraining

### First: resolve the frontend's gameplay and runtime gates

Native LeVJEPA inference, numerical parity and causal streaming checks are
implemented; its first final frozen Pong seed wins but fails the mastery gate. The
[released model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)
describes a 303.1M-parameter ViT-L/16 trained on 16-frame clips, with block-causal
attention. It is much larger than today's ViT-S and cannot be assumed faster or
better for small game objects. Record its code/weight revision, license and
measured memory/latency.

The [LeVJEPA experiment](experiments/2026-09-06-levjepa-pong.md) records the
pinned checkpoint, native reference parity, future-frame invariance and bounded
16-arrival causal chunks. Current-time spatial tokens exclude the clip-level
CLS output. Frozen-feature probes compare position and motion cues against
DINO and compare the projected 14×14 grid with pooled 7×7×64 targets. These
are representation diagnostics, not a trained-RSSM or gameplay comparison.

1. Complete the declared training seeds and retain failed final endpoints.
   Probe the trained recurrent belief and imagined reward/action predictions
   before inferring that weak control requires more temporal input or a larger grid.
2. Hold the causal world objective, actor settings, executable and collection
   protocol fixed in a bounded DINO/video comparison before broad Atari runs.
   Historical DINO versus the new vectorized LeVJEPA trajectory is not that control.
   The active `VectorDreamerAgent` is LeVJEPA-only. An isolated
   [batched DINO candidate](https://github.com/kvark/kindle/blob/d909883032bb5a8e37ef199c6afbba64cb62db47/docs/experiments/2026-09-07-batched-dino.md)
   passes CPU validation (83 Rust / 248 Python tests); frontend/stream GPU
   parity, learning integration and timing remain gates before adoption.
   Keep LeVJEPA primary; this is a control, not a silent frontend fallback.
3. Measure downstream learning and complete playing-plus-training cost, not
   only representation loss. Retain causal/reset/cache checks when changing the
   frontend runtime; version temporal sampling, projection and feature semantics.

Freeze the encoder during initial online learning. Do not load a DINO world
checkpoint or old latent replay into a different representation just because
tensor dimensions match. Generic video pretraining is disclosed prior data,
not target-game learning from scratch. If the released large model fails the
speed/quality gate, investigate a compact/distilled video variant or retain the
explicit DINO control; do not silently rename that fallback LeVJEPA.

### Then: bootstrap the world model, not just perception

Kindle loads pretrained DINO or LeVJEPA encoder weights and complete online
checkpoints. It has no supported video-dataset ingestion or world-only
pretraining workflow.
DreamerCore::learn also constructs imagination/behavior targets and trains the
policy/critic; its private world update is not a turnkey offline trainer.
mind-games has gameplay datasets and other training stacks, but that does not
mean the current Kindle world model is pretrained.

An isolated [native world-only candidate](https://github.com/kvark/kindle/blob/a7995a015ed5d7201314d806cb3faee70b34c306/docs/experiments/2026-09-07-world-pretraining.md)
adds aligned feature-clip learning with explicit missing-label masks and no
actor/critic sessions. It passes 86 Rust CPU tests; a verified dataset reader,
compatible-world initialization and hardware/adaptation gates remain unfinished.
It is not adopted and has not trained a dataset or changed the active campaign.

Separate the sources of prior knowledge:

| Initialization | Data used | What transfers |
| --- | --- | --- |
| Generic video encoder | General video; no game actions required | Perception only; fresh RSSM and behavior |
| Gameplay visual adaptation | Source-game clips with titles/splits recorded | Adapted perception; fresh RSSM and behavior |
| Action-conditioned world pretraining | Aligned observations, executed controls, durations and boundaries | Compatible perception/dynamics/predictor; fresh target-game actor and value/reward heads |
| Gameplay-policy warm start | Logged action supervision or rewarded source-game experience | Compatible world model and policy; evaluated separately from dynamics-only transfer |

The first practical world-pretraining path should reuse aligned mind-games
recordings and, where compatible, [Pixels to Play](https://arxiv.org/abs/2508.14295)
data. Unlabeled video can teach visual dynamics but does not identify which
keyboard/mouse action caused a transition. Missing actions are not NOOP;
inferred actions require a separately evaluated inverse-dynamics path.

Build a small world-only ingestion/update path: contiguous clips, reset masks,
real action/duration alignment and explicit masks for missing reward/terminal
labels. Disable actor/critic learning and replay-value targets when the required
supervision is absent; unknown reward is not zero reward. Keep pretraining
counters separate from online interactions. Verify that intended world parameters
change while policy/critic parameters do not.

Add explicit compatible-weight initialization, distinct from full checkpoint
restore. Preserve strict completeness checks on ordinary restores. Before target
adaptation, clear replay/recurrence and reset incompatible heads, optimizers and
normalizers according to the declared transfer arm. Compare fresh, encoder-only
and encoder-plus-world initialization at equal target interaction budgets, while
reporting offline data and compute too. Useful bootstrap means faster retained
gameplay learning, not merely lower feature prediction error.

## After Atari: use mind-games, starting with vkQuake2 and TMNF

The inspected infrastructure is /x/Code/mind-games at d86cd439. It already
provides Podman game launch, Chronolock, a game-only cgroup freezer, Dullahan
frame capture, uinput and scalar/vector environment contracts. Reuse it.
One lane is sufficient now; vector support is not a reason to run a swarm.

The Kindle submodule there is eb8af7fc, not this Dreamer pivot. Existing
src/actor/kindle.py and kindle_visual.py use the legacy BatchAgent and
structured/EfficientNet paths. Old training scripts also contain game-specific
shaping and action overrides. They are historical evidence, not compatible
Dreamer adapters or unassisted gameplay results. Respect mind-games' separate
BC development plan; do not rewrite its actor stack to host this experiment.

Build one small adapter to the current RGB8/action/reward/terminal boundary.
Forward the action actually executed and both boundary causes; keep engine
observers behind reward/evaluation extraction. Validate ordinary inputs and
fresh frame alignment with the frozen and learning agent. Dullahan's existing
GPU exports are not automatically zero-copy inputs to Blade; start with a
correct boundary and optimize a measured transfer bottleneck.

The game progression is:

- vkQuake: short integration/calibration reference using existing infrastructure
  and results, not another long campaign rediscovering its behavior.
- vkQuake2: preferred next FPS learning target. No Quake2 game config/registered
  adapter was found in this checkout; add launch, clock/capture/input and
  reward/terminal plumbing rather than assuming Quake1's hooks transfer.
  Evaluate kills/deaths and genuine level/objective completion, not movement alone.
- TMNF: reuse the TrackMania configuration and adapter, then audit checkpoint,
  finish and lap-time reward extraction. Test held-out tracks; speed without
  progress or completion is not the primary outcome.
- GOG/Wine: begin with a small existing-catalog panel such as Broforce or Spelunky
  for 2D control and Project Warlock for FPS, subject to fresh launch/input/reward
  checks. A working launcher is not evidence that Kindle can learn the game.

Validate action coverage before blaming learning. A shared compact FPS vocabulary
must allow movement, strafing, yaw/pitch aiming, firing, interaction and necessary
combinations. Calibrate device mappings/cadence identically for all comparison
arms. Do not inherit a six-action navigation-only restriction or bake a solution
into game-specific macros.

Inspect startup scripts too: current Broforce setup includes scripted walking
before handoff. Record assisted gameplay and start accounting consistently;
it must not become unexplained agent progress. Harness resets/menus, natural
deaths, reward-read failures and input overrides all need explicit provenance.
Never reuse per-state save/restore or cloned trajectories as undisclosed online
learning. An exact-start BC diagnostic and one continuing Kindle run are different
protocols.

## Rewards, a general FPS actor, and cross-game transfer

Explicit game feedback remains the primary recipe. Reward sparsity varies:
Pong points, native food, shooter kills and race finishes are not equally sparse.
Adapters may use engine/HUD information to compute event rewards and terminal
flags; privileged coordinates, enemy state and scripted goals do not enter the
policy/world inputs. Keep native game outcome primary and label any additional
human feedback or shaping separately. Freeze reward definitions before runs.

The existing bounded visual-visitation bonus stays off in primary runs. Its
intrinsic-only native result is roughly random and its sparse Atari discovery
does not establish competence. Intrinsic reward is a separate later experiment
with an extrinsic-only control, not a prerequisite for useful transfer.

A general FPS actor must carry behavior, not merely a pretrained encoder.
First distinguish independent per-game learning, source-to-target transfer,
and a single policy trained sequentially across source games. For the latter,
retain shared dynamics and policy across title changes with explicit boundaries
and compatible controls. Measure forgetting when returning to earlier games.

Reserve a genuinely held-out shooter before source training and model selection.
A new map in the same game is not a new title; vkQuake2 ceases to be an unseen
test once used for development. Exclude the held-out title from action training,
gameplay visual adaptation and tuning; disclose unavoidable generic-pretraining
overlap. Audit existing mind-games corpora before calling any split unseen.

Compare three initialization arms on identical target rewards, controls and
interaction budgets:

1. Fresh dynamics and policy, with the common pretrained encoder.
2. Transferred dynamics, fresh target policy.
3. Transferred dynamics and the compatible source FPS policy.

For arm 3, do not reset the very policy whose skill transfer is being tested.
Reset or explicitly retain game-specific reward/value/continuation heads and
their optimizer/normalizer state consistently; record the decision.
Changing action meanings requires a declared mapping, not silent reuse of logits.

Measure frozen zero-shot play, then adaptation at fixed budgets such as
1k/10k/100k target actions, final competence and area under the learning curve.
Compare against scratch and, when available, mind-games' frozen BC actor under
the same environment contract. After adaptation, freeze and reevaluate source
games to quantify retention. Any demonstration-assisted arm is separate.
Transfer is a general capability we want; it is not evidence of a JEPA advantage
without a matched reconstruction/representation comparison.

## Strong single-actor results before swarms

Before experience sharing, require a small GOG portfolio with predeclared
competence on at least three titles across at least two control genres, confirmed
over multiple independent seeds. Include verified Wine execution, videos showing
real task progress, retained final policies, one held-out cross-title adaptation
result and a source-game retention check. Report failures and negative transfer.

Reliable longer runs also need bounded replay/archive storage and recovery:
pack features/context only after numerical checks, retain contiguous histories
with representation/action/reward identity, and measure bytes/hour and sample age.
Current checkpoints omit replay/RNG/live state and are non-atomic; detected
torn saves are not fault-tolerant continuation. Add atomic complete-state recovery
before calling a long exposure an engineering lifetime.

Only then test two independent Kindles sharing immutable experience chunks
against isolated controls. Keep environment version, executed controls,
representation and reward-generator provenance; recompute receiving-agent
recurrent state rather than sharing latent coordinates blindly. Another agent's
novelty is not the receiver's novelty. Show useful transfer without harmful
contamination before any larger swarm or shared-optimizer design.

## Immediate work and invariants

Current work is LeVJEPA gameplay validation, stronger Pong mastery and
single-learner vectorized playing-plus-training throughput. Then broaden the
Atari panel and add the world-only pretraining seam and current-Dreamer mind-games
adapter as their gates are reached. Do not start actor/learner separation,
independent learner swarms or another arbitrary 100-hour run as a substitute
for stronger behavior.

For every experiment retain source/model/encoder hashes, environment and data
manifests, actual action/reward/boundary logs, all seeds, fixed final evaluations,
videos and wall/game/update counts. Report frozen versus learning mode and
recurrent initialization. Serialize GPU-heavy jobs on the shared device.
Apply mind-games' job isolation and environment contract when using its runtime.

Preserve the recovered exp/dreamerv3-baseline-12m work: full-recurrence row
microbatching, F32 gradient safeguards and strict score/provenance checks.
Historical DINO/control anchors are Dreamer e3f02248693a79dc8b0ebd62c93683888ddaccfe,
Meganeura bd6be0882c53b94f65f164f88464cc6b24e9df4d and
Blade b208f3b1f97196c2971436b5726e61e71b149c37.
DINO snapshot 114c1379950215c8b35dfcd4e90a5c251dde0d32 has SHA-256
4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d.
These identify the historical controls. Use the LeVJEPA/vector experiment
manifests for the current source, executable, encoder and backend pins.
