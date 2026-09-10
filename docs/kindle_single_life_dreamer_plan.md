# Kindle: one actor learning to play

Updated 2026-09-10. This is the single project roadmap. Detailed measurements
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

1. Keep Pong as an achieved initial-learning milestone; improve seed reliability
   and runtime before another multi-day gameplay sweep.
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
now also wins under frozen evaluation. All three seeds of the
[vector experiment](experiments/2026-09-06-vector-pong.md) are complete:
**seed 2 passes the declared mastery gate; seeds 0 and 1 fail**. The gate requires
at least 20 natural games, mean return ≥+15 and ≥90% wins for every seed.
Each evaluates its final 200k-action model for 75k sampled actions with zero
updates. Substantial seed variation remains, and the all-seeds gate fails.
Numerical parity and useful learned behavior are demonstrated, not consistent
mastery or a pretrained action-conditioned world model.

| Part | Running implementation | Target and missing work |
| --- | --- | --- |
| Perception | DINOv3 control; native LeVJEPA with three completed frozen seeds, both frontends projected/pooled to 7×7×64 | Diagnose the failed all-seeds mastery gate and test Atari breadth; reduce runtime cost |
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

The new native LeVJEPA vector recipe has a stronger, separately declared gate:

| Seed | Frozen mean / natural wins | Mastery result |
| --- | ---: | --- |
| 0 | +10.2778 / 18 of 18 | Fails game count and mean return |
| 1 | +0.5 / 7 of 12 | Fails game count, mean return and win fraction |
| 2 | +20.4651 / 43 of 43 | Passes all three per-seed criteria |

None has timeouts. Unfinished tails are −4 over 5,408 actions, 0 over 309 and
+11 over 871, respectively; they are not completed games. All accounting and
checkpoint audits pass. Preserve the failed all-seeds decision without extending
the evaluation or lowering the bar. Diagnose held-out recurrent belief and
imagined reward/action predictions, then choose a bounded follow-up. One strong
seed is not a reliable recipe across seeds.

The [all-seed motion diagnostic](experiments/2026-09-08-device-imagination.md#completed-held-out-motion-probes)
finds useful motion information in every trained belief; the strongest player
does not have the strongest linear probe. Do not default to a bigger visual
grid or more temporal input. Training first wins arrive at 160,344 / 193,232 /
86,352 aggregate actions for seeds 0 / 1 / 2. Seed 1 earns only 12 positive
points in its first 80k actions, versus 87 / 76 for seeds 0 / 2. This points to
uneven discovery and learning speed, not an established numerical collapse.
The [complete training-coverage summary](experiments/2026-09-08-world-evaluation.md#completed-training-coverage-not-a-causal-explanation)
finds 86.5% of seed 1's first-80k replay batches have no positive target, versus
63.9% / 65.9% for seeds 0 / 2. All three have nonzero reported absolute advantage
after their first two updates. Count repeated replay samples separately from
unique rewards. Coverage alone does not explain the ranking: seed 0 has more
early positive points than seed 2 but wins much later.
The [separate world-model evaluation](experiments/2026-09-08-world-evaluation.md)
now reproduces each recorded first game exactly, with zero updates. All three
models predict features better with actual controls than with unrelated controls.
Seed 1 has weaker point-reward estimates even after seeing the frame: positive /
negative event MAE is 0.351 / 0.430, versus 0.043 / 0.281 for seed 0. This is a
more concrete diagnostic lead than speculative visual expansion, not proof of
the cause. The common-recording comparison below tests that lead. Preserve
event counts, distinguish prior forecasts from posterior inference, and compare
feature persistence and zero rewards. Good
feature prediction or event ranking alone is not a good policy. Fixed-stride
long forecasts can miss entire sparse classes; terminal accuracy is unestablished.
These correlations do not establish causation. A longer-budget replication
must be declared for all seeds as a new experiment, not an extension that
relabels the failed 200k-action gate.

The common-recording follow-up in `runs/common-world-20260908.7gWHsJ` completed
all nine GPU runs: all three same-model diagonals reproduce the original forecasts
exactly and all six cross-model runs pass common-input checks. Each model sees
the same 11,388 transitions, with zero updates; all memory/coverage checks pass.
The [common-input report](../runs/common-world-report-20260909.O7nqqe/report.html)
is complete and changes the diagnosis: each model predicts scored points and
features best on its own recording. Model 2, the strongest player, has the worst
pooled feature and negative-reward magnitude errors; model 1 has the worst
positive magnitude error but the best negative magnitude error. Four of six
cross-model all-frame prior reward errors are worse than always predicting zero.
Errors remain large even after consuming the target frame. This points toward limited
reward generalization and experience coverage, not a globally broken seed-1
world model or a demonstrated need for more perception. It does not establish
the cause of policy failures. The historical training-coverage summary above
supplies a baseline, not an intervention. Require fixed all-seed budgets,
own-policy frozen gameplay and a new multi-match forecast set before selecting
changed exploration/replay/loss settings. Preserve the completed Freeway
comparison, the 35 common-world pins and all nine rows. Forced controls are diagnostics,
not new gameplay successes; off-policy logged returns are not critic targets.

The [multi-match diagnostic candidate](experiments/2026-09-10-multimatch-world-probe.md)
now reads a fixed first-N stream-zero subset from complete vector evaluations,
preserving full source identity and episode boundaries. Its 633 CPU tests and
25,136 independently replayed actions validate extraction, not native forecasts.
Require serial/vector strict replay, same-model forecast parity and GPU safety
before its proposed first-four-match-per-final-model Pong comparison. Keep the
existing queues fixed; a longer all-seed training budget is still undeclared.

The [Freeway CPU discovery check](experiments/2026-09-08-freeway-discovery.md)
finds no rewards under independent random actions in three 200,004-action arms,
but hundreds when random actions persist for 16 or 64 decisions. These are
exploration controls, not learned competence. The completed plain-policy pilot is
reward-starved; test a bounded persistent-exploration component with explicit
executed-action provenance and unassisted frozen evaluation before simply
spending more GPU time on a longer unchanged run. The isolated
[persistent-exploration candidate](experiments/2026-09-08-persistent-exploration.md)
now passes 95 Rust and 547 Python CPU tests, including actual-action provenance
and unassisted-evaluation guards. The full GPU gate now passes: three native
tests, exact default learning/state/trace parity in both orders, the pixel
override integration, and at least 3,303 MiB directly free across all 13 phases.
Warmed candidate throughput is 8.57–8.59 actions/s, 99.92%/99.98% of paired
controls: no regression, but no speedup. Preserve the completed gate and queue.
Their 31 CPU checks are handoff/binding evidence, not gameplay results.
A separate [conditional learning pilot](experiments/2026-09-09-freeway-persistence.md)
compares hold64 and hold1 at the same .5 exploration probability, seed 0,
200,004 actions per arm and 75,000 unassisted frozen actions, plus a separately
restored untrained control. Hold64 completed training at 13:59 UTC on September 9:
49,651 updates, 96 rewarded natural rounds, mean 14.1146 crossings, still assisted.
Complete final state, exploration accounting and the frozen restore match;
full-training GPU coverage passes with at least 3,302 MiB free. Its completed
unassisted evaluation passes: 36/36 natural rounds reach 25 crossings, mean
31.0556, with no cutoffs or updates and complete checkpoint/replay/video checks.
Hold1 also completes its fixed budgets and passes the frozen gate: 36/36
qualifying natural rounds, mean 29.0278, no cutoffs or updates, with complete
checkpoint/replay/video checks. Both exploration arms succeed on this seed;
hold64's 2.0278-crossing mean advantage does not establish that persistence is
necessary or reliable across seeds. The restored untrained control returns zero
in all 36 natural rounds and partial tails. All six GPU-phase checks pass;
the complete pilot is independently verified and its launcher has exited.
Provisionally carry hold64 into fresh Freeway confirmation for its larger score
margin and rewarded training coverage; retain hold1 as the simpler successful
control, without assuming held actions help other games. The 47 CPU launcher/proof tests
are not gameplay evidence. This paired pilot is
distinct from the required fresh three-seed, five-game replication, and cannot
select a recipe automatically.
Post-hoc [action-order controls](experiments/2026-09-09-freeway-persistence.md#action-ordering-versus-a-simple-up-bias)
do not reproduce its score with constant UP or shuffled per-stream actions;
ordering matters beyond action counts, without establishing visual-feedback dependence.
The [completed-training control diagnostic](experiments/2026-09-09-freeway-zero-signal.md)
finds zero rewards across 96 natural rounds and zero reported absolute advantage
in all 49,651 updates through 200,004 actions. All 241 saved tensor entries are
complete and finite, despite no reward-driven policy signal; predictive training
loss falls substantially.
Imagined-policy entropy remains near its maximum. This supplies a concrete
reason to prioritize reward discovery; good fitting of unrewarded experience
does not establish useful policy learning or a cause for the Pong seed failures.

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
The historical pinned Pong runner and auditor stay unchanged.

The next bounded [five-game campaign](experiments/2026-09-08-atari-five.md)
targets Pong, Boxing, Freeway, Breakout and Qbert. Its completed first experiment
compares replay ratios 64 and 256 on Boxing, with the same LeVJEPA model,
200k fresh interactions per arm and 75k frozen sampled actions. Training and
evaluation both use eight independent streams sharing one learner. Version-2
episode accounting and game-specific scoring are isolated from the historical
Pong controls. R64 seed 0 passes its fixed frozen Boxing gate: 40/40 natural wins,
mean +51.55, no cutoffs or updates, with complete checkpoint and replay audits.
The untrained control hit a zero-update checkpoint restore edge case; preserve
the successful arm and failure. Explicit-zero optimizer saves are now validated
against exact initial actions/parameters and all trained tensor state. The
control now completes 40 natural matches with 21 wins and mean +0.125, versus
R64's +51.55, with complete checkpoint/replay checks and zero learning updates.
R256 now also passes: 162/162 natural wins, mean +92.4877, no cutoffs or updates,
with the original native package and complete checkpoint/declaration/replay
checks. Both policies received the same action budgets; R256 finishes matches
sooner, yielding more completed games. Use R256 provisionally for its much
larger score margin, retaining the faster R64 control. The extra replay costs
roughly 2.29 times the training-loop wall time; this is not a free speedup.
The first-80k diagnostic finds earlier posterior reward-sign separation in R256
at roughly four times the updates, not a held-out forecast or final-policy win.
A one-seed pilot is not reliability. The completed memory/runtime comparison
selects N6 for the unchanged repaired package, without changing the learner.
The declared sparse Freeway pilot in `runs/freeway-pilot-20260908.WWxHEM`
completed 200,004 fresh seed-0 training actions at 04:45 UTC on September 9,
with no rewarded rounds. Its completed 75,000-action sampled frozen evaluation
also returns zero in all 36 natural rounds, with no cutoffs or updates and full
checkpoint/replay checks. It fails the unchanged gate of ≥25 crossings in ≥90%
of ≥20 natural rounds, mean ≥25 and no cutoffs. The separately restored untrained
control also returns zero in all 36 rounds. Both complete replays/checkpoints and
all four GPU-phase coverage/memory checks pass; the pilot is complete. The
[complete failure video and evidence](experiments/2026-09-09-freeway-zero-signal.md#final-frozen-policy-no-learned-crossing-skill)
are available. This pilot is not fresh replication or Freeway competence.
The subsequent [persistent-exploration pilot](experiments/2026-09-09-freeway-persistence.md#completed-unassisted-hold64-result)
does pass the frozen Freeway gate on seed 0: 36/36 qualifying rounds, mean 31.0556,
with complete final-checkpoint and replay checks. The matched hold1 arm also
passes: 36/36 qualifying rounds, mean 29.0278. The completed untrained control
scores zero, and all pilot validation passes. Neither the score difference nor
these two trained passes establishes fresh-seed reliability.
The [fresh Freeway confirmation](experiments/2026-09-10-freeway-confirmation.md)
now declares hold64/probability .5 for roots 1009/2017/3019, with the same
200,004-action training budget and unassisted 75,000-action final evaluations
plus paired untrained controls. It has not started and follows the existing
Boxing, current-runtime and Breakout/Qbert sequence; all predecessor data must
reverify first.
The [fresh Boxing confirmation](experiments/2026-09-10-boxing-confirmation.md)
has started on the qualified current backend: roots 1009/2017/3019 each receive
200,004 training and 75,000 frozen actions, plus a paired untrained control.
The first root is training; there is no confirmation result yet. The
[first Breakout/Qbert pilots](experiments/2026-09-10-breakout-qbert-pilots.md)
are now conditionally declared: 200,004 fresh seed-0 actions per game, no
exploration overrides, final-checkpoint episode-budgeted evaluation and
paired untrained controls. They have not started and require the complete
current-package runtime gate. Fresh-seed replication remains separate; do
not call a positive score a win or let tooling substitute for learning.
The [task observers](experiments/2026-09-08-atari-task-observers.md) now distinguish
match wins, complete Freeway rounds, both Breakout walls and Qbert pyramid
completion. Qbert's first pyramid is only a progress milestone: sustained
competence also needs its declared final-score bar. CPU-only reconstruction
produces complete-stream videos and outcome ledgers without another GPU run.
Breakout's two-wall rule is now verified by an actual scripted 864-point game
and two failing fixtures, not only synthetic threshold tests. None counts as
learned competence or enters training replay.
Separate match/task scorers cover all five declared gates and require final
checkpoint and frozen-replay evidence. This is evaluation readiness, not
completed learning. The candidate campaign checker additionally requires all
15 game/seed records, fixed budgets/config and fresh models; its passing CPU
checks are not actual replication results. The isolated replication-v2 checker
also binds declared vector counts to completed, matching runtime/memory gates;
N4/N6 pass the real evidence and N8 is rejected. Untrained controls remain separate.
Fresh confirmation uses seeds 1009/2017/3019: their live RNG
inputs do not overlap for N≤8 under the existing `seed + stream` rule. Adjacent roots
would share most live RNG streams. This does not explain the old Pong variation
or replace the need for independent model runs and a fixed declared budget.

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

Do not spend weeks on an unprofiled full sweep. The completed campaign cost
3.8 hours per 100k aggregate actions: roughly 300 GPU hours for 26 games × 3
seeds before controls. The new short runtime gate improves that rate, not its
order of magnitude.
Use the panel to resolve failure modes and improve serial throughput first.

## Runtime: uncapped, accelerated, and eventually free-running

| Mode | Meaning | Current status |
| --- | --- | --- |
| Uncapped step-driven learning | Advance the game, then compute without wall-clock pacing | Supported |
| Super-real-time playing plus training | More simulated game seconds than wall seconds, including learning | R64 Boxing exceeds 1× in aggregate; the selected R256 recipe does not |
| Free-running learning without time control | The game continues while the agent computes | Required eventually; not validated by step-driven Atari |
| Frozen evaluation | Act without parameter updates | Supported; not training throughput |

The selected fresh-run control is native LeVJEPA, 12M, **N6/R256/B16/T64**,
full 64-step BPTT, microbatch 16 and F32. One learner shares policy parameters
across six independently initialized environments; each retains its own visual
cache, belief, RNG and contiguous replay sequences. Batched inference and
row-independent replay/head work are implemented without batching away recurrence.

For new experiment declarations use the
[qualified current package](experiments/2026-09-09-meganeura-update.md#use-the-qualified-package)
(`f6a2b6ad…`) with its matching Python modules, runner and auditors. The adopted
backend `4d45ba3a…` carries current upstream `e59bd32d…` plus the required
LeVJEPA frame-prefix attention and cache-alias corrections, with Blade 0.9.0 and
Rust 1.92 minimum. Production numerical checks, exact complete synthetic/pixel
learning-state comparisons and direct-memory gates pass. The main worktree's
backend integration also passes CPU/GPU checks. Its historical Python accounting
interface differs from the isolated Atari package: do not mix them.
The default editable extension and all old models' executables remain intact.
Strict restore identity still requires each historical model's original backend.
Existing campaigns are not silently switched to the new package.

### Measured throughput and memory

The current-backend N6 comparison reaches **8.554–8.599 actions/s**,
0.570–0.573× aggregate real time and 0.0950–0.0955× per stream. It retains at
least 3,302 MiB directly free across all ten GPU phases. Its two throughput ratios
against a7e2efd9 are 1.004068/0.998255: **no measured speedup**. All four complete
Boxing states, learning reports and action/reset traces match exactly. These
short runtime checks do not establish fresh-seed learning reliability.

The earlier a7e2efd9 [forward/reverse comparison](experiments/2026-09-08-atari-five.md#completed-vector-memory-and-runtime-comparison)
keeps the learner configuration fixed. Each trial trains for 3,840 actual
actions and restores for 768 frozen actions; the timed 1,536-action interval
contains exactly 384 updates. Every same-N repeat reproduces all 241 named
checkpoint tensors and action/episode/reset traces exactly.

| Streams | Aggregate actions/s, two orders | Aggregate game/wall time | Minimum directly free VRAM | Decision |
| ---: | ---: | ---: | ---: | --- |
| 4 | 8.423 / 8.432 | 0.562× | 4,889 MiB | Eligible, slower |
| 6 | 8.574 / 8.550 | 0.570–0.572× | 3,302 MiB | Selected; retained after current-backend check |
| 8 | 8.672 / 8.657 | 0.577–0.578× | 1,630 MiB | Fails 2 GiB reserve |

N6 is about 1.2% slower than N8, not a speedup. Per-stream acceleration is
only about 0.095×. Different N changes collection/prefill trajectories; these
short deterministic repeats are not a learning-quality or seed-reliability
comparison. A changed package/configuration requires matching runtime evidence.

The [memory-accounting correction](experiments/2026-09-08-atari-five.md#memory-accounting-correction)
withdraws older reserve-pass claims based on total minus used: that calculation
omitted driver-reserved memory. Require directly sampled free VRAM ≥2,048 MiB
with coverage checks. Preserve original logs and safety flags; they cannot
prove fields they never recorded. Do not lower the gate or silently change
precision, sequence length or learner batch to pass it.

### What actually costs time

The current N6 timed windows spend roughly **74% learning, 25% observation and
less than 1% in the emulator**. Mean GPU activity is 67–69%; activity is not
occupancy or a measurement of idle gaps. Full learner calls are approximately
345–346 ms: world training ~160 ms, imagination ~85 ms, posterior inference
~59 ms, with behavior and synchronization making up most of the remainder.

At 15 aggregate actions/s, R256 and B16×T64 require 3.75 learner updates/s.
The updates alone exceed the one-second budget. Including roughly 30 ms of
observation/other work per action leaves only about **145–150 ms/update** for 1×.
Eliminating host handoffs alone cannot deliver that: world training already
costs more. The simplicity of Atari does not make this replay-heavy computation
cheap.

[Device-resident imagination](experiments/2026-09-08-device-imagination.md)
already improved exact paired pixel throughput by 12.9–13.3%; the subsequent
[backend refresh](experiments/2026-09-08-meganeura-refresh.md) added only
0.65–1.24%. Do not repeat their completed queues. A
[grouped RSSM-gate rewrite](experiments/2026-09-08-grouped-rssm-gates.md) passed
focused output/gradient tests but failed exact full-learning parity from
update 3. It remains experimental; its short timing is not an adopted speedup.

A small [world-sync fan-out candidate](experiments/2026-09-09-world-sync-fanout.md)
removes repeated reads of overlapping weights while retaining backend cache
refresh. It is CPU-tested only; require GPU state/trace equality, memory headroom
and paired timing before adoption. It does not displace the current learning
queue or the larger world-training bottleneck.

Prioritize measured world-training kernels/layout and recurrent handoffs, then
perception. Preserve F32 gradient safeguards and full recurrence. Readback waits
include unfinished producer compute and transfers, not just GPU idle time.
The available external capture resolves queue submissions, not individual
dispatches or calibrated idle gaps; use validated stage timings and untraced
paired throughput meanwhile. Serialize GPU-heavy work, and avoid large CPU
graph builds during learning: an earlier probe caused host-memory pressure.

Smaller models, larger learner batches and lower replay ratios are legitimate
learning-compute ablations, not identical-recipe optimizations. R64 Boxing
reaches about 1.31× aggregate real time and passes its one-seed task gate, but
R256's mean +92.49 has much more margin than R64's +51.55. Retain R64 as the
faster ablation and test quality across seeds before switching the control.

Report actual emulator frames, aggregate/per-stream actions, updates, training
debt, cold construction and end-to-end wall time. Never count vector ticks as
interactions, sum subtimings into their parent totals, or call fast frozen
inference super-real-time learning. The next target is sustained >1× with
retained quality; 2× is a useful stretch target.

Use mind-games' time control for accelerated development. Separately validate
free-running play: timestamp observations and executed inputs, record elapsed
game time and observation gaps, bound training debt and give variable-duration
transitions explicit semantics. Try measured serial scheduling before
actor/learner concurrency. Pausing the game does not satisfy this requirement.

## LeVJEPA and video/world pretraining

### First: resolve the frontend's gameplay and runtime gates

Native LeVJEPA inference, numerical parity and causal streaming checks are
implemented. All three frozen Pong seeds win games, but only seed 2 passes
the predeclared mastery gate; the all-seeds recipe fails. The
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

1. Retain all three completed final endpoints and the failed all-seeds decision.
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

An isolated [native world-only candidate](https://github.com/kvark/kindle/blob/8196bd5666e9a79302ffd2d510b82509da965434/docs/experiments/2026-09-07-world-pretraining.md)
adds aligned feature-clip learning with explicit missing-label masks and no
actor/critic sessions, plus strict dynamics-only initialization of fresh serial
or vector runtimes. It passes 91 Rust CPU tests, but remains unadopted: verified
dataset ingestion, GPU checks and adaptation/retention comparisons are unfinished.
Initialized checkpoints use format 4 to retain the offline source history;
the existing format-3 campaign declarations and Python runners are unchanged.

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

Validate explicit compatible-weight initialization, distinct from full checkpoint
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

The fixed Freeway exploration comparison is complete: hold64 and hold1 both
pass their unassisted frozen gates; the restored untrained control scores zero.
All budgets, checkpoints, replays and GPU phases validate. Carry hold64
provisionally into fresh Freeway confirmation, retaining the successful hold1
control; neither result proves reliability. Do not restart the completed queue.
The corrected v2 three-seed Freeway worker is now declared with 626 pins and 102 CPU
checks, and its actual live-parent refusal is verified. It stays behind the
already declared Boxing/runtime/Breakout/Qbert sequence; no follower or GPU
worker has started. Its all-seed gates and separately restored controls are fixed.
The episode-count evaluation gate is complete through a separately declared
continuation after interruption: full frozen-state/default-trace/prefix parity
and direct-memory checks pass. Preserve the interrupted original and completed
continuation; this is tooling validation, not a new learned result. The requested
latest Meganeura refresh is qualified and adopted, with no measured speedup.
Fresh three-seed Boxing confirmation is now running on that qualified package.
Keep its seeds, 200,004-action budgets, frozen evaluations and controls fixed.
The corrected v2 Breakout/Qbert pilots are declared but not running. Their 521 pins,
92 CPU checks and actual live-parent refusal bind the fixed seed-0 budgets,
unassisted final evaluations and separate untrained controls. The current-package
episode-count candidate is CPU-qualified at `24b2968`, with 580 tests and unchanged
native bytes. Its executable current-package runtime gate is now declared with
498 pins and 47 CPU checks, but has no active follower/GPU worker. The actual
live-parent refusal is verified; run its state/trace/prefix/memory checks only
after the entire Boxing queue completes and its results independently reverify.
Preserve but do not launch the superseded, unstarted 512/617-pin declarations.
Early episode resets can advance replay warmup without adding action credit;
derive each game's exact update count from its complete ledger and bind restored
state to that count. The CPU Breakout counterexample verifies this audit repair,
not a change to learning arithmetic or evidence of task success.
Keep the optional world-sync
comparison separate: its ~16 ms/update scope does not justify postponing learning
again. Rebase/redeclare it against the current backend before any GPU test,
without displacing the live confirmation queue.
The common-distribution world report finds
strong recording-dependent reward errors, including in the best Pong player;
prioritize reward discovery/generalization and measured training coverage.
The completed plain-policy Freeway pilot fails without finding any reward.
Keep Pong's failed all-seeds gate visible; broader Atari learning and fresh three-seed
reliability remain unfinished. Stage runtime candidates separately from the
live experiment. Add world-only pretraining and the current-Dreamer mind-games
adapter as their gates are reached. Do not start actor/learner separation,
independent learner swarms or another arbitrary 100-hour run as a substitute
for stronger behavior.

For the remaining Atari pilots, avoid a fixed evaluation window too short for
20 long games. The [episode-budget candidate](experiments/2026-09-09-episode-evaluation.md)
keeps a fixed per-stream episode target and hard action cap, with every completed
episode scored. Its CPU and complete frozen-state/prefix/memory GPU checks pass
on the previous backend. Latest-package qualification is still required; the
new pilot declaration supplies the four-per-stream target, 600,000-action cap
and longer frozen timeout. Existing evaluations and all five task/three-seed
criteria are unchanged.

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
