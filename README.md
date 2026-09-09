# Kindle

Kindle is an experimental Rust agent that learns while acting. It combines
Dreamer's recurrent world model and imagined actor/critic with frozen DINOv3 or
causal LeVJEPA perception, using [Meganeura](https://github.com/kvark/meganeura) and
[Blade](https://github.com/kvark/blade) for native GPU computation.

The current pivot adds action-conditioned prediction of unseen visual features.
Prediction-only has passed initial own-action learning gates on native GridWorld
and Pong with DINOv3. The native LeVJEPA video frontend is now implemented;
[three vectorized Pong seeds](docs/experiments/2026-09-06-vector-pong.md) have
completed frozen evaluation: one passes the stronger mastery gate, two fail.
LeVJEPA also passes the one-seed Boxing pilot; Freeway's plain-policy pilot
fails its frozen gate. This demonstrates learned play, not a consistently
reliable five-game recipe.
Reconstruction remains the measured control. Next comes
broader single-actor gameplay, accelerated playing plus training, pretraining and
cross-game transfer. Vectorized environments now share batched inference and one
learner; distributed learning and experience sharing remain later milestones.

Read the [single project plan](docs/kindle_single_life_dreamer_plan.md) for the
architecture, evidence and research gates, and [AGENTS.md](AGENTS.md) for working
directions. Superseded planners and training stacks remain in git history.

## Watch the current agent

Local Pong footage: [seed 2 (+20)](runs/levjepa-vector-pong-20260906/seed2-eval-first-game-reconstructed.mp4)
and [seed 0 (+14)](runs/levjepa-vector-pong-20260906/seed0-eval-first-game-reconstructed.mp4).
For comparison, [seed 1 (−1)](runs/levjepa-vector-pong-20260906/seed1-eval-first-game-reconstructed.mp4)
and the [untrained agent (−21)](runs/levjepa-vector-pong-20260906/untrained-first-game-reconstructed.mp4).

These are each run's first completed game, reconstructed from its logged actions
with every reward, boundary and frame count checked—not original screen captures
or selected best games. Videos are local, git-ignored artifacts. See the
[footage record](docs/experiments/2026-09-06-vector-pong.md#reconstructed-gameplay-footage)
for provenance; the full frozen evaluation, not one video, determines mastery.

Boxing: [trained full-stream video](runs/atari-five-continue-20260908.JrdVto/boxing-r256-seed0-eval.mp4)
and [trained/untrained comparison](runs/atari-five-continue-20260908.JrdVto/report.html).
The R256 seed-0 frozen evaluation wins 162/162 natural matches across eight
streams, mean +92.49. This passes the pilot's task gate, not fresh-seed reliability.

Freeway: [failed full-stream video](runs/freeway-pilot-20260908.WWxHEM/evaluation.mp4)
and [evidence](docs/experiments/2026-09-09-freeway-zero-signal.md#final-frozen-policy-no-learned-crossing-skill).
After 200k training actions, all 36 frozen rounds return zero. Both games' videos
are complete CPU reconstructions of stream 0, including its unfinished tail;
their full multi-stream evaluations determine the scores.

The local [forecast-versus-match report](runs/world-evaluation-20260908.Xzx3pN/report.html)
compares the three final Pong world models with their first recorded matches.
It includes feature and
reward baselines, full prediction traces and clickable point times in the videos.

## Architecture

```text
previous belief + action ──► deterministic state ──► forecast frozen features
                                     │
RGB ──► frozen encoder ──► 7×7×64 ───┤
                                     ▼
                            categorical posterior
                                     │
                     reward / continuation / reconstruction
                                     │
                       imagined actor + two-hot critic
```

The forecast reads the deterministic state *before* the current observation
enters the posterior. The feature-reconstruction control reads the posterior.
Both retain balanced KL, categorical sampling, replay-value learning, 15-step
imagination and DreamerV3's LaProp/AGC optimizer ordering.

Three objective settings are exposed in both native and Atari runners:

| Experiment | `--reconstruction-loss-scale` | `--future-prediction-loss-scale` |
| --- | ---: | ---: |
| Calibrated reconstruction control | 0.25 | 0 |
| Predictive auxiliary | 0.25 | 0.25 |
| Prediction only | 0 | 0.25 |

Zero disables and removes the corresponding head's parameters. Future targets
are stop-gradient frozen features; reset observations do not contribute to that
loss. Both heads use the same spatial structure and matched downstream
initialization. These coefficients are experimental settings, not universally
tuned values.

Intentional differences from the pinned upstream DreamerV3 include frozen features
instead of learned RGB perception, f32 computation, 100k-entry replay, and
separate world/behavior optimizer sessions. Library defaults retain BPTT 8 and
train ratio 32 for compatibility. The historical Atari control uses full BPTT 64,
B16×T64, row microbatch 4 and ratio 256; the new matched causal/reconstruction
runs use row batch 16. Keep those results' provenance separate. Microbatching
accumulates gradients before one update; it does not truncate recurrence.

## Build and weights

Use Rust 1.92 or newer and a Blade-supported GPU backend. Linux experiments use
Vulkan. Practical learning needs a hardware accelerator.

DINO weights are supplied separately:
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m),
snapshot `114c1379950215c8b35dfcd4e90a5c251dde0d32`.
The model is externally licensed and is not committed here.

LeVJEPA uses the separately licensed CC-BY-NC-4.0
[`galilai-group/LeVJEPA-VideoMix-Large`](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large),
snapshot `e831a0347737fcaa660b39c57d41c109de399845`. Select it with
`--encoder levjepa` in the Atari runner, `encoder="levjepa"` in Python's Agent,
or `DreamerAgent::with_perception(..., PerceptionKind::LeVJepa, ...)` in Rust.
It processes every arrival in bounded 16-frame causal chunks using F32 native
inference; the world model stays continuous across chunk boundaries.
Checkpoint restore verifies the saved encoder identity and actual weight file.
Format-2 DINO checkpoints remain usable with their original executable, not as
initializers for the new frontend's coordinates.

On multi-adapter hosts, `MEGANEURA_DEVICE_ID` selects a backend-reported numeric
device ID, not an adapter ordinal. Our RTX 5080 uses `0x2c02`; omit the variable
to retain Blade's default selection.

```bash
cargo build --release --workspace --examples

cd python
python -m venv .venv
. .venv/bin/activate
pip install maturin
maturin develop --release --extras test
```

During pinned local experiments, use an isolated package rather than replacing
the historical editable extension. The [backend refresh record](docs/experiments/2026-09-08-meganeura-refresh.md#use-the-adopted-package)
shows how to select the validated current package in this workspace.

## Native use

```rust,ignore
use kindle::{ActionMode, DreamerAgent, DreamerConfig};

let mut config = DreamerConfig::new(environment.action_count());
config.loss_scales.reconstruction = 0.25;
let mut agent = DreamerAgent::new(config, "/models/dino/model.safetensors", None)?;
agent.begin_episode(&environment.reset());

let action = agent.act(ActionMode::Sample, None);
let transition = environment.step(action);
let reward_channels = agent.observe(&transition);
let reports = agent.learn_scheduled(1);
if transition.terminated || transition.truncated {
    agent.begin_episode(&environment.reset());
}
```

The first reset and executed transitions have different replay semantics.
Learning starts after enough complete replay sequences exist. State and learning
persist across natural episodes; `begin_episode` resets belief and visual history.
Validity masks affect live actions, not imagination: give invalid actions a
benign meaning or use the same fixed vocabulary throughout.

GridWorld is a small visual integration task, not a general benchmark:

```bash
cargo run --release -p kindle-gym --example grid_world -- \
  /models/dino/model.safetensors --steps 10000 --model-size 1m \
  --all-actions --world-backprop-length 64 --train-ratio 256 \
  --random-action-steps 10000 --learning-rate 0.001 \
  --behavior-learning-rate 0.0001 --learning-rate-warmup 0 \
  --reconstruction-loss-scale 0.25 \
  --checkpoint checkpoints/grid --output runs/grid.jsonl

cargo run --release -p kindle-gym --example grid_world -- \
  --random --all-actions --persistent --steps 10000

cargo run --release -p kindle-gym --example grid_world -- \
  /models/dino/model.safetensors --all-actions --steps 10000 \
  --restore checkpoints/grid --evaluate --output runs/grid-eval.jsonl
```

This native gate uses diagnostic learning rates and a fixed random training
trajectory before frozen evaluation. Omit `--random-action-steps` to let the
learning agent collect its own data; the persistent reward comparison does so.

`--persistent` disables the artificial episode limit. Exhaustion respawns the
agent through normal dynamics, costs one game reward, and preserves food
progress. In this mode positive/negative reward events count food/deaths.
Keep this flag and action vocabulary matched across training and evaluation.

`--visitation-bonus --intrinsic-reward-scale 0.1` enables an optional bounded
visual-novelty experiment. A fixed 16-bit random-hyperplane hash counts visits;
features are centered on a checkpointed, fixed first-observation reference.
The bonus is `1/sqrt(previous_count + 1)`. Counts survive episode resets and
checkpoints, and saturate instead of wrapping. Collisions reduce novelty.
This is not epistemic uncertainty. Intrinsic and extrinsic channels are logged
separately; neither intrinsic generation nor its learning scale is enabled by
default. `--train-ratio 0` disables scheduled updates for coverage/noise checks;
the native `--fixed-action N --all-actions` supplies a stationary-action control.

## Atari and analysis

From an activated Python environment at the repository root:

```bash
python python/examples/atari.py /models/dino/model.safetensors ALE/Pong-v5 \
  --steps 100000 --seed 0 --atari-protocol published \
  --model-size 12m --world-backprop-length 64 --world-microbatch-size 4 \
  --checkpoint checkpoints/pong --output runs/pong-seed0.jsonl

python python/examples/atari.py /unused ALE/Pong-v5 \
  --steps 100000 --seed 0 --atari-protocol published --random-policy \
  --output runs/pong-random-seed0.jsonl

python python/examples/summarize_atari_scores.py \
  runs/pong-seed0.jsonl runs/pong-seed1.jsonl runs/pong-seed2.jsonl \
  --minimum-kindle-seeds 3 --require-uninterrupted-kindle

python python/examples/summarize_atari_training.py runs/pong-seed0.jsonl
```

`published` uses all 18 actions, no reset no-ops, repeat four, max-pooling,
non-sticky actions, a 100k-frame episode cap and Pillow 64×64 RGB resizing.
`current` follows the pinned upstream source's newer wrapper.
`published-minimal` changes only the action vocabulary. Do not mix protocols.

The conventional score window is 350k–400k nominal frames, with episode scores
averaged within seed before averaging seeds. These are training-window scores,
not frozen endpoint evaluations. `--restore ... --evaluate` runs a frozen
sampled policy; `--greedy` is a separate diagnostic. Restores refill replay and
must not be presented as uninterrupted runs.

For a qualitative check of the final checkpoint, the separate recorder leaves
the training runner unchanged:

```bash
python python/examples/record_atari.py /models/dino/model.safetensors ALE/Pong-v5 \
  --restore checkpoints/pong --evaluate --steps 10000 --seed 0 \
  --atari-protocol published --output runs/pong-video.jsonl --video runs/pong.mp4
```

This requires `ffmpeg` and writes raw-emulator gameplay at nominal 60 Hz, including
reset no-ops, with a hashed `pong.mp4.json` manifest and ordinary evaluation log.
It refuses existing artifacts and training runs. Playback is not wall-clock agent
latency; videos supplement fixed-budget scores, not selected-checkpoint claims.

The recovered 12M Pong runs scored −4, +6 and −2 (seed mean 0), versus matched
random −20.623. Local pinned upstream 1M and 12M controls also exist. See the plan for
the small episode counts, runtime/perception differences and artifact locations.

Read-only native and Atari probes measure reward calibration, representation
separability, open-loop feature error against persistence/unrelated actions,
actor/model agreement and critic calibration. Each example's `--help` describes
the exact protocol.

### Vectorized LeVJEPA collection

`VectorAgent` shares one encoder and learner across independent Atari streams.
LeVJEPA's dense layers and Dreamer's live posterior/policy are GPU-batched;
visual caches, beliefs, RNGs and replay sequences remain separate. CPU emulator
steps are synchronous: measured environment work is below 1% of wall time.
The selected fresh-run configuration is **N6/12M/B16/T64/R256**, full BPTT64,
row microbatch 16 and F32. Its paired RTX 5080 measurements reach 8.55–8.57
aggregate actions/s: 0.570–0.572× the game clock in aggregate and about 0.095×
per stream. All same-N checkpoint tensors and action/episode/reset traces repeat
exactly. GPU activity averages about 69%; learning takes roughly 74% of loop time
and observation 25%. Activity is not occupancy or measured idle time.

The [memory/runtime comparison](docs/experiments/2026-09-08-atari-five.md#completed-vector-memory-and-runtime-comparison)
measures at least 3,302 MiB directly free for N6. N8 retains only 1,630 MiB and
fails the 2 GiB safety gate. The earlier total-minus-used reserve claim omitted
driver reservations and is withdrawn. Require directly sampled free memory and
matching runtime evidence for a changed package or configuration.

R256 playing **with training** remains below real time. The lower-ratio R64
Boxing pilot reaches about 1.31× in aggregate and passes its one-seed task gate;
it is a separate learning-compute ablation, not an identical-recipe speedup.
See the [runtime analysis](docs/kindle_single_life_dreamer_plan.md#what-actually-costs-time)
and [adopted package](docs/experiments/2026-09-08-meganeura-refresh.md#use-the-adopted-package).
The default local editable extension deliberately remains the historical control;
do not overwrite pinned executables or mix them with newer checkpoint identities.

```bash
python python/examples/profile_atari_vector.py /models/levjepa/model.safetensors \
  runs/vector-profile --num-envs 1 2 4

python python/examples/atari_vector.py /models/levjepa/model.safetensors \
  --num-envs 4 --steps 20000 --checkpoint checkpoints/vector-canary \
  --output runs/vector-canary.jsonl

python -m kindle._vector_audit runs/vector-canary.jsonl
```

`--steps` is the **total executed actions across all environments**, not actions
per environment. The runner drains scheduled learner credit after each vector
step and records aggregate and per-stream simulated/wall time separately.
Reset frames enter replay but earn no action credit. SIGINT/SIGTERM stop at a
completed vector round and save when `--checkpoint` is supplied. Output and
checkpoint paths must be fresh. Restore starts fresh environment histories and
replay; it is not exact lifetime recovery. `--restore ... --evaluate` is frozen,
sampled evaluation; a throughput check or accounting audit is not Pong mastery.

`python python/examples/check_atari_adapter.py runs/atari-adapters` checks the
eight-game Atari panel on CPU, comparing two interleaved random-action streams
with fresh serial replay of their images, rewards, boundaries and clocks.
It constructs no agent and makes no learning or competence claim. Positive
terminal return is the vector runner's Pong win rule, not a general Atari win.

## Checkpoints and verification

Checkpoints contain world/behavior parameters and optimizer moments, the slow
critic, configuration, backend/vision provenance, counters, return normalization
and optional visitation counts. Replay, scheduler credit, exact RNG state and
the in-flight environment are absent. Restore is model recovery into a fresh
data segment, not exact lifetime continuation. Backend revisions are validated;
old artifacts require their original checkout. Experimental head and hash
revisions are also checked. Run headers expose model provenance; Atari headers
add the executable extension and runner hashes, preventing mixed implementations
from silently becoming a multi-seed score.

Format-3 pixel-agent checkpoints record the frontend, its causal/preprocessing
semantics and the actual encoder weight-file SHA-256. Restore checks all three
before GPU construction; equal tensor shapes do not make different perceptual
representations compatible. Format-2 checkpoints require their original build.

New saves fingerprint all three tensor files in metadata so restore can reject
damaged or mixed-generation files. Every restore requires all model and optimizer
tensors, but legacy files without hashes cannot prove generation consistency.
Individual file renames are not an atomic generation: a torn hashed save can be
detected but is not automatically recoverable.
Keep a separate known-good checkpoint until generation-based recovery is added.

### Evaluate the world model separately

`probe_atari_dynamics.py` forecasts features, rewards and discounted continuation
before seeing the target frames. `--recorded-run` replays the first complete game
of a frozen single-stream vector evaluation, failing if sampled actions or real
transitions differ. Use the checkpoint's original native build; historical models
cannot be restored by silently changing their backend metadata.

```bash
PYTHONPATH=python MEGANEURA_DEVICE_ID=0x2c02 \
  python python/examples/probe_atari_dynamics.py /models/levjepa/model.safetensors \
  checkpoints/pong-final --recorded-run runs/pong-eval.jsonl \
  --horizon 15 --stride 5 --output runs/world-probe.json \
  --trace runs/world-predictions.jsonl
```

Output paths must be fresh. One-step forecasts cover every action; longer
forecasts start at the declared stride and condition on the recorded controls
without intermediate observations. Compare against feature persistence, unrelated
actions and zero rewards; inspect positive/negative event errors separately.
Posterior reward estimates already see the frame and are not forecasts.
These are LeVJEPA/DINO feature predictions, not rendered imagined video, and
good prediction scores alone do not establish a good policy. Omitting the
recording runs a forced-random coverage diagnostic, not learned-policy evaluation.

`LearnReport.timing` separates replay, posterior, imagination, training and
synchronization wall time. The synthetic canary avoids perception/environment
costs and can export inference and gradient traces. Gradient captures exclude
optimizer/clipping/accumulation and use the final row microbatch. Graphs exceeding
Blade's timestamp capacity retain ordinary wall time and an explicit missing-trace
reason; they are not silently reported as complete GPU profiles:

```bash
MEGANEURA_GPU_TIMING=1 cargo run --release -p kindle --example dreamer_canary -- \
  12m 64 4 --learn --updates 8 --profile-dir runs/profile

cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
cargo test --release -p kindle tiny_ --lib -- --ignored --test-threads=1
KINDLE_DINOV3_WEIGHTS=/models/dino/model.safetensors \
  cargo test --release -p kindle vits16_checkpoint_matches_hugging_face \
  --lib -- --ignored
PYTHONPATH=python python -m pytest python/tests
```

Serialize GPU-heavy tests and experiments on one device. Trace instrumentation
changes launch overhead; report ordinary wall time alongside GPU timestamps.
