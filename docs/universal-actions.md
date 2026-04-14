# Universal Actions, Cross-Environment Training, and Fork-Join Learning

## Goal

Lift IRIS from "one agent per env, one env per agent" to a configuration that
can:

1. Hop between environments at runtime, signalled by a reset.
2. Treat discrete and continuous action spaces uniformly.
3. Train multiple agent copies in parallel envs and merge their experience.
4. Adapt quickly to never-seen environments (ARC-AGI3 target).

This document plans the architecture. No code yet.

## Where we are today

- `AgentConfig` bakes `obs_dim`, `action_dim`, and `action_kind` at agent
  construction. The three GPU sessions (world model, credit, policy) all
  have these dimensions hard-wired into their compiled graphs.
- `Action::Discrete(n) | Continuous(Vec<f32>)` exists at the trait level,
  but the agent's policy graph picks one branch at build time and then
  only handles that one.
- The buffer holds raw `(observation, action, reward, ...)` tuples from a
  single source. There's no notion of "which env did this come from".
- Each `Session::new` allocates its own GPU context, so multiple sessions
  per process is real but expensive.

## Constraints to design against

- **Static graphs.** Meganeura compiles tensor shapes into the dispatch
  sequence at `build_session` time; we cannot resize at runtime. Anything
  variable across envs has to live outside the compiled core.
- **One scalar loss per session.** Autodiff in meganeura runs from a single
  scalar; combining losses across heterogeneous envs in one session means
  packing them into one weighted sum.
- **Single GPU context per Session.** Many small sessions add up; in
  practice we want at most a handful of long-lived sessions.
- **Per-step CPU↔GPU transfer is the per-step cost floor.** Batching helps;
  micro-batching (one transition per step) does not.

## Three changes, in dependency order

### 1. Universal action / observation interface

Pick a maximum action dim `MAX_ACTION_DIM` (say 16) and a maximum observation
projection dim `OBS_TOKEN_DIM` (say 64). The core (encoder, world model,
credit, policy, value) only ever sees:

- An observation token of shape `[batch, OBS_TOKEN_DIM]`.
- An action token of shape `[batch, MAX_ACTION_DIM]`.

What changes per env:

- **Observation adapter** (per-env, CPU side at first): pads / projects /
  one-hot-encodes the env's raw observation into the universal token. For
  envs with `obs_dim ≤ OBS_TOKEN_DIM` we just zero-pad. For larger envs we
  add a learned linear projection (eventually a small per-env GPU head).
- **Action adapter** (per-env, CPU side): translates a universal action
  token back into the env's expected `Action`. Discrete env with N actions
  takes the first N policy outputs, softmaxes, samples. Continuous env
  with K dims takes the first K policy outputs as the Gaussian mean. An
  `action_mask: Vec<bool>` describes which output dims are "live".

Rust shape (sketch):

```rust
pub trait EnvAdapter {
    fn obs_to_token(&self, obs: &Observation) -> Vec<f32>; // len OBS_TOKEN_DIM
    fn token_to_action<R: Rng>(&self, head: &[f32], rng: &mut R) -> Action;
    fn action_to_token(&self, action: &Action) -> Vec<f32>;  // len MAX_ACTION_DIM
    fn id(&self) -> u32;     // stable per-env identifier
}
```

Crucially, **the agent's graphs are built once for the universal sizes and
never rebuilt** when we switch envs. This is what makes cross-env training
cheap.

### 2. Task conditioning

Random observation distributions across envs aren't enough for the encoder
to reliably switch behavioural modes. We add an explicit task embedding:

- Each env gets a `task_embedding: [TASK_DIM]` (say 8 dims).
- The embedding is concatenated to the observation token before the
  encoder, or fed as a separate input that the world model and policy
  attend to.
- Embeddings are learned. When a new env arrives, initialize its embedding
  to the mean of existing embeddings (or zeros) and let it train.

This is the standard multi-task RL pattern (e.g. Multitask DQN). It gives
the core a way to specialize behaviour without forking parameters.

### 3. Reset / env-hop signal

```rust
impl Agent {
    pub fn switch_env(&mut self, adapter: Box<dyn EnvAdapter>);
}
```

Effects:

- Push a sentinel `Transition` to the buffer marking the boundary, so the
  credit assigner doesn't try to attribute reward across the boundary.
- Swap the active adapter.
- Optionally: take a parameter snapshot ("checkpoint") for rollback if the
  new env destabilizes training.

The agent's compiled graphs do not change. The buffer keeps growing.
Visit-count novelty reuses the same hash function but tags by env_id so
that revisiting an env doesn't claim "new state" credit for things we
already explored elsewhere. Essentially: state key is `(env_id, latent_grid)`.

### 4. Fork-join (vectorized) training

Two complementary mechanisms:

#### 4a. Batched envs in a single agent (preferred default)

Build the agent with `batch_size = num_envs` and have a single set of GPU
sessions process all envs in parallel. Each batch element is one env's
current step. This requires:

- The shared universal interface (so all envs produce same-shape inputs).
- The buffer becomes per-env (a `Vec<ExperienceBuffer>`) so per-env
  rewards and credit don't get mixed.
- The policy/value/world-model losses sum across the batch as usual; this
  is just the standard "batch dim is env dim" trick. Meganeura supports
  `batch_size > 1` already; we've just been using `batch_size = 1`.

This gives near-linear speedup on a real GPU and zero parameter sync cost.

#### 4b. Multi-worker fork-join (when batch isn't enough)

When envs run at very different speeds (some block on environment ticks)
or when we want true fault-isolation per worker, run multiple Agent
processes/threads, each with its own session, all reading/writing a
shared parameter store:

```rust
pub struct AgentSwarm {
    workers: Vec<WorkerAgent>,
    sync_interval: usize,
    /// CPU-side parameter store, the source of truth.
    canonical_params: HashMap<String, Vec<f32>>,
}
```

Sync protocol every `sync_interval` steps:

1. Each worker reads its current params via `session.read_param`.
2. Compute delta from the canonical params it last loaded.
3. Workers' deltas are averaged (weighted by sample count) into the
   canonical params.
4. All workers `set_parameter` from canonical.

This is essentially A3C/IMPALA-style parameter averaging. Cost is one
CPU↔GPU transfer of the full parameter vector per worker per sync — fine
if `sync_interval` ≥ 100 steps.

Recommended: get 4a working first, defer 4b until we have a real
multi-env-instance scenario.

## ARC-AGI3 readiness

ARC-AGI3 has the structure: many small environments, very limited
interaction per environment, success requires generalization. Mapping to
the design above:

- **Pretraining**: run the agent in fork-join mode across a curriculum of
  the seven existing built-in envs (and any synthetic ones we add). The
  core encoder, world model, and credit assigner learn generic
  representations and dynamics.
- **Adaptation**: when a new ARC-AGI3 env arrives, freeze the core (set
  encoder LR scale to ~0), let the per-env adapter and task embedding
  train. The buffer collects new transitions; novelty rewards drive
  exploration; world-model surprise tells us when the env diverges from
  what we already know.
- **Evaluation**: zero-shot try the new env first (random or
  policy-from-embedding-mean); only "spend" interaction budget on
  adapter training when zero-shot performance is low.

For this to work, the universal action interface has to be expressive
enough for ARC-AGI3 actions. ARC-AGI3 actions are typically grid clicks
or transformations, which fit naturally as `Discrete(N)` or short
continuous vectors. We don't need a token-vocabulary action space.

## Migration path

Phased rollout that doesn't break existing examples:

1. **Phase A — Adapter trait, no agent changes.** Define `EnvAdapter` and
   write CPU-side adapters for the existing 7 envs. The agent still
   builds graphs with the env's actual dims, but goes through the
   adapter for I/O. No behavioural change yet.

2. **Phase B — Universal sizes.** Move the agent's graph sizes to
   `MAX_ACTION_DIM` and `OBS_TOKEN_DIM`. Examples now build the agent
   with universal sizes; adapters do the per-env padding.

3. **Phase C — Task embedding.** Add `task_embedding` input to all three
   sessions. Each env gets a learned embedding vector; the agent loads
   the right one on `switch_env`.

4. **Phase D — Multi-env buffer + env switching.** Replace
   `ExperienceBuffer` with `MultiEnvBuffer { per_env: Vec<EB>, ... }`,
   add `Agent::switch_env`, write a multi-env example that round-robins
   through GridWorld → CartPole → ... and shows the agent doesn't crash.

5. **Phase E — Batched envs.** Bump `batch_size` to N, change observe()
   to consume N (obs, action) pairs at a time. Each step trains on N
   parallel transitions. This is where the GPU starts paying off.

6. **Phase F — Fork-join workers** (deferred). Only if 5 isn't enough.

## Open questions

- **What's the right `OBS_TOKEN_DIM` and `MAX_ACTION_DIM`?** Big enough to
  accommodate ARC-AGI3 (probably 64 / 16 respectively), small enough to
  keep the core fast. Needs measurement.
- **Should the per-env action mask be a graph input or CPU-side?** Graph
  input lets the loss zero out non-live dims cleanly; CPU-side is simpler
  but loses the cleanness. Probably graph input once we get there.
- **Per-env credit assigners?** The credit assigner's contrastive
  training depends on similar states with different rewards; if we mix
  envs, the contrastive pairs span envs and might be meaningless. Either
  stratify pair sampling by env, or keep one credit assigner per env.
- **Frozen reward circuit and env hopping.** The reward circuit is frozen
  by design, but `homeostatic` is per-env. We need `RewardCircuit` to
  route the right env's homeostatic provider per step — trivial if we
  pass `env: &dyn Environment` to `observe()`, which we already do.
- **Catastrophic forgetting across envs.** Replay mixing already exists
  for within-env. Cross-env replay (sampling from old envs while in a
  new one) is a separate concern; the multi-env buffer enables it.

## What to build first

If we want measurable progress soon, the smallest useful slice is:

- Phase A (adapter trait + adapters for existing envs)
- Phase D (multi-env buffer + `switch_env`)
- A `multi_env` example that hops GridWorld → CartPole → MountainCar
  every 1000 steps and confirms wm_loss stays bounded.

That gets us from "one agent, one env" to "one agent, sequential env
hopping" without touching the core graph compilation. It's the most
informative step to do first because it'll tell us whether the encoder
representations are actually transferable or whether each switch causes
catastrophic interference. Phase B/C/E/F all become more concrete once
we see real numbers from this minimal slice.
