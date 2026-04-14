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

- Each env gets a `task_embedding: [TASK_DIM]` (8 dims).
- The embedding is fed as a graph **input** (named `"task"`) to the
  encoder, summed with the obs projection at the hidden layer (same
  trick we use to avoid concat in the world model).
- Embeddings are **deterministic-random per env_id**, not learned. The
  encoder learns to map `(obs_token, env_embedding)` into per-env
  latents; the embedding itself is just a stable identifier.

  Why not learned? An earlier iteration made the embedding a graph
  parameter so SGD would train it. Meganeura's autodiff over a small
  parameter on this code path was unstable — the world model diverged
  to NaN within a few steps. Making the embedding a fixed input (per
  env_id, picked from a golden-ratio hash) avoids the issue and still
  gives the encoder a unique env signal. The encoder absorbs the
  per-env specialization into its own weights.

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

## Fourth reward primitive: order

The frozen reward circuit has three primitives today: surprise, novelty,
homeostatic. We add a fourth: **order**, which rewards the agent for
reducing entropy in its observations.

Motivation: ARC-AGI-style tasks require the agent to *fix* things —
complete a pattern, unify a palette, resolve an inconsistency. The agent
should prefer states where the observation is more concentrated /
regular. Surprise encourages exploration; order encourages
consolidation.

Definition (v1, intentionally simple):

```
p_i    = |obs_i| / (Σ_j |obs_j| + ε)
H(obs) = -Σ_i p_i · log(p_i + ε)
r_order = -H(obs)
```

The observation is reinterpreted as a probability distribution over its
own components (after normalization). Shannon entropy of that
distribution is low when one or a few components dominate — "ordered" —
and high when the mass is spread — "chaotic". The reward is the
negative of entropy, so ordered states get more reward.

Combined reward becomes:

```
r_t = w_s·r_surprise + w_n·r_novelty + w_h·r_homeo + w_o·r_order
```

Default `w_o = 0` — we don't want to change existing behaviour on envs
where order isn't meaningful (CartPole doesn't have a notion of
order-vs-chaos in the observation). ARC-AGI-ish envs set `w_o > 0` and
let the agent chase state-space concentration.

Known limits of v1:

- **Degenerate on one-hot observations.** GridWorld's observation is
  always one-hot (plus energy scalar) so order is ~constant. Fine —
  set `w_o = 0` for those envs.
- **Latent-space alternative.** A more principled variant measures
  entropy on the encoder output (the latent) instead of the raw
  observation. This gives a learnable "order" but can drive the
  encoder to collapse. Defer.
- **Window-based alternative.** Compare entropy of recent obs window
  to older obs window; reward reductions. Captures "agent is
  progressively reducing env chaos" more cleanly but needs a window
  buffer. Defer to v2.

Implementation is a pure CPU function like the other primitives;
no graph changes.

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

## Open questions (resolved)

- **`OBS_TOKEN_DIM` and `MAX_ACTION_DIM`** — picking from current envs:
  max `obs_dim` = 34 (Taxi), max `action_dim` = 6 (Taxi).
  Set `MAX_ACTION_DIM = 6`, `OBS_TOKEN_DIM = 64`. Both can grow later.
- **Action mask: CPU side.** The graph stays clean; the per-env adapter
  zeros or softmaxes over the live dims before returning an action.
  Revisit if the wasted gradient on dead dims becomes a problem.
- **One shared credit assigner** across all envs. Sample contrastive
  pairs only within the same env (env_id-stratified) so the signal
  stays meaningful. Simpler than per-env credit and keeps transfer of
  temporal-attribution circuitry across envs.

## Still to decide later

- **Frozen reward circuit and env hopping.** The reward circuit is frozen
  by design, but `homeostatic` is per-env. The route is trivial since we
  already pass `env: &dyn Environment` to `observe()`.
- **Catastrophic forgetting across envs.** Replay mixing already exists
  for within-env. Cross-env replay (sampling old envs while in a new
  one) is a separate concern; the multi-env buffer enables it, but the
  right sampling ratio wants measurement.

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
