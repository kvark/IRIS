# Phase E — Batched lanes

## Goal

Lift `Agent` from one env per step to **N envs per step**, where N is baked
into the compiled GPU graphs as `batch_size`. Every per-step dispatch
(world-model forward+backward, policy+value forward+backward) processes all
N lanes in a single kernel launch. For `N = 1` the runtime behaviour is
byte-identical to today — Phase E is a strict superset.

## Where we are today

`AgentConfig.batch_size` exists and is already wired into the graph-shape
inputs (`[batch_size, OBS_TOKEN_DIM]`, etc.), but every `observe()` call
feeds exactly one row. With `batch_size = 1` the agent is single-lane. The
code has no concept of "which lane" — there's one `adapter`, one
`ExperienceBuffer`, one `RewardCircuit`, one `pending_boundary` flag.

## Core change

Internalize multi-lane at the Agent level. Lane count `N` is fixed at
construction and equals `config.batch_size`. The agent owns N adapters and
N per-lane CPU states; the GPU graphs are unchanged in shape — they just
get fed non-trivially populated batch rows.

### What is batched (GPU)

- **World model + encoder.** One forward+backward per step, `[N, …]`
  tensors throughout. The loss is a mean over the batch (status quo — this
  is what `mse_loss` already does when batch_size > 1). The biggest single
  win: one kernel dispatch amortizes GPU launch overhead across N.
- **Policy + value.** Same deal. Gaussian-MSE loss means over the batch.
  Value head's `value_target` becomes an `[N]` input. The continuous head
  output is `[N, MAX_ACTION_DIM]`; each lane's adapter samples its own
  action from its row.

### What stays sequential (CPU-lightweight)

- **Credit assigner.** The history graph is sized for one lane
  (`[history_len × (latent + action + 1)]` flat). We call it N times per
  step, feeding each lane's own history. Credit is ~2 orders of magnitude
  cheaper than WM already; running it N times costs us no observable
  throughput. Rebuilding the credit graph for `[N, history_len, …]` would
  mean a second shape convention and bigger tests for a negligible win.
- **Reward primitives** (surprise, novelty, homeostatic, order). All CPU
  already; each lane computes its own. Order in particular now carries
  per-lane state (recent + reference windows) that cannot be shared across
  lanes without distorting the signal.

### Per-lane state

```text
lanes: Vec<Lane>            // length N, fixed at construction

struct Lane {
    adapter:          Box<dyn EnvAdapter>,
    buffer:           ExperienceBuffer,     // own novelty visit counts
    reward_circuit:   RewardCircuit,        // own order window + phi-seed
    pending_boundary: bool,

    // Cached last-step values for diagnostics & policy advantage.
    last_value:     f32,
    last_entropy:   f32,
    last_surprise:  f32,
    last_novelty:   f32,
    last_homeo:     f32,
    last_order:     f32,
    last_reward:    f32,
}
```

### Shared state

- `task_embeddings: HashMap<u32 /* env_id */, Vec<f32>>` — one vector per
  **env_id**, not per lane. Multiple lanes pointing at the same env share
  one embedding. Each step we stack the active embeddings row-wise:
  `task_input[lane_i * TASK_DIM..][..TASK_DIM] = task_embeddings[lanes[i].adapter.id()]`.
- Single `wm_session`, `credit_session`, `policy_session`. Parameters are
  shared across lanes by construction; that's the whole point.
- Representation-drift probe set: one shared set, probe forwards still use
  the WM session. When the probe count isn't a multiple of N, we pad the
  remainder rows with zeros and ignore their outputs.
- `step_count`, `encoder_lr_scale`, `last_*_loss` fields for global metrics.

## API changes

```rust
impl Agent {
    /// Build an N-lane agent. `adapters.len()` equals `config.batch_size`;
    /// mismatching shapes panic at construction. For a single-lane agent,
    /// pass a single-element vec: `Agent::new(cfg, vec![adapter])`.
    pub fn new(config: AgentConfig, adapters: Vec<Box<dyn EnvAdapter>>) -> Self;

    /// Select one action per lane. `observations.len() == N`.
    /// Returns a `Vec<Action>` of length N, in lane order.
    pub fn act<R: Rng>(&mut self, observations: &[Observation], rng: &mut R) -> Vec<Action>;

    /// Observe one synchronous step across all lanes. All input slices
    /// must have length N.
    pub fn observe<R: Rng>(
        &mut self,
        observations: &[Observation],
        actions: &[Action],
        envs: &[&dyn Environment],
        rng: &mut R,
    );

    /// Swap one lane's adapter. Marks `pending_boundary` on that lane only.
    pub fn switch_lane(&mut self, lane_idx: usize, adapter: Box<dyn EnvAdapter>);

    /// Per-lane diagnostics; length N.
    pub fn diagnostics(&self) -> Vec<Diagnostics>;
}
```

N=1 migration for existing callers: wrap each single value in a one-element
slice. Examples in `kindle-gym/` get a `vec![adapter]` at construction and
`&[obs]` / `&[action]` / `&[&env]` at the call sites.

## Per-step flow

```text
for each step:
    1. Build stacked task input: [N, TASK_DIM], one row per lane.
    2. Build stacked obs_token input: [N, OBS_TOKEN_DIM], one row per lane.
    3. Build stacked z_target input: [N, LATENT_DIM], per-lane previous
       latent (zeros if pending_boundary[lane] || buffer[lane] empty).
    4. Build stacked action input: [N, MAX_ACTION_DIM] from the actions.
    5. wm_session.step() + read loss + read z_t → [N, LATENT_DIM].
    6. For each lane i:
         - pred_error_i    = sqrt(latent_dim · wm_loss).  (loss is mean)
           NOTE: the mean-over-batch loss loses per-lane granularity. See
           "Per-lane surprise" below.
         - surprise_i, novelty_i, homeo_i, order_i computed on lane state.
         - push transition into buffer[i].
    7. For each lane i (credit pass): run credit_session on lane_i history,
       write credits back into buffer[i].
    8. Stack lane actions + rewards + values → policy_session.step() in one
       batched dispatch. Per-lane advantages computed from per-lane
       last_value. Learning rate scale is the mean of per-lane scales (or
       the min; tune).
    9. Replay step (optional): pick one lane at random, run one replay
       minibatch through wm_session using that lane's buffer. Keeps replay
       cost flat in N.
```

### Per-lane surprise

The WM graph emits a **scalar mean loss** over the batch. To recover
per-lane surprise without a graph change, we expose `z_t` (already an
output) and compute `||z_hat − z_target||` per lane on CPU. But `z_hat`
isn't an output (by design — aliasing concerns, see `src/agent.rs`). Two
options:

1. **Approximate per-lane surprise** as `sqrt(latent_dim · wm_loss)` for
   all lanes — the mean-loss surrogate — and accept that per-lane surprise
   collapses to a shared per-step scalar. Cheapest, preserves the current
   reward signal quality when N is small. **Recommended for Phase E.v1.**
2. **Add a second output** `per_lane_loss: [N]` computed inside the graph
   as the per-row L2 of `z_hat − z_target` **before** the mean reduction.
   Only needed if option 1 degrades learning at N ≫ 1. Design stub only.

### Pending-boundary semantics

`switch_lane(i, new_adapter)` flips `lanes[i].pending_boundary = true`.
Next step, lane i's `z_target` row is zeroed and the stored transition is
tagged `env_boundary = true` in lane i's buffer. Other lanes proceed
normally. This matches the single-lane semantics today exactly.

### Replay mixing

Single-lane replay today picks a random `(i, i+1)` pair from the one
buffer and runs `wm_forward_backward` with `batch_size = 1` inputs. Under
Phase E the WM graph has fixed `batch_size = N`, so either:

- **Pad the replay batch with the other N−1 lanes' current-step rows**
  (already on the GPU — cheap). Each replay step trains one row on a
  replayed sample and N−1 rows on the current step (already-trained
  signal). Simple but somewhat wasteful.
- **Sample N independent replay transitions**, one per lane. Slightly more
  principled; matches the "all lanes are training" story; costs one extra
  `flatten` per lane.

Both are valid. Recommendation: sample N per-lane replay transitions — it
preserves the gradient signal density per step and needs no padding hack.

## Things that stay the same

- `AgentConfig` fields, serialization, and defaults — except `batch_size`
  now carries the semantic "N lanes" in addition to its old mechanical
  meaning.
- The public `Diagnostics` struct, just returned as `Vec<Diagnostics>`.
- `RewardCircuit` public API, including the order windowing. Each lane
  gets its own `RewardCircuit` with a distinct `phi` seed (derived from
  lane index) so the per-lane digests don't interfere — or the same seed
  if we want lanes comparable across the run. Recommend lane-index seeds;
  cheaper to debug one lane at a time.
- `EnvAdapter` trait is unchanged. Adapters are still owned 1:1 by a lane.

## Testing

- **N=1 parity test.** Construct an `Agent` with `vec![adapter]`, run
  1000 steps, compare diagnostics to the pre-Phase-E single-lane run on
  the same seed. All fields must match to within numerical noise.
- **N=4 smoke test.** Four CartPole lanes, 5k steps, assert no NaN, assert
  mean `wm_loss` trends down, assert per-lane `buffer_len == step_count`.
- **Heterogeneous lanes.** Lane 0 = GridWorld (discrete 4), lane 1 =
  Pendulum (continuous 1), lane 2 = Acrobot (discrete 3), lane 3 =
  CartPole (discrete 2). Assert training stays bounded and per-lane
  rewards vary by env.
- **Lane switch.** Swap lane 2's adapter mid-run; assert only lane 2's
  next buffer entry carries `env_boundary = true`.

## Out of scope for Phase E

- **Fork-join (multi-session) parallelism.** Covered by Phase F in
  `universal-actions.md`. Phase E is a single-session, single-GPU-context
  multiplexer.
- **Async/pipelined envs.** All N lanes step in lockstep. Envs that block
  (real I/O, human feedback) are a separate concern — solved either by
  batch-wrapping them CPU-side or by dropping to Phase F.
- **Per-lane learning rates.** One global `learning_rate`; lanes vary
  only in their data, not their optimizer schedule.

## What to build first

1. Per-lane state struct + `Vec<Lane>` inside `Agent`. No GPU changes.
   Pipe single-lane behaviour through `lanes[0]` and confirm nothing
   regresses.
2. Stack per-lane obs/action/task/z_target into the existing graph inputs
   (they're already `batch_size`-shaped). This is the only place where
   `N > 1` starts to matter for the GPU path.
3. Per-lane credit + reward on CPU. Diagnostic vector.
4. `switch_lane`, heterogeneous-env smoke test, parity test.
5. Replay with per-lane sampled transitions.
