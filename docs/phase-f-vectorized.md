# Phase F — Vectorized envs and large-N scale-up

## Goal

Crank Phase E's `N` up until the GPU is the bottleneck — not the envs,
not the CPU reward primitives, not the adapter glue. The kernel shapes
are unchanged from Phase E; we run **the same graphs on much larger
batches**. The payoff is SIMD saturation on both forward inference and
backward gradient dispatch, plus the better gradient SNR that comes with
bigger minibatches.

Phase F is entirely single-process, single-session, single-GPU. Multi-
GPU / multi-node scale-out is explicitly out of scope here.

## Why Phase E alone isn't enough at large N

Phase E gives us one GPU dispatch per step for all N lanes. The win is
real at N ∈ [4, 32]. But the step flow is:

```
  per-lane CPU env.step()   ×N       ──┐
  per-lane adapter calls     ×N       │
  per-lane reward primitives ×N       │  CPU-serial
  per-lane credit pass       ×N       │
  buffer pushes              ×N      ──┘
  GPU wm + policy dispatch   ×1       ── GPU
```

At N = 32 the CPU block grows linearly while the GPU block grows sub-
linearly — the GPU swallows the bigger batch almost for free, but the
CPU side still loops N times. Past the crossover (env-specific; usually
around N = 64 on our current envs) the CPU tail dominates and further
increases in N buy nothing. We never see the kernel saturation that's
sitting right there on the GPU.

Phase F removes the CPU-side linear prefix.

## Core change: vectorized everything on the critical path

Three parallel vectorizations, each independent:

### 1. Vectorized environments

A new trait:

```rust
pub trait VectorizedEnv: HomeostaticProviderBatch + Send {
    fn batch_size(&self) -> usize;

    /// Current observation for every lane, written as `[batch, obs_dim]`
    /// row-major into `out`. No allocation.
    fn observe(&self, out: &mut [f32]);

    /// Step every lane with its action. `actions.len() == batch_size`.
    /// Returns observations in the same layout as `observe`.
    fn step(&mut self, actions: &[Action], out: &mut [f32]);

    /// Reset every lane (or just the ones whose episode ended —
    /// per-lane auto-reset is implementation-defined).
    fn reset_all(&mut self);

    /// Per-lane homeostatic variables, flattened row-major.
    fn homeostatic_flat(&self, out: &mut [HomeostaticVariable]);
}
```

Each built-in env in `kindle-gym` grows a `VectorizedXxx` sibling with
SoA state (`positions: Vec<(f32, f32)>` instead of `Vec<Position>`). The
step function becomes a tight loop over contiguous arrays — SIMD-
vectorizable by LLVM, cache-friendly, and free of the per-lane box/heap
traffic that `Vec<Box<dyn Environment>>` would force.

For an env where the natural implementation is already scalar (Taxi's
enum-heavy state, Acrobot's RK4 integrator), the vectorized variant is
just a loop — still a win because it amortizes the per-call overhead and
lets the compiler see the loop shape.

### 2. Vectorized adapter path

Today's `EnvAdapter::obs_to_token` writes one row at a time. Add:

```rust
pub trait EnvAdapter: Send {
    fn obs_to_token_batch(&self, obs_batch: &[f32], obs_dim: usize, out: &mut [f32]) {
        // Default = loop over rows. Adapters can specialize for SIMD.
        // ...
    }
    fn action_to_token_batch(&self, actions: &[Action], out: &mut [f32]);
    fn sample_action_batch(&self, head: &[f32], rng: &mut dyn RngCore, out: &mut [Action]);
}
```

With a single `VectorizedEnv` feeding a single adapter, `obs_to_token_batch`
can operate on one contiguous `[N, obs_dim]` buffer and emit one contiguous
`[N, OBS_TOKEN_DIM]` buffer. No per-lane function-pointer dispatch.

### 3. Vectorized reward primitives

These run on CPU today once per lane. At large N they matter:

- **Novelty.** Visit-count hash lookup per lane → batch version that
  hashes all N latents in one pass; the hash map is still shared.
- **Order.** The digest φ·obs can run as a matmul over `[N, OBS_TOKEN_DIM]
  × [OBS_TOKEN_DIM, DIGEST_DIM]` — a tight dgemm-shaped loop instead of
  N independent small vector multiplies. The per-lane recent/reference
  windows stay per-lane (they have to, for signal grounding), but the
  digest is the hot part.
- **Homeostatic.** Linear in lane count; already trivial, but batching
  removes the pointer-chasing through per-lane `HomeostaticVariable`
  Vecs.

## API shape

Agent stays the single-lane-minded object it is today plus the Phase E
multi-lane upgrade; the new pieces slot in at the edges.

```rust
impl Agent {
    /// Step once against a vectorized env. Replaces the Phase E
    /// `observe(&[Observation], &[Action], &[&dyn Environment], ...)`
    /// when the caller has a vectorized env available; the per-lane
    /// entry point is still there for heterogeneous setups.
    pub fn step_vector<R: Rng>(
        &mut self,
        env: &mut dyn VectorizedEnv,
        rng: &mut R,
    );
}
```

For homogeneous training, `step_vector` is the hot path. For
heterogeneous lanes (mixed env kinds — the `multi_env` example), the
Phase E per-lane API keeps working without modification.

## What gets us the SIMD payoff

- **GPU forward + backward.** Already batched in Phase E; Phase F just
  pushes a larger `batch_size`. Rebuilding the session with a larger N
  is a one-time cost. Backward is batched natively because meganeura
  computes grads from the mean loss — a bigger batch is literally more
  samples contributing to one gradient, so the SNR improves roughly as
  √N (standard minibatch result).
- **Vectorized env step.** N env states in one contiguous pass, SIMD-
  friendly layout. For the physics envs (CartPole, Pendulum, Acrobot,
  MountainCar) this is the part LLVM can actually auto-vectorize; for
  the table-driven envs (Taxi, GridWorld) it's mostly cache locality
  and removing per-call overhead.
- **Vectorized adapter + reward.** Bulk matmul (digest) instead of
  N small matvecs. Shared hashmap probes for novelty instead of N
  separate lookups.

## What doesn't change

- **The compiled graphs.** Same ops, just bigger batch dimension. No
  shape surgery, no new sessions, no parameter sharing games.
- **Frozen reward circuit semantics.** Order, novelty, homeostatic,
  surprise all produce the same per-lane values as Phase E; we just
  compute them with tighter loops.
- **Per-lane state.** Order windows, visit counts, pending-boundary
  flags remain per-lane — they have to, for correctness.
- **Heterogeneous-env support.** Phase E's per-lane adapter path stays
  for cases where one vectorized env isn't the right fit (different
  action kinds per lane, etc.).

## Credit assigner

Credit stays scalar-per-lane, run N times per step, exactly as in Phase
E. It's cheap and the history graph is a totally different shape from
the WM / policy graphs; retargeting it to `[N, history_len, …]` is a
disproportionate complexity hit. We revisit this only if profiling shows
N × credit dominates the step at our target batch sizes.

## Expected crossovers (hypotheses to measure)

- **Step time vs N** on a fast env (CartPole): expect roughly flat in
  the N ∈ [32, 256] range once Phase F lands, versus linear growth
  today.
- **GPU utilization vs N**: target >70% at N = 128 on lavapipe,
  indicating we've crossed out of the launch-overhead regime.
- **Steps-to-converge vs N** on CartPole: should improve roughly as √N
  in the low-to-mid range, flatten past the critical-batch-size where
  the reward signal's own noise ceiling kicks in.

## Testing

- **Parity: N=1 vectorized vs N=1 per-lane.** Identical diagnostics
  across a 2k-step deterministic run.
- **N=128 smoke on CartPole.** No NaN, wm_loss trending down, buffer
  fills correctly per lane.
- **Adapter batch correctness.** Golden-file check: scalar
  `obs_to_token` loop over N rows must equal `obs_to_token_batch` on
  the same input, byte-exact.
- **Order digest batch correctness.** Same: scalar per-lane digest must
  equal the batched matmul output, byte-exact.
- **Vectorized env correctness.** Per-env: running N scalar copies in
  parallel must match `VectorizedEnv::step` row-by-row for the same
  actions + seeds.

## Out of scope

- **Multi-GPU / multi-process / parameter averaging.** Those are a
  different question (fault isolation, independent exploration), and
  they belong in a later phase if and when we have a concrete driver
  for them. Phase F is about SIMD saturation on the GPU we already have.
- **Heterogeneous vectorized envs.** One `VectorizedEnv` implies one
  env kind; mixing kinds in a single batched step would force a union
  action type and is not worth the complexity. Heterogeneous setups
  keep using Phase E's per-lane API.
- **Async env stepping.** Lockstep is fine — with envs vectorized, the
  CPU block shrinks enough that the pipeline argument (Phase F as I
  previously drafted) stops applying.

## What to build first

1. `VectorizedEnv` trait + `VectorizedGridWorld` as the first impl. SoA
   state, contiguous buffers. Exercisable as a unit test without any
   Agent involvement.
2. Vectorized adapter path: default impls loop over rows; specialized
   impl for `GenericAdapter` does bulk copy / fills.
3. `Agent::step_vector` wiring — builds the stacked task / obs / action
   inputs from the vectorized adapter output directly. No per-lane
   buffering on the way in.
4. Batched reward digest (`RewardCircuit::observe_order_batch`) plus
   batched novelty lookup. Preserves the per-lane output shape.
5. `VectorizedCartPole`, `VectorizedPendulum`, the rest. Benchmarks
   against Phase E at matched N.
