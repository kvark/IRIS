# Phase F — Fork-join workers

## Goal

Run **W independent agent workers**, each Phase-E batched (N lanes per
worker), and periodically average their parameters so they converge to a
shared policy + world model while still exploring independently. This is
IMPALA / A3C-style decentralized training; the payoff is multi-GPU / multi-
host scaling and fault isolation per worker. For `W = 1` the runtime is
Phase E exactly — Phase F is a strict superset.

## When to reach for Phase F (not before)

Phase E (batched lanes in one session) already gives near-linear speedup
inside a single GPU context. Phase F is only worth its complexity when at
least one of these is true:

- **Multiple GPUs / hosts.** One session pins one GPU; to light up a
  second device you need a second session, i.e. a second worker.
- **Blocking envs.** A real-robot env, a human-in-the-loop env, or a
  remote simulator where one env stall would starve the whole batch.
  Phase E lanes step in lockstep; Phase F lets each worker stall on its
  own without blocking the others.
- **Fault isolation.** A single worker crashing (OOM, env panic, driver
  reset) takes out its lanes only; the rest of the swarm keeps training.
- **Exploration diversity.** Per-worker RNG + adapter mix yields a wider
  state distribution than any single batched agent can cover, and
  parameter averaging distills the diversity into a shared model.

If none of these apply, stay on Phase E.

## Core change

Introduce `AgentSwarm`, a coordinator that owns W workers and a canonical
CPU-side parameter store. Each worker is a Phase-E `Agent` running in its
own thread (or process — see "Isolation" below) and stepping at its own
rate. Sync happens every `sync_interval` steps.

```text
swarm:
  workers:         Vec<Worker>         // W workers
  canonical:       ParamStore          // HashMap<String, Vec<f32>>
  sync_interval:   usize               // default 500
  sync_mode:       SyncMode             // Average | DeltaAverage

worker:
  agent:           Agent (N lanes)
  step_counter:    usize
  last_sync_snapshot: ParamStore       // params at the last sync
  handle:          JoinHandle<...>     // thread or process handle
  tx:              Sender<Command>     // swarm → worker
  rx:              Receiver<Report>    // worker → swarm
```

### Sync protocol (DeltaAverage, default)

Every `sync_interval` worker-steps, each worker:

1. Reads its current parameters from the session: `theta_w`.
2. Computes `delta_w = theta_w - snapshot_w`, where `snapshot_w` is the
   per-worker copy of the canonical params at the last sync.
3. Sends `delta_w` + `steps_since_last_sync` to the swarm.

The swarm:

4. Weighted-averages the incoming deltas: `Δ = Σ (s_w · delta_w) / Σ s_w`,
   where `s_w` is the worker's step count since last sync.
5. Updates canonical: `canonical += Δ`.
6. Broadcasts `canonical` to all workers.

Each worker:

7. Writes canonical into its session via `set_parameter`.
8. Stores `snapshot_w = canonical`.

Weighting by per-worker step count is what A3C calls "importance-weighted
averaging". Workers that did more work since last sync carry more voice.

### Alternative: Average

Simpler: just average the absolute parameters (no deltas). Use this if the
delta computation turns out to be noisy at small `sync_interval`. The
tradeoff: delta-averaging converges to the per-worker mean update; plain
averaging converges to the centroid of the per-worker states. Empirically
delta tends to be more stable when workers see different envs.

## What is synced vs per-worker

### Synced (canonical)

- **All GPU parameters** across the three sessions (encoder, world model,
  credit assigner, policy, value head). These are the "model". Read via
  `session.read_param`, written via `session.set_parameter`.

### Not synced (per-worker)

- **Per-lane buffers.** Each worker's experience buffer is local. Cross-
  worker replay is possible but wasteful — the parameter sync already
  carries the information across.
- **Visit counts (novelty).** Per-worker. Novelty reward is intentionally
  local: different workers can explore different regions without
  stepping on each other's counts.
- **Order window state.** Per-worker per-lane, same as Phase E.
- **Task embeddings.** Deterministic from `env_id`, so each worker
  independently regenerates the same vector for the same env. Shared by
  construction, no sync needed.
- **RNGs.** Per-worker; the swarm assigns distinct seeds at spawn.
- **Optimizer state** (if meganeura ever adds Adam): **should be
  synced** with the parameters. Until then, SGD has no state to sync.
- **Diagnostics.** Each worker reports its own; the swarm aggregates.

## API

```rust
pub struct SwarmConfig {
    pub num_workers: usize,       // W
    pub sync_interval: usize,     // steps between syncs (default 500)
    pub sync_mode: SyncMode,      // Average | DeltaAverage (default)
    pub agent_config: AgentConfig, // shared; each worker gets N lanes per batch_size
}

impl AgentSwarm {
    /// `adapter_factory(worker_idx, lane_idx) -> adapter` builds the
    /// W × N adapters at construction. Typical pattern: deterministic
    /// env rotation, or per-worker env specialization.
    pub fn new<F>(config: SwarmConfig, adapter_factory: F) -> Self
    where
        F: Fn(usize, usize) -> Box<dyn EnvAdapter>;

    /// Step every worker once against a freshly queried obs + action per
    /// lane per worker. Envs are owned per-worker and never cross thread
    /// boundaries. Returns aggregated diagnostics across the swarm.
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> SwarmDiagnostics;

    /// Force a sync point now, regardless of `sync_interval`.
    pub fn sync(&mut self);

    /// Per-worker diagnostics; length W.
    pub fn diagnostics(&self) -> Vec<Vec<Diagnostics>>;
}
```

For the minimal in-process use case, `step()` drives the workers by
channel messages and blocks until every worker reports back; envs live
inside the worker thread and `step()` doesn't see them directly. A
separate `step_async()` variant returns a future-per-worker for
throughput-optimized callers.

## Isolation: threads vs processes

- **Thread-per-worker (recommended default).** Each worker is a
  `std::thread` with its own meganeura `Session`. GPU contexts can
  coexist in one process (blade-graphics supports this; each session owns
  its own device). Sync is a channel-based `Vec<f32>` transfer — fast.
  Good for multi-GPU-on-one-host.
- **Process-per-worker.** Each worker is a separate OS process; sync
  over Unix-domain sockets / shared memory. Needed for true fault
  isolation and for cross-host training. Deferred until we have a
  concrete use case; thread-per-worker covers most of Phase F's value.

## Per-step flow (thread mode, default)

```text
Swarm main loop (one macro-step):
  for each worker w in parallel:
      worker.step_batch(1)          // runs N-lane Agent::{act, observe}
      if worker.step_count % sync_interval == 0:
          worker.tx.send(Report::ReadyToSync { deltas })
  if any worker signaled sync:
      // barrier: wait for all workers to report
      wait_for_all_reports()
      aggregate → canonical
      for each worker: worker.tx.send(Command::Apply(canonical))
```

### Stragglers

If worker W−1 takes 2× as long as worker 0 (slow env), the barrier stalls
the swarm. Two mitigations:

1. **Lag-tolerant sync.** Workers sync with whichever canonical version
   is current; they don't wait for a global barrier. Their delta is
   applied late but still with step-count weighting. Closer to ASGD than
   A3C; tolerates stragglers but loses the reproducibility of sync'd
   training.
2. **Budget-based step sizes.** Each worker runs until it has completed
   `sync_interval` steps *or* hit a wall-clock budget. Slow workers
   contribute fewer steps; their vote is proportionally smaller (that's
   what importance-weighting already does).

Recommendation: barrier by default, expose `lag_tolerant = true` as an
opt-in for production workloads with heterogeneous envs.

## Determinism

With a single global RNG and deterministic env factory, the swarm is
reproducible — workers see the same sync boundaries, same adapter
assignments, same seeds. The parameter averaging step is deterministic
given fixed worker ordering and fp32 addition semantics (which is
associative enough for our purposes; if bit-exact reproducibility matters
we sort deltas by worker id before summing).

Thread scheduling can introduce nondeterminism only via race conditions,
which we avoid by construction — workers don't share mutable state
except through the sync channel.

## Failure handling

A crashing worker is a first-class concern at this scale.

- **Panic in worker thread.** The thread's JoinHandle surfaces the
  panic on next `step()`. The swarm has two options:
  - Drop the worker, continue with `W − 1` workers for the rest of the
    run. Importance weighting naturally down-weights missing workers.
  - Respawn: start a fresh worker, initialize its session from
    `canonical`, resume. This is the preferred path — we've already paid
    for canonical params CPU-side.
- **Session error (GPU hang / driver reset).** The worker surfaces it
  as a panic via the meganeura session; same handling as above.
- **Env panic.** Scoped to the worker; never kills the swarm.

## What Phase F does not attempt

- **Gradient averaging** (as opposed to parameter averaging). That's
  SGD-style synchronous data parallelism and needs meganeura to expose
  per-step gradients, which it doesn't today. Parameter averaging is
  coarser but works with the current API.
- **Mixed-precision sync.** Deltas fly over the wire as fp32. Halving to
  fp16 / bf16 is a plausible follow-up once profiling says the transfer
  is the bottleneck.
- **Learned mixing weights.** The per-worker weighting is step-count-
  proportional; no meta-learner decides how much each worker contributes.
- **Cross-worker replay buffers.** Each worker keeps its own. Sharing a
  buffer across threads adds lock contention that the parameter sync
  already obviates.

## Testing

- **W=1 parity.** Swarm with one worker, one lane: identical diagnostics
  to Phase E single-lane for N=1 and to Phase E multi-lane for N>1.
- **W=2, N=1 convergence.** Two workers, one lane each, both on
  CartPole. Assert `||theta_a − theta_b||` stays bounded across training;
  assert mean wm_loss trends down as fast as single-worker N=2.
- **W=4 heterogeneous.** Workers 0,1 run GridWorld; workers 2,3 run
  CartPole. After 50k steps the canonical policy should play both
  envs reasonably (within 2× of specialist single-worker runs).
- **Sync barrier correctness.** Inject a deterministic lag into worker
  1; assert the swarm advances in lockstep at `sync_interval`, not
  faster than the slowest worker.
- **Recovery.** Kill worker 2 mid-run, respawn from canonical, assert
  no training regression larger than `sync_interval` steps of lost
  experience.

## What to build first

1. `ParamStore` + `Agent::read_params() -> ParamStore` and
   `Agent::write_params(&ParamStore)`. Pure CPU-side plumbing; exercisable
   as a round-trip test without any swarm involved.
2. `AgentSwarm<W=1>`: a coordinator that wraps one Phase-E Agent,
   triggers `sync()` every `sync_interval` steps, verifies canonical ==
   agent params post-sync. No parallelism yet.
3. Promote to `W=2` with thread-per-worker. Barrier sync, delta
   averaging. Run the convergence test above.
4. Failure handling: catch panics, respawn from canonical.
5. Heterogeneous envs + the ARC-AGI-ish adaptation story from
   `universal-actions.md`.
