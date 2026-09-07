# Batched DINO control candidate

Status: isolated implementation, CPU-validated, **no GPU execution or adoption**.
Branch `exp/batched-dino-control`, based on `7cd60aa`. The active three-seed
LeVJEPA queue and all its pinned code/binaries remain untouched.

## Purpose and scope

The revised plan requires a DINO/video comparison with the same executable,
world objective, actor, collection and learner schedule. The active vector API
only implements LeVJEPA; historical serial DINO runs cannot supply that control.
This candidate closes the implementation gap without changing the primary
frontend or introducing another learner.

- `VectorDreamerAgent::new` and Python `VectorAgent` still default to LeVJEPA.
  Rust `with_perception` and Python's keyword-only `encoder="dinov3"` make the
  control explicit. Both Atari runner and throughput matrix expose `--encoder`.
- Restore uses the checkpoint's validated frontend identity and actual encoder
  SHA-256. The runner rejects frontend overrides on restore. No new checkpoint
  format, encoding revision, objective, optimizer, sampling or replay semantics.
- DINO ViT-S/16 retains its F32 weights, 12 layers, letterbox224, fixed JL64
  projection and 2x2 pooling. Its input is `[N * 196, 768]`; dense transformer
  layers process `[N * 201, 384]`. The five learned prefix tokens are repeated
  per image, and full attention sees exactly one image's 201 tokens.
- One weight set serves all streams. The existing stream slicing/stacking
  helpers are shared with LeVJEPA, whose graph math is unchanged by inspection.
  There is no cross-image attention mask approximation or serial encoder pool.
- Sparse arrivals return in caller order. DINO has no visual history, so its
  reset flags do not affect features; Dreamer's episode belief resets still do.
  Omitted streams do not enter replay or advance live RNG/belief state.
- Projected-token readback, observation pooling and synchronous scheduling stay
  unchanged in purpose. This is a control-enabling change, not a demonstrated
  optimization. Separate per-image attention still means multiple dispatches.

## CPU validation

On 2026-09-07, using the pinned Cargo dependencies:

- `cargo test --workspace --locked -j 2`: 78 Kindle and 5 gym tests pass;
  19 GPU tests are ignored, including the two new hardware gates below.
- Workspace and Python `cargo clippy --all-targets --locked ... -- -D warnings`
  and both Rust formatting checks pass.
- All 248 Python tests pass using a separately built **debug** extension from
  this worktree. No installation or replacement in the active Python checkout.
- Graph tests cover N=1/2/3/8, identical parameter names/shapes/dtypes, batched
  input/output dimensions, one-image attention dimensions and repeated RoPE.
  Invalid zero batches, configuration and encoder names are rejected before GPU
  setup. CPU restore tests reject incorrect weights for both frontends.
- Mocked runner tests verify default/explicit selection, checkpoint-selected
  restore and cleanup without claiming a started run on construction failure.
  The throughput launcher preserves the selected frontend and model provenance.
  The unchanged Pong mastery auditor still rejects DINO-labelled input.

The debug extension SHA-256 is
`1da97a3731940104786f5bab3a34ccec9f25f0c1edf38f4aeee728fc6170d642`.
It was used only for CPU API checks, not gameplay, GPU parity or timing.
Builds/checks overlapped seed 1 approximately 13:10–13:27 UTC; affected live
throughput windows are not quiet-system performance comparisons.

The short-run profiler now requires a complete interaction/replay/update ledger,
consistent window clocks, monotone finite GPU samples with approximately 1 Hz
coverage, and nonnegative stage deltas. It reports actual emulator frames and
both aggregate/per-stream game-clock speed, retains frontend provenance and input
hashes, and distinguishes the measurement-window memory peak from the peak over
the entire per-job monitor trace, including construction. Both are selected-device
memory, not exclusive per-process allocation. A zero process exit with an invalid
ledger or trace is retained as a failed measurement, never a completed benchmark.

Nine added CPU tests cover these reports and rejection paths. A CPU-only reread
of the preserved `runs/vector-20260906-temporal/` N=2/4/8 logs and GPU traces passes
the stronger checks and reproduces all compared historical metrics exactly.
The N=8 window contains 1,024 actions, 256 updates and 4,096 actual frames:
0.491085 aggregate game-clock speed and 0.061386 per stream. This validates the
reporter against old LeVJEPA evidence, not the candidate's GPU implementation.

## Hardware gates after the pinned queue

Run every GPU test in its own process, on adapter `0x2c02`, only after the
current launcher/learner have exited. Do not use `cargo test -- --ignored`
without an exact test filter: the shared device must remain serialized.

1. Re-run `vision::tests::vits16_checkpoint_matches_hugging_face` with the pinned
   DINO checkpoint: SHA-256
   `4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d`.
   This anchors the serial frontend to its existing upstream golden values.
2. Run `vision::tests::dino_batch_matches_serial_with_sparse_and_reordered_arrivals`.
   Six deterministic images include square and non-square frames. N=3 projected
   and pooled outputs must match serial: relative L2 < 1e-4, max absolute error
   < 0.005. Empty, sparse and reordered arrivals are covered. Perturbing only
   stream 1 must change that image while streams 0 and 2 remain bit-identical,
   including when an unchanged image's reset flag changes.
3. Run `dreamer::agent::vector::tests::dino_pixel_vector_matches_serial_and_restores_recorded_frontend`.
   Tiny shared-model N=2 pixel agents must match independent serial live beliefs
   within 1e-4 and sampled actions exactly, with unequal deaths/timeouts and
   reordered arrivals. Restore must select DINO and retain counters but start
   with empty replay and fresh histories. This test deliberately performs zero
   learning updates; it does not establish DINO learning quality.
4. Re-run existing LeVJEPA batched/serial reset-gap parity and the vector-core
   state and learning/checkpoint tests. Compare a fresh, fixed-seed N=8 LeVJEPA
   3,072-action pixel canary against the preserved temporal-batching control,
   requiring identical non-timing transition/update ledgers. A helper move or
   frontend dispatch must not silently change the primary experiment.
5. Build a release extension in isolation, record source/binary hashes, then run
   short DINO N=1/2/4/8 pixel/learning canaries with complete accounting audits.
   Keep B16/T64/full BPTT/row16, R256, 18 actions and the published Atari wrapper;
   report useful aggregate/per-stream speed, memory and sustained update credit.
   Retain the 10% VRAM reserve. No timing claim from the debug API build.

These gates precede a new predeclared matched learning comparison. Compare fresh
models using the same released executable and same aggregate/per-stream budget;
do not restore DINO weights into LeVJEPA state or contrast new DINO curves with
old LeVJEPA curves as though the runtime were matched. Predeclare final frozen
evaluations, seeds and stopping rules. Keep this separate from the active Pong
mastery protocol and from the staged generic Atari v2 accounting change.

LeVJEPA remains the primary target. Better DINO throughput alone would not
establish better control, just as a lower video prediction loss does not establish
stronger gameplay. Complete the trained-belief probe and measured learner profiling
before expanding the architecture or scheduling another multi-day comparison.
