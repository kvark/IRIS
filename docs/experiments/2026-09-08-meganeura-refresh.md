# Meganeura refresh, preserving the LeVJEPA control

Completed and adopted 2026-09-08 in `898e968` and `6019f93`, after hardware,
synthetic and paired pixel validation. This is a backend refresh, not a new
objective, precision mode, frontend or training-ratio experiment.

## Source and compatibility

The latest upstream main observed at declaration was
[`df11bb0c5ff074c643d593e4acf41dc7d58bbd28`](https://github.com/kvark/meganeura/commit/df11bb0c5ff074c643d593e4acf41dc7d58bbd28).
Kindle's previous `35a410ce…` pin was not an ancestor of main: it carried two
LeVJEPA patches on `levjepa-block-cache`. Replacing it with main alone loses
cached query-block support and cache aliases through views.

The isolated refresh carries only `0ff33a4` and `35a410c` onto main, and registers
their cached-block test in main's newly consolidated regression target. The
result is
[`a7e2efd9d2d1c14e654658fa1b736582d96e16cd`](https://github.com/kvark/meganeura/commit/a7e2efd9d2d1c14e654658fa1b736582d96e16cd),
pushed as `kindle-refresh-20260908`. No PR was merged and the user's local
Meganeura worktree was not changed.

The refresh includes upstream's batched partial cooperative-convolution fix,
Winograd cache sharing, logical checkpoint validation and training-state
hardening. Kindle now resolves the same published Blade 0.9.0 package as
Meganeura, with Naga 30.0.1 in both lockfiles. Its Rust minimum rises to 1.92.
Blade provenance records the registry version and checksum; a CPU test requires
one matching graphics-crate source in each Cargo root, avoiding incompatible
GPU-context types from simultaneous registry/git copies.

New Meganeura checkpoints store logical shapes and omit derived Winograd caches.
Kindle's restore guard previously required these execution caches as if they
were saved learned parameters. It now identifies only Winograd-derived buffers
from the compiled plan, still requiring every real weight and both optimizer
moments. A genuine parameter named `kernel:winograd` remains required. A new
hardware test saves tied convolution weights without their shared cache,
restores them, regenerates the cache and checks exactly equal outputs.

Meganeura's embedded checkpoint format 3 is distinct from Kindle's outer
format-3 model metadata. Backend identity remains strict: old final Pong models
require their original executable. Nothing rewrites their metadata, resaves
them under a new identity or weakens the torn-save checks.

## Validation and immutable artifacts

The pre-hardware declaration is
`runs/meganeura-refresh-20260908.ltOGRe/declaration.md`. The tested Rust candidate
is `exp/meganeura-refresh` at `26ad84ac8c40726ed113369879f26fc4c4cabf64`,
isolated in `/x/Code/.kindle-meganeura-refresh`. It builds from the adopted
device-resident-imagination baseline, not the older editable Python extension.

- Workspace fmt, all-target Clippy/check and 80 library CPU tests pass; 19 GPU tests
  remain ignored by the ordinary suite. Python Clippy and 229 Python CPU tests
  pass against the actual isolated refresh package. The separate world-probe
  changes raise the main Python suite to 253 passing tests. Final all-target
  Rust validation also runs 12 example tests, for 92 passing CPU tests.
- Eight serialized hardware tests pass in
  `runs/meganeura-hardware-20260908.Ni24H4/`: device-copy packing, logical cache
  restore, act/learn/restore, vector-one equivalence, full scalar continuation
  reduction, production B16/T64 loss/all-gradient parity, LeVJEPA reference and
  chunk/reset parity, and batched/serial asymmetric stream reset/gap parity.
  Worst per-parameter relative gradient L2 in the production temporal check is
  0.000746, within its unchanged tolerance. This is numerical correctness,
  not a throughput benchmark.
- Two fresh AB/BA eight-update 12M/B16/T64/R256 prediction-only pairs pass in
  `runs/meganeura-canary-20260908.u7tbei/`. Every non-timing report and every
  named tensor payload agrees exactly: 164 world, 66 behavior and 11 slow-value
  tensors. All states are finite, optimizer counters match and second moments
  are nonnegative. Only the two declared backend identities and native
  flattened-to-logical checkpoint representation differ.

The backend-specific read-only comparator validates the complete logical layout
and all names, fingerprints, values and moments. It does not drop tensors,
relax numerical tolerance, alter the previous pinned comparator or rewrite
checkpoint files. Every original control pin remains unchanged.

The actual candidate Python extension is
`runs/meganeura-refresh-20260908.ltOGRe/package-v2/kindle/_native.cpython-314-x86_64-linux-gnu.so`,
SHA-256 `0d1c3fd6f33757fe9e51cb0ebf56ae1859bbf8f8aff7a009b4e3d4dbcbea0ea8`.
The comparison package is
`runs/device-imagination-python-20260908.UvQQjL/package`, native SHA-256
`2b5bc9d1c26630896de8ecde5ed70cb8fd87fb024f59e46e415fbead6691d08a`.
The earlier `package/` in the refresh root predates the logical-restore fix;
do not use it as the validated candidate.

## Completed fixed-ratio pixel gate

`runs/meganeura-pixel-20260908.RK3iqu/` declares fresh AB then BA runs, each
3,072 actual actions, N8/B16/T64/R256 and 387 updates with zero debt. The final
1,024 actions/256 updates form the warmed window. Keep the 2 GiB GPU reserve,
actual emulator clocks, action/reward/reset equality, complete checkpoints and
exact non-timing reports. No tracing or competing compute overlaps the timing
jobs. Both pairs pass with all 241 named tensor payloads, all 387 non-timing
reports and every action/reward/reset exactly equal. The refresh is adopted.

The first control completed normally, but the new CPU auditor incorrectly read
`perception` at the top level of the historical raw header instead of inside
`model_provenance`. It stopped the gate before launching the candidate. Preserve
that `KeyError`, original auditor and complete control in the original root.
The corrected auditor in `runs/meganeura-pixel-resume-20260908.YEAjKb/` revalidates
and pins the retained control, then continues AB/BA without rerunning or selecting
another control. Its changes affect bookkeeping only; all headers, action logs,
checkpoints and raw timing samples remain unchanged. The longer intervening
idle period is visible in the two launch records; both use the same declared
warmed window and the required reverse-order pair also passes.

The continuation and its `summary.json` retain both complete pairs:

| Pair / order | Actions/s, old → refreshed | Aggregate real time, old → refreshed | Full learner ms, old → refreshed | GPU activity %, old → refreshed |
| --- | ---: | ---: | ---: | ---: |
| 0 / AB | 8.588 → 8.644 | 0.5725× → 0.5763× | 347.6 → 344.7 | 67.4 → 65.9 |
| 1 / BA | 8.598 → 8.704 | 0.5732× → 0.5803× | 347.2 → 343.2 | 66.7 → 70.1 |

Peak GPU memory stays 14,212 MiB in every run. The former 2,091 MiB headroom
calculation subtracted usage from total memory but omitted driver reservations;
it does not prove the declared 2 GiB free-memory gate. The later
[memory audit](2026-09-08-atari-five.md#memory-accounting-correction) measures only
1,631 MiB free in the live pilot and withdraws that reserve-pass claim, without
changing these raw timing or exact-parity results.
Refreshed per-stream real time is 0.0720–0.0725×. Observed
throughput rises 0.65% and 1.24%; two short pairs do not establish a precise
population speedup, and this is not a substantial acceleration. Coarse activity
fluctuates in both directions and does not identify idle gaps or occupancy.
World training still costs 161–162 ms/update, posterior inference 59 ms,
imagination 84 ms, behavior training 19 ms and parameter synchronization 19 ms.
The larger throughput work remains necessary.

## Use the adopted package

`runs/meganeura-package-20260908.Ec48N4/package` combines the current Python
world-probe helpers with the **identical** native bytes tested above. It is
repacked from `7391563` on the isolated branch; the main Rust, lockfiles and
Python implementation match that source. Its manifest and final validation logs
record the package, wheel and original-control hashes. All 253 Python tests and
92 Rust CPU tests pass. The retained DINO reference/projection/pooling hardware
test also passes, bringing the focused backend hardware checks to nine. This
does not switch the active frontend away from LeVJEPA.

The subsequent [zero-update save repair](2026-09-08-atari-five.md#preserved-zero-update-restore-failure)
keeps this backend pin and strict restore checks, but ensures fresh checkpoints
contain explicit zero optimizer moments. Its isolated package is
`runs/zero-update-checkpoint-20260908.yxaalA/package`, native SHA-256
`9cd176c1e293d23ca553285008ce19b3507c0b16e848fbbd26bd72d9b1831bb9`.
It passes the CPU/GPU and real 12M checkpoint checks linked above. Preserve the
original refresh package as an artifact; use the repaired package for fresh work.

Keep the older editable extension for historical checkpoints; do not replace it
in place or change an active experiment's pinned package. For a new experiment:

```bash
PYTHONPATH=/x/Code/kindle/runs/zero-update-checkpoint-20260908.yxaalA/package \
  python/.venv/bin/python python/examples/atari_vector.py \
  /models/levjepa/model.safetensors --num-envs 8 --steps 3072 \
  --output runs/new-backend-check.jsonl
```

Paths and outputs must be chosen for the new experiment. This example is a
short integration run, not another declared learning-quality campaign. All
packages and experiment artifacts are local and git-ignored; other checkouts
can build the pinned source normally with Rust 1.92 or newer.

The [separate world-model diagnostic](2026-09-08-world-evaluation.md) is already
complete on the original executable. Its seed findings are not attributed to
this refresh, and backend parity does not repair the failed three-seed mastery
result or establish faster-than-real-time learning.
