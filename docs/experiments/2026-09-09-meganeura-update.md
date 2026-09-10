# Current upstream refresh: adopted after exact qualification

The September 9 remote check resolves Meganeura main to
[`e59bd32d`](https://github.com/kvark/meganeura/commit/e59bd32d2aeb200eb00b1a15698a777d7df8db4f).
Unlike the earlier metadata-only `970da8e3` review, this includes runtime changes:
softplus negative-tail values/gradients, new multimodal/cache operations, native
capture support and pipeline-label reuse, and weighted/small-tile matmul epilogues.
These fixes do not by themselves explain the old Pong seed failures.
The September 10 00:30 UTC recheck still resolves the same upstream revision.

## Preserve the encoding contract

Upstream's new cached block attention masks queries by token position. LeVJEPA
needs every patch in the current frame to attend to the entire current frame,
plus earlier frames. Substituting the new operator would silently change the
representation. Main also lacks the carried early cache-write alias correction:
a view of the write result can otherwise point at a separate unused buffer.

The new downstream revision
[`4d45ba3a`](https://github.com/kvark/meganeura/commit/4d45ba3a1830107769ae07fabcf3b95d0762973c)
retains the validated tiled F32 cached-query generator with a distinct shader
identity, alongside upstream's token-causal operator. It resolves both ordinary
and prefix cache-write aliases before allocating views. The existing 1/3/33/196
query CPU-reference/reset hardware fixture is retained unchanged; a new small
CPU regression checks views of both write forms. No PR was merged and the
user's `megakernel-probe` worktree remains untouched.

Kindle's isolated candidate is `90b4763` on `exp/meganeura-refresh-20260909`,
starting at the completed exploration package's `9fc8108`. Its only source changes are the
dependency pin, both lockfiles and reported backend identity. Both locks retain
the same remaining dependencies, including Blade 0.9.0. The world-sync fan-out
candidate is not included. Historical binaries and restore identities remain
strict and unchanged.

## Gates and current status

Preparation and logs are in `runs/meganeura-refresh-20260909.xfF3AZ`.
Workspace/Python formatting and all-target Clippy pass. The Kindle workspace
has 95 passing CPU tests (22 hardware tests ignored); all 547 Python tests pass
with the actual isolated extension imported and rechecked after the suite.
The backend's 21 code-generation tests and small cache-alias CPU test pass.
The first locked build failed because the fresh upstream worktree has no tracked
Cargo.lock; the separate retry generated an offline lock and completed. Preserve
both logs. A separate 16-test fabricated-state suite checks the full-learning
comparator, including missing moments, incorrect identities, torn saves, shape
changes, nonfinite state and any changed non-timing report.

The isolated package is `runs/meganeura-refresh-20260909.xfF3AZ/package`, native
SHA-256 `f6a2b6ad6256fde2ad5209f90d5e94533537421a2adf603d0834d7c082fada74`.
The wheel, built library and imported bytes agree; all six Python files match
source. `cpu-evidence.json` binds 61 inputs. The historical/control extensions
remain unchanged; this is not an instruction to replace them.

All 18 declared GPU tests completed at 23:41 UTC. An independent CPU audit
rechecks all 261 pins, command exits/output hashes and complete direct-memory
coverage. The production B16/T64 loss/all-gradient test passes at its unchanged
tolerance (worst relative gradient L2 0.00074572). LeVJEPA passes its pinned
reference/reset and asymmetric-stream tests; N4/N6/N8 each have zero measured
dense-feature error against serial encoding. Ordinary and zero-update checkpoint
roundtrips, device transfer, vector learning and executed-action overrides pass.
The hardware group retains at least 6,545 MiB directly free; this is not combined
learner memory or a throughput measurement.

[`hardware-result.json`](../../runs/meganeura-refresh-20260909.xfF3AZ/hardware-result.json)
has SHA-256 `549f96ea00a70bb83130ff43553057222b627e9caabb7450b612b7ef63e92f09`.
Both fresh eight-update 12M/B16/T64 canary pairs completed at 23:47 UTC in
parent/candidate then candidate/parent order. All 241 tensors, complete optimizer
state, normalizers and eight non-timing reports match exactly in both pairs;
only the declared backend identity and report timing differ. The independent
audit rechecks all 309 pins, ten successful commands and four complete GPU
windows, with at least 9,559 MiB directly free. This synthetic run excludes the
visual frontend, so it does not establish combined learner memory or speed.
`canary-result.json` has SHA-256
`35e69e2be0690a228b296c77d3793673f96a3c99a10b6d16cb63b6e04b26ef34`.
The first preflight found a missing standalone candidate example before creating
any declaration or GPU job; the unchanged example was then built explicitly.
That preparation failure is retained, not a numerical failure or partial run.

The separately declared N6 pixel comparison completed at 00:37 UTC on September
10, with 374 input pins and 22 passing CPU comparison tests. All four fresh
Boxing trials reproduce the same complete logical state, all 610 non-timing
reports and training/frozen action/reward/reset traces exactly. Each trains
3,840 actions, then restores the final checkpoint for 768 unassisted sampled
actions. The fixed 2,304–3,840 timing window contains exactly 384 updates and
no unpaid training debt. No numerical tolerance or historical identity was
changed to pass. The candidate Freeway action-override/frozen integration also
passes, including the mixed exploration ledger and complete final state.

| Order / backend | Actions/s | Aggregate real time | GPU activity in timed window | Full learner ms/update |
| --- | ---: | ---: | ---: | ---: |
| AB / parent | 8.5643 | 0.5710× | 68.20% | 345.25 |
| AB / candidate | 8.5991 | 0.5733× | 67.42% | 344.98 |
| BA / candidate | 8.5542 | 0.5703× | 69.28% | 345.76 |
| BA / parent | 8.5691 | 0.5713× | 67.20% | 344.98 |

Candidate/parent ratios are **1.004068 and 0.998255**: the predeclared 98%
regression guard passes, but the requirement for both ratios above 1.005 does
not. This is a correctness update, **not a speedup**. Candidate per-stream real
time is 0.0950–0.0955×. Learning remains about 74% of elapsed window time and
observation about 25%; emulation is below 0.5%. World training costs roughly
160–161 ms/update, imagination 85 ms, posterior inference 59 ms and world sync
16 ms. These subtimings are contained in the outer learner total, not additional
wall time. GPU activity is neither occupancy nor calibrated idle-gap coverage.

All ten complete native phases pass direct-memory coverage, with at least
**3,302 MiB directly free** overall; each timed Boxing window retains 3,303 MiB.
Driver reservations remain 462 MiB. The independent CPU audit freshly rechecks
all 20 command exits/output hashes, ten ledgers, complete checkpoint files,
four-way parity and raw GPU samples; it constructs no agent. The completed
`pixel-result.json` SHA-256 is
`7568c3541251b76088f6de8e68267f0576c944b3180e92a05bcdc27b69ba8f40`.
The declaration, raw evidence, `pixel-independent-audit.json` and
`runtime-readout.json` are retained in the same directory. Do not restart the
completed worker. No new long training, game competence or seed reliability is
claimed by these short integration runs.

The four dependency/identity files are now applied to the main worktree and
match the isolated candidate byte-for-byte. Main-worktree integration checks
are recorded separately; historical packages and default editable extension
remain untouched. The optional exploration implementation still lives in the
isolated package; this source integration changes only the backend.

Main formatting, workspace/Python Clippy and all 92 Rust CPU tests pass. The
first Python integration attempt mixed the newer Atari package with main's
historical Pong auditor: 250 tests passed and three failed looking for the old
`natural_wins` accounting field. The same three failures reproduce with the old
a7e2efd9 package, so this is an incompatible Python pairing, not new backend
arithmetic. Preserve both negative checks and the original failed check
sequence. No auditor or test expectation was edited. A fresh main-source wheel
in `main-package` matches its own five Python files and built native library;
all 253 main Python tests pass with that exact extension imported. This wheel
is integration evidence, not a substitute for the separately pixel-qualified
Atari package. All three focused main-worktree GPU checks completed at 00:46
UTC: act/learn, serial-versus-vector live belief/policy, and vector-one scheduled
learning/checkpoint restore. The continuation preserves full direct-memory
coverage; its tiny synthetic fixtures are not another throughput measurement.
`main-continuation-result.json` binds the completed checks. The source-matched
main extension is `6c630ecb64fe8b67047518852cfa6f867ef59859052d84b1c86ccb6784183df6`;
the qualified Atari extension remains `f6a2b6ad…`. The dependency update is
adopted without changing the learning recipe, frontend, historical restore
identity or optional exploration's isolation.

This user-requested update now precedes the still-unrun world-sync comparison.
Do not rerun the completed Freeway, common-world or episode-evaluation queues.
The [world-model reports](2026-09-08-world-evaluation.md) deliberately continue
to use each historical model's original executable, not this new backend.

## New facilities are not automatic optimizations

Kindle already constructs its shared context through `GpuOptions::from_env`, so
the new `MEGANEURA_GPU_CAPTURE=1` can enable native-tool names/debug information
in a separately declared capture. It does not require enabling per-dispatch
timing or changing the production schedule. It is disabled throughout this gate.
Usable imported coverage and numerical qualification remain necessary; labels
alone do not resolve the old queue-only/idle-gap limitations.

Upstream also exposes `Session::share_parameter_from`. Do not replace world
synchronization with it blindly: equal physical byte counts are weaker than
matching logical shape/format, and sharing an allocation does not refresh a
target session's derived parameters or Winograd caches. Any such optimization
needs its own mutation/restore/cache checks and serialized learning comparison.
The current update uses neither sharing nor the staged read-once fan-out.

## Use the qualified package

For newly declared experiments, select
`runs/meganeura-refresh-20260909.xfF3AZ/package` explicitly and use its matching
runner at `/x/Code/.kindle-meganeura-20260909/python/examples/atari_vector.py`.
Keep its Python modules, runner and auditors together: the main worktree's
historical Pong accounting interface is different.
From the Kindle root, this import check constructs no GPU agent:

```sh
PYTHONPATH=runs/meganeura-refresh-20260909.xfF3AZ/package \
  python/.venv/bin/python -c 'import kindle._native as native; print(native.__file__)'
```

Do not use that package to restore old a7e2efd9 or historical f663 models:
restore deliberately retains strict backend identity. Keep their original
executables. Existing campaign declarations also remain pinned to their original
package and protocol; qualify a new declaration before long training.
