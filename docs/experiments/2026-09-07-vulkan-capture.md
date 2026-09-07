# External Vulkan capture: tooling preflight

Prepared while the pinned LeVJEPA Pong queue is running. No live process was
attached to or restarted, no native agent was constructed, and no gameplay or
GPU-workload capture has been validated here. Hardware checks remain serialized
after the queue. This complements the isolated host/readback timer candidate;
it does not replace its numerical or overhead gates.

## Local tool and permissions

The installed Nsight Systems CLI and QdstrmImporter both report
`2023.4.4.54-234433681190v0`. The CLI resolves to
`/usr/lib/x86_64-linux-gnu/nsight-systems/target-linux-x64/nsys`; the matching
importer is `/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter`.
Hardware compatibility with this RTX 5080 / 595.71.05 driver is still untested.

`nsys status --environment` finds a usable timestamp counter, but CPU sampling
fails: `perf_event_paranoid=4` and `perf_event_open` is unavailable. NVIDIA's
`RmProfilingAdminOnly` is also 1. Leave both restrictions unchanged. The proposed
capture disables CPU sampling/context-switch collection, GPU metrics/context
switches and event sampling. A failed CPU sampling check does not by itself
exclude API tracing ([NVIDIA installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#optional-setting-up-the-cli)).

Use Vulkan API tracing with `--vulkan-gpu-workload=batch`, which records
submission-batch GPU ranges rather than individual workloads. These options
were verified against the installed CLI help. Batch capture is a first
diagnostic choice, not a demonstrated low-overhead mode on Kindle.

## CPU-only result

At 12:14 UTC on September 7, a fresh capture targeted **`/usr/bin/true`**:

```sh
nsys profile --trace=vulkan --vulkan-gpu-workload=batch \
  --sample=none --cpuctxsw=none --gpu-metrics-device=none \
  --gpuctxsw=false --event-sample=none --force-overwrite=false \
  --kill=none --wait=primary --stop-on-exit=true \
  --output=/x/Code/kindle/runs/nsight-cli-20260907.c1rW83/noop /usr/bin/true
```

The CLI exits zero but automatic import fails: it cannot find the importer and
its dependencies. It produces only `noop.qdstrm`, not a usable report. The
installed importer's dependencies resolve and direct invocation succeeds:

```sh
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
  --input-file /x/Code/kindle/runs/nsight-cli-20260907.c1rW83/noop.qdstrm \
  --output-file /x/Code/kindle/runs/nsight-cli-20260907.c1rW83/noop.nsys-rep
nsys export --type=sqlite --force-overwrite=false --quiet=true \
  --output=/x/Code/kindle/runs/nsight-cli-20260907.c1rW83/noop.sqlite \
  /x/Code/kindle/runs/nsight-cli-20260907.c1rW83/noop.nsys-rep
```

These paths now exist; use fresh paths for any repeat. No system installation,
library search path, driver, kernel or security setting was changed. The report
imports and exports successfully; read-only SQLite `quick_check` returns `ok`.
There are no Vulkan/GPU workload tables, as expected for this target. That is
**not** GPU trace validation. `nsys sessions list` is empty after completion.

`runs/nsight-cli-20260907.c1rW83/preflight.json` retains the exact executed
commands, failure output, tool/artifact hashes, schema inventory and explicit
unmeasured gates. Raw Nsight artifacts include host/environment metadata; keep
them in ignored run storage and review before any external sharing. This small
CPU diagnostic overlaps seed-1 training; it is not a quiet-system benchmark.

## Post-queue hardware gate

1. Confirm the pinned launcher and learner have exited. Do not attach to their
   processes or replace their extension. Verify the prebuilt parent/candidate
   hashes in `runs/readback-profile-20260907/builds.json` before execution.
2. Complete the host/readback candidate's hardware tests and eight-update
   parent/candidate parity check. Preserve the existing
   [validation plan](https://github.com/kvark/kindle/blob/ec074a5/docs/experiments/2026-09-07-readback-profile.md).
3. Capture a separate eight-update **parent** `dreamer_canary` with the flags
   above, adapter `MEGANEURA_DEVICE_ID=0x2c02`, and arguments
   `12m 64 16 --learn --prediction-only --updates 8 --checkpoint PATH`.
   Start from fresh state and use fresh output/checkpoint paths. Keep target
   JSON stdout separate from profiler output. Do not combine this first capture
   with `MEGANEURA_GPU_TIMING=1` or `--profile-dir`; measure one instrument at
   a time. Let the finite target exit naturally, with `--kill=none`.
4. Require successful import, valid SQLite and nonempty Vulkan API **and GPU
   workload** records for the expected process/device/queue. A zero exit code,
   API-only trace or empty no-op report is insufficient. Retain failed artifacts.
   Use the explicit matching importer if automatic discovery fails again.
5. Require eight complete reports with learner steps 1 through 8. Compare every
   non-timing report and named model/optimizer tensor with the untraced parent
   control using `runs/kickoff-compare-checkpoints.py`. Equal report lengths
   alone are insufficient. Record capture wall/memory overhead and repeat
   alternating warmed controls before interpreting any speed difference.
6. Only then capture a short separately declared N=8 pixel run. The synthetic
   canary excludes LeVJEPA and real game execution, so it cannot validate
   end-to-end playing-plus-training cost or gameplay quality.

NVIDIA warns that command-buffer range ends on Compute/Transfer queues can
precede actual completion ([Vulkan trace notes](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#vulkan-gpu-trace-notes)).
Check actual queue identity and timing validity before attributing apparent gaps.
GPU queue activity is not SM occupancy or proof that no other queue is running.
Cross-check native pass durations and host stage timings, retain unclassified
time, and measure instrumentation overhead. If this tool/version cannot provide
trustworthy workload ranges, preserve that failure instead of presenting its
drawn timeline as an exact idle-time measurement.
