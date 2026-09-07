# Learner readback timing candidate

Staged separately on `exp/learner-readback-profile`, based on `73273df`.
The active LeVJEPA three-seed queue retains its original binary, runners,
auditors and recipe. This candidate has not run on the GPU or been adopted.

## Measurement boundary

`LearnTiming.posterior_readback` and `imagination_readback` report the actual
number of nonempty readback submissions, requested output bytes, and host wall
time spent preparing the transfer, submitting it, waiting for completion, and
copying completed CPU-visible outputs. These are subsets of the existing
posterior/imagination totals; do not add them to those totals.

Preparation includes validation, buffer growth and command recording. Wait
time includes unfinished producer computation and the GPU transfer, not just
idle bubbles. Requested bytes are payload, not measured PCIe bus traffic.
The `Readback::read` wrapper allocation, input packing/writes, producer dispatch,
CPU sampling/decoding, training graphs and parameter synchronization are not
separately timed by this patch. GPU kernel time still needs hardware timestamps.

The patch adds clocks and counters around existing operations, without adding
a wait, changing submission order, moving arithmetic or consuming RNG draws.
Live/diagnostic reads are discarded before the posterior stage so they cannot
inflate its counts. Each learner stage drains its own counters.

## Validation before measurement

CPU checks pass: 76 Kindle tests and five gym tests, workspace/Python-library
Clippy, and Rust formatting. Seventeen GPU tests are explicitly skipped by the
ordinary test command. New hardware assertions check batched/prefix/empty reads,
counter reset, exact per-stage calls/bytes, and phase totals bounded by the
parent wall time during both fresh and restored learning.

After the active queue exits, run each relevant GPU test in a fresh process on
adapter `0x2c02`: the two `dreamer::readback::tests` hardware tests and
`dreamer::agent::tests::tiny_agent_completes_an_act_and_learn_cycle`. Do not
replace the running Python extension to test this branch.

Then compare the parent and candidate with the existing `dreamer_canary`
12M/B16/T64/row16 prediction-only workload, eight updates and final tensors.
All non-timing reports and model/optimizer tensors must remain identical.
At this shape, require 64 posterior reads / 10 MiB and 31 imagination reads /
199 MiB per update; the hardware counters must confirm the source inventory.
Run repeated alternating parent/candidate timings after warmup to quantify
instrumentation overhead before interpreting the profile. The synthetic
canary excludes the video encoder and is not a gameplay-quality comparison.

Only then capture the same counters in a short, separately declared N=8 pixel
run and compare with normal session GPU timestamps. Inspect world-gradient and
perception cost too: removing the entire posterior and imagination stages at
zero cost still cannot reach aggregate real time in the current Pong recipe.
No speedup, numerical parity or hardware result is claimed for this candidate yet.
