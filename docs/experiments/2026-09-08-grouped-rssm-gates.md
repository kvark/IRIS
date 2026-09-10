# Grouped RSSM gates: not adopted

Completed 2026-09-08. The candidate passed focused numerical checks but failed
the declared exact full-learning comparison. Production source is restored;
no Python package or gameplay run uses this rewrite.

## Candidate and tests

The grouped GRU already uses block-linear weights. The candidate additionally
treats `(batch, block)` as rows for reset/candidate/update elementwise gates,
replacing eight repeated slices and a concatenation with common operations.
Weights, normalization, F32 equations, recurrence and learning settings remain
unchanged. The smaller graph is useful only if its composed learning behavior
also satisfies the declared checks.

The isolated branch is `exp/rssm-gate-batching`, commit `f03fb93`, in
`/x/Code/.kindle-grouped-rssm`. Artifacts are in
`runs/grouped-rssm-20260908.Vodj6w`.

- Formatting, Clippy and all 93 ordinary Rust CPU tests pass; 22 GPU tests are
  ignored by that ordinary suite.
- The focused GPU test compares legacy and grouped gates, including all input
  gradients, against independent f64 equations. It covers training rows 1, 3,
  6 and 16, plus 1,024 inference rows and the production eight-block geometry.
  Every output/gradient is finite; unchanged absolute/relative-L2 tolerances
  are 3e-6/3e-5. It passes in 3.57 seconds.
- A separate declaration then requires two full eight-update AB/BA canary pairs,
  using 12M/B16/T64/R256 and prediction-only training, exact non-timing reports
  and every named checkpoint tensor. This runs only after the vector-memory
  comparison releases the GPU; no workload overlaps it.

## Full-learning failure

The first AB pair finished at 21:59:52 UTC. Both runs have complete finite
241-tensor checkpoints, matching native optimizer counters and valid return
normalizers. Their first two non-timing reports match exactly. Reports diverge
from update 3, and the final weights/optimizer moments do not all match.
The original strict comparator rejects the pair; BA is not launched.

The single pair's warmed native learner means are 336.59 ms for the parent and
323.22 ms for the candidate. This is **not a validated speedup**: equivalence
failed and the reverse-order check did not run. It is also a synthetic native
timing, not complete pixel playing-plus-training throughput. Focused gate parity
does not establish composed training parity, and the cause of the divergence
is not established by this test.

Keep the original declaration, output, comparison failure and binaries. Do not
loosen the exact comparator, relabel the failure as harmless rounding or use
the candidate in a long learning run. A future investigation may locate the
first differing gradient/update; that would be a new bounded diagnostic.

## Artifact identities

The tested candidate network source is preserved unchanged on its isolated
branch, SHA-256 `3358d1bbed948a3faaa88d11e4e88d8a1dd52e5061d9be3be59e8707a97d8826`.
The archived `grouped-canary` is
`a3bf11341aab2072f098f254554fdcbfa04d29d55a7c10d52671f49e168e45ed`;
`grouped-tests` is
`ccfdfc441fb5a932f5d7bdbbb83198e6388c1c9baeb18373b0dadbf3a2d6c6c3`.
The validated parent canary is
`f1a9c0bcd940ef7c49e0161782686a4a3449758d2f8f4b8417b173014daf1797`.
Its only other Dreamer-source difference is the zero-update save repair, whose
new branch is not taken after eight learner updates.

Main `networks.rs` is restored to
`49b228e7bc70d69be3a9f9f22c8a5278f2134f195f0e991ce2729d6fcfd70902`.
That intentionally differs from the candidate's original source-path pin;
the archived branch/binary retain the tested bytes. Existing root release
binaries still contain the rejected candidate. Rebuild current source before
using them, or explicitly use the unchanged validated `9cd176c1…` Python
package. The Freeway pilot uses that validated package, not this candidate.

## September 10 CPU postmortem

The read-only comparison in
[`runs/grouped-rssm-postmortem-20260910.FiITco/analysis.json`](../../runs/grouped-rssm-postmortem-20260910.FiITco/analysis.json)
binds twenty closed inputs. The original parent exactly matches the earlier
`meganeura-canary-20260908.u7tbei` a7e2efd9 candidate: all 241 named tensors,
native optimizer and logical metadata, file integrity and eight non-timing
reports. The grouped candidate comparison also reproduces its original saved
failure exactly. This verifies repeatability for these controls, not arbitrary
runtime determinism or a new numerical gate.

The source rewrite changes neither parameter shapes nor AGC groups. The first
reported scalar differences are still at update 3, but that does not locate
the first differing gradient. Only the final update-8 checkpoints were saved.
Historical optimizer indexing starts at zero: the first update has zero learning
rate under the declared 1,000-step warmup, while LaProp still updates moments.
Matching two aggregate scalar reports cannot substitute for those missing states.

No cause or safe fix is established. A future bounded diagnostic should inspect
forward/recurrent inputs, raw and clipped gradients, and complete optimizer
state beginning at update 1, separating world training from posterior/imagination
inference. Do not jump directly to update 3, relax the comparator or reinterpret
this as harmless rounding. No GPU job is scheduled by this postmortem; the
active Atari learning queue and its serial follower retain priority.
