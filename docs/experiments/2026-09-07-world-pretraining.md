# World-only pretraining candidate

Staged on `exp/world-pretraining`, based on `190a80c`. Not adopted, and no
candidate GPU test, dataset training or transfer evaluation has run. The active
LeVJEPA campaign and queued readback worker retain their pinned inputs.

`WorldPretrainer` owns two native sessions: world training and posterior
sampling. It reuses the RSSM, causal predictor, initialization, optimizer and
row microbatching. It has no actor, critic, imagination, online replay or
environment scheduler. Its counters record sampled offline observations and
transitions, including repeated visits, never online interactions.

The Rust input is independent contiguous clips of frozen `Observation` features.
Each resulting observation carries its executed previous action, game tick,
reset/boundary flags and optional reward/terminal labels. Actions must be known;
missing actions are not NOOP. The source declares ordered action names, feature
identity, dataset fingerprint, ticks per second and fixed action ticks. Atari
can use 60 ticks/s and four ticks/action without nanosecond rounding. Timing
gaps and variable-duration transitions, including short terminal action repeats,
are rejected rather than silently modeled as equal-duration transitions.

A clip starts with a cold posterior and no incoming prediction/action/reward;
this context reset need not be an actual episode reset. Later reset markers
must agree with the preceding episode boundary. Missing reward and terminal
labels have zero loss weight. Known zero rewards remain supervised, and a
known timeout is nonterminal. Loss normalization remains B*T, including masked
rows; report label counts when comparing datasets or mask densities.

Only the offline graph uses masked continuation BCE from logits. The original
online graph is unchanged byte-for-byte in five checked configurations: Tiny
reconstruction, prediction-only and combined objectives, plus 12M/T64 with
16 and four world rows. The original source is `world.rs` at `190a80c`, SHA-256
`d22f4ce2ac3881770a64c04a2f2076ca95f01b03cce8b50c5e3fa06ade8d818e`.
Node/output fingerprints retain that CPU regression proof without shipping a
duplicate world implementation. These are source-graph checks, not GPU parity.

CPU checks cover ingestion alignment, reset/gap rejection, mask semantics,
source namespaces, early configuration errors and graph differentiation/
compilation. The two new ignored GPU tests must verify masked continuation
loss/gradients over 513 and 1,024 rows, actual world updates, unchanged initially
unlabeled heads, known-zero reward learning and full-row/microbatch parity.
Existing online numerical and causal regression tests remain required.

`export_world` writes a fresh `world.safetensors` plus `pretraining.json`, not an
online format-3 checkpoint. It records source identity, configuration, offline
counters, backend revisions and a tensor-file fingerprint. It does not save RNG
state or provide resume/transfer. Nonfinite losses disable further use/export of
that runtime; an invalid clip is rejected before changing its state.

Still required: a content-verified dataset reader/causal feature exporter,
explicit compatible-world initialization with fresh target heads/optimizers,
GPU validation and equal-budget adaptation/retention controls. Existing
mind-games DINO caches and continuous/toggle controls cannot be relabeled as
these LeVJEPA features and categorical actions. Do not describe this native
building block as supported video pretraining or useful skill transfer yet.
