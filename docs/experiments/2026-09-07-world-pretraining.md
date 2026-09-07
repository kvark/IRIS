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
source namespaces, early configuration errors, graph differentiation/compilation
and strict initialization loading. All 86 Kindle and five gym CPU tests pass;
workspace and Python-library Clippy and formatting pass. Four new ignored GPU
tests still need execution: masked continuation loss/gradients over 513 and
1,024 rows; actual updates with initially unlabeled heads, known-zero reward
learning and full-row/microbatch parity; serial initialization and full restore;
and vector posterior synchronization with unchanged independent live state.
Existing online numerical and causal regression tests remain required. Test
fixtures use synthetic features and declared identities, not real dataset training.

`export_world` writes a fresh `world.safetensors` plus `pretraining.json`, not an
online format-3 checkpoint. It records source identity, configuration, offline
counters, algorithm/backend revisions and a tensor-file fingerprint. It does not
save RNG state or provide exact offline resume. Nonfinite losses disable further
use/export of that runtime; an invalid clip is rejected before changing its state.

`WorldInitialization::load` validates the bundle on CPU before target construction:
the explicitly expected dataset/encoder/action-order/timing contract, revisions,
offline counters, complete F32 parameter/moment tensors and selected target
parameter schemas. It hashes and parses the same bytes. Even discarded heads
and moments must be present and finite; second moments must be nonnegative.
The adapter remains responsible for enforcing the declared game's action and
clock semantics; a matching declaration is not evidence about an external game.

`DreamerCore::initialize_world` and `VectorDreamerAgent::initialize_world` accept
this object only on a newly constructed, inactive runtime. They copy dynamics,
representation and causal-predictor weights, then synchronize inference sessions.
Actor, value/slow-value, reward, continuation and reconstruction heads stay fresh,
as do optimizer moments, normalizer, replay, beliefs, RNGs and online counters.
Started streams, repeated initialization and even zero-counter restored agents
are rejected. Target configuration and any bound actual encoder must match.

Ordinary online saves remain format 3. Initialized saves use format 4 and require
the complete source record plus its original metadata fingerprint. Old executables
reject this new format instead of silently losing offline-history provenance.
Complete online restore is still strict; initialization is not partial restore.
The active Python runners/auditors do not support format 4 or expose this new API.

Still required: a content-verified dataset reader/causal feature exporter,
hardware validation of the initializer and masked world updates, adapter/API
integration and equal-budget adaptation/retention controls. Existing
mind-games DINO caches and continuous/toggle controls cannot be relabeled as
these LeVJEPA features and categorical actions. Do not describe this native
building block as supported video pretraining or useful skill transfer yet.
