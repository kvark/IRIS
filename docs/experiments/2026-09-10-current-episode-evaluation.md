# Current-backend episode-count evaluation candidate

This isolated candidate carries exactly the four Python implementation/test
files from `4281242` onto the qualified current Atari source `90b4763`.
Rust, backend identity, Cargo inputs and native learning arithmetic are unchanged.
The archived f6a2b6ad extension is copied into a fresh import bundle; this is
not a rebuilt wheel or an overwritten historical/editable package.

The optional frozen-only `--episodes-per-env` protocol remains v4: restore a
checkpoint, run every stream until each reaches the declared completed-episode
target, retain every completed episode, and enforce a hard action cap. A cap
without the episode target is incomplete, not success. Training, default v2
evaluation and v3 exploration remain separate and unchanged. No task gate is
weakened by the stopping rule.

All 580 Python CPU tests pass with the actual matched current native preloaded.
The three implementation files and test file are byte-identical to the earlier
candidate. Artifacts and the six source-matched Python modules plus unchanged
native live in `runs/current-episode-package-20260910.etyDN4` in the primary
checkout. This establishes CPU compatibility, not latest-package GPU parity,
combined learner memory, runtime adoption or Breakout/Qbert competence.

Before using this package for a new learning protocol, separately declare and
run a serialized current-native gate: exact default-training full state/reports/
traces against the retained current-backend pixel control; matching frozen
default and v4 prefixes; a negative cap case; complete frozen-state checks;
and >=2,048 MiB directly free with complete GPU coverage. No native rebuild is
necessary when its actual bytes and all native source inputs are unchanged.
Do not alter the pinned Boxing confirmation, and do not run this gate alongside
it. No follower or GPU queue is started by preparing this candidate.

Breakout/Qbert pilots still need their own fixed training budgets and declared
evaluation target (proposed: four completed episodes per six streams, with a
600,000-action cap). The task gates and later fresh roots 1009/2017/3019 remain
unchanged. CPU stopping fixtures are not learned task completions.
