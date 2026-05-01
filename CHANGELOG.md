## v0.2 (TBD)

- PPO clipped-surrogate + end-to-end encoder restored
  (`build_ppo_policy_graph_e2e`). Removed in v0.1 era after never
  converging; revalidated post-autodiff-bug-fix and now trains stably
  on LunarLander (29.7% landings post-drop, vs 0% before fix). The
  earlier failure was likely an autodiff-bug-induced symptom rather
  than a clipped-surrogate pathology.
- GRPO (Group-Relative Policy Optimization) added via `use_grpo`
  config flag. No new graph: replaces V-baseline with raw n-step
  return, lets the existing `advantage_normalize` cross-lane
  mean/std produce the GRPO advantage. Mutually exclusive with
  `use_ppo` / `use_kl_ppo`; requires `advantage_normalize=true`.
  Initial LunarLander validation shows weak performance vs A2C and
  KL-PPO (6.5% vs 45.2%) — no V baseline at 8 lanes loses too much
  signal. Worth revisiting at higher lane counts or with a
  per-episode formulation.

## v0.1 (14 Apr 2026)

- Cold-start continually-self-training RL agent on meganeura: encoder,
  world model, credit assigner, policy+value head, all trained from first
  contact with no pretraining.
- Frozen four-primitive reward circuit: surprise (WM prediction error),
  novelty (1/√N visit counts), homeostatic balance (env-provided target
  variables), and **order** (causal entropy reduction — frozen random
  digest of the universal obs token, sliding recent/reference windows,
  `r_order = H_reference − H_recent`).
- Universal obs/action tokens + per-env `EnvAdapter`; one compiled graph
  handles discrete and continuous action spaces. `Agent::switch_env`
  hops between environments with no graph rebuild; per-env deterministic
  task embeddings keep representations disambiguated.
- Continual-learning mechanics: experience replay mixing, representation
  drift monitor with reactive encoder-LR scaling, policy entropy floor.
- Numerical confidence tooling: finite-difference gradient checks,
  e-graph opt-level parity tests, CartPole / random-walk / world-model
  canaries, long-run stability probes.
- Seven built-in environments in `kindle-gym`: grid_world, cart_pole,
  mountain_car, pendulum, acrobot, taxi, random_walk.
- Optional Python bindings (`kindle-py`, pyo3 + maturin): `kindle.Agent`
  with `train(gym_env, steps)` and `diagnostics()`, drop-in for any
  gymnasium loop.
- Cargo workspace layout: `kindle` (core, publishable), `kindle-gym`
  (envs + runnable examples), `python` (maturin-built extension, out of
  the default workspace build).
