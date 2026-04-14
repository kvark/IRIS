## v0.1 (TBD)

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
