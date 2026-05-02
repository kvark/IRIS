## v0.2 (TBD)

- PPO clipped-surrogate + end-to-end encoder restored
  (`build_ppo_policy_graph_e2e`). Removed in v0.1 era after never
  converging; revalidated post-autodiff-bug-fix and now trains stably
  on LunarLander (29.7% landings post-drop, vs 0% before fix). The
  earlier failure was likely an autodiff-bug-induced symptom rather
  than a clipped-surrogate pathology.
- GRPO (Group-Relative Policy Optimization) added via `use_grpo`
  config flag. Per-step variant uses n-step return with no V
  bootstrap, normalized across the policy batch (cross-lane mean/std).
  After the bootstrap-noise fix (early version had random V_n
  contaminating the n-step return), per-step GRPO reaches 23% landings
  on LunarLander vs A2C 42.8% and KL-PPO 45.2%.

  Optional `use_grpo_episode` enables per-episode GRPO: each transition
  in a completed episode is annotated retroactively with the episode's
  total return, then advantage is normalized cross-batch. Composes
  with `use_ppo` (PPO clip + GRPO advantage = canonical DeepSeek-R1
  GRPO) and `use_kl_ppo`. Per-episode mode is degenerate at small
  batch sizes (8-32 lanes): clustered episode returns yield zero
  advantage after std normalization. The std-skip fix
  (`use_grpo_episode` only mean-centers) helps but doesn't unlock
  meaningful learning at our scale.

- Self-Imitation Learning (Oh et al. 2018) added via `use_sil`. The
  agent maintains a buffer of `(obs, action, R_to_go, V_at_collect)`
  from "successful" episodes (return > EMA baseline). After each
  policy update, an additional supervised CE step runs on a sampled
  batch with per-sample weight = `sil_loss_coef × max(0, R_to_go - V)`.
  The advantage is clamped to `advantage_clamp` to keep the SIL
  gradient comparable in magnitude to the regular policy gradient
  (without the clamp, raw R_to_go on dense-reward envs produces
  100× stronger gradients that destabilize training).
  Validated end-to-end via diagnostics (`sil_buffer_size`,
  `sil_counters`, `sil_last_param_change`); confirmed SIL changes
  policy parameters and policy entropy.

  Empirical finding on LunarLander: SIL eliminates the hover/timeout
  failure mode (13.7% → 0-4%) but at the cost of more crashes (42% →
  66-88%). Hover-state and landing-state require genuinely different
  optimal actions (mid-air vs on-ground); SIL imports the landed-noop
  action into hover-state where it causes freefall and crash. Adding
  Gaussian noise to obs (encoder smoothness regularization, harness-
  side) moderates this somewhat but doesn't beat the no-SIL baseline
  total non-crash rate (44+14=58% non-crash for baseline vs ~27-34%
  for SIL variants).

  The deeper conclusion: pure on-policy A2C with SIL cannot bridge
  state distributions that genuinely require different optimal
  actions. Off-policy methods (Q-learning, SAC) would be needed to
  evaluate counterfactual actions in those states.

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
