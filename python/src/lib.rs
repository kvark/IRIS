//! Python bindings for the kindle RL agent.
//!
//! Built as a maturin extension module. Exposes a thin `Agent` wrapper
//! with a gymnasium-native API: `act(obs)`, `observe(next_obs, action,
//! homeostatic=...)`, `mark_boundary()`, and a convenience `run(env,
//! steps, homeo_fn=...)` that drives any `gymnasium.Env`.
//!
//! Build: `maturin develop -m python/Cargo.toml`

// pyo3 0.22's `#[pymethods]` macro expansion emits `.into()` calls on
// PyErr values, which clippy flags as useless. The code is in a macro
// we don't own; silence the lint at the crate level.
#![allow(clippy::useless_conversion)]

use kindle::adapter::{GenericAdapter, OBS_TOKEN_DIM};
use kindle::env::{
    Action, Environment, HomeostaticProvider, HomeostaticVariable, Observation, StepResult,
};
use kindle::{
    agent::{ApproachRankBy, EncoderKind, OutcomeBonus, OutcomeTarget},
    Agent, AgentConfig,
};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// A kindle agent wired to a Python-side environment.
///
/// Marked `unsendable` because the inner GPU session holds raw pointers
/// that are not `Send`. Python callers must keep the `Agent` on the thread
/// that created it.
#[pyclass(name = "Agent", module = "kindle", unsendable)]
pub struct PyAgent {
    agent: Agent,
    rng: StdRng,
    num_actions: usize,
}

#[pymethods]
impl PyAgent {
    /// Create a new agent.
    ///
    /// Args:
    ///     obs_dim (int): env observation vector length. Must be ≤ OBS_TOKEN_DIM.
    ///     num_actions (int): number of discrete actions.
    ///     env_id (int): stable identifier for this env (default 0).
    ///     seed (int): RNG seed (default 0).
    #[new]
    #[pyo3(signature = (obs_dim, num_actions, env_id = 0, seed = 0))]
    fn new(obs_dim: usize, num_actions: usize, env_id: u32, seed: u64) -> PyResult<Self> {
        if obs_dim > OBS_TOKEN_DIM {
            return Err(PyValueError::new_err(format!(
                "obs_dim {obs_dim} exceeds kindle OBS_TOKEN_DIM {OBS_TOKEN_DIM}"
            )));
        }
        let adapter = Box::new(GenericAdapter::discrete(env_id, obs_dim, num_actions));
        let agent = Agent::new(AgentConfig::default(), vec![adapter]);
        Ok(Self {
            agent,
            rng: StdRng::seed_from_u64(seed),
            num_actions,
        })
    }

    /// Sample a discrete action for the given observation.
    ///
    /// `obs` is any 1-D iterable of floats (list, tuple, numpy array).
    /// Returns the action index.
    fn act(&mut self, obs: &Bound<'_, PyAny>) -> PyResult<usize> {
        let obs_vec = parse_obs(obs)?;
        let observation = Observation::new(obs_vec);
        let action = self
            .agent
            .act(std::slice::from_ref(&observation), &mut self.rng)
            .remove(0);
        match action {
            Action::Discrete(i) => Ok(i),
            Action::Continuous(_) => Err(PyRuntimeError::new_err(
                "agent produced a continuous action; PyAgent expects a discrete action space",
            )),
        }
    }

    /// Observe the transition `(prev_obs, action) -> next_obs` and train.
    ///
    /// `action` is the integer index returned by a prior `act()` call.
    /// `homeostatic` is an optional list of dicts with keys `"value"`,
    /// `"target"`, `"tolerance"` — these drive the homeostatic reward
    /// primitive. Without them, homeostatic reward is zero for this step
    /// and the agent trains on surprise + novelty + order alone.
    #[pyo3(signature = (next_obs, action, homeostatic = None))]
    fn observe(
        &mut self,
        next_obs: &Bound<'_, PyAny>,
        action: usize,
        homeostatic: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        if action >= self.num_actions {
            return Err(PyValueError::new_err(format!(
                "action {action} out of range for num_actions={}",
                self.num_actions
            )));
        }
        let obs_vec = parse_obs(next_obs)?;
        let observation = Observation::new(obs_vec);
        let homeo = match homeostatic {
            Some(h) => parse_homeo(h)?,
            None => Vec::new(),
        };
        let proxy = ProxyEnv {
            obs: &observation,
            homeo,
        };
        let action = Action::Discrete(action);
        let proxy_ref: &dyn Environment = &proxy;
        self.agent.observe(
            std::slice::from_ref(&observation),
            std::slice::from_ref(&action),
            std::slice::from_ref(&proxy_ref),
            &mut self.rng,
        );
        Ok(())
    }

    /// Mark the next observed transition as the start of a new episode.
    /// Call this after a gymnasium `terminated | truncated` reset so the
    /// world model and credit assigner don't attribute across the reset.
    fn mark_boundary(&mut self) {
        self.agent.mark_boundary(0);
    }

    /// Convenience: drive a gymnasium-style env for `steps` iterations.
    ///
    /// Handles both the modern gymnasium API (`reset() -> (obs, info)`,
    /// `step() -> (obs, reward, terminated, truncated, info)`) and the
    /// older 4-tuple shape. Returns a list of completed-episode returns
    /// (sums of extrinsic env reward). Extrinsic reward is **not** used
    /// for training — it's returned for monitoring only.
    ///
    /// Args:
    ///     env: any object with `reset()` and `step(action)` methods.
    ///     steps: number of agent steps to run.
    ///     homeo_fn: optional callable `obs -> list[{value, target, tolerance}]`
    ///         mapping each observation to homeostatic targets.
    #[pyo3(signature = (env, steps, homeo_fn = None))]
    fn run(
        &mut self,
        py: Python<'_>,
        env: &Bound<'_, PyAny>,
        steps: usize,
        homeo_fn: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f32>> {
        let reset_ret = env.call_method0("reset")?;
        let mut obs_vec = unpack_reset(&reset_ret)?;

        let mut returns: Vec<f32> = Vec::new();
        let mut episode_return = 0.0f32;

        for step in 0..steps {
            // Allow Ctrl-C during long runs.
            if step % 256 == 0 {
                py.check_signals()?;
            }

            let observation = Observation::new(obs_vec.clone());
            let action = self
                .agent
                .act(std::slice::from_ref(&observation), &mut self.rng)
                .remove(0);
            let action_idx = match &action {
                Action::Discrete(i) => *i,
                Action::Continuous(_) => 0,
            };

            let step_args = PyTuple::new(py, [action_idx])?;
            let step_ret = env.call_method1("step", step_args)?;
            let (next_vec, reward, terminated, truncated) = unpack_step(&step_ret)?;
            episode_return += reward;

            let homeo = match homeo_fn {
                Some(fun) => {
                    let obs_list = PyList::new(py, next_vec.iter().copied())?;
                    let ret = fun.call1((obs_list,))?;
                    parse_homeo(&ret)?
                }
                None => Vec::new(),
            };
            let next_observation = Observation::new(next_vec.clone());
            let proxy = ProxyEnv {
                obs: &next_observation,
                homeo,
            };
            let proxy_ref: &dyn Environment = &proxy;
            self.agent.observe(
                std::slice::from_ref(&next_observation),
                std::slice::from_ref(&action),
                std::slice::from_ref(&proxy_ref),
                &mut self.rng,
            );

            if terminated || truncated {
                returns.push(episode_return);
                episode_return = 0.0;
                let reset_ret = env.call_method0("reset")?;
                obs_vec = unpack_reset(&reset_ret)?;
                self.agent.mark_boundary(0);
            } else {
                obs_vec = next_vec;
            }
        }
        Ok(returns)
    }

    /// Current agent step count.
    fn step_count(&self) -> usize {
        self.agent.step_count()
    }

    /// Diagnostics as a plain Python dict (lane 0).
    ///
    /// The underlying agent is multi-lane (Phase E); `PyAgent` currently
    /// exposes a single-lane wrapper, so this returns lane 0's snapshot.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let diags = self.agent.diagnostics();
        let d = diags
            .first()
            .ok_or_else(|| PyRuntimeError::new_err("agent has no lanes"))?;
        let json = serde_json::to_string(d).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        json_mod.call_method1("loads", (json,))
    }
}

/// Lightweight Environment proxy: holds a reference to the current obs
/// and an owned list of homeostatic variables built from Python input.
struct ProxyEnv<'a> {
    obs: &'a Observation,
    homeo: Vec<HomeostaticVariable>,
}

impl<'a> HomeostaticProvider for ProxyEnv<'a> {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl<'a> Environment for ProxyEnv<'a> {
    fn observation_dim(&self) -> usize {
        self.obs.dim()
    }
    fn num_actions(&self) -> usize {
        0
    }
    fn observe(&self) -> Observation {
        self.obs.clone()
    }
    fn step(&mut self, _action: &Action) -> StepResult {
        StepResult {
            observation: self.obs.clone(),
            homeostatic: self.homeo.clone(),
        }
    }
    fn reset(&mut self) {}
}

fn parse_obs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Fast path: Python list of floats.
    if let Ok(list) = obj.cast::<PyList>() {
        let mut v = Vec::with_capacity(list.len());
        for item in list.iter() {
            v.push(item.extract::<f32>()?);
        }
        return Ok(v);
    }
    // Fallback: any iterable (tuple, numpy array, etc.).
    let iter = obj.try_iter()?;
    let mut v = Vec::new();
    for item in iter {
        let item: Bound<'_, PyAny> = item?;
        v.push(item.extract::<f32>()?);
    }
    Ok(v)
}

fn parse_homeo(obj: &Bound<'_, PyAny>) -> PyResult<Vec<HomeostaticVariable>> {
    let mut out = Vec::new();
    for item in obj.try_iter()? {
        let item: Bound<'_, PyAny> = item?;
        out.push(parse_homeo_one(&item)?);
    }
    Ok(out)
}

fn parse_homeo_one(obj: &Bound<'_, PyAny>) -> PyResult<HomeostaticVariable> {
    // Accept either a {"value", "target", "tolerance"} dict or a 3-tuple/list.
    if let Ok(dict) = obj.cast::<PyDict>() {
        let value = dict_get_f32(dict, "value")?;
        let target = dict_get_f32(dict, "target")?;
        let tolerance = dict_get_f32(dict, "tolerance")?;
        return Ok(HomeostaticVariable {
            value,
            target,
            tolerance,
        });
    }
    if let Ok(tup) = obj.cast::<PyTuple>() {
        if tup.len() != 3 {
            return Err(PyValueError::new_err(
                "homeostatic tuple must have length 3: (value, target, tolerance)",
            ));
        }
        return Ok(HomeostaticVariable {
            value: tup.get_item(0)?.extract()?,
            target: tup.get_item(1)?.extract()?,
            tolerance: tup.get_item(2)?.extract()?,
        });
    }
    Err(PyValueError::new_err(
        "homeostatic entry must be a dict with keys value/target/tolerance or a 3-tuple",
    ))
}

fn dict_get_f32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f32> {
    match dict.get_item(key)? {
        Some(v) => v.extract::<f32>(),
        None => Err(PyKeyError::new_err(format!(
            "homeostatic dict missing key '{key}'"
        ))),
    }
}

/// Gymnasium: `reset() -> (obs, info)`. Older gym: `reset() -> obs`.
fn unpack_reset(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(tup) = obj.cast::<PyTuple>() {
        if !tup.is_empty() {
            return parse_obs(&tup.get_item(0)?);
        }
    }
    parse_obs(obj)
}

/// Gymnasium: `(obs, reward, terminated, truncated, info)`.
/// Older gym:  `(obs, reward, done, info)`.
fn unpack_step(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, f32, bool, bool)> {
    let tup = obj
        .cast::<PyTuple>()
        .map_err(|_| PyValueError::new_err("env.step() must return a tuple"))?;
    let n = tup.len();
    if n < 4 {
        return Err(PyValueError::new_err(
            "env.step() must return at least (obs, reward, done, info)",
        ));
    }
    let obs = parse_obs(&tup.get_item(0)?)?;
    let reward: f32 = tup.get_item(1)?.extract()?;
    let (terminated, truncated) = if n >= 5 {
        let t: bool = tup.get_item(2)?.extract()?;
        let tr: bool = tup.get_item(3)?.extract()?;
        (t, tr)
    } else {
        let done: bool = tup.get_item(2)?.extract()?;
        (done, false)
    };
    Ok((obs, reward, terminated, truncated))
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgent>()?;
    m.add_class::<PyBatchAgent>()?;
    m.add("OBS_TOKEN_DIM", OBS_TOKEN_DIM)?;
    Ok(())
}

/// Multi-lane ("batched") kindle agent — N concurrent envs share a single
/// set of compiled GPU graphs.
///
/// Construction:
///     BatchAgent(obs_dim, num_actions, batch_size, env_ids=None, seed=0)
///
/// Each step drives N envs synchronously: `act(list_of_obs)` returns a
/// list of N action indices; `observe(list_of_next_obs, list_of_actions,
/// homeostatic=list_of_lists)` trains across all lanes with one batched
/// world-model and policy dispatch. `diagnostics()` returns a list of per-lane
/// dicts; `mark_boundary(lane_idx)` marks a single-lane episode reset.
#[pyclass(name = "BatchAgent", module = "kindle", unsendable)]
pub struct PyBatchAgent {
    agent: Agent,
    rng: StdRng,
    num_actions: usize,
    batch_size: usize,
}

#[pymethods]
impl PyBatchAgent {
    #[new]
    #[pyo3(signature = (
        obs_dim,
        num_actions,
        batch_size,
        env_ids = None,
        seed = 0,
        learning_rate = None,
        warmup_steps = None,
        latent_dim = None,
        hidden_dim = None,
        action_repeat = None,
        lr_policy = None,
        lr_credit = None,
        entropy_beta = None,
        entropy_floor = None,
        advantage_clamp = None,
        policy_loss_watchdog_threshold = None,
        replay_ratio = None,
        label_smoothing = None,
        num_options = None,
        option_horizon = None,
        per_option_heads = None,
        option_entropy_beta = None,
        diayn_reward_alpha = None,
        diayn_hidden_dim = None,
        diayn_lr = None,
        history_len = None,
        gamma = None,
        n_step = None,
        outcome_reward_alpha = None,
        lr_outcome = None,
        outcome_baseline_ema = None,
        outcome_clamp = None,
        outcome_target = None,
        outcome_bonus = None,
        outcome_window = None,
        outcome_max_episode_len = None,
        approach_reward_alpha = None,
        approach_buffer_size = None,
        approach_top_frac = None,
        approach_update_interval = None,
        approach_warmup_episodes = None,
        approach_distance_clamp = None,
        approach_confidence_saturation = None,
        homeo_confidence_taper = None,
        approach_rank_by = None,
        encoder_kind = None,
        encoder_channels = None,
        encoder_height = None,
        encoder_width = None,
        reward_surprise = None,
        reward_novelty = None,
        reward_homeostatic = None,
        reward_order = None,
        grid_resolution = None,
        rnd_reward_alpha = None,
        rnd_feature_dim = None,
        rnd_hidden_dim = None,
        rnd_lr = None,
        coord_action_alpha = None,
        coord_hidden_dim = None,
        coord_sigma = None,
        coord_lr = None,
        delta_goal_alpha = None,
        delta_goal_threshold = None,
        delta_goal_merge_radius = None,
        delta_goal_bank_size = None,
        delta_goal_distance_clamp = None,
        delta_goal_surprise_threshold = None,
        xeps_reward_alpha = None,
        xeps_grid_resolution = None,
        extrinsic_reward_alpha = None,
        policy_adv_global_clip = None,
        policy_lr_adaptive_target = None,
        policy_lr_adaptive_ema = None,
        value_bootstrap = None,
        gae_lambda = None,
        value_loss_coef = None,
        policy_update_interval = None,
        advantage_normalize = None,
        use_ppo = None,
        ppo_clip_eps = None,
        use_grpo = None,
        use_grpo_episode = None,
        use_sil = None,
        sil_loss_coef = None,
        sil_buffer_capacity = None,
        sil_baseline_decay = None,
        use_kl_ppo = None,
        kl_beta = None,
        kl_use_snapshot = None,
        ppo_n_epochs = None,
        policy_warmup_steps = None,
        recompute_base_v = None,
        end_to_end_encoder = None,
        rollout_length = None,
        value_clip_scale = None,
        bootstrap_value_clamp = None,
        value_target_clamp = None,
        terminal_proximity_k = None,
        terminal_proximity_bonus = None,
        terminal_proximity_threshold = None,
        recon_loss_coef = None,
        reward_pred_loss_coef = None,
        planner_horizon = None,
        planner_samples = None,
        planner_refresh_interval = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        obs_dim: usize,
        num_actions: usize,
        batch_size: usize,
        env_ids: Option<Vec<u32>>,
        seed: u64,
        learning_rate: Option<f32>,
        warmup_steps: Option<usize>,
        latent_dim: Option<usize>,
        hidden_dim: Option<usize>,
        action_repeat: Option<usize>,
        lr_policy: Option<f32>,
        lr_credit: Option<f32>,
        entropy_beta: Option<f32>,
        entropy_floor: Option<f32>,
        advantage_clamp: Option<f32>,
        policy_loss_watchdog_threshold: Option<f32>,
        replay_ratio: Option<f32>,
        label_smoothing: Option<f32>,
        num_options: Option<usize>,
        option_horizon: Option<usize>,
        per_option_heads: Option<bool>,
        option_entropy_beta: Option<f32>,
        diayn_reward_alpha: Option<f32>,
        diayn_hidden_dim: Option<usize>,
        diayn_lr: Option<f32>,
        history_len: Option<usize>,
        gamma: Option<f32>,
        n_step: Option<usize>,
        outcome_reward_alpha: Option<f32>,
        lr_outcome: Option<f32>,
        outcome_baseline_ema: Option<f32>,
        outcome_clamp: Option<f32>,
        outcome_target: Option<String>,
        outcome_bonus: Option<String>,
        outcome_window: Option<usize>,
        outcome_max_episode_len: Option<usize>,
        approach_reward_alpha: Option<f32>,
        approach_buffer_size: Option<usize>,
        approach_top_frac: Option<f32>,
        approach_update_interval: Option<usize>,
        approach_warmup_episodes: Option<usize>,
        approach_distance_clamp: Option<f32>,
        approach_confidence_saturation: Option<usize>,
        homeo_confidence_taper: Option<f32>,
        approach_rank_by: Option<String>,
        encoder_kind: Option<String>,
        encoder_channels: Option<u32>,
        encoder_height: Option<u32>,
        encoder_width: Option<u32>,
        reward_surprise: Option<f32>,
        reward_novelty: Option<f32>,
        reward_homeostatic: Option<f32>,
        reward_order: Option<f32>,
        grid_resolution: Option<f32>,
        rnd_reward_alpha: Option<f32>,
        rnd_feature_dim: Option<usize>,
        rnd_hidden_dim: Option<usize>,
        rnd_lr: Option<f32>,
        coord_action_alpha: Option<f32>,
        coord_hidden_dim: Option<usize>,
        coord_sigma: Option<f32>,
        coord_lr: Option<f32>,
        delta_goal_alpha: Option<f32>,
        delta_goal_threshold: Option<f32>,
        delta_goal_merge_radius: Option<f32>,
        delta_goal_bank_size: Option<usize>,
        delta_goal_distance_clamp: Option<f32>,
        delta_goal_surprise_threshold: Option<f32>,
        xeps_reward_alpha: Option<f32>,
        xeps_grid_resolution: Option<f32>,
        extrinsic_reward_alpha: Option<f32>,
        policy_adv_global_clip: Option<f32>,
        policy_lr_adaptive_target: Option<f32>,
        policy_lr_adaptive_ema: Option<f32>,
        value_bootstrap: Option<bool>,
        gae_lambda: Option<f32>,
        value_loss_coef: Option<f32>,
        policy_update_interval: Option<usize>,
        advantage_normalize: Option<bool>,
        use_ppo: Option<bool>,
        ppo_clip_eps: Option<f32>,
        use_grpo: Option<bool>,
        use_grpo_episode: Option<bool>,
        use_sil: Option<bool>,
        sil_loss_coef: Option<f32>,
        sil_buffer_capacity: Option<usize>,
        sil_baseline_decay: Option<f32>,
        use_kl_ppo: Option<bool>,
        kl_beta: Option<f32>,
        kl_use_snapshot: Option<bool>,
        ppo_n_epochs: Option<usize>,
        policy_warmup_steps: Option<usize>,
        recompute_base_v: Option<bool>,
        end_to_end_encoder: Option<bool>,
        rollout_length: Option<usize>,
        value_clip_scale: Option<f32>,
        bootstrap_value_clamp: Option<f32>,
        value_target_clamp: Option<f32>,
        terminal_proximity_k: Option<usize>,
        terminal_proximity_bonus: Option<f32>,
        terminal_proximity_threshold: Option<f32>,
        recon_loss_coef: Option<f32>,
        reward_pred_loss_coef: Option<f32>,
        planner_horizon: Option<usize>,
        planner_samples: Option<usize>,
        planner_refresh_interval: Option<usize>,
    ) -> PyResult<Self> {
        if obs_dim > OBS_TOKEN_DIM {
            return Err(PyValueError::new_err(format!(
                "obs_dim {obs_dim} exceeds kindle OBS_TOKEN_DIM {OBS_TOKEN_DIM}"
            )));
        }
        if batch_size == 0 {
            return Err(PyValueError::new_err("batch_size must be >= 1"));
        }
        let ids: Vec<u32> = match env_ids {
            Some(v) => {
                if v.len() != batch_size {
                    return Err(PyValueError::new_err(format!(
                        "env_ids length {} does not match batch_size {}",
                        v.len(),
                        batch_size
                    )));
                }
                v
            }
            None => (0..batch_size as u32).collect(),
        };
        let adapters: Vec<Box<dyn kindle::EnvAdapter>> = ids
            .into_iter()
            .map(|id| {
                Box::new(GenericAdapter::discrete(id, obs_dim, num_actions))
                    as Box<dyn kindle::EnvAdapter>
            })
            .collect();
        let mut config = AgentConfig {
            batch_size,
            ..AgentConfig::default()
        };
        if let Some(lr) = learning_rate {
            config.learning_rate = lr;
            // Scale dependent LRs proportionally to preserve the 0.3×/0.5×
            // ratios documented in the agent module.
            config.lr_credit = lr * 0.3;
            config.lr_policy = lr * 0.5;
        }
        if let Some(w) = warmup_steps {
            config.warmup_steps = w;
        }
        if let Some(ld) = latent_dim {
            config.latent_dim = ld;
        }
        if let Some(hd) = hidden_dim {
            config.hidden_dim = hd;
        }
        if let Some(k) = action_repeat {
            if k == 0 {
                return Err(PyValueError::new_err("action_repeat must be >= 1"));
            }
            config.action_repeat = k;
        }
        // Per-head LR overrides apply *after* the `learning_rate`-derived
        // scaling above, so callers can target a specific head without
        // retuning the other two. Useful for LunarLander-style problems
        // where the policy needs a bigger LR than the world model.
        if let Some(lrp) = lr_policy {
            config.lr_policy = lrp;
        }
        if let Some(lrc) = lr_credit {
            config.lr_credit = lrc;
        }
        if let Some(ef) = entropy_floor {
            config.entropy_floor = ef;
        }
        if let Some(ac) = advantage_clamp {
            config.advantage_clamp = ac;
        }
        if let Some(wd) = policy_loss_watchdog_threshold {
            config.policy_loss_watchdog_threshold = wd;
        }
        if let Some(rr) = replay_ratio {
            config.replay_ratio = rr;
        }
        if let Some(eb) = entropy_beta {
            config.entropy_beta = eb;
        }
        if let Some(ls) = label_smoothing {
            config.label_smoothing = ls;
        }
        if let Some(no) = num_options {
            config.num_options = no;
        }
        if let Some(oh) = option_horizon {
            config.option_horizon = oh;
        }
        if let Some(p) = per_option_heads {
            config.per_option_heads = p;
        }
        if let Some(b) = option_entropy_beta {
            config.option_entropy_beta = b;
        }
        if let Some(a) = diayn_reward_alpha {
            config.diayn_reward_alpha = a;
        }
        if let Some(h) = diayn_hidden_dim {
            config.diayn_hidden_dim = h;
        }
        if let Some(lr) = diayn_lr {
            config.diayn_lr = Some(lr);
        }
        if let Some(h) = history_len {
            config.history_len = h;
        }
        if let Some(g) = gamma {
            config.gamma = g;
        }
        if let Some(ns) = n_step {
            config.n_step = ns;
        }
        if let Some(a) = outcome_reward_alpha {
            config.outcome_reward_alpha = a;
        }
        if let Some(lr) = lr_outcome {
            config.lr_outcome = Some(lr);
        }
        if let Some(ema) = outcome_baseline_ema {
            config.outcome_baseline_ema = ema;
        }
        if let Some(c) = outcome_clamp {
            config.outcome_clamp = c;
        }
        if let Some(t) = outcome_target {
            config.outcome_target = match t.as_str() {
                "episode_sum" | "episode-sum" => OutcomeTarget::EpisodeSum,
                "terminal_reward" | "terminal-reward" | "terminal" => OutcomeTarget::TerminalReward,
                "reward_to_go" | "reward-to-go" | "rtg" => OutcomeTarget::RewardToGo,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "outcome_target must be 'episode_sum', 'terminal_reward' or \
                         'reward_to_go', got {other:?}"
                    )));
                }
            };
        }
        if let Some(b) = outcome_bonus {
            config.outcome_bonus = match b.as_str() {
                "raw" => OutcomeBonus::Raw,
                "potential_delta" | "potential-delta" | "delta" => OutcomeBonus::PotentialDelta,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "outcome_bonus must be 'raw' or 'potential_delta', got {other:?}"
                    )));
                }
            };
        }
        if let Some(w) = outcome_window {
            config.outcome_window = w.max(1);
        }
        if let Some(l) = outcome_max_episode_len {
            config.outcome_max_episode_len = l;
        }
        if let Some(a) = approach_reward_alpha {
            config.approach_reward_alpha = a;
        }
        if let Some(s) = approach_buffer_size {
            config.approach_buffer_size = s;
        }
        if let Some(tf) = approach_top_frac {
            config.approach_top_frac = tf;
        }
        if let Some(ui) = approach_update_interval {
            config.approach_update_interval = ui;
        }
        if let Some(we) = approach_warmup_episodes {
            config.approach_warmup_episodes = we;
        }
        if let Some(dc) = approach_distance_clamp {
            config.approach_distance_clamp = dc;
        }
        if let Some(cs) = approach_confidence_saturation {
            config.approach_confidence_saturation = cs;
        }
        if let Some(ht) = homeo_confidence_taper {
            config.homeo_confidence_taper = ht;
        }
        if let Some(rb) = approach_rank_by {
            config.approach_rank_by = match rb.as_str() {
                "return" => ApproachRankBy::Return,
                "novelty" => ApproachRankBy::Novelty,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "approach_rank_by must be 'return' or 'novelty', got {other:?}"
                    )));
                }
            };
        }
        if let Some(v) = reward_surprise {
            config.reward_weights.surprise = v;
        }
        if let Some(v) = reward_novelty {
            config.reward_weights.novelty = v;
        }
        if let Some(v) = reward_homeostatic {
            config.reward_weights.homeostatic = v;
        }
        if let Some(v) = reward_order {
            config.reward_weights.order = v;
        }
        if let Some(gr) = grid_resolution {
            config.grid_resolution = gr;
        }
        if let Some(v) = rnd_reward_alpha {
            config.rnd_reward_alpha = v;
        }
        if let Some(v) = rnd_feature_dim {
            config.rnd_feature_dim = v;
        }
        if let Some(v) = rnd_hidden_dim {
            config.rnd_hidden_dim = v;
        }
        if let Some(v) = rnd_lr {
            config.rnd_lr = Some(v);
        }
        if let Some(v) = coord_action_alpha {
            config.coord_action_alpha = v;
        }
        if let Some(v) = coord_hidden_dim {
            config.coord_hidden_dim = v;
        }
        if let Some(v) = coord_sigma {
            config.coord_sigma = v;
        }
        if let Some(v) = coord_lr {
            config.coord_lr = Some(v);
        }
        if let Some(v) = delta_goal_alpha {
            config.delta_goal_alpha = v;
        }
        if let Some(v) = delta_goal_threshold {
            config.delta_goal_threshold = v;
        }
        if let Some(v) = delta_goal_merge_radius {
            config.delta_goal_merge_radius = v;
        }
        if let Some(v) = delta_goal_bank_size {
            config.delta_goal_bank_size = v;
        }
        if let Some(v) = delta_goal_distance_clamp {
            config.delta_goal_distance_clamp = v;
        }
        if let Some(v) = delta_goal_surprise_threshold {
            config.delta_goal_surprise_threshold = v;
        }
        if let Some(v) = xeps_reward_alpha {
            config.xeps_reward_alpha = v;
        }
        if let Some(v) = xeps_grid_resolution {
            config.xeps_grid_resolution = Some(v);
        }
        if let Some(v) = extrinsic_reward_alpha {
            config.extrinsic_reward_alpha = v;
        }
        if let Some(v) = policy_adv_global_clip {
            config.policy_adv_global_clip = v;
        }
        if let Some(v) = policy_lr_adaptive_target {
            config.policy_lr_adaptive_target = v;
        }
        if let Some(v) = policy_lr_adaptive_ema {
            config.policy_lr_adaptive_ema = v;
        }
        if let Some(v) = value_bootstrap {
            config.value_bootstrap = v;
        }
        if let Some(v) = gae_lambda {
            config.gae_lambda = v;
        }
        if let Some(v) = value_loss_coef {
            config.value_loss_coef = v;
        }
        if let Some(v) = policy_update_interval {
            config.policy_update_interval = v;
        }
        if let Some(v) = advantage_normalize {
            config.advantage_normalize = v;
        }
        if let Some(v) = use_ppo {
            config.use_ppo = v;
        }
        if let Some(v) = ppo_clip_eps {
            config.ppo_clip_eps = v;
        }
        if let Some(v) = use_grpo {
            config.use_grpo = v;
        }
        if let Some(v) = use_grpo_episode {
            config.use_grpo_episode = v;
        }
        if let Some(v) = use_sil {
            config.use_sil = v;
        }
        if let Some(v) = sil_loss_coef {
            config.sil_loss_coef = v;
        }
        if let Some(v) = sil_buffer_capacity {
            config.sil_buffer_capacity = v;
        }
        if let Some(v) = sil_baseline_decay {
            config.sil_baseline_decay = v;
        }
        if let Some(v) = use_kl_ppo {
            config.use_kl_ppo = v;
        }
        if let Some(v) = kl_beta {
            config.kl_beta = v;
        }
        if let Some(v) = kl_use_snapshot {
            config.kl_use_snapshot = v;
        }
        if let Some(v) = ppo_n_epochs {
            config.ppo_n_epochs = v;
        }
        if let Some(v) = policy_warmup_steps {
            config.policy_warmup_steps = v;
        }
        if let Some(v) = recompute_base_v {
            config.recompute_base_v = v;
        }
        if let Some(v) = end_to_end_encoder {
            config.end_to_end_encoder = v;
        }
        if let Some(v) = rollout_length {
            config.rollout_length = v;
        }
        if let Some(v) = value_clip_scale {
            config.value_clip_scale = v;
        }
        if let Some(v) = bootstrap_value_clamp {
            config.bootstrap_value_clamp = v;
        }
        if let Some(v) = value_target_clamp {
            config.value_target_clamp = v;
        }
        if let Some(v) = terminal_proximity_k {
            config.terminal_proximity_k = v;
        }
        if let Some(v) = terminal_proximity_bonus {
            config.terminal_proximity_bonus = v;
        }
        if let Some(v) = terminal_proximity_threshold {
            config.terminal_proximity_threshold = v;
        }
        if let Some(v) = recon_loss_coef {
            config.recon_loss_coef = v;
        }
        if let Some(v) = reward_pred_loss_coef {
            config.reward_pred_loss_coef = v;
        }
        if let Some(v) = planner_horizon {
            config.planner_horizon = v;
        }
        if let Some(v) = planner_samples {
            config.planner_samples = v;
        }
        if let Some(v) = planner_refresh_interval {
            config.planner_refresh_interval = v;
        }
        if let Some(ek) = encoder_kind {
            config.encoder_kind = match ek.as_str() {
                "mlp" => EncoderKind::Mlp,
                "cnn" => {
                    let channels = encoder_channels.ok_or_else(|| {
                        PyValueError::new_err("encoder_kind='cnn' requires encoder_channels")
                    })?;
                    let height = encoder_height.ok_or_else(|| {
                        PyValueError::new_err("encoder_kind='cnn' requires encoder_height")
                    })?;
                    let width = encoder_width.ok_or_else(|| {
                        PyValueError::new_err("encoder_kind='cnn' requires encoder_width")
                    })?;
                    EncoderKind::Cnn {
                        channels,
                        height,
                        width,
                    }
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "encoder_kind must be 'mlp' or 'cnn', got {other:?}"
                    )));
                }
            };
        }
        let agent = Agent::new(config, adapters);
        Ok(Self {
            agent,
            rng: StdRng::seed_from_u64(seed),
            num_actions,
            batch_size,
        })
    }

    /// Sample one discrete action per lane. `obs_list` must be a sequence
    /// of length `batch_size` where each entry is a 1-D sequence of floats.
    /// Returns a list of integer action indices, length `batch_size`.
    fn act(&mut self, obs_list: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let obs_vecs = parse_obs_list(obs_list, self.batch_size)?;
        let observations: Vec<Observation> = obs_vecs.into_iter().map(Observation::new).collect();
        let actions = self.agent.act(&observations, &mut self.rng);
        actions
            .into_iter()
            .map(|a| match a {
                Action::Discrete(i) => Ok(i),
                Action::Continuous(_) => Err(PyRuntimeError::new_err(
                    "agent produced a continuous action; BatchAgent expects discrete actions",
                )),
            })
            .collect()
    }

    /// Observe one synchronous step across all lanes.
    ///
    /// `next_obs_list` and `actions_list` must have length `batch_size`.
    /// `homeostatic` is an optional list-of-lists (one list per lane) — each
    /// inner list is a sequence of `{"value", "target", "tolerance"}` dicts
    /// (or 3-tuples).
    #[pyo3(signature = (next_obs_list, actions_list, homeostatic = None))]
    fn observe(
        &mut self,
        next_obs_list: &Bound<'_, PyAny>,
        actions_list: &Bound<'_, PyAny>,
        homeostatic: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let obs_vecs = parse_obs_list(next_obs_list, self.batch_size)?;
        let observations: Vec<Observation> = obs_vecs.into_iter().map(Observation::new).collect();

        // Parse actions list.
        let actions_outer: Vec<Bound<'_, PyAny>> = {
            let mut v = Vec::with_capacity(self.batch_size);
            for item in actions_list.try_iter()? {
                v.push(item?);
            }
            v
        };
        if actions_outer.len() != self.batch_size {
            return Err(PyValueError::new_err(format!(
                "actions_list length {} must equal batch_size {}",
                actions_outer.len(),
                self.batch_size
            )));
        }
        let mut actions = Vec::with_capacity(self.batch_size);
        for (i, item) in actions_outer.into_iter().enumerate() {
            let a: usize = item.extract()?;
            if a >= self.num_actions {
                return Err(PyValueError::new_err(format!(
                    "action {a} at lane {i} out of range for num_actions={}",
                    self.num_actions
                )));
            }
            actions.push(Action::Discrete(a));
        }

        // Parse optional per-lane homeostatic lists.
        let homeos: Vec<Vec<HomeostaticVariable>> = match homeostatic {
            None => (0..self.batch_size).map(|_| Vec::new()).collect(),
            Some(outer) => {
                let mut out: Vec<Vec<HomeostaticVariable>> = Vec::with_capacity(self.batch_size);
                for (i, inner) in outer.try_iter()?.enumerate() {
                    if i >= self.batch_size {
                        return Err(PyValueError::new_err(
                            "homeostatic list longer than batch_size",
                        ));
                    }
                    let inner: Bound<'_, PyAny> = inner?;
                    out.push(parse_homeo(&inner)?);
                }
                while out.len() < self.batch_size {
                    out.push(Vec::new());
                }
                out
            }
        };

        // Build proxy envs (hold the observations + homeostats for this step).
        let proxies: Vec<ProxyEnv> = observations
            .iter()
            .zip(homeos.into_iter())
            .map(|(obs, homeo)| ProxyEnv { obs, homeo })
            .collect();
        let env_refs: Vec<&dyn Environment> =
            proxies.iter().map(|p| p as &dyn Environment).collect();

        self.agent
            .observe(&observations, &actions, &env_refs, &mut self.rng);
        Ok(())
    }

    /// Mark lane `lane_idx` as starting a new episode on the next observe.
    fn mark_boundary(&mut self, lane_idx: usize) {
        self.agent.mark_boundary(lane_idx);
    }

    /// Current agent step count.
    fn step_count(&self) -> usize {
        self.agent.step_count()
    }

    /// Number of lanes.
    fn num_lanes(&self) -> usize {
        self.agent.num_lanes()
    }

    /// Per-lane diagnostics as a list of plain Python dicts.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let diags = self.agent.diagnostics();
        let json =
            serde_json::to_string(&diags).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        json_mod.call_method1("loads", (json,))
    }

    /// Cheap per-step getter: list of `R̂(z_t)` values, one per lane.
    /// Avoids the JSON round-trip of `diagnostics()` so per-step
    /// M6 instrumentation in a Python harness stays in the noise
    /// budget. Returns zeros when M6 is disabled.
    fn r_hats(&self) -> Vec<f32> {
        self.agent.r_hats()
    }

    /// Per-lane value baseline V(s_t) from the most recent act().
    /// Used for diagnosing whether the value head is discriminating
    /// across states or just predicting the mean return everywhere
    /// (degenerate case → advantage ≈ 0 → policy can't learn).
    fn values(&self) -> Vec<f32> {
        self.agent.values()
    }

    /// Per-lane policy entropy from the most recent act().
    fn entropies(&self) -> Vec<f32> {
        self.agent.entropies()
    }

    /// Visual-encoder input setter for `encoder_kind='cnn'` agents.
    /// Accepts a 1-D sequence of length `batch_size · channels · height · width`
    /// laid out as flat NCHW. Must be called before each
    /// `observe()` when the agent's encoder is CNN; no-op for MLP
    /// agents.
    fn set_visual_obs(&mut self, visual: &Bound<'_, PyAny>) -> PyResult<()> {
        // Fast path: a contiguous float32 buffer (numpy array, bytes,
        // memoryview, etc.) — read directly via the buffer protocol
        // without materializing an intermediate Vec<f32>. For Atari's
        // 28224 floats / lane / step this is ~50× cheaper than
        // `Vec::<f32>::extract`, which iterates a Python sequence.
        if let Ok(buf) = pyo3::buffer::PyBuffer::<f32>::get(visual) {
            let py = visual.py();
            if let Some(slice) = buf.as_slice(py) {
                // `ReadOnlyCell<f32>` has the same layout as `f32`;
                // cast and hand kindle an ordinary slice.
                let ptr = slice.as_ptr() as *const f32;
                let len = slice.len();
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                self.agent.set_visual_obs(slice);
                return Ok(());
            }
        }
        // Fallback: extract to a Vec<f32>. Accepts any iterable of
        // floats (legacy `list[float]` usage).
        let v: Vec<f32> = visual.extract()?;
        self.agent.set_visual_obs(&v);
        Ok(())
    }

    /// Set per-lane extrinsic (env) reward for the next `observe()`
    /// call. `rewards` is a length-`batch_size` buffer (numpy array
    /// preferred for zero-copy; any iterable also accepted). No-op
    /// when `extrinsic_reward_alpha == 0.0` at agent construction.
    fn set_extrinsic_reward(&mut self, rewards: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(buf) = pyo3::buffer::PyBuffer::<f32>::get(rewards) {
            let py = rewards.py();
            if let Some(slice) = buf.as_slice(py) {
                let ptr = slice.as_ptr() as *const f32;
                let len = slice.len();
                let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                self.agent.set_extrinsic_reward(slice);
                return Ok(());
            }
        }
        let v: Vec<f32> = rewards.extract()?;
        self.agent.set_extrinsic_reward(&v);
        Ok(())
    }

    /// Update the base learning rate at runtime. Picked up by every
    /// per-step `set_learning_rate` across all sessions on the next
    /// `observe()`. Use case: drop LR after sustained-solve detected,
    /// to head off the on-policy AC post-solve crash.
    ///
    /// **Footgun:** the policy session reads `lr_policy`, not
    /// `learning_rate`. Calling only this setter is usually a silent
    /// no-op for an LR schedule (the policy is the post-solve
    /// destabilizer). Prefer `set_all_learning_rates` for a single-
    /// knob LR drop.
    fn set_learning_rate(&mut self, lr: f32) {
        self.agent.set_learning_rate(lr);
    }

    /// Update the policy-session learning rate at runtime. Independent
    /// of `set_learning_rate`. Most direct lever for stabilizing a
    /// committed-good policy.
    fn set_lr_policy(&mut self, lr: f32) {
        self.agent.set_lr_policy(lr);
    }

    /// Update base + policy + credit learning rates together,
    /// preserving the constructor's 0.5× / 0.3× ratios. The
    /// recommended setter for LR schedules — avoids the
    /// `set_learning_rate`-without-`set_lr_policy` no-op footgun.
    fn set_all_learning_rates(&mut self, lr: f32) {
        self.agent.set_all_learning_rates(lr);
    }

    /// Update entropy bonus weight at runtime. Currently only effective
    /// for the e2e graph (where entropy_beta is a graph input). Use case:
    /// anneal entropy from high → 0 over training to maintain exploration
    /// early then commit late. Standard PPO/A2C entropy schedule.
    fn set_entropy_beta(&mut self, beta: f32) {
        self.agent.set_entropy_beta(beta);
    }

    /// Update KL-PPO trust-region weight at runtime. Use Schulman 2017's
    /// adaptive rule: read `last_kl()`, double β if KL > target × 1.5,
    /// halve if KL < target / 1.5.
    fn set_kl_beta(&mut self, beta: f32) {
        self.agent.set_kl_beta(beta);
    }

    /// Most recent observed KL(π_new ‖ π_old) from the last training
    /// step. 0.0 when KL-PPO is off.
    fn last_kl(&self) -> f32 {
        self.agent.last_kl()
    }

    /// Print a summary of the largest policy-session gradient norms
    /// (descending), with grad/weight ratios. Diagnostic for which
    /// parameter is driving instability. Print goes to stderr.
    fn dump_policy_grad_summary(&self, top_n: usize) {
        self.agent.dump_policy_grad_summary(top_n);
    }

    /// Bulk read of (param_name, grad_norm) pairs from the policy session.
    /// Useful for plotting or anomaly detection without printing.
    fn policy_grad_norms(&self) -> Vec<(String, f32)> {
        self.agent.policy_grad_norms()
    }

    /// Bulk read of (param_name, weight_norm) pairs from the policy session.
    fn policy_weight_norms(&self) -> Vec<(String, f32)> {
        self.agent.policy_weight_norms()
    }

    /// Set a per-parameter LR multiplier on the policy session by
    /// name prefix. Use to rebalance asymmetric gradient flow between
    /// encoder and policy head — see Agent::set_policy_lr_multiplier
    /// for the diagnostic context.
    fn set_policy_lr_multiplier(&mut self, prefix: &str, mul: f32) {
        self.agent.set_policy_lr_multiplier(prefix, mul);
    }

    /// Clear all per-parameter LR multipliers on the policy session.
    fn clear_policy_lr_multipliers(&mut self) {
        self.agent.clear_policy_lr_multipliers();
    }

    /// Writable `memoryview` over the world-model session's
    /// `visual_obs` input buffer. Meganeura allocates that buffer
    /// as `Memory::Shared` (device-local + host-visible + host-
    /// coherent), so a Python write through this memoryview lands
    /// directly in the GPU-side memory — no staging, no explicit
    /// upload, no `VK_EXT_external_memory_host` import.
    ///
    /// Typical usage:
    ///
    /// ```python
    /// mv = agent.visual_obs_memoryview()
    /// buf = np.frombuffer(mv, dtype=np.float32).reshape(
    ///     batch, channels, height, width
    /// )
    /// np.divide(raw_u8, 255.0, out=buf, dtype=np.float32, casting="unsafe")
    /// agent.observe(...)  # reads the just-written frames
    /// ```
    ///
    /// Returns an error when the encoder is MLP (no visual slot).
    /// The memoryview lifetime is tied to the agent; the agent
    /// must not be dropped while Python still holds the view.
    ///
    /// Ordering: writes through this pointer must complete before
    /// the next `observe()` call. `observe()` internally waits for
    /// the previous step's GPU reads to finish before returning,
    /// so any write performed between two `observe()` calls is
    /// safe.
    fn visual_obs_memoryview<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (ptr, size) = self.agent.visual_obs_host_ptr().ok_or_else(|| {
            PyRuntimeError::new_err(
                "visual_obs_memoryview: no visual_obs slot (encoder is MLP, not CNN)",
            )
        })?;
        // Safety: `ptr`/`size` are stable for the lifetime of the
        // agent (backed by meganeura's owned graph buffer). The
        // memoryview is a non-owning view; Python must not outlive
        // the BatchAgent, which matches typical harness usage.
        let mv = unsafe {
            let ffi_mv = pyo3::ffi::PyMemoryView_FromMemory(
                ptr as *mut std::os::raw::c_char,
                size as pyo3::ffi::Py_ssize_t,
                pyo3::ffi::PyBUF_WRITE,
            );
            if ffi_mv.is_null() {
                return Err(PyErr::fetch(py));
            }
            Bound::from_owned_ptr(py, ffi_mv)
        };
        Ok(mv)
    }

    /// Byte size of the `visual_obs` input buffer. Matches the
    /// length of the memoryview returned by `visual_obs_memoryview`.
    /// Zero when the encoder is MLP.
    fn visual_obs_host_size(&self) -> usize {
        self.agent.visual_obs_host_size()
    }

    /// Re-initialize the RND predictor. Used to re-activate
    /// curiosity when the agent enters a new state distribution
    /// (e.g. on level-up in ARC-AGI-3). No-op when RND is
    /// disabled.
    fn reset_rnd_predictor(&mut self) {
        self.agent.reset_rnd_predictor();
    }

    /// Sample per-lane `(x, y)` from the coord head, each in
    /// `[−1, 1]`. Harness rescales to the target env's coord
    /// range. Returns zeros per lane when the coord head is
    /// disabled. Call before the env step; kindle caches sample
    /// state so `train_coord_head` can apply the REINFORCE
    /// update once the resulting reward is known.
    fn sample_coords(&mut self) -> Vec<(f32, f32)> {
        self.agent.sample_coords(&mut self.rng)
    }

    /// Train the coord head on the per-step rewards cached during
    /// the last `observe()`. Uses an EMA reward baseline inside
    /// the agent so the harness doesn't need to supply
    /// advantages. No-op when the coord head is disabled.
    fn train_coord_head(&mut self) {
        self.agent.train_coord_head();
    }

    /// Current size of the M8 delta-goal bank. Returns 0 when
    /// M8 is disabled.
    fn delta_goal_bank_size(&self) -> usize {
        self.agent.delta_goal_bank_size()
    }

    /// Number of M8 goal-events recorded during the most recent
    /// `observe()` call, summed across lanes. Returns 0 when M8
    /// is disabled.
    fn last_delta_goal_events(&self) -> usize {
        self.agent.last_delta_goal_events()
    }

    /// Distinct `(quantized_state, action)` pairs tracked by the
    /// cross-episode memory. Returns 0 when disabled.
    fn xeps_distinct_pairs(&self) -> usize {
        self.agent.xeps_distinct_pairs()
    }

    /// Run the Track 3 model-based planner for every lane with an
    /// empty planner queue. Samples `planner_samples` random action
    /// sequences of length `planner_horizon`, rolls each out
    /// through the WM, and queues the highest-novelty-score
    /// sequence. Call from the harness before `act()` when you
    /// want the next K actions to come from planning rather than
    /// the policy. No-op when `planner_horizon == 0`.
    fn plan_and_queue(&mut self, num_actions: usize) {
        self.agent.plan_and_queue(num_actions, &mut self.rng);
    }

    /// Total queued planner actions across all lanes. Diagnostic.
    fn planner_queue_len(&self) -> usize {
        self.agent.planner_queue_len()
    }

    /// Snapshot the most recent `n` transitions of `lane_idx`, oldest
    /// first, as a list of dicts. Each dict contains:
    ///   observation (list[float]), latent (list[float]),
    ///   action (list[float]), reward (float), credit (float),
    ///   pred_error (float), value (float), prob_taken (float),
    ///   option_idx (int), env_id (int), env_boundary (bool).
    /// Diagnostic-only — used to inspect rollout buffers offline
    /// (e.g. compare V(s_t) trajectory to discounted return, plot
    /// pred_error over time, locate landing windows by reward spikes).
    fn dump_buffer<'py>(
        &self,
        py: Python<'py>,
        lane_idx: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let snap = self.agent.recent_transitions(lane_idx, n);
        let out = PyList::empty(py);
        for t in snap.iter() {
            let d = PyDict::new(py);
            d.set_item("observation", t.observation.clone())?;
            d.set_item("latent", t.latent.clone())?;
            d.set_item("action", t.action.clone())?;
            d.set_item("reward", t.reward)?;
            d.set_item("credit", t.credit)?;
            d.set_item("pred_error", t.pred_error)?;
            d.set_item("value", t.value)?;
            d.set_item("prob_taken", t.prob_taken)?;
            d.set_item("option_idx", t.option_idx)?;
            d.set_item("env_id", t.env_id)?;
            d.set_item("env_boundary", t.env_boundary)?;
            out.append(d)?;
        }
        Ok(out)
    }
}

/// Parse an outer sequence of length `expected` whose entries are each a
/// 1-D obs vector (list/tuple/ndarray of floats).
fn parse_obs_list(obj: &Bound<'_, PyAny>, expected: usize) -> PyResult<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(expected);
    for item in obj.try_iter()? {
        let item: Bound<'_, PyAny> = item?;
        out.push(parse_obs(&item)?);
    }
    if out.len() != expected {
        return Err(PyValueError::new_err(format!(
            "expected outer sequence of length {expected}, got {}",
            out.len()
        )));
    }
    Ok(out)
}
