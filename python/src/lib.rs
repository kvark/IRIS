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
use kindle::{Agent, AgentConfig};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rand::SeedableRng;
use rand::rngs::StdRng;

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

            let step_args = PyTuple::new_bound(py, [action_idx]);
            let step_ret = env.call_method1("step", step_args)?;
            let (next_vec, reward, terminated, truncated) = unpack_step(&step_ret)?;
            episode_return += reward;

            let homeo = match homeo_fn {
                Some(fun) => {
                    let obs_list = PyList::new_bound(py, next_vec.iter().copied());
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
        let json =
            serde_json::to_string(d).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import_bound("json")?;
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
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut v = Vec::with_capacity(list.len());
        for item in list.iter() {
            v.push(item.extract::<f32>()?);
        }
        return Ok(v);
    }
    // Fallback: any iterable (tuple, numpy array, etc.).
    let iter = obj.iter()?;
    let mut v = Vec::new();
    for item in iter {
        let item: Bound<'_, PyAny> = item?;
        v.push(item.extract::<f32>()?);
    }
    Ok(v)
}

fn parse_homeo(obj: &Bound<'_, PyAny>) -> PyResult<Vec<HomeostaticVariable>> {
    let mut out = Vec::new();
    for item in obj.iter()? {
        let item: Bound<'_, PyAny> = item?;
        out.push(parse_homeo_one(&item)?);
    }
    Ok(out)
}

fn parse_homeo_one(obj: &Bound<'_, PyAny>) -> PyResult<HomeostaticVariable> {
    // Accept either a {"value", "target", "tolerance"} dict or a 3-tuple/list.
    if let Ok(dict) = obj.downcast::<PyDict>() {
        let value = dict_get_f32(dict, "value")?;
        let target = dict_get_f32(dict, "target")?;
        let tolerance = dict_get_f32(dict, "tolerance")?;
        return Ok(HomeostaticVariable {
            value,
            target,
            tolerance,
        });
    }
    if let Ok(tup) = obj.downcast::<PyTuple>() {
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
    if let Ok(tup) = obj.downcast::<PyTuple>() {
        if !tup.is_empty() {
            return parse_obs(&tup.get_item(0)?);
        }
    }
    parse_obs(obj)
}

/// Gymnasium: `(obs, reward, terminated, truncated, info)`.
/// Older gym:  `(obs, reward, done, info)`.
fn unpack_step(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, f32, bool, bool)> {
    let tup = obj.downcast::<PyTuple>().map_err(|_| {
        PyValueError::new_err("env.step() must return a tuple")
    })?;
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
    m.add("OBS_TOKEN_DIM", OBS_TOKEN_DIM)?;
    Ok(())
}
