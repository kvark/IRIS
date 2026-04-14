//! Python bindings for the kindle RL agent.
//!
//! Built as a maturin extension module. Exposes a minimal `Agent` wrapper
//! plus a `GymEnvAdapter` that takes a `gymnasium.Env` and trains a
//! kindle agent against it.
//!
//! Build: `maturin develop -m python/Cargo.toml`

use kindle::adapter::{GenericAdapter, OBS_TOKEN_DIM};
use kindle::env::{
    Action, Environment, HomeostaticProvider, HomeostaticVariable, Observation, StepResult,
};
use kindle::{Agent, AgentConfig};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// A kindle agent wired to a Python-side environment.
///
/// Marked `unsendable` because the inner GPU session holds raw pointers that
/// are not `Send`. Python callers must keep the `Agent` on the thread that
/// created it.
#[pyclass(name = "Agent", module = "kindle", unsendable)]
pub struct PyAgent {
    agent: Agent,
    rng: StdRng,
}

#[pymethods]
impl PyAgent {
    /// Create a new agent bound to a gymnasium-style env.
    ///
    /// Args:
    ///     obs_dim (int): env observation vector length.
    ///     num_actions (int): number of discrete actions.
    ///     env_id (int): stable identifier for this env (default 0).
    ///     seed (int): RNG seed (default 0).
    #[new]
    #[pyo3(signature = (obs_dim, num_actions, env_id = 0, seed = 0))]
    fn new(obs_dim: usize, num_actions: usize, env_id: u32, seed: u64) -> PyResult<Self> {
        if obs_dim > OBS_TOKEN_DIM {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "obs_dim {obs_dim} exceeds kindle OBS_TOKEN_DIM {OBS_TOKEN_DIM}"
            )));
        }
        let adapter = Box::new(GenericAdapter::discrete(env_id, obs_dim, num_actions));
        let agent = Agent::new(AgentConfig::default(), adapter);
        Ok(Self {
            agent,
            rng: StdRng::seed_from_u64(seed),
        })
    }

    /// Step the agent against a Python gym env for `steps` iterations.
    ///
    /// Expects `env` to expose `reset()` and `step(action)` following the
    /// gymnasium API. Observations must be 1-D sequences of floats.
    fn train(&mut self, py: Python<'_>, env: &Bound<'_, PyAny>, steps: usize) -> PyResult<()> {
        let reset_obs = env.call_method0("reset")?;
        let mut obs_vec = parse_obs(&reset_obs)?;

        for _ in 0..steps {
            let observation = Observation::new(obs_vec.clone());
            let proxy = ProxyEnv::new(&observation);
            let action = self.agent.act(&observation, &mut self.rng);
            let action_idx = match &action {
                Action::Discrete(i) => *i,
                Action::Continuous(_) => 0,
            };
            let args = PyTuple::new_bound(py, [action_idx]);
            let step_ret = env.call_method1("step", args)?;
            let next_obs = step_ret.get_item(0)?;
            let next_vec = parse_obs(&next_obs)?;
            let next_observation = Observation::new(next_vec.clone());

            self.agent
                .observe(&next_observation, &action, &proxy, &mut self.rng);
            obs_vec = next_vec;
        }
        Ok(())
    }

    /// Current step count.
    fn step_count(&self) -> usize {
        self.agent.step_count()
    }

    /// Diagnostics as a plain Python dict.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let d = self.agent.diagnostics();
        let json = serde_json::to_string(&d)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import_bound("json")?;
        json_mod.call_method1("loads", (json,))
    }
}

/// Lightweight env wrapper that exposes an Observation + empty homeostatic vars.
struct ProxyEnv<'a> {
    obs: &'a Observation,
    empty: Vec<HomeostaticVariable>,
}

impl<'a> ProxyEnv<'a> {
    fn new(obs: &'a Observation) -> Self {
        Self {
            obs,
            empty: Vec::new(),
        }
    }
}

impl<'a> HomeostaticProvider for ProxyEnv<'a> {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.empty
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
            homeostatic: Vec::new(),
        }
    }
    fn reset(&mut self) {}
}

fn parse_obs(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Accept any iterable of numbers (list, tuple, numpy array via __iter__).
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut v = Vec::with_capacity(list.len());
        for item in list.iter() {
            v.push(item.extract::<f32>()?);
        }
        return Ok(v);
    }
    let iter = obj.iter()?;
    let mut v = Vec::new();
    for item in iter {
        let item: Bound<'_, PyAny> = item?;
        v.push(item.extract::<f32>()?);
    }
    Ok(v)
}

#[pymodule]
fn _native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAgent>()?;
    m.add("OBS_TOKEN_DIM", OBS_TOKEN_DIM)?;
    Ok(())
}
