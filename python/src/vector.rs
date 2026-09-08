use super::*;
use kindle::VectorDreamerAgent;

#[pyclass(name = "VectorAgent", module = "kindle", unsendable)]
pub(crate) struct PyVectorAgent {
    inner: VectorDreamerAgent,
}

fn validate_streams(streams: &[usize], count: usize, items: usize) -> PyResult<()> {
    if streams.len() != items {
        return Err(PyValueError::new_err(
            "streams and frames must have equal lengths",
        ));
    }
    let mut seen = vec![false; count];
    for &stream in streams {
        if stream >= count || seen[stream] {
            return Err(PyValueError::new_err("invalid or repeated stream index"));
        }
        seen[stream] = true;
    }
    Ok(())
}

#[pymethods]
impl PyVectorAgent {
    /// The config is the complete dictionary returned by kindle.default_config.
    #[new]
    fn new(encoder_checkpoint: &str, num_envs: usize, config: &Bound<'_, PyAny>) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        let encoded: String = config
            .py()
            .import("json")?
            .call_method1("dumps", (config,))?
            .extract()?;
        let config: DreamerConfig =
            serde_json::from_str(&encoded).map_err(|e| PyValueError::new_err(e.to_string()))?;
        config.check().map_err(PyValueError::new_err)?;
        let inner = VectorDreamerAgent::new(config, num_envs, encoder_checkpoint)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[classmethod]
    fn restore(
        _class: &Bound<'_, PyType>,
        dreamer_checkpoint: &str,
        encoder_checkpoint: &str,
        num_envs: usize,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        let inner = VectorDreamerAgent::restore(dreamer_checkpoint, num_envs, encoder_checkpoint)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn begin_episodes(
        &mut self,
        streams: Vec<usize>,
        frames: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        validate_streams(&streams, self.inner.stream_count(), frames.len())?;
        let frames = streams
            .into_iter()
            .zip(frames)
            .map(|(id, frame)| Ok((id, parse_rgb_frame(&frame)?)))
            .collect::<PyResult<Vec<_>>>()?;
        self.inner.begin_episodes(&frames);
        Ok(())
    }

    #[pyo3(signature = (greedy = false, action_overrides = None))]
    fn act(
        &mut self,
        greedy: bool,
        action_overrides: Option<Vec<Option<usize>>>,
    ) -> PyResult<Vec<usize>> {
        let mode = if greedy {
            ActionMode::Greedy
        } else {
            ActionMode::Sample
        };
        match action_overrides {
            Some(actions) => self
                .inner
                .act_with_overrides(mode, &actions)
                .map_err(PyValueError::new_err),
            None => Ok(self.inner.act(mode)),
        }
    }

    fn observe(
        &mut self,
        streams: Vec<usize>,
        frames: Vec<Bound<'_, PyAny>>,
        rewards: Vec<f32>,
        terminated: Vec<bool>,
        truncated: Vec<bool>,
    ) -> PyResult<Vec<(f32, f32)>> {
        validate_streams(&streams, self.inner.stream_count(), frames.len())?;
        let count = frames.len();
        if rewards.len() != count || terminated.len() != count || truncated.len() != count {
            return Err(PyValueError::new_err(
                "every transition field must have the same length",
            ));
        }
        if !rewards.iter().all(|r| r.is_finite()) {
            return Err(PyValueError::new_err("rewards must be finite"));
        }
        let transitions = streams
            .into_iter()
            .zip(frames)
            .enumerate()
            .map(|(row, (id, frame))| {
                Ok((
                    id,
                    Transition {
                        frame: parse_rgb_frame(&frame)?,
                        reward: Reward {
                            extrinsic: rewards[row],
                            intrinsic: 0.0,
                        },
                        terminated: terminated[row],
                        truncated: truncated[row],
                    },
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(self
            .inner
            .observe(&transitions)
            .into_iter()
            .map(|r| (r.extrinsic, r.intrinsic))
            .collect())
    }

    /// Drain all due updates by default; vector ticks are not individual actions.
    #[pyo3(signature = (maximum_updates = usize::MAX))]
    fn learn_scheduled<'py>(
        &mut self,
        py: Python<'py>,
        maximum_updates: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        reports_to_python(py, &self.inner.learn_scheduled(maximum_updates))
    }

    fn save_checkpoint(&mut self, path: &str) -> PyResult<()> {
        self.inner
            .save_checkpoint(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, self.inner.config())
    }
    #[getter]
    fn provenance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.provenance())
    }
    #[getter]
    fn gpu_device<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_python(py, &self.inner.gpu_device())
    }
    #[getter]
    fn trainable_parameter_counts(&self) -> (usize, usize) {
        self.inner.trainable_parameter_counts()
    }
    #[getter]
    fn learner_step(&self) -> u64 {
        self.inner.learner_step()
    }
    #[getter]
    fn environment_step(&self) -> u64 {
        self.inner.environment_step()
    }
    #[getter]
    fn replay_len(&self) -> usize {
        self.inner.replay_len()
    }
    #[getter]
    fn training_debt(&self) -> f32 {
        self.inner.training_debt()
    }
    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.stream_count()
    }

    #[getter]
    fn cpu_worker_threads(&self) -> usize {
        self.inner.cpu_worker_threads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validate_stream_ids_before_any_native_mutation() {
        assert!(validate_streams(&[2, 0], 3, 2).is_ok());
        assert!(validate_streams(&[0, 0], 3, 2).is_err());
        assert!(validate_streams(&[3], 3, 1).is_err());
        assert!(validate_streams(&[0], 3, 2).is_err());
    }
}
