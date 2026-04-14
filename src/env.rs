//! Environment traits defining the boundary between IRIS and any world.

/// A single homeostatic variable exposed by the environment.
#[derive(Clone, Debug)]
pub struct HomeostaticVariable {
    pub value: f32,
    pub target: f32,
    pub tolerance: f32,
}

/// Provides homeostatic signals to the agent.
pub trait HomeostaticProvider {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable];
}

/// Raw observation from the environment.
#[derive(Clone, Debug)]
pub struct Observation {
    pub data: Vec<f32>,
}

impl Observation {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

/// Agent action — discrete or continuous.
#[derive(Clone, Debug)]
pub enum Action {
    Discrete(usize),
    Continuous(Vec<f32>),
}

impl Action {
    /// One-hot encode a discrete action into a vector of length `num_actions`.
    pub fn to_one_hot(&self, num_actions: usize) -> Vec<f32> {
        match *self {
            Action::Discrete(i) => {
                let mut v = vec![0.0; num_actions];
                v[i] = 1.0;
                v
            }
            Action::Continuous(ref v) => v.clone(),
        }
    }

    pub fn dim(&self, num_actions: usize) -> usize {
        match *self {
            Action::Discrete(_) => num_actions,
            Action::Continuous(ref v) => v.len(),
        }
    }
}

/// Outcome of a single environment step.
pub struct StepResult {
    pub observation: Observation,
    pub homeostatic: Vec<HomeostaticVariable>,
}

/// Action space kind for agent configuration.
///
/// - `Discrete { n }`: categorical policy, `n` mutually-exclusive actions,
///   cross-entropy loss against one-hot target.
/// - `Continuous { dim, scale }`: diagonal-Gaussian policy with fixed unit
///   variance, `dim`-dimensional action vectors sampled as `μ + ε·scale`
///   where `ε ~ N(0, 1)`. Loss is MSE between predicted μ and taken action,
///   which is equivalent to the negative log-likelihood up to a constant.
#[derive(Clone, Copy, Debug)]
pub enum ActionKind {
    Discrete { n: usize },
    Continuous { dim: usize, scale: f32 },
}

impl ActionKind {
    /// Dimensionality of the action vector fed to the world model and policy.
    pub fn dim(&self) -> usize {
        match *self {
            ActionKind::Discrete { n } => n,
            ActionKind::Continuous { dim, .. } => dim,
        }
    }
}

/// The environment trait that any world must implement.
pub trait Environment: HomeostaticProvider {
    /// Dimensionality of the observation vector.
    fn observation_dim(&self) -> usize;

    /// Number of discrete actions (for discrete action spaces).
    fn num_actions(&self) -> usize;

    /// Current observation without advancing state.
    fn observe(&self) -> Observation;

    /// Apply an action and advance one timestep.
    fn step(&mut self, action: &Action) -> StepResult;

    /// Reset to initial state.
    fn reset(&mut self);
}
