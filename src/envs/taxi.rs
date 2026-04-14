//! Taxi-v3 from OpenAI Gymnasium.
//!
//! A 5×5 grid with walls, four pickup/dropoff locations, and a passenger
//! needing to be moved between them. Classic discrete navigation task
//! with long-horizon credit assignment:
//!
//! 1. Navigate to the passenger's location
//! 2. Pick them up
//! 3. Navigate to the destination
//! 4. Drop them off
//!
//! Actions (6): south, north, east, west, pickup, dropoff
//! Observation (34-dim): one-hot taxi position (25) + one-hot passenger
//! location (5, incl. "in-taxi") + one-hot destination (4)
//! Homeostatic: Manhattan distance to current sub-goal (target 0)

use crate::env::*;

pub const GRID_SIZE: usize = 5;
pub const NUM_LOCS: usize = 4;
pub const OBS_DIM: usize = GRID_SIZE * GRID_SIZE + (NUM_LOCS + 1) + NUM_LOCS;
pub const NUM_ACTIONS: usize = 6;

/// Pickup/dropoff location coordinates (row, col).
const LOCS: [(usize, usize); NUM_LOCS] = [(0, 0), (0, 4), (4, 0), (4, 3)];

/// Walls as (row, col_between_a_b). A wall between cols c and c+1 at row r
/// means movement from (r, c)↔(r, c+1) is blocked.
const WALLS: &[(usize, usize)] = &[
    // Top half: wall between col 1 and col 2, rows 0-1
    (0, 1),
    (1, 1),
    // Bottom half: wall between col 0 and col 1, rows 3-4
    (3, 0),
    (4, 0),
    // Bottom half: wall between col 2 and col 3, rows 3-4
    (3, 2),
    (4, 2),
];

fn wall_between(row: usize, col_a: usize, col_b: usize) -> bool {
    let left = col_a.min(col_b);
    WALLS.iter().any(|&(r, c)| r == row && c == left)
}

pub struct Taxi {
    /// Taxi position (row, col)
    pub pos: (usize, usize),
    /// Passenger location: 0..NUM_LOCS = at that location, NUM_LOCS = in taxi
    pub passenger: usize,
    /// Destination index 0..NUM_LOCS
    pub destination: usize,
    step_count: usize,
    max_steps: usize,
    /// Initial state cycle counter for deterministic-enough resets
    reset_counter: usize,
    homeo: Vec<HomeostaticVariable>,
}

impl Taxi {
    pub fn new() -> Self {
        let mut env = Self {
            pos: (2, 2),
            passenger: 0,
            destination: 1,
            step_count: 0,
            max_steps: 200,
            reset_counter: 0,
            homeo: Vec::new(),
        };
        env.update_homeo();
        env
    }

    fn goal_position(&self) -> (usize, usize) {
        if self.passenger < NUM_LOCS {
            // Need to pick up
            LOCS[self.passenger]
        } else {
            // Carrying passenger — head to destination
            LOCS[self.destination]
        }
    }

    fn manhattan_to_goal(&self) -> f32 {
        let (gr, gc) = self.goal_position();
        ((self.pos.0 as i32 - gr as i32).abs() + (self.pos.1 as i32 - gc as i32).abs()) as f32
    }

    fn update_homeo(&mut self) {
        self.homeo = vec![HomeostaticVariable {
            value: self.manhattan_to_goal(),
            target: 0.0,
            tolerance: 0.5,
        }];
    }
}

impl HomeostaticProvider for Taxi {
    fn homeostatic_variables(&self) -> &[HomeostaticVariable] {
        &self.homeo
    }
}

impl Environment for Taxi {
    fn observation_dim(&self) -> usize {
        OBS_DIM
    }

    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }

    fn observe(&self) -> Observation {
        let mut data = vec![0.0f32; OBS_DIM];
        // Taxi position one-hot (25 dims)
        data[self.pos.0 * GRID_SIZE + self.pos.1] = 1.0;
        // Passenger location one-hot (5 dims: 4 locs + in-taxi)
        data[GRID_SIZE * GRID_SIZE + self.passenger] = 1.0;
        // Destination one-hot (4 dims)
        data[GRID_SIZE * GRID_SIZE + (NUM_LOCS + 1) + self.destination] = 1.0;
        Observation::new(data)
    }

    fn step(&mut self, action: &Action) -> StepResult {
        let &Action::Discrete(a) = action else {
            panic!("Taxi uses discrete actions");
        };

        let (row, col) = self.pos;
        let mut terminated = false;

        match a {
            0 => {
                // south
                if row < GRID_SIZE - 1 {
                    self.pos.0 = row + 1;
                }
            }
            1 => {
                // north
                if row > 0 {
                    self.pos.0 = row - 1;
                }
            }
            2 => {
                // east
                if col < GRID_SIZE - 1 && !wall_between(row, col, col + 1) {
                    self.pos.1 = col + 1;
                }
            }
            3 => {
                // west
                if col > 0 && !wall_between(row, col - 1, col) {
                    self.pos.1 = col - 1;
                }
            }
            4 => {
                // pickup
                if self.passenger < NUM_LOCS && self.pos == LOCS[self.passenger] {
                    self.passenger = NUM_LOCS; // now in taxi
                }
            }
            5 => {
                // dropoff
                if self.passenger == NUM_LOCS && self.pos == LOCS[self.destination] {
                    // Successful delivery
                    self.passenger = self.destination;
                    terminated = true;
                }
            }
            _ => {}
        }

        self.step_count += 1;
        if terminated || self.step_count >= self.max_steps {
            self.reset();
        }

        self.update_homeo();
        StepResult {
            observation: self.observe(),
            homeostatic: self.homeo.clone(),
        }
    }

    fn reset(&mut self) {
        // Cycle through (passenger, destination, starting_position) tuples
        // deterministically. 4 passenger locs × 4 destinations × 25 positions
        // = 400 combinations, but we skip passenger==destination.
        self.reset_counter = self.reset_counter.wrapping_add(1);
        let n = self.reset_counter;

        let pass = n % NUM_LOCS;
        let mut dest = (n / NUM_LOCS) % NUM_LOCS;
        if dest == pass {
            dest = (dest + 1) % NUM_LOCS;
        }
        let pos_idx = (n / (NUM_LOCS * NUM_LOCS)) % (GRID_SIZE * GRID_SIZE);

        self.pos = (pos_idx / GRID_SIZE, pos_idx % GRID_SIZE);
        self.passenger = pass;
        self.destination = dest;
        self.step_count = 0;
        self.update_homeo();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxi_basics() {
        let env = Taxi::new();
        assert_eq!(env.observation_dim(), OBS_DIM);
        assert_eq!(env.num_actions(), NUM_ACTIONS);
        // Observation should have 3 hot bits (pos, passenger, destination)
        let obs = env.observe();
        let hot: f32 = obs.data.iter().sum();
        assert!((hot - 3.0).abs() < 1e-5);
    }

    #[test]
    fn taxi_walls_block_movement() {
        let mut env = Taxi::new();
        // Place taxi at (0, 1), try to move east — wall between col 1 and col 2
        env.pos = (0, 1);
        env.step(&Action::Discrete(2)); // east
        assert_eq!(env.pos, (0, 1), "should be blocked by wall");

        // (0, 0) east should succeed (no wall between 0 and 1)
        env.pos = (0, 0);
        env.step(&Action::Discrete(2));
        assert_eq!(env.pos, (0, 1));
    }

    #[test]
    fn taxi_pickup_and_dropoff() {
        let mut env = Taxi::new();
        env.pos = LOCS[0]; // passenger starts at loc 0
        env.passenger = 0;
        env.destination = 1;

        // Pickup
        env.step(&Action::Discrete(4));
        assert_eq!(env.passenger, NUM_LOCS, "passenger should be in taxi");

        // Move to destination (loc 1 = (0,4))
        env.pos = LOCS[1];
        env.step(&Action::Discrete(5)); // dropoff
        // After dropoff, episode terminates and env resets
        assert_eq!(env.step_count, 0);
    }

    #[test]
    fn taxi_illegal_pickup_is_noop() {
        let mut env = Taxi::new();
        env.pos = (2, 2); // not at passenger location
        env.passenger = 0;
        env.step(&Action::Discrete(4)); // pickup
        assert_eq!(env.passenger, 0, "passenger should not be picked up");
    }
}
