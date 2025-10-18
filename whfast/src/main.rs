use nalgebra::Vector3;
mod stumpff;
use num_traits::pow::Pow;
/// Represents an N-body system with masses, positions, and velocities
#[derive(Debug)]
struct System {
    masses: Vec<f64>,
    positions: Vec<Vector3<f64>>,
    velocities: Vec<Vector3<f64>>,
    t: f64,
    dt: f64,
    t_max: f64,
    cumulative_masses: Vec<f64>,
    n: usize,
}

#[derive(Debug)]
pub enum SystemError {
    EmptySystem,
    InconsistentLengths,
    ZeroOrNegativeMass(usize),
}

impl std::fmt::Display for SystemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SystemError::EmptySystem => write!(f, "System must contain at least one body"),
            SystemError::InconsistentLengths => {
                write!(
                    f,
                    "Masses, positions, and velocities must have the same length"
                )
            }
            SystemError::ZeroOrNegativeMass(i) => {
                write!(f, "Mass at index {} must be positive", i)
            }
        }
    }
}

impl std::error::Error for SystemError {}

impl System {
    pub fn new(
        masses: Vec<f64>,
        positions: Vec<Vector3<f64>>,
        velocities: Vec<Vector3<f64>>,
        t: f64,
        dt: f64,
        t_max: f64,
    ) -> Result<Self, SystemError> {
        if masses.is_empty() {
            return Err(SystemError::EmptySystem);
        }

        let n = masses.len();

        if positions.len() != n || velocities.len() != n {
            return Err(SystemError::InconsistentLengths);
        }

        for (i, &m) in masses.iter().enumerate() {
            if m <= 0.0 {
                return Err(SystemError::ZeroOrNegativeMass(i));
            }
        }

        let mut cumulative_masses = Vec::with_capacity(n);
        let mut sum = 0.0;
        for &m in &masses {
            sum += m;
            cumulative_masses.push(sum);
        }

        Ok(System {
            masses,
            positions,
            velocities,
            t,
            dt,
            t_max,
            cumulative_masses,
            n,
        })
    }

    fn transform_cartesian_to_jacobi(
        vec: &mut Vec<Vector3<f64>>,
        masses: &[f64],
        cumulative: &[f64],
    ) {
        let n = vec.len();

        if n == 0 {
            return;
        }

        for i in 0..n {
            let mut r = masses[i] * vec[i];

            for j in (i + 1)..n {
                let cumulative_prev = cumulative[j - 1];

                if cumulative_prev.abs() < f64::EPSILON {
                    continue;
                }

                let r_prime_j = vec[j] - r / cumulative_prev;
                r = r * (1.0 + masses[j] / cumulative_prev) + masses[j] * r_prime_j;
            }

            let total_mass = cumulative[n - 1];
            if total_mass.abs() > f64::EPSILON {
                vec[i] = r / total_mass;
            }
        }
    }

    /// Generic helper function to transform vectors from Jacobi to Cartesian coordinates
    fn transform_jacobi_to_cartesian(
        vec: &mut Vec<Vector3<f64>>,
        masses: &[f64],
        cumulative: &[f64],
    ) {
        let n = vec.len();

        if n == 0 {
            return;
        }

        let mut r = vec[0] * cumulative[n - 1];

        for i in (1..n).rev() {
            let cumulative_prev = cumulative[i - 1];

            if cumulative_prev.abs() > f64::EPSILON {
                r = (r - masses[i] * vec[i]) / cumulative_prev;
                vec[i] = vec[i] + r;
                r = r * cumulative_prev;
            }
        }

        if masses[0].abs() > f64::EPSILON {
            vec[0] = r / masses[0];
        }
    }

    /// Public API Methods

    /// Transform position coordinates from Cartesian to Jacobi coordinates
    pub fn transform_coordinates_from_cartesian_to_jacobi(&mut self) {
        Self::transform_cartesian_to_jacobi(
            &mut self.positions,
            &self.masses,
            &self.cumulative_masses,
        );
    }

    /// Transform velocity coordinates from Cartesian to Jacobi coordinates
    pub fn transform_velocities_from_cartesian_to_jacobi(&mut self) {
        Self::transform_cartesian_to_jacobi(
            &mut self.velocities,
            &self.masses,
            &self.cumulative_masses,
        );
    }

    /// Transform position coordinates from Jacobi to Cartesian coordinates
    pub fn transform_coordinates_from_jacobi_to_cartesian(&mut self) {
        Self::transform_jacobi_to_cartesian(
            &mut self.positions,
            &self.masses,
            &self.cumulative_masses,
        );
    }

    /// Transform velocity coordinates from Jacobi to Cartesian coordinates
    pub fn transform_velocities_from_jacobi_to_cartesian(&mut self) {
        Self::transform_jacobi_to_cartesian(
            &mut self.velocities,
            &self.masses,
            &self.cumulative_masses,
        );
    }

    // Accessor Methods

    pub fn positions(&self) -> &[Vector3<f64>] {
        &self.positions
    }

    pub fn velocities(&self) -> &[Vector3<f64>] {
        &self.velocities
    }

    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    pub fn time(&self) -> f64 {
        self.t
    }
    /// G-functions (Stiefel & Scheifele 1971)
    pub fn stiefel_scheifele(&self, n: u8, beta: f64, x: f64) -> f64 {
        x.pow(n) * stumpff::stumpff(beta * x.pow(2), n)
    }
    /// We are following the notation of the paper
    pub fn h_kepler(&mut self, dt: f64, output: bool) {
        for i in 1..self.n {
            // loop through all of the bodies
            // note this method takes a dt input, so when calling this function we should give dt/2 since it also updates the positions
            let m_total = self.masses[0] + self.masses[i];
            let r_0 = self.positions[i];
            let r_0_magnitude = r_0.norm();
            let r_0_magnitude_sq = r_0.norm_squared();
            let v_0 = self.velocities[i];
            let v_0_magnitude_sq = v_0.norm_squared(); // we only need the squared norm
            let beta = 2.0 * m_total / r_0_magnitude - v_0_magnitude_sq;
            let eta_0 = r_0.dot(&v_0); // dot product
            let zeta_0 = m_total - beta * r_0_magnitude;
            let eta = eta_0 * dt / r_0_magnitude_sq;
            let mut x: f64 = (dt / r_0_magnitude) * (1.0 - 0.5 * eta);
            let mut x_prev1 = 0.0;
            loop {
                //Newton's method
                let x_prev2 = x_prev1;
                x_prev1 = x;
                x = (x
                    * (eta_0 * self.stiefel_scheifele(1, beta, x)
                        + zeta_0 * self.stiefel_scheifele(2, beta, x))
                    - eta_0 * self.stiefel_scheifele(2, beta, x)
                    - zeta_0 * self.stiefel_scheifele(3, beta, x)
                    + dt)
                    / (r_0_magnitude
                        + eta_0 * self.stiefel_scheifele(1, beta, x)
                        + zeta_0 * self.stiefel_scheifele(2, beta, x));
                if x == x_prev1 || x == x_prev2 {
                    break;
                }
            }
            let r = r_0_magnitude
                + eta_0 * self.stiefel_scheifele(1, beta, x)
                + zeta_0 * self.stiefel_scheifele(2, beta, x);
            let f_hat = -m_total * self.stiefel_scheifele(2, beta, x) / r_0_magnitude;
            let g = dt - m_total * self.stiefel_scheifele(3, beta, x);
            self.positions[i] = (f_hat * r_0 + g * v_0) + r_0;
            if output {
                let f_derivative =
                    -m_total * self.stiefel_scheifele(1, beta, x) / (r_0_magnitude * r);
                let g_hat_derivative = -m_total * self.stiefel_scheifele(2, beta, x) / r;
                self.velocities[i] = (f_derivative * r_0 + g_hat_derivative * v_0) + v_0;
            }
        }
    }
    pub fn h_interaction_1(&mut self, dt: f64) {
        for i in 2..self.n {
            let mut sum_until_i = 0.0;
            for j in 0..i {
                sum_until_i += self.masses[j];
            }
            let a = sum_until_i * self.positions[i]
                / (self.positions[i].norm_squared() * self.positions[i].norm());
            self.velocities[i] += a * dt;
        }
    }
    pub fn h_interaction_2(&self) -> Vec<f64> {
        let mut accelerations: Vec<f64> = Vec::new();
        for i in 0..self.n {
            for j in i + 1..self.n {
                if j == 1 {
                    break; // exclude j = 1
                }
            }
        }

        Vec::new()
    }

    pub fn update(&self) {}
}

fn main() {
    print!("Hello, world!")
}
