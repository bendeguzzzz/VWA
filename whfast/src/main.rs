//TODO: Optimize the code, use cache and precalculated vecs
// + remove new instance of vec for each iteration,
// use util functions instead, seperate the codebase for more visibility,
//
// also use buffer for often used stuff between iterations

use std::f64;

/// SOURCE: Rein & Tamayo 2015 (https://arxiv.org/pdf/1506.01084)
use nalgebra::Vector3;
mod stumpff;
use num_traits::pow::Pow;
/// Represents an N-body system with masses, positions, and velocities
#[derive(Debug)]
struct System {
    masses: Vec<f64>,
    positions: Vec<Vector3<f64>>,
    velocities: Vec<Vector3<f64>>,
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

        let mut r = masses[0] * vec[0];

        for i in 1..n {
            let r_prime_i = vec[i] - r / cumulative[i - 1];
            r = r * (1.0 + masses[i] / cumulative[i - 1]) + masses[i] * r_prime_i;
        }

        vec[0] = r / cumulative[n - 1]; // Center of mass
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
    pub fn _transform_velocities_from_cartesian_to_jacobi(&mut self) {
        Self::transform_cartesian_to_jacobi(
            &mut self.velocities,
            &mut self.masses,
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
    pub fn _transform_velocities_from_jacobi_to_cartesian(&mut self) {
        Self::transform_jacobi_to_cartesian(
            &mut self.velocities,
            &self.masses,
            &self.cumulative_masses,
        );
    }

    // Accessor Methods

    pub fn _positions(&self) -> &[Vector3<f64>] {
        &self.positions
    }

    pub fn _velocities(&self) -> &[Vector3<f64>] {
        &self.velocities
    }

    pub fn _masses(&self) -> &[f64] {
        &self.masses
    }

    /// G-functions (Stiefel & Scheifele 1971)
    pub fn stiefel_scheifele(&self, n: u8, beta: f64, x: f64) -> f64 {
        x.pow(n) * stumpff::stumpff(beta * x.pow(2), n)
    }
    /// We are following the notation of the paper
    pub fn h_kepler(&mut self, dt: f64, output: bool) {
        println!("h_kepler called with dt={}, output={}", dt, output);
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
            println!("Initial guess: {}", x);
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
                println!("{}", x);

                if (x - x_prev1).abs() <= f64::EPSILON || (x - x_prev2).abs() <= f64::EPSILON {
                    println!("Difference is smaller than f64::EPSILON");
                    break;
                }
                //println!("{}", x);
            }
            println!("Finished newton's method");
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
            let a = sum_until_i * self.positions[i] / self.positions[i].norm().powi(3);
            self.velocities[i] += a * dt;
        }
    }
    pub fn h_interaction_2(&self) -> Vec<Vector3<f64>> {
        let mut accelerations: Vec<Vector3<f64>> = Vec::with_capacity(self.n);
        let mut accelerations_jacobi: Vec<Vector3<f64>> = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut a: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
            for j in i + 1..self.n {
                if j == 1 {
                    continue; // exclude j = 1
                }
                let r_diff = self.positions[i] - self.positions[j];
                a += self.masses[j] * r_diff / r_diff.norm().powi(3);
            }
            accelerations.push(a);
        }
        let mut a_total: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
        let mut m_total_until_i = 0.0;
        for i in 0..self.n {
            a_total += self.masses[i] * accelerations[0];
            m_total_until_i += self.masses[i];
        }
        accelerations_jacobi.push(accelerations[0] - a_total / m_total_until_i);
        for i in 1..self.n {
            let mut m_total_until_i = 0.0;
            let mut f_j: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
            for j in 0..i {
                m_total_until_i += self.masses[j];
                f_j += self.masses[j] * accelerations[j];
            }
            accelerations_jacobi.push(accelerations[i] - f_j / m_total_until_i);
        }

        //convert to jacobi

        accelerations_jacobi
    }

    pub fn dkd(&mut self, dt: f64, mut t: f64, t_max: f64, _output_t: f64) {
        println!("Starting DKD: t={}, t_max={}, dt={}", t, t_max, dt);
        self.transform_coordinates_from_cartesian_to_jacobi();
        self.h_kepler(dt / 2.0, false);

        let mut iteration = 0;
        while t < t_max {
            println!("Iteration {}: t={}", iteration, t);
            iteration += 1;

            if iteration > 1000 {
                panic!("Too many iterations!");
            }

            self.h_interaction_1(dt);
            self.transform_coordinates_from_jacobi_to_cartesian();
            let accelerations = self.h_interaction_2();
            for i in 0..self.n {
                self.velocities[i] += accelerations[i] * dt;
            }
            self.transform_coordinates_from_cartesian_to_jacobi();

            if t + dt < t_max {
                self.h_kepler(dt, false);
            }

            t += dt;
            println!("After increment: t={}", t);
        }

        println!("Exiting loop");
        self.h_kepler(dt / 2.0, true);
        self.transform_coordinates_from_jacobi_to_cartesian();
    }
}

fn main() {
    let m_0 = 1.0;
    let m_1 = 0.000003;
    let m: Vec<f64> = vec![m_0, m_1];

    let r_0: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
    let r_1: Vector3<f64> = Vector3::new(1.0, 0.0, 0.0);
    let r: Vec<Vector3<f64>> = vec![r_0, r_1];

    let v_0: Vector3<f64> = Vector3::new(0.0, 0.0, 0.0);
    let v_1: Vector3<f64> = Vector3::new(0.0, 1.0, 0.0);
    let v: Vec<Vector3<f64>> = vec![v_0, v_1];

    let t = 0.0;
    let dt = 0.01; // Smaller timestep for better accuracy
    let t_max = 6.283185307179586; // One full orbit (2π)

    let mut system = System::new(m, r, v).unwrap();

    // Calculate initial energy
    let initial_energy = calculate_energy(&system);
    println!("Initial energy: {}", initial_energy);
    println!("Initial position: {:?}", system.positions[1]);
    println!("Initial velocity: {:?}", system.velocities[1]);

    system.dkd(dt, t, t_max, 0.0);

    // Calculate final energy
    let final_energy = calculate_energy(&system);
    println!("\nFinal energy: {}", final_energy);
    println!("Final position: {:?}", system.positions[1]);
    println!("Final velocity: {:?}", system.velocities[1]);
    println!(
        "Energy error: {:.2e}",
        (final_energy - initial_energy).abs() / initial_energy.abs()
    );
    println!("Distance from origin: {}", system.positions[1].norm());
}

fn calculate_energy(system: &System) -> f64 {
    let mut kinetic = 0.0;
    let mut potential = 0.0;

    for i in 0..system.n {
        kinetic += 0.5 * system.masses[i] * system.velocities[i].norm_squared();
    }

    for i in 0..system.n {
        for j in i + 1..system.n {
            let r_diff = system.positions[i] - system.positions[j];
            potential -= system.masses[i] * system.masses[j] / r_diff.norm();
        }
    }

    kinetic + potential
}
