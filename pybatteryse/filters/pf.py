"""Implementation of Particle Filter"""

import numpy as np

from tqdm import tqdm

from pybatteryid.dataclasses import Model

from ..statespace import get_matrices, StateSpace


# pylint: disable=R0902
class ParticleFilter:
    """
    Particle Filter for battery state estimation with current sensor bias.

    Attributes:
        num_particles: Number of particles
        particles: State particles (num_particles, state_dim)
        weights: Normalised particle weights (num_particles,)
        eta_particles: Current sensor bias estimates per particle (num_particles,)
        statespace: StateSpace representation of the battery model
    """

    # Class variables
    num_particles: int

    eta_bound_min: float
    eta_bound_max: float
    eta_particles: np.ndarray

    sigma_ny_ne: float

    statespace: StateSpace
    emf_function: callable
    state_dim: int

    matrix_a_stack: np.ndarray
    matrix_b_stack: np.ndarray
    matrix_c_stack: np.ndarray
    matrix_d_stack: np.ndarray

    particles: np.ndarray
    weights: np.ndarray

    def __init__(self, model: Model, num_particles: int, eta_bounds: tuple[float, float],
                 sigma_ny_ne: float):
        """
        Initialise particle filter.

        Args:
            model: Battery model (Model dataclass) to create StateSpace from
            num_particles: Number of particles to use
            eta_bounds: Tuple of (min, max) for current sensor bias
            sigma_ny_ne: Measurement noise variance (scalar)
        """
        self.num_particles = num_particles
        self.eta_bound_min, self.eta_bound_max = eta_bounds
        self.sigma_ny_ne = sigma_ny_ne

        # Create state space representation from model
        self.statespace = StateSpace(model)
        self.emf_function = model.emf_function
        self.state_dim = model.model_order + 1

        # Preallocate particle attributes
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.eta_particles = np.zeros(self.num_particles)

        # Preallocate matrices
        self.matrix_a_stack = np.zeros((self.num_particles, self.state_dim, self.state_dim))
        self.matrix_b_stack = np.zeros((self.num_particles, self.state_dim))
        self.matrix_c_stack = np.zeros((self.num_particles, self.state_dim))
        self.matrix_d_stack = np.zeros(self.num_particles)

        # Will be initialised when run() is called
        self.particles = None


    # pylint: disable=R0913, R0914, R0917
    def step(self, previous_temperature: float, previous_input: float, current_temperature: float,
             current_input: float, current_voltage: float):
        """
        Execute one prediction-update cycle.

        Args:
            previous_temperature: Temperature at time k-1
            previous_input: Current at time k-1
            current_temperature: Temperature at time k
            current_input: Current at time k
            current_voltage: Voltage measurement at time k
        """
        # Compute bias-corrected currents
        previous_input_true = previous_input - self.eta_particles

        # Propagation step: compute A, B matrices
        invalid_particles = self.compute_propagation_matrices(previous_input_true,
                                                              previous_temperature)

        # Propagate particles (no process noise)
        particles_next = (np.einsum('nij,nj->ni', self.matrix_a_stack, self.particles) +
                          self.matrix_b_stack * previous_input_true[:, None])

        soc_next = particles_next[:, 0]

        # Sample new current sensor bias for time k
        eta_k = np.random.uniform(self.eta_bound_min, self.eta_bound_max, size=self.num_particles)
        current_input_true = current_input - eta_k

        # Measurement update: compute C, D matrices
        invalid_particles |= self.compute_measurement_matrices(soc_next,
                                                               current_input_true,
                                                               current_temperature,
                                                               invalid_particles)

        # Invalidate out-of-bounds SOC
        invalid_particles |= (soc_next < 0) | (soc_next > 1)
        valid_particles = ~invalid_particles

        # Compute weights using vectorised likelihood
        weights_unnormalised = self.compute_weights(current_voltage, soc_next, particles_next,
                                                    current_input_true, valid_particles)

        # Normalise and resample
        weights_normalised = self.normalise_weights(weights_unnormalised)
        resampled_indices = np.random.choice(self.num_particles, size=self.num_particles,
                                             p=weights_normalised, replace=True)

        # Update particle set
        self.particles = particles_next[resampled_indices]
        self.weights[:] = 1.0 / self.num_particles
        self.eta_particles = eta_k[resampled_indices]


    def compute_propagation_matrices(self, previous_input_true, previous_temperature):
        """Compute A and B matrices for all particles. Returns invalid mask."""
        #
        invalid_particles = np.zeros(self.num_particles, dtype=bool)

        for particle_idx in range(self.num_particles):
            try:
                matrix_a, matrix_b, _, _ = (mat.squeeze() for mat in
                             get_matrices(self.statespace,
                                          self.particles[particle_idx, 0],
                                          previous_input_true[particle_idx],
                                          previous_temperature))
                self.matrix_a_stack[particle_idx] = matrix_a
                self.matrix_b_stack[particle_idx] = matrix_b
            except ValueError:
                invalid_particles[particle_idx] = True

        return invalid_particles


    def compute_measurement_matrices(self, soc_next, current_input_true, current_temperature,
                                     invalid_particles):
        """Compute C and D matrices for all valid particles. Returns updated invalid mask."""
        #
        for particle_idx in range(self.num_particles):
            if invalid_particles[particle_idx]:
                continue
            try:
                _, _, matrix_c, matrix_d = (mat.squeeze() for mat in
                             get_matrices(self.statespace, soc_next[particle_idx],
                                        current_input_true[particle_idx], current_temperature))
                self.matrix_c_stack[particle_idx] = matrix_c
                self.matrix_d_stack[particle_idx] = matrix_d
            except ValueError:
                invalid_particles[particle_idx] = True

        return invalid_particles

    # pylint: disable=R0913, R0917
    def compute_weights(self, current_voltage, soc_next, particles_next, current_input_true,
                        valid_particles):
        """Compute unnormalised weights using Gaussian likelihood."""
        #
        weights_unnormalised = np.zeros(self.num_particles)

        if np.any(valid_particles):
            # Vectorised measurement prediction
            emf_vals = self.emf_function(soc_next[valid_particles])
            voltage_pred = (emf_vals + np.einsum('ni,ni->n', self.matrix_c_stack[valid_particles],
                                                 particles_next[valid_particles]) +
                                                 self.matrix_d_stack[valid_particles] *
                                                 current_input_true[valid_particles])

            # Vectorised Gaussian likelihood
            measurement_residuals = current_voltage - voltage_pred
            weights_unnormalised[valid_particles] = (
                np.exp(-0.5 * measurement_residuals**2 / self.sigma_ny_ne) /
                np.sqrt(2 * np.pi * self.sigma_ny_ne)
            )

        return weights_unnormalised


    def normalise_weights(self, weights_unnormalised):
        """Normalise weights with degeneracy check."""
        weight_sum = np.sum(weights_unnormalised)
        if weight_sum == 0:
            raise ValueError("Particle filter degeneracy: all weights zero")
        return weights_unnormalised / weight_sum


    def estimate(self):
        """Return weighted mean state estimate."""
        return np.average(self.particles, axis=0, weights=self.weights)


    def run(self, initial_particles: np.ndarray, temperature_values: np.ndarray,
            current_values: np.ndarray, voltage_values: np.ndarray):
        """
        Run particle filter over entire measurement sequence.

        Args:
            initial_particles: Initial particle states array (num_particles, state_dim)
            temperature_values: Temperature sequence (num_timesteps,)
            current_values: Current measurements (num_timesteps,)
            voltage_values: Voltage measurements (num_timesteps,)

        Returns:
            estimates: State estimates at each time step (num_timesteps, state_dim)
        """
        # Validate initial_particles shape
        if initial_particles.shape[0] != self.num_particles:
            raise ValueError(
                f"initial_particles has {initial_particles.shape[0]} particles, "
                f"but filter was initialised with num_particles={self.num_particles}"
            )
        if initial_particles.shape[1] != self.state_dim:
            raise ValueError(
                f"initial_particles has state dimension {initial_particles.shape[1]}, "
                f"but filter expects state_dim={self.state_dim} (model_order + 1)"
            )

        # Initialise particles
        self.particles = initial_particles

        # Reset weights and eta_particles to initial state
        self.weights[:] = 1.0 / self.num_particles
        self.eta_particles[:] = 0.0

        # Get time horizon
        num_timesteps = min(len(voltage_values), len(current_values), len(temperature_values))

        state_estimates = np.zeros((num_timesteps, self.state_dim))
        state_estimates[0] = self.estimate()

        for k in tqdm(range(1, num_timesteps), desc="PF Progress", unit="step", ncols=100):
            try:
                self.step(temperature_values[k-1], current_values[k-1],
                         temperature_values[k], current_values[k],
                         voltage_values[k])
            except ValueError as e:
                raise RuntimeError(
                    f"Particle filter failed at time step {k}: {str(e)}"
                ) from e
            state_estimates[k] = self.estimate()

        return state_estimates
