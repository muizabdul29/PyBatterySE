"""Implementation of Particle Filter."""

import numpy as np

from tqdm import tqdm

from ..statespace import StateSpace, SocSource


# pylint: disable=R0902
class ParticleFilter:
    """Bootstrap Particle Filter for battery state estimation with current
    sensor bias.

    Operates on a :class:`pybatteryse.statespace.StateSpace` instance and
    propagates a weighted particle cloud through the state-transition and
    measurement functions ``f`` and ``h`` defined by that statespace. Each
    step has three stages: (1) propagate every particle via
    ``evaluate_next_state`` using a bias-corrected previous current, then
    add Gaussian process noise to any random-walk entries; (2) sample a
    fresh current-sensor bias per particle, bias-correct the present
    current, and score particles with a Gaussian likelihood on the voltage
    residual via ``evaluate_predicted_measurement``; (3) multinomially
    resample particles (and their bias values) according to the normalised
    weights. Before each statespace call the filter syncs model parameters
    and capacity from the particle's random-walk entries via
    ``StateSpace.update_model_from_state``, so each particle's
    linearization reflects its own belief about those quantities.

    The current-sensor bias ``eta`` is modelled as a latent variable
    sampled fresh each step from ``U(eta_bound_min, eta_bound_max)``; its
    resampled values are carried alongside particles so the posterior over
    bias informs subsequent steps. Random-walk dynamics for extended
    states (``theta_{i}``, capacity) contribute Gaussian process noise
    parameterized by ``variance_eta_theta`` and ``variance_eta_capacity``;
    the ``s`` and ``overpotentials`` blocks are deterministic in
    propagation. Particles whose state contains NaN / inf after
    propagation, or whose statespace call raises, are flagged invalid and
    receive zero weight; resampling removes them implicitly.

    Attributes:
        statespace: StateSpace representation of the battery model.
        num_particles: Number of particles in the cloud.
        eta_bound_min, eta_bound_max: Bounds of the uniform distribution
            from which the current-sensor bias ``eta`` is sampled each
            step.
        variance_eta_y_e: Variance of the Gaussian voltage-measurement
            noise used in the likelihood.
        variance_eta_theta: Variance of the random-walk process noise on
            ``theta_{i}`` states. Required when any ``theta_{i}`` is in
            state_components, ``None`` otherwise.
        variance_eta_capacity: Variance of the random-walk process noise
            on the ``capacity`` state. Required when ``capacity`` is in
            state_components, ``None`` otherwise.
        particles: Current particle states, shape
            ``(num_particles, state_dimension)``. ``None`` until
            ``run`` is called. Holds the post-resample cloud after each
            ``step``.
        weights: Normalised particle weights from the most recent
            measurement update (pre-resample), shape
            ``(num_particles,)``. Initialised uniform; refreshed by
            ``step``. These index the propagated particle cloud at the
            update, not the post-resample ``particles`` field.
        eta_particles: Current-sensor-bias estimates per particle, shape
            ``(num_particles,)``, carried through resampling.
    """

    statespace: StateSpace

    variance_eta_y_e: float
    variance_eta_theta: float | None
    variance_eta_capacity: float | None

    # Number of particles
    num_particles: int

    # Input uncertainty bounds
    eta_bound_min: float
    eta_bound_max: float

    particles: np.ndarray
    weights: np.ndarray
    eta_particles: np.ndarray

    # pylint: disable=R0913, R0917
    def __init__(self,
                 statespace: StateSpace,
                 num_particles: int,
                 eta_bounds: tuple[float, float],
                 variance_eta_y_e: float,
                 variance_eta_theta: float | None = None,
                 variance_eta_capacity: float | None = None):
        """
        Initialise particle filter.

        Args:
            statespace: Pre-constructed StateSpace for the battery model.
                Its state_components determines the state-vector layout
                used by the filter.
            num_particles: Number of particles to use.
            eta_bounds: Tuple of (min, max) for current sensor bias.
            variance_eta_y_e: Measurement noise variance (scalar).
            variance_eta_theta: Process noise variance for random-walk
                theta parameters. Must be provided (non-None) when any
                'theta_{i}' appears in statespace.state_components; pass
                None otherwise.
            variance_eta_capacity: Process noise variance for the
                random-walk capacity state. Must be provided (non-None)
                when 'capacity' appears in statespace.state_components;
                pass None otherwise.
        """
        self.statespace = statespace
        self.num_particles = num_particles
        self.eta_bound_min, self.eta_bound_max = eta_bounds
        self.variance_eta_y_e = variance_eta_y_e
        self.variance_eta_theta = variance_eta_theta
        self.variance_eta_capacity = variance_eta_capacity

        self._validate_variances()

        # Per-state-entry process-noise std-devs; zero for deterministic blocks.
        self._process_noise_std = self._build_process_noise_std_diag()

        # Preallocate particle attributes
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.eta_particles = np.zeros(self.num_particles)

        # Will be initialised when run() is called
        self.particles = None


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def _validate_variances(self) -> None:
        """Non-negativity for all provided variances; presence gated on state_components."""
        def check_nonnegative(name: str, value: float) -> None:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

        check_nonnegative("variance_eta_y_e", self.variance_eta_y_e)

        components = self.statespace.state_components
        optional = [
            ("variance_eta_theta",    self.variance_eta_theta,
             any(c.startswith("theta_") for c in components), "any 'theta_*' entry"),
            ("variance_eta_capacity", self.variance_eta_capacity,
             "capacity" in components, "'capacity'"),
        ]
        for name, value, required, context in optional:
            if required:
                if value is None:
                    raise ValueError(
                        f"{name} is required when state_components contains {context}."
                    )
                check_nonnegative(name, value)


    def _build_process_noise_std_diag(self) -> np.ndarray:
        """Length-state_dimension vector of random-walk process-noise std-devs.

        Deterministic blocks ('s', 'overpotentials') get zero noise;
        random-walk blocks ('theta_{i}', 'capacity') get the user-supplied
        std-dev (sqrt of variance).
        """
        std = np.zeros(self.statespace.state_dimension)
        offset = 0
        # pylint: disable=protected-access
        block_sizes = self.statespace._block_sizes
        for component in self.statespace.state_components:
            size = block_sizes[component]
            if component.startswith('theta_'):
                std[offset:offset + size] = np.sqrt(self.variance_eta_theta)
            elif component == 'capacity':
                std[offset:offset + size] = np.sqrt(self.variance_eta_capacity)
            # 's' and 'overpotentials' stay zero.
            offset += size
        return std


    # ----- main PF step -----


    # pylint: disable=R0913, R0914, R0917
    def step(self, previous_temperature: float, previous_input: float, present_temperature: float,
             present_input: float, present_voltage: float,
             previous_soc: float | None = None,
             present_soc: float | None = None):
        """
        Execute one prediction-update cycle.

        After this call, ``self.weights`` holds the normalised
        per-particle weights from the measurement update (pre-resample),
        which index the propagated particle cloud at this step rather
        than the resampled ``self.particles``. Useful for diagnostics
        like effective sample size or weight degeneracy.

        Args:
            previous_temperature: Temperature at time k-1 (may be None for
                temperature-independent models).
            previous_input: Current at time k-1.
            present_temperature: Temperature at time k (may be None).
            present_input: Current at time k.
            present_voltage: Voltage measurement at time k.
            previous_soc: SOC at time k-1, used during propagation.
                Required when the underlying StateSpace uses exogenous
                SOC (i.e. 's' is not in state_components); ignored
                otherwise.
            present_soc: SOC at time k, used during the measurement
                update. Required when the StateSpace uses exogenous SOC;
                ignored otherwise.
        """
        # pylint: disable=protected-access
        if self.statespace._soc_source is not SocSource.STATE \
                and (previous_soc is None or present_soc is None):
            raise ValueError(
                "previous_soc and present_soc are required when the StateSpace "
                "uses SocSource.EXOGENOUS ('s' not in state_components)."
            )
        # Bias-correct the previous-step current for each particle
        previous_input_true = previous_input - self.eta_particles

        # Propagate each particle through f(x, i, T, soc)
        particles_next, invalid_particles = self._propagate_particles(
            previous_input_true, previous_temperature, previous_soc
        )

        # Add process noise to random-walk components
        if np.any(self._process_noise_std > 0):
            noise = np.random.randn(self.num_particles, self.statespace.state_dimension) \
                * self._process_noise_std[None, :]
            particles_next = particles_next + noise

        # Flag any particle whose state contains NaN or inf after
        # propagation + noise. Out-of-domain SOC, divergent dynamics,
        # or similar pathologies typically surface this way.
        invalid_particles |= ~np.all(np.isfinite(particles_next), axis=1)

        # Sample fresh current sensor bias for time k
        eta_k = np.random.uniform(self.eta_bound_min, self.eta_bound_max,
                                  size=self.num_particles)
        present_input_true = present_input - eta_k

        # Compute predicted voltage per particle via h(x, i, T, soc).
        # Particles whose statespace calls raise (e.g. out-of-domain SOC)
        # are caught and flagged inside _compute_weights.
        valid_particles = ~invalid_particles
        weights_unnormalised = self._compute_weights(
            present_voltage, particles_next, present_input_true,
            present_temperature, present_soc, valid_particles
        )

        # Normalise and resample
        weights_normalised = self._normalise_weights(weights_unnormalised)
        resampled_indices = np.random.choice(
            self.num_particles, size=self.num_particles,
            p=weights_normalised, replace=True
        )

        # Update particle set. self.weights holds the pre-resample
        # normalised weights for diagnostics; self.particles is the
        # post-resample cloud.
        self.particles = particles_next[resampled_indices]
        self.weights = weights_normalised
        self.eta_particles = eta_k[resampled_indices]


    # ----- propagation -----


    def _propagate_particles(self, previous_input_true: np.ndarray,
                             previous_temperature: float | None,
                             previous_soc: float | None):
        """Run f(x, i, T, soc) for every particle; return (next_states, invalid_mask).

        previous_soc is forwarded to StateSpace.evaluate_next_state only
        when the StateSpace uses exogenous SOC; otherwise it is ignored
        by the StateSpace. Passing None in STATE-SOC mode is fine.
        """
        particles_next = np.empty_like(self.particles)
        invalid_particles = np.zeros(self.num_particles, dtype=bool)

        for particle_idx in range(self.num_particles):
            particle_state = self.particles[particle_idx]
            try:
                # Sync the shared statespace's model parameters to this
                # particle's random-walk entries (theta_*, capacity).
                # No-op when neither is in state_components.
                self.statespace.update_model_from_state(particle_state)

                particles_next[particle_idx] = self.statespace.evaluate_next_state(
                    particle_state,
                    current_value=previous_input_true[particle_idx],
                    temperature_value=previous_temperature,
                    soc_value=previous_soc,
                )
            except (ValueError, KeyError, IndexError):
                invalid_particles[particle_idx] = True
                particles_next[particle_idx] = particle_state  # placeholder

        return particles_next, invalid_particles


    # ----- measurement likelihood -----


    # pylint: disable=R0913, R0917
    def _compute_weights(self, present_voltage: float, particles_next: np.ndarray,
                         present_input_true: np.ndarray,
                         present_temperature: float | None,
                         present_soc: float | None,
                         valid_particles: np.ndarray) -> np.ndarray:
        """Compute unnormalised Gaussian likelihood weights per particle.

        present_soc is forwarded to StateSpace.evaluate_predicted_measurement
        only when the StateSpace uses exogenous SOC; otherwise it is ignored
        by the StateSpace.
        """
        weights_unnormalised = np.zeros(self.num_particles)

        if not np.any(valid_particles):
            return weights_unnormalised

        # Local copy so failures during measurement eval don't leak back
        # into the caller's invalid_particles mask.
        valid_local = valid_particles.copy()

        voltage_pred = np.zeros(self.num_particles)
        for particle_idx in np.flatnonzero(valid_local):
            particle_state = particles_next[particle_idx]
            try:
                self.statespace.update_model_from_state(particle_state)
                voltage_pred[particle_idx] = self.statespace.evaluate_predicted_measurement(
                    particle_state,
                    current_value=present_input_true[particle_idx],
                    temperature_value=present_temperature,
                    soc_value=present_soc,
                )
            except (ValueError, KeyError, IndexError):
                valid_local[particle_idx] = False

        if not np.any(valid_local):
            return weights_unnormalised

        residuals = present_voltage - voltage_pred[valid_local]
        weights_unnormalised[valid_local] = (
            np.exp(-0.5 * residuals**2 / self.variance_eta_y_e)
            / np.sqrt(2 * np.pi * self.variance_eta_y_e)
        )
        return weights_unnormalised


    def _normalise_weights(self, weights_unnormalised: np.ndarray) -> np.ndarray:
        """Normalise weights with degeneracy check."""
        weight_sum = np.sum(weights_unnormalised)
        if weight_sum == 0:
            raise ValueError("Particle filter degeneracy: all weights zero")
        return weights_unnormalised / weight_sum


    # ----- estimation and driver -----


    def estimate(self) -> np.ndarray:
        """Return the mean state estimate over the post-resample cloud.

        Importance weighting is already baked into ``self.particles`` by
        the resampling step, so a plain mean is the correct estimator
        here. ``self.weights`` carries the pre-resample weights for
        diagnostics and indexes a different cloud, so it is intentionally
        not used.
        """
        return self.particles.mean(axis=0)


    def _generate_initial_particles(self) -> np.ndarray:
        """Sample initial particles uniformly per state_components block.

        Bounds per block:
            - 's'           : U(0, 1)
            - 'overpotentials' : 0
            - 'theta_{i}'   : U(theta_i - sqrt(var_theta), theta_i + sqrt(var_theta))
            - 'capacity'    : U(capacity - sqrt(var_capacity), capacity + sqrt(var_capacity))
        """
        particles = np.zeros((self.num_particles, self.statespace.state_dimension))

        offset = 0
        for component in self.statespace.state_components:
            # pylint: disable-next=protected-access
            size = self.statespace._block_sizes[component]
            if component == 's':
                particles[:, offset] = np.random.uniform(0.0, 1.0,
                                                         size=self.num_particles)
            elif component.startswith('theta_'):
                theta_index = int(component[len('theta_'):]) - 1
                centre = float(self.statespace.model_estimate[theta_index])
                half_width = np.sqrt(self.variance_eta_theta)
                particles[:, offset] = np.random.uniform(
                    centre - half_width, centre + half_width,
                    size=self.num_particles
                )
            elif component == 'capacity':
                centre = float(self.statespace.battery_capacity)
                half_width = np.sqrt(self.variance_eta_capacity)
                particles[:, offset] = np.random.uniform(
                    centre - half_width, centre + half_width,
                    size=self.num_particles
                )
            # 'overpotentials' stays zero.
            offset += size

        return particles


    # pylint: disable-next=too-many-branches
    def run(self, dataset: dict, initial_particles: np.ndarray | str = 'auto'):
        """
        Run particle filter over the entire measurement sequence.

        Args:
            dataset: Dict with measurement sequences. Expected keys:
                    - 'voltage_values': (num_timesteps,)
                    - 'current_values': (num_timesteps,)
                    - 'temperature_values': (num_timesteps,) or None for
                      temperature-independent models
                    - 'soc_values': (num_timesteps,), required only when
                      the StateSpace uses exogenous SOC
            initial_particles: Either a (num_particles, state_dim) array
                of initial states, or the string 'auto' to sample them
                uniformly per state_components block. See
                _generate_initial_particles for bounds.

        Returns:
            Tuple ``(estimates, weight_trajectories)``:
                - estimates: State estimates at each time step,
                  shape ``(num_timesteps, state_dim)``.
                - weight_trajectories: Normalised particle weights from
                  each measurement update *before* resampling, shape
                  ``(num_timesteps, num_particles)``. Row 0 is uniform
                  (1/num_particles) since no update has occurred yet;
                  rows 1..num_timesteps-1 contain the post-likelihood,
                  pre-resample weights and pair with the propagated
                  particle cloud at that step.
        """
        try:
            current_values = np.asarray(dataset["current_values"], dtype=float)
            voltage_values = np.asarray(dataset["voltage_values"], dtype=float)
        except KeyError as exc:
            raise KeyError(
                f"dataset is missing required key {exc.args[0]!r}; "
                f"'current_values' and 'voltage_values' are always required."
            ) from exc

        temperature_values = dataset.get("temperature_values")
        if temperature_values is not None:
            temperature_values = np.asarray(temperature_values, dtype=float)

        soc_values = dataset.get("soc_values")
        # pylint: disable=protected-access
        if self.statespace._soc_source is not SocSource.STATE and soc_values is None:
            raise KeyError(
                "dataset['soc_values'] is required when the statespace uses "
                "SocSource.EXOGENOUS (i.e. 's' is not in state_components)."
            )
        if soc_values is not None:
            soc_values = np.asarray(soc_values, dtype=float)

        # Resolve initial particles
        if isinstance(initial_particles, str):
            if initial_particles != 'auto':
                raise ValueError(
                    f"initial_particles string must be 'auto', got {initial_particles!r}."
                )
            initial_particles = self._generate_initial_particles()

        # Shape validation
        if initial_particles.shape[0] != self.num_particles:
            raise ValueError(
                f"initial_particles has {initial_particles.shape[0]} particles, "
                f"but filter was initialised with num_particles={self.num_particles}"
            )
        if initial_particles.shape[1] != self.statespace.state_dimension:
            raise ValueError(
                f"initial_particles has state dimension {initial_particles.shape[1]}, "
                f"but StateSpace expects state_dimension={self.statespace.state_dimension} "
                f"(state_components={self.statespace.state_components})."
            )

        num_timesteps = min(len(current_values), len(voltage_values))
        if temperature_values is not None:
            num_timesteps = min(num_timesteps, len(temperature_values))
        if soc_values is not None:
            num_timesteps = min(num_timesteps, len(soc_values))

        if num_timesteps < 2:
            raise ValueError(
                f"dataset has {num_timesteps} samples but the filter "
                f"needs at least 2: sample 1 (k=0) drives the first "
                f"propagation via the predict step, and sample 2 (k=1) "
                f"is used in the update step to produce the first state "
                f"estimate."
            )

        # Initialise particles (copy to avoid aliasing caller's array)
        self.particles = np.array(initial_particles, dtype=float, copy=True)

        # Reset per-run particle auxiliaries
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.eta_particles[:] = 0.0

        state_estimates = np.zeros((num_timesteps, self.statespace.state_dimension))
        weight_trajectories = np.zeros((num_timesteps, self.num_particles))

        state_estimates[0] = self.estimate()
        # No measurement update has occurred at k=0 — record the uniform
        # prior weights to keep the array shape-aligned with state_estimates.
        weight_trajectories[0] = self.weights

        def scalar(arr, k):
            return None if arr is None else float(arr[k])

        for k in tqdm(range(1, num_timesteps), desc="PF Progress", unit="step", ncols=100):
            try:
                self.step(
                    previous_temperature=scalar(temperature_values, k - 1),
                    previous_input=scalar(current_values, k - 1),
                    present_temperature=scalar(temperature_values, k),
                    present_input=scalar(current_values, k),
                    present_voltage=scalar(voltage_values, k),
                    previous_soc=scalar(soc_values, k - 1),
                    present_soc=scalar(soc_values, k),
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Particle filter failed at time step {k}: {exc}"
                ) from exc
            state_estimates[k] = self.estimate()
            weight_trajectories[k] = self.weights

        return state_estimates, weight_trajectories
