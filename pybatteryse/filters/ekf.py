"""Implementation of Extended Kalman Filter."""

import warnings
from contextlib import contextmanager

import numpy as np
from tqdm import tqdm

from ..statespace import StateSpace, SocSource


# pylint: disable=R0902,R0913
class ExtendedKalmanFilter:
    """Correlated-noise Extended Kalman Filter for battery models
    identified using ``PyBatteryID``.

    Implements a correlated-noise EKF that operates on a
    :class:`pybatteryse.statespace.StateSpace` instance. Process and
    measurement noise share a common input-current noise source eta_u, which
    induces a non-zero cross-covariance ``S = B D^T * var_eta_u`` between
    them. The filter handles this via the standard correlated-noise
    Kalman-filter formulation (Anderson & Moore, *Optimal Filtering*,
    ch. 3): the predict step consumes the k-1 innovation to partially resolve
    the shared noise realization, and the covariance propagation includes
    the ``Q - S R^-1 S^T`` correction. With the correlation absorbed into
    predict, the update step reduces to the textbook EKF update.

    The state vector always includes the overpotential sub-system and may
    additionally contain state of charge, model parameters ``theta_{i}``,
    and battery capacity, in any order. Random-walk dynamics for the
    extended states (``theta_{i}``, capacity) contribute diagonal entries
    to the process-noise covariance parameterized by ``variance_eta_theta``
    and ``variance_eta_capacity``; the ``s`` and ``overpotentials`` blocks
    inherit their process noise from the input-current noise via
    ``B B^T * var_eta_u``. Before each predict and update the filter syncs
    the statespace's model parameters and capacity from the current state
    via ``StateSpace.update_model_from_state``, so the linearizations reflect
    the filter's current belief about those quantities.
    """

    statespace: StateSpace

    variance_eta_u: float
    variance_eta_y_e: float
    variance_eta_theta: float | None
    variance_eta_capacity: float | None

    state_estimate: np.ndarray
    error_covariance: np.ndarray

    _soc_min: float
    _soc_max: float

    # pylint: disable=R0917
    def __init__(
        self,
        statespace: StateSpace,
        variance_eta_u: float,
        variance_eta_y_e: float,
        variance_eta_theta: float | None = None,
        variance_eta_capacity: float | None = None,
    ):
        self.statespace = statespace
        self.variance_eta_u = variance_eta_u
        self.variance_eta_y_e = variance_eta_y_e
        self.variance_eta_theta = variance_eta_theta
        self.variance_eta_capacity = variance_eta_capacity

        self._validate_variances()

        self._soc_min = float(np.min(self.statespace.emf_function.voltage_func.x)) + 1e-3
        self._soc_max = float(np.max(self.statespace.emf_function.voltage_func.x)) - 1e-3

        self.state_estimate = None
        self.error_covariance = None

        self._extended_state_noise_diag = self._build_extended_state_noise_diag()


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    # pylint: disable=R0914
    def predict(
        self,
        previous_current: float,
        previous_voltage: float,
        previous_temperature: float | None = None,
        previous_soc: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict step (correlated-noise EKF with k-1 measurement feedback).
        """
        state = self.state_estimate
        cov = self.error_covariance

        self.statespace.update_model_from_state(state)

        # Linearizations via central finite differences.
        matrix_a = self._jacobian_wrt_state(
            self.statespace.evaluate_next_state,
            state, previous_current, previous_temperature, previous_soc,
        )
        matrix_b = self._jacobian_wrt_input(
            self.statespace.evaluate_next_state,
            state, previous_current, previous_temperature, previous_soc,
        )
        matrix_c = self._jacobian_wrt_state(
            self.statespace.evaluate_predicted_measurement,
            state, previous_current, previous_temperature, previous_soc,
        )
        matrix_d = self._jacobian_wrt_input(
            self.statespace.evaluate_predicted_measurement,
            state, previous_current, previous_temperature, previous_soc,
        )

        # Noise covariances.
        #   Q = B B^T * var_eta_u  (+ random-walk diag for extended states)
        #   R = D^2 * var_eta_u + var_eta_y_e      (scalar)
        #   S = B D * var_eta_u                    (shape (n,))
        matrix_q = np.outer(matrix_b, matrix_b) * self.variance_eta_u
        matrix_q[np.diag_indices_from(matrix_q)] += self._extended_state_noise_diag

        scalar_r = (matrix_d * matrix_d) * self.variance_eta_u + self.variance_eta_y_e
        vector_s = matrix_b * matrix_d * self.variance_eta_u

        # Covariance with correlated-noise correction:
        #   P+ = (A - S R^-1 C) P (A - S R^-1 C)^T + Q - S R^-1 S^T
        feedback_gain = np.outer(vector_s, matrix_c) / scalar_r
        a_eff = matrix_a - feedback_gain
        predicted_covariance = (
            a_eff @ cov @ a_eff.T
            + matrix_q
            - np.outer(vector_s, vector_s) / scalar_r
        )

        # Nonlinear mean propagation + measurement-feedback correction.
        predicted_mean = self.statespace.evaluate_next_state(
            state, previous_current,
            temperature_value=previous_temperature,
            soc_value=previous_soc,
        )
        predicted_measurement = self.statespace.evaluate_predicted_measurement(
            state, previous_current,
            temperature_value=previous_temperature,
            soc_value=previous_soc,
        )
        previous_innovation = previous_voltage - predicted_measurement
        predicted_state = predicted_mean + vector_s * (previous_innovation / scalar_r)

        self._clip_soc_inplace(predicted_state)

        return predicted_state, predicted_covariance


    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        present_current: float,
        present_voltage: float,
        present_temperature: float | None = None,
        present_soc: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update step (standard EKF update; correlation was absorbed in predict).
        """
        self.statespace.update_model_from_state(predicted_state)

        matrix_c = self._jacobian_wrt_state(
            self.statespace.evaluate_predicted_measurement,
            predicted_state, present_current, present_temperature, present_soc,
        )
        matrix_d = self._jacobian_wrt_input(
            self.statespace.evaluate_predicted_measurement,
            predicted_state, present_current, present_temperature, present_soc,
        )

        scalar_r = (matrix_d * matrix_d) * self.variance_eta_u + self.variance_eta_y_e
        innovation_variance = float(matrix_c @ predicted_covariance @ matrix_c) + scalar_r
        kalman_gain = (predicted_covariance @ matrix_c) / innovation_variance

        predicted_measurement = self.statespace.evaluate_predicted_measurement(
            predicted_state, present_current,
            temperature_value=present_temperature,
            soc_value=present_soc,
        )
        innovation = present_voltage - predicted_measurement

        updated_state = predicted_state + kalman_gain * innovation

        # Joseph-form covariance update: P+ = (I - K C) P (I - K C)^T + K R K^T
        identity = np.eye(updated_state.shape[0])
        i_minus_kc = identity - np.outer(kalman_gain, matrix_c)
        updated_covariance = (
            i_minus_kc @ predicted_covariance @ i_minus_kc.T
            + np.outer(kalman_gain, kalman_gain) * scalar_r
        )

        self._clip_soc_inplace(updated_state)

        return updated_state, updated_covariance


    def run(
        self,
        dataset: dict,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Run the EKF on a full dataset.

        Parameters
        ----------
        dataset : dict
            Measurement arrays keyed by name. Recognized keys:
                - 'current_values'     (required, length T)
                - 'voltage_values'     (required, length T)
                - 'temperature_values' (optional; required only when the
                  model is temperature-dependent)
                - 'soc_values'         (required iff the statespace uses
                  SocSource.EXOGENOUS)
        initial_state : np.ndarray
            Initial state estimate, shape (state_dimension,).
        initial_covariance : np.ndarray
            Initial error covariance, shape (state_dimension, state_dimension).

        Returns
        -------
        state_estimates : np.ndarray
            Estimated state trajectory, shape (T, state_dimension). On
            divergence, returns the partial trajectory up to (but not
            including) the diverged step.
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

        num_timesteps = min(len(current_values), len(voltage_values))
        if temperature_values is not None:
            num_timesteps = min(num_timesteps, len(temperature_values))
        if soc_values is not None:
            num_timesteps = min(num_timesteps, len(soc_values))

        if num_timesteps < 2:
            raise ValueError(
                f"dataset has {num_timesteps} samples but the filter "
                f"needs at least 2 (one initial condition + one filter step)."
            )

        self.state_estimate = np.asarray(initial_state, dtype=float).copy()
        self.error_covariance = np.asarray(initial_covariance, dtype=float).copy()
        self.statespace.update_model_from_state(self.state_estimate)

        state_estimates = np.zeros((num_timesteps, self.statespace.state_dimension))
        state_estimates[0, :] = self.state_estimate

        def scalar(arr, k):
            return None if arr is None else float(arr[k])

        for k in tqdm(range(1, num_timesteps),
                      desc="EKF Progress", unit="step", ncols=100):
            predicted_state, predicted_covariance = self.predict(
                previous_current=float(current_values[k - 1]),
                previous_voltage=float(voltage_values[k - 1]),
                previous_temperature=scalar(temperature_values, k - 1),
                previous_soc=scalar(soc_values, k - 1),
            )

            updated_state, updated_covariance = self.update(
                predicted_state,
                predicted_covariance,
                present_current=float(current_values[k]),
                present_voltage=float(voltage_values[k]),
                present_temperature=scalar(temperature_values, k),
                present_soc=scalar(soc_values, k),
            )

            if np.any(np.isnan(updated_state)):
                warnings.warn(
                    f"Extended Kalman Filter diverged at timestep {k}/{num_timesteps}. "
                    f"Returning partial results with {k} timesteps.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return state_estimates[:k, :]

            self.state_estimate = updated_state
            self.error_covariance = updated_covariance
            state_estimates[k, :] = updated_state

        return state_estimates


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def _validate_variances(self) -> None:
        """Non-negativity for all provided variances; presence gated on state_components."""
        def check_nonnegative(name: str, value: float) -> None:
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")

        check_nonnegative("variance_eta_u",   self.variance_eta_u)
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


    def _build_extended_state_noise_diag(self) -> np.ndarray:
        """Length-state_dimension vector of random-walk process-noise variances."""
        diag = np.zeros(self.statespace.state_dimension, dtype=np.float64)

        offset = 0
        # pylint: disable=protected-access
        block_sizes = self.statespace._block_sizes
        for component in self.statespace.state_components:
            size = block_sizes[component]
            if component.startswith("theta_"):
                diag[offset] = self.variance_eta_theta
            elif component == "capacity":
                diag[offset] = self.variance_eta_capacity
            offset += size

        return diag


    def _clip_soc_inplace(self, state: np.ndarray) -> None:
        """Clip the SOC entry to [soc_min, soc_max] if 's' is in the state."""
        # pylint: disable=protected-access
        soc_offset = self.statespace._soc_offset
        if soc_offset is not None:
            state[soc_offset] = np.clip(
                state[soc_offset], self._soc_min, self._soc_max
            )

    # --- Finite-difference Jacobians -----------------------------------

    _SQRT_EPS = float(np.sqrt(np.finfo(np.float64).eps))

    @classmethod
    def _fd_step(cls, value: float) -> float:
        return cls._SQRT_EPS * max(abs(float(value)), 1.0)


    def _with_model_synced_to(self, state: np.ndarray):
        """Context manager: sync statespace model to `state`, restore on exit.

        Needed inside finite-difference Jacobians so that perturbing theta_i
        or capacity entries of the state actually flows through into func's
        output. Snapshots model_estimate, battery_capacity, and the
        subterm.parameter values that update_model_from_state mutates, and
        restores them when the block exits.
        """

        @contextmanager
        def _cm():
            ss = self.statespace
            # pylint: disable=protected-access
            saved_model_estimate = ss.model_estimate.copy()
            saved_capacity = ss.battery_capacity
            saved_subterm_params = [
                (subterm, subterm.parameter)
                for subterm, _ in ss._theta_subterm_map
            ]
            try:
                ss.update_model_from_state(state)
                yield
            finally:
                ss.model_estimate[:] = saved_model_estimate
                ss.battery_capacity = saved_capacity
                for subterm, original in saved_subterm_params:
                    subterm.parameter = original

        return _cm()


    def _jacobian_wrt_state(
        self,
        func,
        state: np.ndarray,
        current_value: float,
        temperature_value: float | None,
        soc_value: float | None,
    ) -> np.ndarray:
        """Central-difference Jacobian of ``func`` with respect to ``state``."""
        n = state.shape[0]
        jacobian: np.ndarray | None = None

        for i in range(n):
            step = self._fd_step(state[i])
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += step
            state_minus[i] -= step

            with self._with_model_synced_to(state_plus):
                f_plus = np.atleast_1d(func(
                    state_plus, current_value,
                    temperature_value=temperature_value, soc_value=soc_value,
                ))
            with self._with_model_synced_to(state_minus):
                f_minus = np.atleast_1d(func(
                    state_minus, current_value,
                    temperature_value=temperature_value, soc_value=soc_value,
                ))

            if jacobian is None:
                jacobian = np.empty((f_plus.size, n), dtype=np.float64)
            jacobian[:, i] = (f_plus - f_minus) / (2.0 * step)

        return jacobian[0] if jacobian.shape[0] == 1 else jacobian


    def _jacobian_wrt_input(
        self,
        func,
        state: np.ndarray,
        current_value: float,
        temperature_value: float | None,
        soc_value: float | None,
    ) -> np.ndarray | float:
        """Central-difference derivative of ``func`` w.r.t. scalar ``current_value``."""
        step = self._fd_step(current_value)
        f_plus = np.atleast_1d(func(
            state, current_value + step,
            temperature_value=temperature_value, soc_value=soc_value,
        ))
        f_minus = np.atleast_1d(func(
            state, current_value - step,
            temperature_value=temperature_value, soc_value=soc_value,
        ))
        derivative = (f_plus - f_minus) / (2.0 * step)
        return float(derivative[0]) if derivative.size == 1 else derivative
