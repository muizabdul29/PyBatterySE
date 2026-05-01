"""Implementation of Extended Kalman Filter."""

import warnings

import numpy as np
from tqdm import tqdm

from ..statespace import StateSpace
from ._common import check_nonnegative, validate_optional_variances, parse_dataset


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
    ``B B^T * var_eta_u``.

    Jacobians of f and h are obtained analytically from
    ``StateSpace.linearize``, which exploits the LPV structure of the model
    (companion-form overpotentials, linear SOC integrator, random-walk
    extended states). The only finite difference left is a 1-D central
    difference on the SOC scalar to capture the basis-function dependence
    on SOC; everything else is closed form. Before each predict and update
    the filter syncs the statespace's model parameters and capacity from
    the current state via ``StateSpace.update_model_from_state``, so the
    linearizations and nonlinear evaluations both reflect the filter's
    current belief about those quantities.
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

        # Buffer the SOC clip range slightly inside the EMF domain to keep
        # state evaluations away from the knot endpoints. The raw domain
        # bounds come from the statespace; we just add the buffer here.
        # pylint: disable=protected-access
        self._soc_min = self.statespace._soc_domain_min + 1e-3
        self._soc_max = self.statespace._soc_domain_max - 1e-3

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

        lin = self.statespace.linearize(
            state, previous_current,
            temperature_value=previous_temperature, soc_value=previous_soc,
        )

        # Noise covariances.
        #   Q = (df/du)(df/du)^T * var_eta_u  (+ random-walk diag for extended states)
        #   R = (dh/du)^2 * var_eta_u + var_eta_y_e        (scalar)
        #   S = (df/du)(dh/du) * var_eta_u                 (shape (n,))
        matrix_q = np.outer(lin.df_du, lin.df_du) * self.variance_eta_u
        matrix_q[np.diag_indices_from(matrix_q)] += self._extended_state_noise_diag

        scalar_r = (lin.dh_du ** 2) * self.variance_eta_u + self.variance_eta_y_e
        vector_s = lin.df_du * lin.dh_du * self.variance_eta_u

        # Covariance with correlated-noise correction:
        #   P+ = (df/dx - S R^-1 dh/dx) P (...)^T + Q - S R^-1 S^T
        feedback_gain = np.outer(vector_s, lin.dh_dx) / scalar_r
        df_dx_eff = lin.df_dx - feedback_gain
        predicted_covariance = (
            df_dx_eff @ cov @ df_dx_eff.T
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

        lin = self.statespace.linearize(
            predicted_state, present_current,
            temperature_value=present_temperature, soc_value=present_soc,
        )


        scalar_r = (lin.dh_du ** 2) * self.variance_eta_u + self.variance_eta_y_e
        innovation_variance = float(lin.dh_dx @ predicted_covariance @ lin.dh_dx) + scalar_r
        kalman_gain = (predicted_covariance @ lin.dh_dx) / innovation_variance

        predicted_measurement = self.statespace.evaluate_predicted_measurement(
            predicted_state, present_current,
            temperature_value=present_temperature,
            soc_value=present_soc,
        )
        innovation = present_voltage - predicted_measurement

        updated_state = predicted_state + kalman_gain * innovation

        # Joseph-form covariance update: P+ = (I - K dh/dx) P (...)^T + K R K^T
        identity = np.eye(updated_state.shape[0])
        i_minus_k_dhdx = identity - np.outer(kalman_gain, lin.dh_dx)
        updated_covariance = (
            i_minus_k_dhdx @ predicted_covariance @ i_minus_k_dhdx.T
            + np.outer(kalman_gain, kalman_gain) * scalar_r
        )

        self._clip_soc_inplace(updated_state)

        return updated_state, updated_covariance


    def run(
        self,
        dataset: dict,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the EKF on a full dataset.

        Returns
        -------
        state_estimates : np.ndarray
            Estimated state trajectory, shape (T, state_dimension). On
            divergence, returns the partial trajectory up to (but not
            including) the diverged step.
        error_covariances : np.ndarray
            Error covariance trajectory, shape (T, state_dimension,
            state_dimension). Truncated consistently with state_estimates
            on divergence.
        """
        # pylint: disable=protected-access
        current_values, voltage_values, temperature_values, soc_values, num_timesteps = (
            parse_dataset(dataset, self.statespace._soc_source)
        )

        self.state_estimate = np.asarray(initial_state, dtype=float).copy()
        self.error_covariance = np.asarray(initial_covariance, dtype=float).copy()
        self.statespace.update_model_from_state(self.state_estimate)

        n = self.statespace.state_dimension
        state_estimates = np.zeros((num_timesteps, n))
        error_covariances = np.zeros((num_timesteps, n, n))
        state_estimates[0, :] = self.state_estimate
        error_covariances[0, :, :] = self.error_covariance

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
                return state_estimates[:k, :], error_covariances[:k, :, :]

            self.state_estimate = updated_state
            self.error_covariance = updated_covariance
            state_estimates[k, :] = updated_state
            error_covariances[k, :, :] = updated_covariance

        return state_estimates, error_covariances


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------


    def _validate_variances(self) -> None:
        """Non-negativity for all provided variances; presence gated on state_components."""
        check_nonnegative("variance_eta_u",   self.variance_eta_u)
        check_nonnegative("variance_eta_y_e", self.variance_eta_y_e)
        validate_optional_variances(self.statespace,
                                    self.variance_eta_theta, self.variance_eta_capacity)


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
