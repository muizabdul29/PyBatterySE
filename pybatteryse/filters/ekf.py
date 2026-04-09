"""Implementation of Extended Kalman Filter"""

import warnings

import numpy as np
from numpy.linalg import inv
from tqdm import tqdm

from pybatteryid.dataclasses import Model

from ..statespace import get_matrices, StateSpace
from ..coefficient import update_model_parameters


# pylint: disable=R0902
class ExtendedKalmanFilter:
    """EKF"""

    statespace: StateSpace

    soc_min: float
    soc_max: float

    sigma_nu: float
    sigma_ny_ne: float

    state_estimate: float
    error_covariance: float

    emf_function: any

    def __init__(
        self,
        model: Model,
        sigma_nu: float,
        sigma_ny_ne: float,
    ):
        """
        Initialize the Extended Kalman Filter.

        Parameters
        ----------
        statespace : StateSpaceRepresentation
            State-space representation object containing model_order and other parameters
        sigma_nu : float
            Process noise intensity
        sigma_ny_ne : float
            Measurement noise variance
        soc_min : float
            Minimum SOC constraint
        soc_max : float
            Maximum SOC constraint
        emf_function_func : callable
            Function to compute EMF from SOC and temperature
        """
        self.statespace = StateSpace(model)
        self.sigma_nu = sigma_nu
        self.sigma_ny_ne = sigma_ny_ne


        # Store function reference
        self.emf_function = model.emf_function
        #
        self.soc_min = np.min(self.emf_function.voltage_func.x) + 1e-3
        self.soc_max = np.max(self.emf_function.voltage_func.x) - 1e-3

        # State variables
        self.state_estimate = None
        self.error_covariance = None

        # Parameter estimation attributes
        self.full_parameter_vector = model.model_estimate.copy()
        self.extended_state_names = []
        self.extended_state_variances = []
        self.theta_index_map = {}  # Maps 'theta_i' -> index in full_parameter_vector
        self._build_theta_index_map()


    def _build_theta_index_map(self) -> None:
        """
        Build mapping from 'theta_i' string to index in full_parameter_vector.

        Flattens coefficients in order: a_1, a_2, ..., a_n, b_0, b_1, ..., b_n
        """
        self.theta_index_map = {}
        param_idx = 0

        # Get model order
        n = self.statespace.model_order

        # Build list of coefficient keys in order
        coeff_keys = []
        for i in range(1, n + 1):
            coeff_keys.append(f'a_{i}')
        for i in range(n + 1):
            coeff_keys.append(f'b_{i}')

        # Iterate through coefficients and build mapping
        for key in coeff_keys:
            if key in self.statespace.coefficients:
                coeff_terms = self.statespace.coefficients[key]
                for _ in coeff_terms:
                    param_idx += 1
                    self.theta_index_map[f'theta_{param_idx}'] = param_idx - 1


    def modify_model_parameters(
        self,
        params: list[float] | dict[str, float]
    ) -> None:
        """
        Modify the internal parameter vector.

        Parameters
        ----------
        params : list[float] or dict[str, float]
            If list: full parameter vector (must match model.model_estimate length)
            If dict: selective updates {parameter_name: value}

        Examples
        --------
        # Update all parameters
        ekf.modify_model_parameters([θ_1, θ_2, ..., θ_20])

        # Update specific parameters
        ekf.modify_model_parameters({'theta_1': 0.6, 'theta_5': -0.3})

        # Update capacity
        ekf.modify_model_parameters({'capacity': 2.8})
        """
        if isinstance(params, list):
            # Full update
            if len(params) != len(self.full_parameter_vector):
                raise ValueError(
                    f"Parameter list length {len(params)} does not match "
                    f"expected length {len(self.full_parameter_vector)}"
                )
            self.full_parameter_vector = list(params)  # Make a copy

        elif isinstance(params, dict):
            # Selective update
            for param_name, param_value in params.items():
                if param_name == 'capacity':
                    # Update capacity directly in statespace
                    self.statespace.battery_capacity = param_value
                elif param_name in self.theta_index_map:
                    # Update theta parameter
                    idx = self.theta_index_map[param_name]
                    self.full_parameter_vector[idx] = param_value
                else:
                    raise ValueError(
                        f"Unknown parameter name: {param_name}. "
                        f"Valid names are 'capacity' or 'theta_i' where i is in "
                        f"range 1 to {len(self.full_parameter_vector)}"
                    )
        else:
            raise TypeError(
                f"params must be list[float] or dict[str, float], got {type(params)}"
            )


    def extend_state(
        self,
        state_names: list[str],
        state_variances: list[float]
    ) -> None:
        """
        Extend the state vector to include additional parameters for estimation.

        Parameters
        ----------
        state_names : list[str]
            Names of parameters to estimate, e.g., ['capacity', 'theta_1', 'theta_5']
        state_variances : list[float]
            Process noise variances for each extended state, same order as state_names

        Examples
        --------
        ekf.extend_state(['capacity', 'theta_1', 'theta_5'], [1e-8, 1e-6, 1e-6])
        """
        # Validate lengths match
        if len(state_names) != len(state_variances):
            raise ValueError(
                f"Length mismatch: state_names has {len(state_names)} elements "
                f"but state_variances has {len(state_variances)} elements"
            )

        # Validate no duplicates
        if len(state_names) != len(set(state_names)):
            raise ValueError("state_names contains duplicate entries")

        # Validate each state name
        for name in state_names:
            #
            if name == 'capacity':
                continue  # capacity is valid
            #
            if name in self.theta_index_map:
                continue  # valid theta index
            #
            raise ValueError(
                f"Invalid state name: {name}. "
                f"Valid names are 'capacity' or 'theta_i' where i is in "
                f"range 1 to {len(self.full_parameter_vector)}"
            )

        # Store configuration
        self.extended_state_names = list(state_names)
        self.extended_state_variances = list(state_variances)


    def _update_model_from_state(self, state: np.ndarray) -> None:
        """
        Update model parameters and capacity from the augmented state vector.

        Parameters
        ----------
        state : np.ndarray
            Augmented state vector including extended parameters
        """
        if not self.extended_state_names:
            return  # No extended states, nothing to update

        # Extract extended state values
        base_dim = self.statespace.model_order + 1  # soc + RC states

        for i, name in enumerate(self.extended_state_names):
            value = float(state[base_dim + i, 0])

            if name == 'capacity':
                # Update battery capacity
                self.statespace.battery_capacity = value
            else:  # theta parameter
                # Update theta parameter in full vector
                idx = self.theta_index_map[name]
                self.full_parameter_vector[idx] = value

        # Update model coefficients if any theta parameters were modified
        if any(name.startswith('theta_') for name in self.extended_state_names):
            update_model_parameters(self.statespace.coefficients, self.full_parameter_vector)


    # pylint: disable=R0913, R0914, R0917
    def compute_state_jacobian(
        self,
        state,
        current,
        temperature,
        step_size_base=1e-2,
        step_size_param=1e-3,
        step_size_cap=1
    ):
        """
        Compute state transition Jacobian F = ∂f/∂x for the nonlinear dynamics
            f(x,u,T) = A(s,u,T) x + B(s,u,T) u

        For extended states (parameters), uses random walk model: p_{k+1} = p_k

        The Jacobian is computed as:
            F = A(s,u,T) + [∂A/∂s x + ∂B/∂s u, 0, ..., 0]
        where the bracketed term modifies only the first column of F.

        For augmented state [soc, x1, ..., xn, p1, ..., pm], F includes parameter derivatives:
            F = [F_base         ∂f/∂p    ]
                [  0          I_{m×m}    ]

        where ∂f/∂p_i is computed by perturbing parameter p_i and recomputing A, B matrices.

        Parameters
        ----------
        state : ndarray, shape (n+1+m, 1) or (n+1, 1)
            State column vector [soc, x1, ..., x_{n}, p1, ..., pm]^T where the first element
            is the state of charge (SOC) and p_i are extended parameters
        current : float
            Applied current in amperes (positive for discharge, negative for charge)
        temperature : float
            Operating temperature in Kelvin or Celsius (depending on model)
        step_size_base : float, optional
            Finite-difference step size for SOC derivatives (default: 1e-2)
        step_size_param : float, optional
            Finite-difference step size for parameter derivatives (default: 1e-3)
        step_size_cap : float, optional
            Finite-difference step size for capacity derivatives in As (default: 10.0)

        Returns
        -------
        state_jacobian : ndarray, shape (n+1+m, n+1+m) or (n+1, n+1)
            State transition Jacobian matrix F = ∂f/∂x

        Notes
        -----
        Uses adaptive finite differencing near SOC boundaries:
        - Forward difference when soc < soc_min + step_size_base
        - Backward difference when soc > soc_max - step_size_base
        - Central difference otherwise (second-order accurate)
        """
        # Extract base state (SOC + RC states)
        base_dim = self.statespace.model_order + 1
        base_state = state[:base_dim, :]

        # Extract SOC from state vector
        soc = float(base_state[0, 0])

        # Get base matrices at current operating point
        matrix_a, matrix_b, _, _ = get_matrices(self.statespace, soc, current, temperature)

        # Initialize base Jacobian with A matrix
        state_jacobian_base = matrix_a.copy()

        # Compute numerical derivatives of A and B with respect to SOC
        # Use boundary-aware finite differencing
        if soc <= self.soc_min + step_size_base:
            # Forward difference at lower boundary
            matrix_a_plus, matrix_b_plus, _, _ = get_matrices(self.statespace,
                                                              soc + step_size_base,
                                                              current,
                                                              temperature)
            da_dsoc = (matrix_a_plus - matrix_a) / step_size_base
            db_dsoc = (matrix_b_plus - matrix_b) / step_size_base

        elif soc >= self.soc_max - step_size_base:
            # Backward difference at upper boundary
            matrix_a_minus, matrix_b_minus, _, _ = get_matrices(self.statespace,
                                                                soc - step_size_base,
                                                                current,
                                                                temperature)
            da_dsoc = (matrix_a - matrix_a_minus) / step_size_base
            db_dsoc = (matrix_b - matrix_b_minus) / step_size_base

        else:
            # Central difference in interior (second-order accurate)
            matrix_a_plus, matrix_b_plus, _, _ = get_matrices(self.statespace,
                                                              soc + step_size_base,
                                                              current,
                                                              temperature)
            matrix_a_minus, matrix_b_minus, _, _ = get_matrices(self.statespace,
                                                                soc - step_size_base,
                                                                current,
                                                                temperature)
            da_dsoc = (matrix_a_plus - matrix_a_minus) / (2 * step_size_base)
            db_dsoc = (matrix_b_plus - matrix_b_minus) / (2 * step_size_base)

        # Compute additional contribution to first column from SOC dependency
        # This represents ∂f/∂soc = ∂A/∂soc · x + ∂B/∂soc · u
        current_column = np.array([[current]])
        soc_contribution = da_dsoc @ base_state + db_dsoc @ current_column

        # Update first column of Jacobian
        state_jacobian_base[:, 0] += soc_contribution.ravel()

        # Handle extended states if present
        if self.extended_state_names:
            num_extended = len(self.extended_state_names)
            total_dim = base_dim + num_extended

            # Create augmented Jacobian
            state_jacobian = np.zeros((total_dim, total_dim))
            state_jacobian[:base_dim, :base_dim] = state_jacobian_base

            # Compute derivatives with respect to each extended parameter
            for i, param_name in enumerate(self.extended_state_names):
                # Store original parameter value
                if param_name == 'capacity':
                    original_value = self.statespace.battery_capacity

                    # Perturb capacity and compute matrices
                    self.statespace.battery_capacity = original_value + step_size_cap
                    matrix_a_plus, matrix_b_plus, _, _ = get_matrices(
                        self.statespace, soc, current, temperature
                    )

                    # Restore capacity
                    self.statespace.battery_capacity = original_value

                    param_step = step_size_cap

                else:  # theta parameter
                    idx = self.theta_index_map[param_name]
                    original_value = self.full_parameter_vector[idx]

                    # Perturb parameter
                    self.full_parameter_vector[idx] = original_value + step_size_param
                    update_model_parameters(
                        self.statespace.coefficients, self.full_parameter_vector
                    )

                    # Compute matrices with perturbed parameter
                    matrix_a_plus, matrix_b_plus, _, _ = get_matrices(
                        self.statespace, soc, current, temperature
                    )

                    # Restore parameter
                    self.full_parameter_vector[idx] = original_value
                    update_model_parameters(
                        self.statespace.coefficients, self.full_parameter_vector
                    )

                    param_step = step_size_param

                # Compute derivative: ∂f/∂p = (∂A/∂p) x + (∂B/∂p) u
                da_dp = (matrix_a_plus - matrix_a) / param_step
                db_dp = (matrix_b_plus - matrix_b) / param_step

                df_dp = da_dp @ base_state + db_dp @ current_column

                # Store in Jacobian (column corresponding to this parameter)
                state_jacobian[:base_dim, base_dim + i] = df_dp.ravel()

            # Extended parameters follow random walk: p_{k+1} = p_k
            # So ∂p_{k+1}/∂p_k = I
            state_jacobian[base_dim:, base_dim:] = np.eye(num_extended)

            return state_jacobian
        #
        return state_jacobian_base


    # pylint: disable=R0914, R0915
    def compute_measurement_jacobian(
        self,
        state,
        current,
        temperature,
        step_size_base=1e-2,
        step_size_param=1e-3,
        step_size_cap=1
    ):
        """
        Compute measurement Jacobian H = ∂h/∂x for the output equation
            h(x,u,T) = V_emf(s,T) + C(s,u,T) x + D(s,u,T) u

        The Jacobian is computed as:
            H = C(s,u,T) + [∂V_emf/∂s + ∂C/∂s x + ∂D/∂s u, 0, ..., 0]
        where the bracketed term modifies only the first column of H.

        For augmented state [soc, x1, ..., xn, p1, ..., pm], H includes parameter derivatives:
            H = [H_base, ∂h/∂p1, ..., ∂h/∂pm]

        where ∂h/∂p_i is computed by perturbing parameter p_i and recomputing C, D matrices.

        Parameters
        ----------
        state : ndarray, shape (n+1+m, 1) or (n+1, 1)
            State column vector [soc, x1, ..., x_{n}, p1, ..., pm]^T where the first element
            is the state of charge (SOC) and p_i are extended parameters
        current : float
            Applied current in amperes (positive for discharge, negative for charge)
        temperature : float
            Operating temperature in Kelvin or Celsius (depending on model)
        step_size_base : float, optional
            Finite-difference step size for SOC derivatives (default: 1e-2)
        step_size_param : float, optional
            Finite-difference step size for parameter derivatives (default: 1e-3)
        step_size_cap : float, optional
            Finite-difference step size for capacity derivatives in As (default: 10.0)

        Returns
        -------
        measurement_jacobian : ndarray, shape (1, n+1+m) or (1, n+1)
            Measurement Jacobian matrix H = ∂h/∂x

        Notes
        -----
        Uses adaptive finite differencing near SOC boundaries:
        - Forward difference when soc < soc_min + step_size_base
        - Backward difference when soc > soc_max - step_size_base
        - Central difference otherwise (second-order accurate)
        """
        # Ensure state is column vector
        state = np.atleast_2d(state).reshape(-1, 1)

        # Extract base state (SOC + RC states)
        base_dim = self.statespace.model_order + 1
        base_state = state[:base_dim, :]
        soc = float(base_state[0, 0])

        # Get base matrices at current operating point
        _, _, matrix_c, matrix_d = get_matrices(self.statespace, soc, current, temperature)

        # Ensure C, D are 2D row vectors
        matrix_c = np.atleast_2d(matrix_c).reshape(1, -1)  # shape (1, n)
        matrix_d = np.atleast_2d(matrix_d).reshape(1, -1)  # shape (1, m_u)

        # Initialize base Jacobian with C matrix
        measurement_jacobian_base = matrix_c.copy()  # shape (1, n)

        # Compute numerical derivatives of V_emf, C, and D with respect to SOC
        # Use boundary-aware finite differencing
        if soc <= self.soc_min + step_size_base:
            # Forward difference at lower boundary
            _, _, matrix_c_plus, matrix_d_plus = get_matrices(
                self.statespace,
                soc + step_size_base,
                current,
                temperature
            )
            dc_dsoc = (np.atleast_2d(matrix_c_plus).reshape(1, -1) - matrix_c) / step_size_base
            dd_dsoc = (np.atleast_2d(matrix_d_plus).reshape(1, -1) - matrix_d) / step_size_base
            dv_dsoc = (self.emf_function(soc + step_size_base, temperature) -
                       self.emf_function(soc, temperature)) / step_size_base

        elif soc >= self.soc_max - step_size_base:
            # Backward difference at upper boundary
            _, _, matrix_c_minus, matrix_d_minus = get_matrices(
                self.statespace,
                soc - step_size_base,
                current,
                temperature
            )
            dc_dsoc = (matrix_c - np.atleast_2d(matrix_c_minus).reshape(1, -1)) / step_size_base
            dd_dsoc = (matrix_d - np.atleast_2d(matrix_d_minus).reshape(1, -1)) / step_size_base
            dv_dsoc = (self.emf_function(soc, temperature) -
                       self.emf_function(soc - step_size_base, temperature)) / step_size_base

        else:
            # Central difference in interior (second-order accurate)
            _, _, matrix_c_plus, matrix_d_plus = get_matrices(
                self.statespace,
                soc + step_size_base,
                current,
                temperature
            )
            _, _, matrix_c_minus, matrix_d_minus = get_matrices(
                self.statespace,
                soc - step_size_base,
                current,
                temperature
            )
            dc_dsoc = (np.atleast_2d(matrix_c_plus).reshape(1, -1) -
                       np.atleast_2d(matrix_c_minus).reshape(1, -1)) / (2 * step_size_base)
            dd_dsoc = (np.atleast_2d(matrix_d_plus).reshape(1, -1) -
                       np.atleast_2d(matrix_d_minus).reshape(1, -1)) / (2 * step_size_base)
            dv_dsoc = (self.emf_function(soc + step_size_base, temperature) -
                       self.emf_function(soc - step_size_base, temperature)) / (2 * step_size_base)

        # Compute additional contribution to first column from SOC dependency
        # This represents ∂h/∂soc = ∂V_emf/∂soc + ∂C/∂soc · x + ∂D/∂soc · u
        soc_contribution = dv_dsoc + (dc_dsoc @ base_state).item() + (dd_dsoc * current).item()
        # Update first column of Jacobian
        measurement_jacobian_base[0, 0] += soc_contribution

        # Handle extended states if present
        if self.extended_state_names:
            num_extended = len(self.extended_state_names)
            total_dim = base_dim + num_extended

            # Extended Jacobian
            measurement_jacobian = np.zeros((1, total_dim))
            measurement_jacobian[0, :base_dim] = measurement_jacobian_base

            # Compute derivatives with respect to each extended parameter
            for i, param_name in enumerate(self.extended_state_names):
                # Store original parameter value
                if param_name == 'capacity':
                    original_value = self.statespace.battery_capacity

                    # Perturb capacity and compute matrices
                    self.statespace.battery_capacity = original_value + step_size_cap
                    _, _, matrix_c_plus, matrix_d_plus = get_matrices(
                        self.statespace, soc, current, temperature
                    )

                    # Restore capacity
                    self.statespace.battery_capacity = original_value

                    # Capacity doesn't affect EMF directly (usually)
                    dv_dp = 0.0

                    param_step = step_size_cap

                else:  # theta parameter
                    idx = self.theta_index_map[param_name]
                    original_value = self.full_parameter_vector[idx]

                    # Perturb parameter
                    self.full_parameter_vector[idx] = original_value + step_size_param
                    update_model_parameters(
                        self.statespace.coefficients, self.full_parameter_vector
                    )

                    # Compute matrices with perturbed parameter
                    _, _, matrix_c_plus, matrix_d_plus = get_matrices(
                        self.statespace, soc, current, temperature
                    )

                    # Restore parameter
                    self.full_parameter_vector[idx] = original_value
                    update_model_parameters(
                        self.statespace.coefficients, self.full_parameter_vector
                    )

                    # Parameters don't affect EMF directly
                    dv_dp = 0.0

                    param_step = step_size_param

                # Ensure C, D are 2D row vectors
                matrix_c_plus = np.atleast_2d(matrix_c_plus).reshape(1, -1)
                matrix_d_plus = np.atleast_2d(matrix_d_plus).reshape(1, -1)

                # Compute derivative: ∂h/∂p = ∂V_emf/∂p + (∂C/∂p) x + (∂D/∂p) u
                dc_dp = (matrix_c_plus - matrix_c) / param_step
                dd_dp = (matrix_d_plus - matrix_d) / param_step

                dh_dp = dv_dp + (dc_dp @ base_state).item() + (dd_dp * current).item()

                # Store in Jacobian (column corresponding to this parameter)
                measurement_jacobian[0, base_dim + i] = dh_dp

            return measurement_jacobian
        #
        return measurement_jacobian_base


    # pylint: disable=R0914
    def predict(
        self,
        previous_temperature: float,
        previous_input: float,
        previous_voltage: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prediction step of the EKF.

        Parameters
        ----------
        previous_temperature : float
            Temperature at k-1
        previous_input : float
            Current input at k-1
        previous_voltage : float
            Voltage measurement at k-1

        Returns
        -------
        predicted_state : np.ndarray
            Predicted state estimate
        predicted_covariance : np.ndarray
            Predicted error covariance
        """
        # Update model parameters from current state estimate (if extended states present)
        self._update_model_from_state(self.state_estimate)

        # Extract dimensions
        base_dim = self.statespace.model_order + 1
        num_extended = len(self.extended_state_names)
        total_dim = base_dim + num_extended

        # Extract base state for matrix computation
        base_state = self.state_estimate[:base_dim, :]
        previous_soc_estimate = base_state[0, 0]

        # Compute Jacobians
        state_jacobian = self.compute_state_jacobian(
            self.state_estimate, previous_input, previous_temperature
        )
        measurement_jacobian = self.compute_measurement_jacobian(
            self.state_estimate, previous_input, previous_temperature
        )

        # Get linearized system matrices at previous time step
        matrix_a, matrix_b, matrix_c, matrix_d = get_matrices(
            self.statespace, previous_soc_estimate, previous_input, previous_temperature
        )

        # Compute base noise covariance matrices
        matrix_q_base = np.atleast_2d(
            matrix_b @ matrix_b.T * self.sigma_nu
        ).astype(np.float64)
        matrix_r = np.atleast_2d(
            matrix_d @ matrix_d.T * self.sigma_nu
        ).astype(np.float64) + self.sigma_ny_ne
        matrix_s_base = np.atleast_2d(
            matrix_b @ matrix_d.T * self.sigma_nu
        ).astype(np.float64)

        # Extend process noise covariance Q and cross-covariance S for extended states
        if num_extended > 0:
            # Process noise for extended states (random walk)
            matrix_q = np.zeros((total_dim, total_dim))
            matrix_q[:base_dim, :base_dim] = matrix_q_base
            # Add parameter process noise on diagonal
            for i, variance in enumerate(self.extended_state_variances):
                matrix_q[base_dim + i, base_dim + i] = variance

            # Cross-covariance S (parameters don't correlate with measurement noise)
            matrix_s = np.zeros((total_dim, 1))
            matrix_s[:base_dim, :] = matrix_s_base
        else:
            matrix_q = matrix_q_base
            matrix_s = matrix_s_base

        # Covariance propagation with correlated noise
        feedback_gain = matrix_s @ inv(matrix_r) @ measurement_jacobian
        predicted_covariance = (
            (state_jacobian - feedback_gain) @ self.error_covariance
            @ (state_jacobian - feedback_gain).T
            + matrix_q
            - matrix_s @ inv(matrix_r) @ matrix_s.T
        )

        # State propagation with measurement feedback from k-1
        previous_emf = self.emf_function(base_state[0, 0], previous_temperature)
        previous_innovation = (
            previous_voltage
            - previous_emf
            - matrix_c @ base_state
            - matrix_d * previous_input
        )

        # Predict base states
        predicted_base_state = (
            matrix_a @ base_state
            + matrix_b * previous_input
            + matrix_s_base @ inv(matrix_r) @ np.atleast_2d(previous_innovation)
        )

        # Clip SOC
        predicted_base_state[0, 0] = np.clip(
            predicted_base_state[0, 0], self.soc_min, self.soc_max
        )

        # Predict extended states (random walk: no change)
        if num_extended > 0:
            predicted_state = np.zeros((total_dim, 1))
            predicted_state[:base_dim, :] = predicted_base_state
            predicted_state[base_dim:, :] = self.state_estimate[base_dim:, :]
        else:
            predicted_state = predicted_base_state

        return predicted_state, predicted_covariance


    # pylint: disable=R0913, R0914, R0917
    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        current_temperature: float,
        current_input: float,
        current_voltage: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Update step of the EKF.

        Parameters
        ----------
        predicted_state : np.ndarray
            Predicted state from prediction step
        predicted_covariance : np.ndarray
            Predicted covariance from prediction step
        current_temperature : float
            Temperature at k
        current_input : float
            Current input at k
        current_voltage : float
            Voltage measurement at k

        Returns
        -------
        updated_state : np.ndarray
            Updated state estimate
        updated_covariance : np.ndarray
            Updated error covariance
        """
        # Update model parameters from predicted state (if extended states present)
        self._update_model_from_state(predicted_state)

        # Extract dimensions
        base_dim = self.statespace.model_order + 1
        predicted_base_state = predicted_state[:base_dim, :]

        # Compute measurement Jacobian
        measurement_jacobian = self.compute_measurement_jacobian(
            predicted_state, current_input, current_temperature
        )

        # Re-linearize around predicted state estimate
        _, _, matrix_c, matrix_d = get_matrices(
            self.statespace, predicted_base_state[0, 0], current_input, current_temperature
        )

        # Update measurement noise covariance
        measurement_noise_covariance = np.atleast_2d(
            matrix_d @ matrix_d.T * self.sigma_nu
        ).astype(np.float64) + self.sigma_ny_ne

        # Compute Kalman gain
        innovation_covariance = (
            measurement_jacobian @ predicted_covariance @ measurement_jacobian.T
            + measurement_noise_covariance
        )
        kalman_gain = predicted_covariance @ measurement_jacobian.T @ inv(innovation_covariance)

        # Predict measurement and compute innovation
        current_emf = self.emf_function(predicted_base_state[0, 0], current_temperature)
        predicted_voltage = (
            current_emf + matrix_c @ predicted_base_state + matrix_d * current_input
        )
        current_innovation = current_voltage - predicted_voltage

        # Update state estimate
        updated_state = predicted_state + kalman_gain * current_innovation

        # Clip SOC
        updated_state[0, 0] = np.clip(updated_state[0, 0], self.soc_min, self.soc_max)

        # Update covariance (Joseph form for numerical stability)
        total_dim = updated_state.shape[0]
        identity_matrix = np.eye(total_dim)
        updated_covariance = (
            (identity_matrix - kalman_gain @ measurement_jacobian) @ predicted_covariance
            @ (identity_matrix - kalman_gain @ measurement_jacobian).T
            + kalman_gain @ measurement_noise_covariance @ kalman_gain.T
        )

        return updated_state, updated_covariance

    def run(
        self,
        temperature_values: np.ndarray,
        current_values: np.ndarray,
        voltage_values: np.ndarray,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Run the Extended Kalman Filter on the given dataset.

        Parameters
        ----------
        temperature_values : np.ndarray
            Array of temperature measurements (length T)
        current_values : np.ndarray
            Array of current measurements (length T)
        voltage_values : np.ndarray
            Array of voltage measurements (length T)
        initial_state : np.ndarray
            Initial state estimate, shape (n+1+m, 1) where n is model order,
            m is number of extended states (0 if no extended states)
        initial_covariance : np.ndarray
            Initial error covariance matrix, shape (n+1+m, n+1+m)

        Returns
        -------
        state_estimates : np.ndarray
            Estimated state trajectories, shape (T, n+1+m)
            First column contains SOC estimates accessible via state_estimates[:, 0]
            If extended states are present, they appear in columns after RC states

        Notes
        -----
        The filter will terminate early if NaN values are detected in the state estimate,
        returning estimates up to the point of divergence with a warning.

        If extended states are configured via extend_state(), the initial_state should
        include initial values for those parameters, which will override the corresponding
        values in the internal parameter vector.
        """
        # Initialize state and covariance
        self.state_estimate = initial_state.copy()
        self.error_covariance = initial_covariance.copy()

        # Update internal parameter vector from initial extended states
        if self.extended_state_names:
            base_dim = self.statespace.model_order + 1
            for i, name in enumerate(self.extended_state_names):
                value = float(initial_state[base_dim + i, 0])

                if name == 'capacity':
                    self.statespace.battery_capacity = value
                elif name in self.theta_index_map:
                    idx = self.theta_index_map[name]
                    self.full_parameter_vector[idx] = value

            # Update model coefficients if theta parameters were modified
            if any(name.startswith('theta_') for name in self.extended_state_names):
                update_model_parameters(
                    self.statespace.coefficients, self.full_parameter_vector
                )

        # Get time horizon
        num_timesteps = min(len(voltage_values), len(current_values), len(temperature_values))

        # Determine state dimension
        state_dim = self.statespace.model_order + 1 + len(self.extended_state_names)

        # Preallocate storage (T, n+m)
        state_estimates = np.zeros((num_timesteps, state_dim))

        # Store initial condition
        state_estimates[0, :] = self.state_estimate.ravel()

        # Main filtering loop with progress bar
        for k in tqdm(range(1, num_timesteps), desc="EKF Progress", unit="step", ncols=100):
            # Extract measurements at current and previous time steps
            current_temperature = temperature_values[k]
            previous_temperature = temperature_values[k - 1]
            current_input = current_values[k]
            previous_input = current_values[k - 1]
            current_voltage = voltage_values[k]
            previous_voltage = voltage_values[k - 1]

            # Prediction step: propagate state from k-1 to k
            predicted_state, predicted_covariance = self.predict(
                previous_temperature,
                previous_input,
                previous_voltage,
            )

            # Update step: correct prediction with measurement at k
            updated_state, updated_covariance = self.update(
                predicted_state,
                predicted_covariance,
                current_temperature,
                current_input,
                current_voltage,
            )

            # Check for NaN (early termination if filter diverges)
            if np.isnan(updated_state[0, 0]):
                warnings.warn(
                    f"Extended Kalman Filter diverged at timestep {k}/{num_timesteps}. "
                    f"Returning partial results with {k} timesteps.",
                    RuntimeWarning,
                    stacklevel=2
                )
                # Return partial results up to divergence point
                return state_estimates[:k, :]

            # Store results in filter state
            self.state_estimate = updated_state
            self.error_covariance = updated_covariance

            # Save to history
            state_estimates[k, :] = self.state_estimate.ravel()

        return state_estimates
