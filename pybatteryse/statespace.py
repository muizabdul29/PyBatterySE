"""Utilities concerning model state-space representation."""

import re
import copy
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np

from pybatteryid.basisfunctions import generate_basis_function_signals, generate_signal_trajectories
from pybatteryid.dataclasses import BasisFunction, Model, Signal, SignalVector, VoltageFunction

from .coefficient import evaluate_coefficient, extract_model_coefficients, Coefficients


class SocSource(Enum):
    """Where the state of charge comes from for coefficient/EMF evaluation."""
    STATE = 'state'           # 's' is in state_components; SOC is read from the state vector.
    EXOGENOUS = 'exogenous'   # 's' is not in state_components; the caller supplies SOC.


@dataclass
class Linearization:
    """Analytical Jacobians of the state-space model at a linearization point.

    All four fields are evaluated at a fixed (state, current, temperature, soc)
    tuple, with the model parameters synced from the state. Matches the
    return shape of the previous finite-difference Jacobians in
    ``ExtendedKalmanFilter`` and is consumed by it directly.

    Attributes:
        df_dx: Jacobian of the state-transition function f wrt the state x.
            Shape (state_dimension, state_dimension).
        dh_dx: Jacobian of the measurement function h (scalar) wrt x.
            Shape (state_dimension,).
        df_du: Derivative of f wrt the scalar input current u.
            Shape (state_dimension,).
        dh_du: Derivative of h wrt u. Equals b_0 evaluated at the lin point.
    """
    df_dx: np.ndarray
    dh_dx: np.ndarray
    df_du: np.ndarray
    dh_du: float


# pylint: disable=R0902
class StateSpace:
    """State-space representation of an identified battery model.

    The overpotential sub-system is always realized as a companion-form
    state-space block, so 'overpotentials' must appear in state_components.
    The remaining configuration axis is the SOC source
    (self._soc_source): whether state of charge is tracked inside the state
    vector (SocSource.STATE, when 's' is in state_components) or provided by
    the caller as an exogenous input (SocSource.EXOGENOUS). 'theta_{i}' and
    'capacity' can be included in any combination as random-walk states.

    Measurement argument conventions:
        - Present values (current_value, temperature_value, soc_value) are
          scalars. temperature_value is optional and should be None when the
          model is not temperature-dependent.
        - soc_value is unused when the SOC source is SocSource.STATE
          (silently ignored) and required when it is SocSource.EXOGENOUS.
        - temperature_value is passed through without inspection. For a
          temperature-independent model, pass None; a temperature-dependent
          model called without a real temperature will fail downstream when
          a basis function tries to read 'T', which is not this class's
          concern.

    Attributes:
        battery_capacity: Battery capacity used for SOC integration.
        sampling_period: Discrete-time sampling period.
        emf_function: Open-circuit (EMF) voltage as a function of SOC and temperature.
        model_order: Order of the overpotential sub-system.
        model_estimate: Flat parameter vector theta.
        model_terms: Ordered list of symbol-product strings indexing into
            the model parameters.
        basis_functions: Basis functions used by the model.
        coefficients: Model coefficients keyed by name (a_1, ..., b_0, b_1, ...).
        state_components: Ordered list of string identifiers specifying
            the components of the state vector, block by block. Must contain
            'overpotentials'. Valid entries:
                - 's'              : state of charge (scalar).
                - 'overpotentials' : overpotential sub-system states x_1, ..., x_n.
                - 'theta_{i}'      : the i-th model parameter (1 <= i <= N).
                - 'capacity'       : battery capacity.
        state_dimension: Length of the state vector x_k. Computed as the
            sum of per-component block sizes. Use this to allocate initial
            state arrays: `np.zeros(ss.state_dimension)`.
    """

    battery_capacity: float
    sampling_period: int
    emf_function: VoltageFunction
    model_order: int
    model_estimate: np.ndarray
    model_terms: list[str]
    basis_functions: list[BasisFunction]
    coefficients: Coefficients
    state_components: list[str]
    state_dimension: int

    _theta_info: list[tuple[object, str]]
    _soc_domain_min: float
    _soc_domain_max: float
    _soc_source: SocSource
    _block_sizes: dict[str, int]
    _soc_offset: int | None
    _overpotentials_offset: int
    _capacity_offset: int | None
    _theta_offsets: dict[int, int]
    # Step size for the SOC finite differences in _soc_derivatives. Be very
    # careful with smaller step size since it can introduce various
    # numerical issues.
    _SOC_FD_STEP = 1e-3


    def __init__(self, model: Model, state_components: list[str]) -> None:
        # Immutable scalars - direct assignment is safe (no aliasing risk).
        self.battery_capacity = model.battery_capacity
        self.sampling_period = model.sampling_period
        self.model_order = model.model_order

        # ndarray of floats - np.array always copies the buffer.
        self.model_estimate = np.array(model.model_estimate, dtype=float)

        # Compound objects (and object-dtype arrays) - deep copy to isolate
        # nested state and contained elements.
        self.model_terms = copy.deepcopy(model.model_terms)
        self.emf_function = copy.deepcopy(model.emf_function)
        self.basis_functions = copy.deepcopy(model.basis_functions)

        # SOC domain over which emf_function and the basis functions are
        # defined. _soc_derivatives uses these to clamp the FD step away
        # from the boundary (falling back to one-sided differences when a
        # central pair would query out-of-domain).
        soc_grid = np.asarray(self.emf_function.voltage_func.x, dtype=float)
        self._soc_domain_min: float = float(soc_grid.min())
        self._soc_domain_max: float = float(soc_grid.max())

        self.coefficients = extract_model_coefficients(
            self.model_terms, self.model_estimate
        )
        self.state_components = self._validate_and_deduplicate_state_components(state_components)

        # Flat list indexed by theta_index (i.e. the index into
        # model_estimate). _theta_info[k] = (subterm, coefficient_key) where
        # subterm is the unique subterm that parameter theta_{k+1} owns and
        # coefficient_key (e.g. 'a_1', 'b_0') identifies the coefficient it
        # lives in. The a-coefficient flag is derived at use sites via
        # key.startswith('a_').
        self._theta_info: list[tuple[object, str]] = [
            (subterm, coefficient_key)
            for coefficient_key, subterms in self.coefficients.items()
            for subterm in subterms
        ]
        self._soc_source: SocSource = (
            SocSource.STATE if 's' in self.state_components else SocSource.EXOGENOUS
        )
        self._block_sizes: dict[str, int] = {
            component: self._compute_block_size(component)
            for component in self.state_components
        }
        self._soc_offset: int | None = self._compute_offset_of('s')
        self._overpotentials_offset: int = self._compute_offset_of('overpotentials')
        self._capacity_offset: int | None = self._compute_offset_of('capacity')
        self._theta_offsets: dict[int, int] = {
            int(c[len('theta_'):]) - 1: self._compute_offset_of(c)
            for c in self.state_components if c.startswith('theta_')
        }
        self.state_dimension = sum(self._block_sizes.values())


    # ----- public API -----


    def evaluate_next_state(self, state: np.ndarray,
                            current_value: float,
                            temperature_value: float | None = None,
                            soc_value: float | None = None) -> np.ndarray:
        """Advance the state by one step: x_{k+1} = f(x_k, i_k, T_k).

        This is the state-transition function f of the state-space model
        (also called F or f_func in filterpy conventions).

        temperature_value is optional; pass None for temperature-independent
        models.
        """
        self._validate_state_dimension(state)
        soc_scalar = self._resolve_soc_scalar(state, soc_value)

        trajectories = self._build_signal_trajectories(
            soc_values=soc_scalar,
            current_values=current_value,
            temperature_values=temperature_value,
        )

        next_state = np.empty_like(state)
        offset = 0
        for component in self.state_components:
            size = self._block_sizes[component]
            if component == 's':
                next_state[offset] = state[offset] \
                    + (self.sampling_period / self.battery_capacity) * current_value
            elif component == 'overpotentials':
                next_state[offset:offset + size] = self._evaluate_overpotentials(
                    state[offset:offset + size], current_value, trajectories
                )
            elif component == 'capacity' or component.startswith('theta_'):
                next_state[offset] = state[offset]
            offset += size

        return next_state


    def evaluate_predicted_measurement(self, state: np.ndarray,
                                       current_value: float,
                                       temperature_value: float | None = None,
                                       soc_value: float | None = None) -> float:
        """Compute the scalar predicted terminal voltage:
        V_k = EMF(s_k, T_k) + v_k.

        This is the measurement function h of the state-space model
        (also called h_func in filterpy conventions). v_k is the
        companion-form overpotential v_k = x_1 + b_0 * i_k; EMF is the
        open-circuit component.

        temperature_value is optional; pass None for temperature-independent
        models.
        """
        self._validate_state_dimension(state)
        soc_scalar = self._resolve_soc_scalar(state, soc_value)

        emf = float(self.emf_function(soc_scalar, temperature_value))

        trajectories = self._build_signal_trajectories(
            soc_values=soc_scalar,
            current_values=current_value,
            temperature_values=temperature_value,
        )

        x1 = float(state[self._overpotentials_offset])
        b0_value = evaluate_coefficient(self.coefficients['b_0'], trajectories, 0)
        return emf + float(x1 + b0_value * current_value)


    def update_model_from_state(self, state: np.ndarray) -> None:
        """Sync model parameters and battery capacity from a state vector.

        Walks self.state_components in order and, for each component that
        represents a random-walk parameter ('theta_{i}' or 'capacity'),
        extracts the corresponding entry of `state` and writes it back into
        the model: 'theta_{i}' updates self.model_estimate[i-1] and the
        matching subterm in self.coefficients (with the a-coefficient sign
        flip); 'capacity' updates self.battery_capacity.

        Partial updates are supported: any subset of theta_i may appear in
        state_components, in any order. Entries of self.model_estimate whose
        indices are not referenced by the state are left untouched. 's' and
        'overpotentials' blocks are skipped. No-op when neither theta_* nor
        capacity is in state_components.
        """
        self._validate_state_dimension(state)

        offset = 0
        for component in self.state_components:
            size = self._block_sizes[component]
            if component.startswith('theta_'):
                theta_index = int(component[len('theta_'):]) - 1
                if not 0 <= theta_index < len(self._theta_info):
                    raise ValueError(
                        f"State component {component!r} refers to parameter "
                        f"index {theta_index + 1}, but the model only has "
                        f"{len(self._theta_info)} parameters."
                    )
                new_value = float(state[offset])
                self.model_estimate[theta_index] = new_value
                subterm, coefficient_key = self._theta_info[theta_index]
                is_a_coefficient = coefficient_key.startswith('a_')
                subterm.parameter = -new_value if is_a_coefficient else new_value
            elif component == 'capacity':
                self.battery_capacity = float(state[offset])
            # 's' and 'overpotentials': nothing to write back.
            offset += size


    def linearize(self, state: np.ndarray,
                  current_value: float,
                  temperature_value: float | None = None,
                  soc_value: float | None = None) -> Linearization:
        """Compute analytical Jacobians of f and h at the given lin point.

        Builds the LPV companion-form blocks A, B and the measurement
        primitives C = e_1^T (implicit), D = b_0 once at the linearization
        point and assembles them into full df_dx, dh_dx, df_du, dh_du
        according to the active state_components layout.

        SOC handling:
        - When self._soc_source is SocSource.STATE, derivatives wrt the SOC
          coefficient-dependence are computed via 1-D central finite
          differences on the SOC scalar (basis functions don't expose
          analytic derivatives), yielding the SOC-column entries of df_dx
          and dh_dx. This costs two extra trajectory rebuilds plus an EMF
          derivative evaluation.
        - When SOC is exogenous, SOC has no column in the Jacobians and the
          1-D FD pass is skipped entirely.

        Theta and capacity columns/rows are filled analytically from the
        coefficient/companion-form structure; no FD is used for those.

        Caller is responsible for calling update_model_from_state(state)
        beforehand if the model parameters or capacity may be out of sync
        with `state`. This matches the convention used by
        evaluate_next_state and evaluate_predicted_measurement.
        """
        self._validate_state_dimension(state)
        soc_scalar = self._resolve_soc_scalar(state, soc_value)

        n = self.model_order
        op_offset = self._overpotentials_offset
        x_op = state[op_offset:op_offset + n]

        # Coefficients at the linearization point.
        trajectories = self._build_signal_trajectories(
            soc_values=soc_scalar,
            current_values=current_value,
            temperature_values=temperature_value,
        )
        a_value, b_value, b0_value = self._evaluate_lpv_coefficients(trajectories)

        # Companion-form Jacobian of the overpotentials block wrt x_op.
        # Row i-1 (i = 1..n): -a_i * x_op[0] + (b_i - a_i * b_0) * u + (x_op[i] if i < n else 0)
        matrix_a_op = np.zeros((n, n), dtype=float)
        matrix_a_op[:, 0] = -a_value
        if n > 1:
            # Super-diagonal shift: row i-1 gets x_op[i] for i = 1..n-1.
            matrix_a_op[np.arange(n - 1), np.arange(1, n)] = 1.0
        vector_b_op = b_value - a_value * b0_value  # shape (n,)

        # Allocate full Jacobians.
        df_dx = np.zeros((self.state_dimension, self.state_dimension), dtype=float)
        dh_dx = np.zeros(self.state_dimension, dtype=float)
        df_du = np.zeros(self.state_dimension, dtype=float)

        # ---- Overpotentials block. -----------------------------------
        df_dx[op_offset:op_offset + n, op_offset:op_offset + n] = matrix_a_op
        df_du[op_offset:op_offset + n] = vector_b_op
        # Measurement: v_k = x_op[0] + b_0 * u  =>  dh/dx_op = e_1^T, dh/du = b_0.
        dh_dx[op_offset] = 1.0

        # ---- Random-walk diagonals (theta_*, capacity). --------------
        for _, theta_offset in self._theta_offsets.items():
            df_dx[theta_offset, theta_offset] = 1.0
        if self._capacity_offset is not None:
            df_dx[self._capacity_offset, self._capacity_offset] = 1.0

        # ---- SOC integrator row and SOC column (only when SOC is a state).
        if self._soc_source is SocSource.STATE:
            soc_offset = self._soc_offset

            # Integrator row:
            #   df_dx[s, s]        = 1
            #   df_du[s]           = T_s / Q
            #   df_dx[s, capacity] = -(T_s / Q^2) * u   (if capacity is in state)
            df_dx[soc_offset, soc_offset] = 1.0
            df_du[soc_offset] = self.sampling_period / self.battery_capacity
            if self._capacity_offset is not None:
                df_dx[soc_offset, self._capacity_offset] = (
                    -(self.sampling_period / self.battery_capacity ** 2) * current_value
                )

            # SOC-column entries: capture coefficient dependence on SOC via
            # 1-D FD plus the EMF derivative.
            da_ds, db_ds, db0_ds, demf_ds = self._soc_derivatives(
                soc_scalar=soc_scalar,
                current_value=current_value,
                temperature_value=temperature_value,
            )
            # Overpotentials rows:
            #   row i-1: -da_i/ds * x_op[0] + (db_i/ds - da_i/ds * b_0 - a_i * db_0/ds) * u
            df_dx[op_offset:op_offset + n, soc_offset] = (
                -da_ds * x_op[0]
                + (db_ds - da_ds * b0_value - a_value * db0_ds) * current_value
            )
            # Measurement: dh/ds = dEMF/ds + db_0/ds * u
            dh_dx[soc_offset] = demf_ds + db0_ds * current_value

        # ---- Theta columns. ------------------------------------------
        # For each theta_k in state_components, route its column based on
        # which coefficient it parameterises. d_coef is d(coef_value)/d(theta_k)
        # at the lin point in *post-flip* convention (matching the sign
        # written into self.coefficients by update_model_from_state).
        # Routing rules:
        #   theta_k in a_i:        row i-1 of op block gets -d_coef * (x_op[0] + b_0 * u)
        #   theta_k in b_0:        every op row i gets -a_i * d_coef * u (cross-cutting),
        #                          measurement gets +d_coef * u
        #   theta_k in b_i (i>=1): row i-1 of op block gets +d_coef * u
        for theta_index, theta_offset in self._theta_offsets.items():
            _, coefficient_key = self._theta_info[theta_index]
            d_coef = self._theta_coefficient_derivative(theta_index, trajectories)

            if coefficient_key.startswith('a_'):
                i = int(coefficient_key[len('a_'):])
                df_dx[op_offset + (i - 1), theta_offset] = (
                    -d_coef * (x_op[0] + b0_value * current_value)
                )
            elif coefficient_key == 'b_0':
                df_dx[op_offset:op_offset + n, theta_offset] = (
                    -a_value * d_coef * current_value
                )
                dh_dx[theta_offset] = d_coef * current_value
            else:  # b_i for i >= 1
                i = int(coefficient_key[len('b_'):])
                df_dx[op_offset + (i - 1), theta_offset] = d_coef * current_value

        return Linearization(
            df_dx=df_dx, dh_dx=dh_dx, df_du=df_du, dh_du=b0_value,
        )


    # ----- linearization helpers -----


    def _soc_derivatives(self, soc_scalar: float,
                         current_value: float,
                         temperature_value: float | None,
                         ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """FD of (a_1..a_n, b_1..b_n, b_0, EMF) wrt SOC at fixed current
        and temperature.

        Returns (da_ds, db_ds, db0_ds, demf_ds) where da_ds and db_ds have
        shape (n,) and the rest are scalars. Uses a fixed step
        (_SOC_FD_STEP) rather than the sqrt(eps) rule because the
        underlying functions are piecewise in SOC.

        Boundary handling: when soc_scalar is within _SOC_FD_STEP of the
        EMF domain edges, falls back to a forward / backward difference
        on the affected side so that no evaluation queries out-of-domain
        SOC. The one-sided variant is O(h) accurate vs. O(h^2) for the
        central case; this asymmetry is unavoidable at the boundary.
        """
        step = self._SOC_FD_STEP

        def coefficients_at(soc_val: float) -> tuple[np.ndarray, np.ndarray, float]:
            trajectories = self._build_signal_trajectories(
                soc_values=soc_val,
                current_values=current_value,
                temperature_values=temperature_value,
            )
            return self._evaluate_lpv_coefficients(trajectories)

        def emf_at(soc_val: float) -> float:
            return float(self.emf_function(soc_val, temperature_value))

        plus_in_domain = (soc_scalar + step) <= self._soc_domain_max
        minus_in_domain = (soc_scalar - step) >= self._soc_domain_min

        if plus_in_domain and minus_in_domain:
            # Central difference (interior, the common case).
            a_plus, b_plus, b0_plus = coefficients_at(soc_scalar + step)
            a_minus, b_minus, b0_minus = coefficients_at(soc_scalar - step)
            denom = 2.0 * step
            da_ds = (a_plus - a_minus) / denom
            db_ds = (b_plus - b_minus) / denom
            db0_ds = (b0_plus - b0_minus) / denom
            demf_ds = (emf_at(soc_scalar + step) - emf_at(soc_scalar - step)) / denom
        elif plus_in_domain:
            # Forward difference: lin point too close to soc_domain_min, so
            # we evaluate at (s, s + step). Querying s - step would be
            # out-of-domain.
            a_lin, b_lin, b0_lin = coefficients_at(soc_scalar)
            a_plus, b_plus, b0_plus = coefficients_at(soc_scalar + step)
            da_ds = (a_plus - a_lin) / step
            db_ds = (b_plus - b_lin) / step
            db0_ds = (b0_plus - b0_lin) / step
            demf_ds = (emf_at(soc_scalar + step) - emf_at(soc_scalar)) / step
        elif minus_in_domain:
            # Backward difference: lin point too close to soc_domain_max,
            # so we evaluate at (s - step, s). Querying s + step would be
            # out-of-domain.
            a_minus, b_minus, b0_minus = coefficients_at(soc_scalar - step)
            a_lin, b_lin, b0_lin = coefficients_at(soc_scalar)
            da_ds = (a_lin - a_minus) / step
            db_ds = (b_lin - b_minus) / step
            db0_ds = (b0_lin - b0_minus) / step
            demf_ds = (emf_at(soc_scalar) - emf_at(soc_scalar - step)) / step
        else:
            # Neither side is in domain: the step is wider than the valid
            # SOC range. This shouldn't happen with a sensible step
            # (1e-3) and a normal SOC range; raise rather than return a
            # silently-degenerate answer.
            raise ValueError(
                f"SOC FD step {step} is wider than the valid SOC range "
                f"[{self._soc_domain_min}, {self._soc_domain_max}]; cannot "
                f"compute SOC derivatives at soc_scalar={soc_scalar}."
            )

        return da_ds, db_ds, db0_ds, demf_ds


    def _theta_coefficient_derivative(self, theta_index: int,
                                      trajectories: dict) -> float:
        """d(coefficient_value)/d(theta_k) at the linearization point.

        For a model coefficient like a_i = sum_j theta_j * basis_j, the
        derivative wrt theta_k is just basis_k evaluated at the current
        trajectories -- with a sign flip if it's an a-coefficient (because
        update_model_from_state writes -theta_k into the subterm parameter).

        Implementation: temporarily set the subterm's parameter to 1 (or -1
        for a-coefficients) and evaluate the single-element coefficient list.
        Restore the original parameter on exit.
        """
        subterm, coefficient_key = self._theta_info[theta_index]
        is_a_coefficient = coefficient_key.startswith('a_')
        saved = subterm.parameter
        try:
            subterm.parameter = -1.0 if is_a_coefficient else 1.0
            value = float(evaluate_coefficient([subterm], trajectories, 0))
        finally:
            subterm.parameter = saved
        return value


    # ----- construction helpers -----


    def _validate_and_deduplicate_state_components(self,
                                                state_components: list[str]) -> list[str]:
        """Validate state_components and deduplicate with a warning.

        Empty input raises ValueError. Invalid entries raise ValueError.
        Missing 'overpotentials' raises ValueError. Duplicate entries
        (valid but repeated) trigger a UserWarning and are dropped,
        preserving first-occurrence order.
        """
        if not state_components:
            raise ValueError(
                "state_components must not be empty. Provide at least "
                "'overpotentials', optionally combined with 's', 'capacity', "
                "or 'theta_{i}'."
            )

        invalid = [
            c for c in state_components
            if c not in {'s', 'overpotentials', 'capacity'}
            and not re.match(r'^theta_\d+$', c)
        ]
        if invalid:
            raise ValueError(
                f"Invalid entries in state_components: {invalid}. "
                f"Valid entries are 's', 'overpotentials', 'capacity', "
                f"and 'theta_{{i}}' for any positive integer i."
            )

        if 'overpotentials' not in state_components:
            raise ValueError(
                "state_components must contain 'overpotentials'. It is a "
                "required entry rather than an implicit one because the "
                "order of state_components defines the block layout of the "
                "state vector, and you will index into that layout when "
                "building initial states, reading filter output, and "
                "interpreting covariances. Listing 'overpotentials' "
                "explicitly keeps the full layout visible at the call site."
            )

        seen: set[str] = set()
        deduped: list[str] = []
        duplicates: list[str] = []
        for component in state_components:
            if component in seen:
                duplicates.append(component)
            else:
                seen.add(component)
                deduped.append(component)

        if duplicates:
            warnings.warn(
                f"Duplicate entries in state_components were ignored: {duplicates}. "
                f"The deduplicated layout is: {deduped}.",
                UserWarning,
                stacklevel=3,
            )

        return deduped


    # ----- argument validation -----


    def _validate_state_dimension(self, state: np.ndarray) -> None:
        """Check that `state` is a 1-D array of the expected length."""
        if state.ndim != 1:
            raise ValueError(
                f"state must be a 1-D array, got shape {state.shape}."
            )
        if state.shape[0] != self.state_dimension:
            raise ValueError(
                f"state has length {state.shape[0]}, expected {self.state_dimension} "
                f"(from state_components {self.state_components})."
            )


    def _resolve_soc_scalar(self, state: np.ndarray,
                            soc_value: float | None) -> float:
        """Return the SOC scalar according to the active SOC source."""
        if self._soc_source is SocSource.STATE:
            return float(state[self._soc_offset])

        if soc_value is None:
            raise ValueError(
                "'soc_value' is required when SOC source is SocSource.EXOGENOUS."
            )
        return float(soc_value)


    # ----- state-vector layout helpers -----


    def _compute_block_size(self, component: str) -> int:
        if component == 'overpotentials':
            return self.model_order
        return 1


    def _compute_offset_of(self, target: str) -> int | None:
        if target not in self.state_components:
            return None
        offset = 0
        for component in self.state_components:
            if component == target:
                return offset
            offset += self._compute_block_size(component)
        return None  # unreachable


    # ----- trajectory + dynamics helpers -----


    def _build_signal_trajectories(
            self,
            soc_values: float | list,
            current_values: float | list,
            temperature_values: float | list | None) -> dict:
        """Build a merged signal-trajectories dict for coefficient evaluation.

        Accepts scalars (for single-point evaluation) or lists (for windowed
        evaluation). The trajectory length L is inferred from soc_values /
        current_values.

        temperature_values = None skips the 'T' signal entirely (for
        temperature-independent models).

        Returns a single dict suitable for passing to evaluate_coefficient,
        formed by merging the (io, p, h) trajectory dicts that
        generate_signal_trajectories returns.
        """
        def as_list(v):
            if isinstance(v, list):
                return v
            if isinstance(v, np.ndarray):
                return v.tolist()
            return [v]

        soc_list = as_list(soc_values)
        current_list = as_list(current_values)

        signals = [
            Signal('s', soc_list, lambda x: x),
            Signal('i', current_list, lambda x: x),
            Signal('d', [np.sign(i) for i in current_list], lambda x: x),
        ]
        if temperature_values is not None:
            signals.append(Signal('T', as_list(temperature_values), lambda x: x))

        signal_vector = SignalVector(signals)

        basis_function_signals = generate_basis_function_signals(
            self.basis_functions, signal_vector
        )

        input_output_signals = [signal_vector.find('i')]

        order = len(soc_list) - 1
        io_traj, p_traj, h_traj = generate_signal_trajectories(
            (input_output_signals, basis_function_signals.signals, []),
            model_order=order, no_of_initial_values=order,
        )
        return io_traj | p_traj | h_traj


    def _evaluate_lpv_coefficients(
            self, trajectories: dict
        ) -> tuple[np.ndarray, np.ndarray, float]:
        """Evaluate the LPV coefficients (a_1..a_n, b_1..b_n, b_0) at the
        given trajectories.

        Returns (a_value, b_value, b0_value), where a_value and b_value have
        shape (model_order,) and b0_value is a scalar. Centralizes the
        evaluation pattern used by linearize, _soc_derivatives, and
        _evaluate_overpotentials.
        """
        n = self.model_order
        a_value = np.array([
            evaluate_coefficient(self.coefficients[f'a_{i}'], trajectories, 0)
            for i in range(1, n + 1)
        ], dtype=float)
        b_value = np.array([
            evaluate_coefficient(self.coefficients[f'b_{i}'], trajectories, 0)
            for i in range(1, n + 1)
        ], dtype=float)
        b0_value = float(evaluate_coefficient(self.coefficients['b_0'], trajectories, 0))
        return a_value, b_value, b0_value


    def _evaluate_overpotentials(self, x: np.ndarray, current_value: float,
                                 trajectories: dict) -> np.ndarray:
        """One-step companion-form update for the overpotentials sub-vector."""
        n = self.model_order
        a_value, b_value, b0_value = self._evaluate_lpv_coefficients(trajectories)

        x_next = np.empty(n)
        for delay_index in range(1, n + 1):
            a_contrib = -a_value[delay_index - 1] * x[0]
            b_contrib = (b_value[delay_index - 1]
                         - a_value[delay_index - 1] * b0_value) * current_value
            shift_contrib = x[delay_index] if delay_index < n else 0.0
            x_next[delay_index - 1] = a_contrib + b_contrib + shift_contrib

        return x_next
