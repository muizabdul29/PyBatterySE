"""Utilities concerning model state-space representation."""

import re
import warnings
from enum import Enum

import numpy as np

from pybatteryid.basisfunctions import generate_basis_function_signals, generate_signal_trajectories
from pybatteryid.dataclasses import BasisFunction, Model, Signal, SignalVector, VoltageFunction

from .coefficient import evaluate_coefficient, extract_model_coefficients, Coefficients


class SocSource(Enum):
    """Where the state of charge comes from for coefficient/EMF evaluation."""
    STATE = 'state'           # 's' is in state_components; SOC is read from the state vector.
    EXOGENOUS = 'exogenous'   # 's' is not in state_components; the caller supplies SOC.


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

    _theta_subterm_map: list[tuple[object, bool]]

    def __init__(self, model: Model, state_components: list[str]) -> None:
        self.battery_capacity = model.battery_capacity
        self.sampling_period = model.sampling_period
        self.emf_function = model.emf_function
        self.model_order = model.model_order
        self.model_estimate = np.asarray(model.model_estimate, dtype=float)
        self.model_terms = list(model.model_terms)
        self.basis_functions = model.basis_functions
        self.coefficients = extract_model_coefficients(
            self.model_terms, self.model_estimate
        )
        self.state_components = self._validate_and_deduplicate_state_components(state_components)

        self._theta_subterm_map = [
            (subterm, coefficient_key.startswith('a'))
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

        io_traj, p_traj, h_traj = self._build_signal_trajectories(
            soc_values=soc_scalar,
            current_values=current_value,
            temperature_values=temperature_value,
        )
        trajectories = io_traj | p_traj | h_traj

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

        io_traj, p_traj, h_traj = self._build_signal_trajectories(
            soc_values=soc_scalar,
            current_values=current_value,
            temperature_values=temperature_value,
        )
        trajectories = io_traj | p_traj | h_traj

        x1 = float(state[self._overpotentials_offset])
        b0 = evaluate_coefficient(self.coefficients['b_0'], trajectories, 0)
        return emf + float(x1 + b0 * current_value)


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
                if not 0 <= theta_index < len(self._theta_subterm_map):
                    raise ValueError(
                        f"State component {component!r} refers to parameter "
                        f"index {theta_index + 1}, but the model only has "
                        f"{len(self._theta_subterm_map)} parameters."
                    )
                new_value = float(state[offset])
                self.model_estimate[theta_index] = new_value
                subterm, is_a_coefficient = self._theta_subterm_map[theta_index]
                subterm.parameter = -new_value if is_a_coefficient else new_value
            elif component == 'capacity':
                self.battery_capacity = float(state[offset])
            # 's' and 'overpotentials': nothing to write back.
            offset += size


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
            temperature_values: float | list | None) -> tuple[dict, dict, dict]:
        """Build signal trajectories for coefficient evaluation.

        Accepts scalars (for single-point evaluation) or lists (for windowed
        evaluation). The trajectory length L is inferred from soc_values /
        current_values.

        temperature_values = None skips the 'T' signal entirely (for
        temperature-independent models).

        Returns the tuple (io_trajectories, p_trajectories, h_trajectories)
        as produced by generate_signal_trajectories. Callers merge these
        dicts with `io | p | h` when they need a single lookup for
        evaluate_coefficient.
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
        return generate_signal_trajectories(
            (input_output_signals, basis_function_signals.signals, []),
            model_order=order, no_of_initial_values=order,
        )


    def _evaluate_overpotentials(self, x: np.ndarray, current_value: float,
                                 trajectories: dict) -> np.ndarray:
        """One-step companion-form update for the overpotentials sub-vector."""
        n = self.model_order
        x_next = np.empty(n)

        b0 = evaluate_coefficient(self.coefficients['b_0'], trajectories, 0)

        for delay_index in range(1, n + 1):
            a_coefficient = evaluate_coefficient(
                self.coefficients[f'a_{delay_index}'], trajectories, 0
            )
            b_coefficient = evaluate_coefficient(
                self.coefficients[f'b_{delay_index}'], trajectories, 0
            )
            a_contrib = -a_coefficient * x[0]
            b_contrib = (b_coefficient - a_coefficient * b0) * current_value
            shift_contrib = x[delay_index] if delay_index < n else 0.0
            x_next[delay_index - 1] = a_contrib + b_contrib + shift_contrib

        return x_next
