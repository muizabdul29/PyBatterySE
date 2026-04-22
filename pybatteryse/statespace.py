"""Utilities concerning model state-space representation."""

import re
import warnings
from enum import Enum

import numpy as np
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from pybatteryid.basisfunctions import generate_basis_function_signals, generate_signal_trajectories
from pybatteryid.dataclasses import BasisFunction, Model, Signal, SignalVector, VoltageFunction

from .coefficient import evaluate_coefficient, extract_model_coefficients, Coefficients


_VALID_STATE_LITERALS = {'s', 'overpotentials', 'capacity'}
_THETA_PATTERN = re.compile(r'^theta_\d+$')


class SocSource(Enum):
    """Where the state of charge comes from for coefficient/EMF evaluation."""
    STATE = 'state'           # 's' is in state_components; SOC is read from the state vector.
    EXOGENOUS = 'exogenous'   # 's' is not in state_components; the caller supplies SOC.


class OverpotentialEquation(Enum):
    """Which equation produces the overpotential v_k.

    The terminal voltage V_k = EMF(s_k, T_k) + v_k is assembled by
    evaluate_predicted_measurement; this axis only controls how v_k is
    computed.
    """
    STATE_SPACE = 'state_space'     # v_k = x_1 + b_0 * i_k           (companion-form realization)
    INPUT_OUTPUT = 'input_output'   # v_k = Σ_j θ_j · φ_j(p_k)         (input-output form)


# pylint: disable=R0902
class StateSpace:
    """State-space representation of an identified battery model.

    The class selects its parameterization along two independent axes
    based on state_components:

    - SOC source (self._soc_source): whether state of charge is tracked inside
      the state vector (SocSource.STATE, when 's' is in state_components) or
      provided by the caller as an exogenous input (SocSource.EXOGENOUS).

    - Overpotential equation (self._overpotential_equation): whether the
      overpotential v_k is produced from a state-space realization
      (OverpotentialEquation.STATE_SPACE, the companion-form block when
      'overpotentials' is in state_components) or from the input-output
      equation (OverpotentialEquation.INPUT_OUTPUT,
      v_k = Σ_j θ_j · φ_j(p_k), otherwise).

    All four combinations of the two axes are valid filter models. 'theta_{i}'
    and 'capacity' can be included in any combination as random-walk states.

    Measurement argument conventions:
        - Present values (current_value, temperature_value, soc_value) are
          scalars. temperature_value is optional and should be None when the
          model is not temperature-dependent.
        - Past values (past_current_values, past_voltage_values,
          past_temperature_values, past_soc_values) are 1-D arrays of length
          model_order, oldest-first (index 0 = oldest sample, index -1 =
          most recent past).
        - past_voltage_values holds TERMINAL voltages V_{k-j}
          (i.e. the filter's observations); this class converts them
          internally to overpotentials v_{k-j} = V_{k-j} - EMF(s_{k-j}, T_{k-j})
          when the input-output equation is active.
        - soc_value / past_soc_values are unused when the SOC source is
          SocSource.STATE (silently ignored).
        - past_* arrays are required only when the overpotential equation is
          OverpotentialEquation.INPUT_OUTPUT; they are silently ignored
          otherwise.
        - temperature_value is passed through without inspection. For a
          temperature-independent model, pass None (past_temperature_values
          can likewise be a list/array of Nones); a temperature-dependent
          model called without a real temperature will fail downstream when
          a basis function tries to read 'T', which is not this class's
          concern.

    The caller is responsible for passing arguments of the correct shape
    and length; no input validation is performed beyond the SOC
    required-when-exogenous check.

    Attributes:
        battery_capacity: Battery capacity used for SOC integration.
        sampling_period: Discrete-time sampling period.
        emf_function: Open-circuit (EMF) voltage as a function of SOC and temperature.
        model_order: Order of the overpotential sub-system; also the history
            window length whenever OverpotentialEquation.INPUT_OUTPUT is active.
        model_estimate: Flat parameter vector theta used for the input-output
            equation. Kept in sync with self.coefficients by
            update_model_parameters.
        model_terms: Ordered list of symbol-product strings (e.g.
            'v(k-1)×s(k)×log[s](k)') that index into the trajectory dict
            when evaluating the input-output equation.
        basis_functions: Basis functions used by the model.
        coefficients: Model coefficients keyed by name (a_1, ..., b_0, b_1, ...).
        state_components: Ordered list of string identifiers specifying
            the components of the state vector, block by block. Valid
            entries:
                - 's'              : state of charge.
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

        # The following attributes are derived from state_components: the two
        # configuration axes (SOC source, overpotential equation) and the state-vector
        # layout sizing (per-component sizes and the offsets of 's' and 'overpotentials').
        self._soc_source: SocSource = (
            SocSource.STATE if 's' in self.state_components else SocSource.EXOGENOUS
        )
        self._overpotential_equation: OverpotentialEquation = (
            OverpotentialEquation.STATE_SPACE
            if 'overpotentials' in self.state_components
            else OverpotentialEquation.INPUT_OUTPUT
        )
        self._block_sizes: dict[str, int] = {
            component: self._compute_block_size(component)
            for component in self.state_components
        }
        self._soc_offset: int | None = self._compute_offset_of('s')
        self._overpotentials_offset: int | None = self._compute_offset_of('overpotentials')
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
                if self._overpotential_equation is OverpotentialEquation.STATE_SPACE:
                    next_state[offset] = state[offset] \
                        + (self.sampling_period / self.battery_capacity) * current_value
                else:
                    # 's' alone: shift register.
                    # [s_k, ..., s_{k-n}] -> [s_{k+1}, s_k, ..., s_{k-n+1}]
                    soc_k = state[offset]
                    next_state[offset] = soc_k \
                        + (self.sampling_period / self.battery_capacity) * current_value
                    next_state[offset + 1:offset + size] = state[offset:offset + size - 1]
            elif component == 'overpotentials':
                next_state[offset:offset + size] = self._evaluate_overpotentials(
                    state[offset:offset + size], current_value, trajectories
                )
            elif component == 'capacity' or component.startswith('theta_'):
                next_state[offset] = state[offset]
            offset += size

        return next_state


    # pylint: disable=R0913,R0914,R0917
    def evaluate_predicted_measurement(self, state: np.ndarray,
                                       current_value: float,
                                       temperature_value: float | None = None,
                                       soc_value: float | None = None,
                                       past_current_values: np.ndarray | None = None,
                                       past_voltage_values: np.ndarray | None = None,
                                       past_temperature_values: np.ndarray | None = None,
                                       past_soc_values: np.ndarray | None = None) -> float:
        """Compute the scalar predicted terminal voltage:
        V_k = EMF(s_k, T_k) + v_k.

        This is the measurement function h of the state-space model
        (also called h_func in filterpy conventions). v_k is the overpotential
        (state-space or input-output form, depending on configuration); EMF
        is the open-circuit component.

        Argument conventions:
            - past_voltage_values are past TERMINAL voltages V_{k-n}, ..., V_{k-1},
              i.e. what the filter has observed. When the input-output
              overpotential equation is active, this method converts them
              internally to overpotentials v_{k-j} = V_{k-j} - EMF(s_{k-j}, T_{k-j})
              before handing them to the input-output equation; callers
              therefore do not subtract EMF themselves.
            - past_current_values, past_temperature_values, past_soc_values
              carry the corresponding past inputs at the same time steps.
            - past_* arrays have length model_order, oldest-first.

        temperature_value is optional; pass None for temperature-independent
        models. past_temperature_values is concatenated with temperature_value
        without inspection; if either is None, the resulting list carries
        Nones and will fail downstream only if some basis function actually
        reads the 'T' signal.
        """
        self._validate_state_dimension(state)
        soc_scalar = self._resolve_soc_scalar(state, soc_value)

        emf = float(self.emf_function(soc_scalar, temperature_value))

        if self._overpotential_equation is OverpotentialEquation.STATE_SPACE:
            return emf + self._evaluate_overpotential_using_state_space(
                state, current_value, temperature_value, soc_scalar
            )

        # INPUT_OUTPUT path: assemble oldest-first lists.
        if self._soc_source is SocSource.STATE:
            # SOC history in state is newest-first; reverse for oldest-first.
            window = self.model_order + 1
            soc_values = state[self._soc_offset:self._soc_offset + window][::-1].tolist()
        else:
            soc_values = list(past_soc_values) + [soc_value]

        current_values = list(past_current_values) + [current_value]
        temperature_values = list(past_temperature_values) + [temperature_value]

        # Convert past terminal voltages V_{k-j} to overpotentials
        # v_{k-j} = V_{k-j} - EMF(s_{k-j}, T_{k-j}) before handing them to the
        # input-output equation. soc_values and temperature_values have length
        # model_order + 1; slice off the current-step entry at index -1.
        past_overpotential_values = [
            V_past - float(self.emf_function(s_past, T_past))
            for V_past, s_past, T_past in zip(
                past_voltage_values,
                soc_values[:-1],
                temperature_values[:-1],
            )
        ]

        return emf + self._evaluate_overpotential_using_input_output(
            soc_values=soc_values,
            current_values=current_values,
            past_overpotential_values=past_overpotential_values,
            temperature_values=temperature_values,
        )


    def update_model_parameters(self, model_estimate: list[float]) -> None:
        """Update coefficient parameters in-place from a flat model_estimate vector.

        Both self.coefficients (with flipped signs for a-terms) and
        self.model_estimate (unflipped) are updated so they stay in sync.
        """
        all_keyed_subterms = [
            (subterm, coefficient_key.startswith('a'))
            for coefficient_key, subterms in self.coefficients.items()
            for subterm in subterms
        ]
        if len(model_estimate) != len(all_keyed_subterms):
            raise ValueError(
                f"Parameter count mismatch: expected {len(all_keyed_subterms)} parameters "
                f"but got {len(model_estimate)}."
            )
        for (subterm, is_a_coefficient), new_parameter in zip(all_keyed_subterms, model_estimate):
            subterm.parameter = -new_parameter if is_a_coefficient else new_parameter

        self.model_estimate = np.asarray(model_estimate, dtype=float)


    # pylint: disable=R0912
    def print_equations(self) -> None:
        """Print the state-transition and measurement equations for this configuration.

        Shows the signatures of f and h (with or without an explicit input u_k,
        depending on the overpotential equation) along with the concrete
        composition of x_k and the scheduling variable p_k (LPV representation).
        """
        ss_form = self._overpotential_equation is OverpotentialEquation.STATE_SPACE
        n = self.model_order

        console = Console()

        # Equation signatures.
        if ss_form:
            eq_lines = [
                Text("x_{k+1} = f(x_k, p_k, u_k)", style="bold"),
                Text("V_k     = h(x_k, p_k, u_k)", style="bold"),
            ]
        else:
            eq_lines = [
                Text("x_{k+1} = f(x_k, p_k)", style="bold"),
                Text("V_k     = h(x_k, p_k)", style="bold"),
            ]

        decomposition_note = Text(
            "where V_k = EMF(s_k, T_k) + v_k   "
            "(terminal voltage = open-circuit + overpotential)",
            style="dim italic",
        )

        # x_k composition, block by block, in the order of state_components.
        x_k_parts: list[str] = []
        for component in self.state_components:
            if component == 's':
                if 'overpotentials' in self.state_components:
                    x_k_parts.append("s_k")
                else:
                    if n <= 4:
                        x_k_parts.append("s_k")
                        x_k_parts.extend(f"s_{{k-{i}}}" for i in range(1, n + 1))
                    else:
                        x_k_parts.extend(["s_k", "s_{k-1}", "...", "s_{k-n}"])
            elif component == 'overpotentials':
                if n <= 4:
                    x_k_parts.extend(f"x_{i},k" for i in range(1, n + 1))
                else:
                    x_k_parts.extend(["x_1,k", "x_2,k", "...", f"x_{n},k"])
            elif component == 'capacity':
                x_k_parts.append("capacity_k")
            elif component.startswith('theta_'):
                idx = component[len('theta_'):]
                x_k_parts.append(f"θ_{idx},k")

        x_k_line = Text.assemble(
            ("x_k = [", "bold"),
            (", ".join(x_k_parts), "bold cyan"),
            ("]", "bold"),
            (f" with dim(x_k) = {self.state_dimension}.", "dim italic"),
        )

        # p_k composition.
        if ss_form:
            p_k_line = Text.assemble(
                ("p_k = [", "bold"),
                ("s_k, i_k, T_k", "bold cyan"),
                ("]", "bold"),
            )
        else:
            history_parts = [
                f"s_{{k-{n}..k}}",
                f"i_{{k-{n}..k}}",
                f"v_{{k-{n}..k-1}}",
                f"T_{{k-{n}..k}}",
            ]
            p_k_line = Text.assemble(
                ("p_k = [", "bold"),
                (", ".join(history_parts), "bold cyan"),
                ("]", "bold"),
            )

        # u_k, only when it appears in the signatures.
        u_k_line: Text | None = (
            Text.assemble(("u_k = ", "bold"), ("i_k", "bold cyan"))
            if ss_form
            else None
        )

        # Assemble the single panel.
        body_items = [*eq_lines, Text(""), decomposition_note, Text(""), x_k_line, p_k_line]
        if u_k_line is not None:
            body_items.append(u_k_line)

        console.print(Panel(
            Group(*body_items),
            title="[bold]State-space configuration[/bold]",
            border_style="magenta",
            expand=False,
        ))


    # ----- construction helpers -----


    def _validate_and_deduplicate_state_components(self,
                                                state_components: list[str]) -> list[str]:
        """Validate state_components and deduplicate with a warning.

        Empty input raises ValueError. Invalid entries raise ValueError.
        Duplicate entries (valid but repeated) trigger a UserWarning and are
        dropped, preserving first-occurrence order.
        """
        if not state_components:
            raise ValueError(
                "state_components must not be empty. Provide at least one of "
                "'s', 'overpotentials', 'capacity', or 'theta_{i}'."
            )

        invalid = [
            c for c in state_components
            if c not in _VALID_STATE_LITERALS and not _THETA_PATTERN.match(c)
        ]
        if invalid:
            raise ValueError(
                f"Invalid entries in state_components: {invalid}. "
                f"Valid entries are 's', 'overpotentials', 'capacity', "
                f"and 'theta_{{i}}' for any positive integer i."
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


    # ----- overpotential equations -----


    def _evaluate_overpotential_using_state_space(self, state: np.ndarray,
                                                  current_value: float,
                                                  temperature_value: float | None,
                                                  soc_value: float) -> float:
        """State-space form: v_k = x_1 + b_0 * i_k."""
        io_traj, p_traj, h_traj = self._build_signal_trajectories(
            soc_values=soc_value,
            current_values=current_value,
            temperature_values=temperature_value,
        )
        trajectories = io_traj | p_traj | h_traj

        x1 = float(state[self._overpotentials_offset])
        b0 = evaluate_coefficient(self.coefficients['b_0'], trajectories, 0)
        return float(x1 + b0 * current_value)


    def _evaluate_overpotential_using_input_output(self, soc_values: list,
                                                   current_values: list,
                                                   past_overpotential_values: list,
                                                   temperature_values: list) -> float:
        """Input-output equation: v_k = Σ_j θ_j · φ_j(p_k).

        Each model term is a product of lag-annotated symbols like
        'v(k-1)×s(k)×log[s](k)', where the lag is encoded in the symbol name.
        Signals are expanded into lag-annotated entries by
        _build_signal_trajectories, then each symbol is looked up at its
        single resolved time step. The dot product with self.model_estimate
        gives the scalar overpotential output.

        past_overpotential_values holds v_{k-n}, ..., v_{k-1}
        (overpotentials, not terminal voltages). The current-step
        overpotential v_k is a None placeholder since no model term reads it.

        All arguments must be 1-D Python lists; soc_values, current_values,
        and temperature_values have length model_order + 1,
        past_overpotential_values has length model_order. Oldest-first.
        """
        overpotential_values = past_overpotential_values + [None]  # v_k placeholder

        io_traj, p_traj, h_traj = self._build_signal_trajectories(
            soc_values=soc_values,
            current_values=current_values,
            temperature_values=temperature_values,
            overpotential_values=overpotential_values,
        )
        trajectories = io_traj | p_traj | h_traj

        phi = [
            np.prod([trajectories[sym][0] for sym in term.split('×')])
            for term in self.model_terms
        ]

        return float(np.dot(phi, self.model_estimate))

    # ----- state-vector layout helpers -----


    def _compute_block_size(self, component: str) -> int:
        if component == 'overpotentials':
            return self.model_order
        if component == 's' and 'overpotentials' not in self.state_components:
            return self.model_order + 1  # 's'-alone shift register
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
            temperature_values: float | list | None,
            overpotential_values: list | None = None) -> tuple[dict, dict, dict]:
        """Build signal trajectories for coefficient evaluation or input-output assembly.

        Accepts scalars (for single-point evaluation) or lists (for windowed
        evaluation). The trajectory length L is inferred from soc_values /
        current_values.

        temperature_values = None skips the 'T' signal entirely (for
        temperature-independent models). overpotential_values is optional;
        when provided, a 'v' signal is included in the input-output signals
        (needed for the input-output overpotential equation path).

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
        if overpotential_values is not None:
            input_output_signals.append(Signal('v', overpotential_values, lambda x: x))

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
