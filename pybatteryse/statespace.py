"""Utilities concerning model state-space representation."""

from dataclasses import dataclass, field, InitVar

import numpy as np

from pybatteryid.basisfunctions import generate_basis_function_signals, \
    generate_signal_trajectories
from pybatteryid.dataclasses import Model, BasisFunction, Signal, SignalVector

from .coefficient import extract_model_coefficients, evaluate_coefficient, Coefficients


@dataclass
class StateSpace:
    """State-space representation."""

    # Constructor inputs via InitVar — passed to __post_init__, not stored as fields
    model: InitVar[Model]

    # Derived fields (populated by __post_init__; no default -> must precede defaulted fields)
    model_order: int = field(init=False)
    basis_functions: list[BasisFunction] = field(init=False)
    coefficients: Coefficients = field(init=False)
    battery_capacity: float = field(init=False)
    sampling_period: int = field(init=False)

    def __post_init__(self, model: Model) -> None:
        self.model_order = model.model_order
        self.basis_functions = model.basis_functions
        self.battery_capacity = model.battery_capacity
        self.sampling_period = model.sampling_period
        self.coefficients = extract_model_coefficients(model.model_terms, model.model_estimate)


# pylint: disable=too-many-locals
def get_matrices(statespace_representation: StateSpace,
                 soc_value: float, current_value: float,
                 temperature_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute discrete-time state-space matrices (A, B, C, D) at a given operating point.

    The state vector is [SOC, x_1, x_2, ..., x_n].
    SOC dynamics are augmented as an integrator driven by current / battery_capacity.
    """

    ss = statespace_representation

    operating_point_signals = SignalVector([
        Signal('s', [soc_value], lambda x: x),
        Signal('i', [current_value], lambda x: x),
        Signal('d', [np.sign(current_value)], lambda x: x),
        Signal('T', [temperature_value], lambda x: x),
    ])

    basis_function_signals = generate_basis_function_signals(ss.basis_functions,
                                                             operating_point_signals)
    input_output_signals = [operating_point_signals.find('i')]

    io_traj, p_traj, h_traj = generate_signal_trajectories((input_output_signals,
                                                            basis_function_signals.signals,
                                                            []),
                                                            model_order=0, no_of_initial_values=0)
    all_signal_trajectories = io_traj | p_traj | h_traj

    # Build companion-form matrices for the voltage sub-system (without SOC row/col yet)
    matrix_a = np.zeros((ss.model_order, ss.model_order))
    matrix_a[:ss.model_order - 1, 1:] = np.eye(ss.model_order - 1)

    matrix_b = np.zeros((ss.model_order, 1))
    matrix_c = np.zeros(ss.model_order)
    matrix_c[0] = 1.0
    matrix_d = np.zeros((1, 1))

    b0 = evaluate_coefficient(ss.coefficients['b_0'], all_signal_trajectories, 0)

    traj = all_signal_trajectories
    for delay_index in range(1, ss.model_order + 1):
        a_coefficient = evaluate_coefficient(ss.coefficients[f'a_{delay_index}'], traj, 0)
        b_coefficient = evaluate_coefficient(ss.coefficients[f'b_{delay_index}'], traj, 0)
        matrix_a[delay_index - 1, 0] = -a_coefficient
        matrix_b[delay_index - 1, 0] = b_coefficient - a_coefficient * b0

    matrix_d[0, 0] = b0

    # Augment with SOC integrator state: x = [SOC, x_1, ..., x_n]
    matrix_a = np.block([
        [1, np.zeros((1, ss.model_order))],
        [np.zeros((ss.model_order, 1)), matrix_a],
    ])
    matrix_b = np.vstack([ss.sampling_period / ss.battery_capacity, matrix_b])
    matrix_c = np.concatenate([[0], matrix_c])

    return matrix_a, matrix_b, matrix_c, matrix_d
