"""
Testing placeholder
"""

import numpy as np

from pybatteryid.dataclasses import Model
from pybatteryse.statespace import StateSpace
from pybatteryse.coefficient import evaluate_coefficient


def test_state_transition_soc(model: Model):
    """Test state transition for SOC state"""

    ss = StateSpace(model=model, state_components=['s'])

    state = np.array([0.5, 0.51, 0.52, 0.53])
    current_value = -1
    temperature_value = 20
    #
    next_state = ss.evaluate_next_state(state=state,
                                        current_value=current_value,
                                        temperature_value=temperature_value)

    # Coulomb counting: SOC(k+1) = SOC(k) - (I * Ts) / Q
    # Negative current = discharge, so with current_value=-1 SOC increases
    expected_soc = state[0] + (current_value * model.sampling_period) / model.battery_capacity
    expected_next_state = np.array([expected_soc, 0.5, 0.51, 0.52])
    #
    np.testing.assert_allclose(next_state, expected_next_state)


# pylint: disable=R0914
def test_state_transition_soc_w_overpotentials(model: Model):
    """Test state transition for SOC state w/ overpotentials"""

    ss = StateSpace(model=model, state_components=['s', 'overpotentials'])

    state = np.array([0.5, 0.1, 0.2, 0.3])
    current_value = -1
    temperature_value = 20
    #
    next_state = ss.evaluate_next_state(state=state,
                                        current_value=current_value,
                                        temperature_value=temperature_value)

    # Coulomb counting: SOC(k+1) = SOC(k) - (I * Ts) / Q
    # Negative current = discharge, so with current_value=-1 SOC increases
    expected_soc = state[0] + (current_value * model.sampling_period) / model.battery_capacity
    #
    io_traj, p_traj, h_traj = ss._build_signal_trajectories(
        soc_values=state[0],
        current_values=current_value,
        temperature_values=temperature_value,
    )
    trajectories = io_traj | p_traj | h_traj
    #
    a_1 = evaluate_coefficient(ss.coefficients['a_1'], trajectories, 0)
    a_2 = evaluate_coefficient(ss.coefficients['a_2'], trajectories, 0)
    a_3 = evaluate_coefficient(ss.coefficients['a_3'], trajectories, 0)
    b_0 = evaluate_coefficient(ss.coefficients['b_0'], trajectories, 0)
    b_1 = evaluate_coefficient(ss.coefficients['b_1'], trajectories, 0)
    b_2 = evaluate_coefficient(ss.coefficients['b_2'], trajectories, 0)
    b_3 = evaluate_coefficient(ss.coefficients['b_3'], trajectories, 0)
    #
    first_overpotential_state = -a_1 * state[1] + state[2] + (b_1 - a_1 * b_0) * current_value
    second_overpotential_state = -a_2 * state[1] + state[3] + (b_2 - a_2 * b_0) * current_value
    third_overpotential_state = -a_3 * state[1] + (b_3 - a_3 * b_0) * current_value
    #
    expected_next_state = np.array([expected_soc,
                                    first_overpotential_state,
                                    second_overpotential_state,
                                    third_overpotential_state])
    #
    np.testing.assert_allclose(next_state, expected_next_state)


def test_predicted_measurement_using_io_equation(model: Model):
    """Test predicted measurement with SOC state"""

    ss = StateSpace(model=model, state_components=['s'])

    state = np.array([0.5, 0.51, 0.52, 0.53])
    current_value = -1
    temperature_value = 20
    past_current_values = [-0.1, 0.1, -0.5]
    past_temperature_values = [19, 19.2, 19.5]
    past_voltage_values = [3.8, 3.81, 3.82]
    #
    predicted_output = ss.evaluate_predicted_measurement(state=state,
                                                         current_value=current_value,
                                                         temperature_value=temperature_value,
                                                         past_current_values=past_current_values,
                                                         past_temperature_values=past_temperature_values,
                                                         past_voltage_values=past_voltage_values)

    io_traj, p_traj, h_traj = ss._build_signal_trajectories(
        soc_values=state[::-1],
        current_values=past_current_values + [current_value],
        temperature_values=past_temperature_values + [temperature_value],
    )
    trajectories = io_traj | p_traj | h_traj
    #
    a_1 = evaluate_coefficient(ss.coefficients['a_1'], trajectories, -1)
    a_2 = evaluate_coefficient(ss.coefficients['a_2'], trajectories, -2)
    a_3 = evaluate_coefficient(ss.coefficients['a_3'], trajectories, -3)
    b_0 = evaluate_coefficient(ss.coefficients['b_0'], trajectories, 0)
    b_1 = evaluate_coefficient(ss.coefficients['b_1'], trajectories, -1)
    b_2 = evaluate_coefficient(ss.coefficients['b_2'], trajectories, -2)
    b_3 = evaluate_coefficient(ss.coefficients['b_3'], trajectories, -3)
    #
    past_overpotential_values = [
        V_past - float(ss.emf_function(s_past, T_past))
        for V_past, s_past, T_past in zip(
            past_voltage_values,
            state[::-1][:-1],
            past_temperature_values,
        )
    ]
    #
    output = model.emf_function(state[0], temperature_value) \
        - a_1 * past_overpotential_values[-1] - a_2 * past_overpotential_values[-2] \
        - a_3 * past_overpotential_values[-3] + b_0 * current_value \
        + b_1 * past_current_values[-1] + b_2 * past_current_values[-2] \
        + b_3 * past_current_values[-3]

    np.testing.assert_allclose(output, predicted_output)

def test_predicted_measurement_using_ss(model: Model):
    """Test predicted measurement with SOC and overpotential states"""

    ss = StateSpace(model=model, state_components=['s', 'overpotentials'])

    state = np.array([0.5, 0.01, 0.02, 0.03])
    current_value = -1
    temperature_value = 25
    #
    predicted_output = ss.evaluate_predicted_measurement(state=state,
                                                         current_value=current_value,
                                                         temperature_value=temperature_value)

    io_traj, p_traj, h_traj = ss._build_signal_trajectories(
        soc_values=state[0],
        current_values=current_value,
        temperature_values=temperature_value,
    )
    trajectories = io_traj | p_traj | h_traj
    #
    b_0 = evaluate_coefficient(ss.coefficients['b_0'], trajectories, 0)
    #
    output = model.emf_function(state[0], temperature_value) \
        + state[1] + b_0 * current_value

    np.testing.assert_allclose(output, predicted_output)
