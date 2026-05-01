"""Shared helpers for EKF and PF filter implementations."""

import numpy as np

from ..statespace import SocSource


def check_nonnegative(name: str, value: float) -> None:
    """Raise ValueError if value is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


def validate_optional_variances(statespace, variance_eta_theta, variance_eta_capacity) -> None:
    """Validate that optional process-noise variances are supplied
    when their state components are active.

    Raises ValueError if a required variance is missing or negative.
    """
    components = statespace.state_components
    optional = [
        ("variance_eta_theta",    variance_eta_theta,
         any(c.startswith("theta_") for c in components), "any 'theta_*' entry"),
        ("variance_eta_capacity", variance_eta_capacity,
         "capacity" in components, "'capacity'"),
    ]
    for name, value, required, context in optional:
        if required:
            if value is None:
                raise ValueError(
                    f"{name} is required when state_components contains {context}."
                )
            check_nonnegative(name, value)


def parse_dataset(dataset: dict, soc_source) -> tuple:
    """Parse and validate a filter dataset dict.

    Returns (current_values, voltage_values, temperature_values, soc_values, num_timesteps).
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
    if soc_source is not SocSource.STATE and soc_values is None:
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

    return current_values, voltage_values, temperature_values, soc_values, num_timesteps
