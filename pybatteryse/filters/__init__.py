"""Filter loading utilities.

This module provides a factory function for instantiating state estimation
filters by name, along with a registry of supported filter types.
"""

from .ekf import ExtendedKalmanFilter
from .pf import ParticleFilter


SUPPORTED_FILTERS = {
    "extended_kalman_filter": ExtendedKalmanFilter,
    "particle_filter": ParticleFilter,
}


def load_filter(filter_name, **filter_parameters):
    """Load a filter by name with the given parameters.

    Args:
        filter_name: Name of the filter to load. Must be a key in
            SUPPORTED_FILTERS.
        **filter_parameters: Keyword arguments passed to the filter's
            constructor.

    Returns:
        An instantiated filter object.

    Raises:
        ValueError: If filter_name is not a recognized filter.
    """
    try:
        filter_cls = SUPPORTED_FILTERS[filter_name]
    except KeyError as exc:
        valid = ", ".join(sorted(SUPPORTED_FILTERS))
        raise ValueError(
            f"Unknown filter '{filter_name}'. Valid options: {valid}"
        ) from exc

    return filter_cls(**filter_parameters)
