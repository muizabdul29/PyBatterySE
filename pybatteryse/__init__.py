"""State estimation package for battery models.

Public API
----------
load_statespace_representation : Build a StateSpace from a model and state components.
load_filter                     : Instantiate a named estimation filter.
"""

from pybatteryid.dataclasses import Model

from .statespace import StateSpace
from .filters import ExtendedKalmanFilter, ParticleFilter


SUPPORTED_FILTERS = {
    "extended_kalman_filter": ExtendedKalmanFilter,
    "particle_filter": ParticleFilter,
}


def load_statespace_representation(model: Model, state_components: list[str]) -> StateSpace:
    """Build a StateSpace representation for the given model and state components.

    Args:
        model: Identified battery model.
        state_components: Names of the state variables (e.g. ``["soc", "rc1"]``).

    Returns:
        A :class:`StateSpace` ready to pass to :func:`load_filter`.
    """
    return StateSpace(model, state_components)


def load_filter(statespace: StateSpace, filter_name: str, **filter_parameters):
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

    return filter_cls(statespace, **filter_parameters)
