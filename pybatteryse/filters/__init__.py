"""Filter loading utilities.

This module provides a factory function for instantiating state estimation
filters by name, along with a registry of supported filter types.
"""

from .ekf import ExtendedKalmanFilter
from .pf import ParticleFilter
