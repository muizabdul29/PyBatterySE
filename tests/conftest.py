"""Configure tests."""

import os
import pytest

from pybatteryid.utilities import load_model_from_file


@pytest.fixture(scope="session")
def model():
    """Load an example model identified using PyBatteryID. """
    #
    parent_directory = os.path.dirname(os.path.dirname(__file__))
    m = load_model_from_file(f'{parent_directory}/examples/data/'
                             'nmc_soc_estimation/model_lpv_n,l=3,3.npy')
    return m
