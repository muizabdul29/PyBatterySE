# PyBatterySE

<div>

[![release](https://img.shields.io/github/v/release/muizabdul29/PyBatterySE)](https://github.com/muizabdul29/PyBatterySE/releases)
[![Pylint](https://github.com/muizabdul29/PyBatterySE/actions/workflows/pylint.yml/badge.svg)](https://github.com/muizabdul29/PyBatterySE/actions/workflows/pylint.yml)

</div>


**PyBatterySE** — a shorthand for **Py**thon **Battery** **S**tate **E**stimation, is an open-source library for state estimation using Bayesian filters and linear parameter-varying (LPV) battery models.

> ⚠️ **Beta:** This package is in beta. Core functionality works, but APIs may change before the stable release. Feedback and bug reports are welcome!

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyBatterySE.

```bash
pip install pybatteryse
```

## Requirements

PyBatterySE is a companion package to PyBatteryID, and utilises the models identified using PyBatteryID for state estimation. Whilst it is possible to use a custom model that follows the same format as PyBatteryID, it is recommended to use models identified using PyBatteryID for state estimation with PyBatterySE.

## Basic usage

In the following, an example usage of PyBatterySE has been demonstrated for performing state estimation of batteries, including SOC estimation, and ageing-aware parameter (and capacity) estimation.

#### 1. SOC estimation

For SOC estimation, we currently have two options, namely (i) extended Kalman filter (EKF), and (ii) particle filter (PF). In both cases, the state corresponding to overpotential model obtained using PyBatteryID is augmented with an extra SOC state, that is, $x = [s \ x_1 \ \cdots \ x_n]^\top$. Below, we give two basic example usages for SOC estimation using EKF and PF.

**Example 1: SOC estimation using EKF**

```python
from pybatteryid.utilities import load_model_from_file
from pybatteryse.filters.ekf import ExtendedKalmanFilter

model = load_model_from_file('path/to/model.npy')
dataset = helper.load_npy_datasets(f'path/to/dataset.npy')
bad_current_measurements = [ ... ] # Bad current measurements

# Initialize EKF
ekf = ExtendedKalmanFilter(
    model=model,
    sigma_nu=1e-3,        # Input noise variance
    sigma_ny_ne=1e-3      # Measurement noise variance
)
 
# Prepare initial conditions
x0 = np.array([[0.5],    # Initial SOC
               [0.0],    # state 1
               [0.0],    # state 2
               [0.0]])   # state 3
 
P0 = np.diag([1, 0.1, 0.1, 0.1])  # Initial covariance
 
# Run filter
state_estimates = ekf.run(
    temperature_values=dataset['temperature_values'],
    current_values=bad_current_measurements,
    voltage_values=dataset['voltage_values'],
    initial_state=x0,
    initial_covariance=P0
)
 
# Extract SOC estimates
soc_estimates = state_estimates[:, 0]
```

**Example 2: SOC estimation using PF**

```python
from pybatteryid.utilities import load_model_from_file
from pybatteryse.filters.ekf import ParticleFilter

model = load_model_from_file('path/to/model.npy')
dataset = helper.load_npy_datasets(f'path/to/dataset.npy')
bad_current_measurements = [ ... ] # Bad current measurements

# Initialize PF
pf = ParticleFilter(
    model,
    num_particles=5,
    eta_bounds=(-4, -1),
    sigma_ny_ne=1e-3
)
 
initial_particles = np.hstack((
    np.random.uniform(0.0001, 0.9999, size=(pf.num_particles, 1)),
    np.zeros((pf.num_particles, model.model_order))
))

state_estimates = pf.run(initial_particles=initial_particles,
                         temperature_values=temperature_values,
                         current_values=bad_current_measurements,
                         voltage_values=voltage_values)

soc_estimates = state_estimates[:, 0]

```

#### 2. Ageing-aware model estimation

We can perform ageing-aware model estimation using recursive state estimation. The detailed methodology is explained in [X], which proposes alternative approaches as well. An example has been provided in the examples folder, which can be consulted for more details.

