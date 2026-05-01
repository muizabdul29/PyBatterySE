# PyBatterySE

<div>

[![release](https://img.shields.io/github/v/release/muizabdul29/PyBatterySE)](https://github.com/muizabdul29/PyBatterySE/releases)
[![Pylint](https://github.com/muizabdul29/PyBatterySE/actions/workflows/pylint.yml/badge.svg)](https://github.com/muizabdul29/PyBatterySE/actions/workflows/pylint.yml)

</div>


**PyBatterySE** — a shorthand for **Py**thon **Battery** **S**tate **E**stimation, is an open-source library for state estimation using Bayesian filters and linear parameter-varying (LPV) battery models.


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install PyBatterySE.

```bash
pip install pybatteryse
```

## Requirements

PyBatterySE is a companion package to PyBatteryID, and utilises the models identified using PyBatteryID for state estimation. Whilst it is possible to use a custom model that follows the same format as PyBatteryID, it is recommended to use models identified using PyBatteryID for state estimation with PyBatterySE.

## Basic usage

In the following, example usages of PyBatterySE have been demonstrated for performing battery state estimation, including SOC estimation, SOC observability analysis, and ageing-aware parameter and capacity estimation.

#### 1. SOC estimation

For SOC estimation, two filter options are available: (i) extended Kalman filter (EKF), and (ii) particle filter (PF). In both cases, a `StateSpace` object is first constructed from the identified model, specifying which quantities form the state vector. The state is then augmented with the SOC: $x = [s \ x_1 \ \cdots \ x_n]^\top$.

**Example 1: SOC estimation using EKF**

```python
import numpy as np
from pybatteryid.utilities import load_model_from_file
from pybatteryse.statespace import StateSpace
from pybatteryse.filters import ExtendedKalmanFilter

model = load_model_from_file('path/to/model.npy')
dataset = {
    'current_values': current_values,   # may contain bias/noise
    'voltage_values': voltage_values,
    'temperature_values': temperature_values,
}

# Build state-space representation
ss = StateSpace(model, state_components=['s', 'overpotentials'])

# Initialize EKF
ekf = ExtendedKalmanFilter(
    statespace=ss,
    variance_eta_u=1e-2,        # Input (current) noise variance
    variance_eta_y_e=1e-3,      # Measurement (voltage) noise variance
)

# Initial conditions
initial_state = np.array([0.5] + [0.0] * model.model_order)
initial_covariance = np.diag([1.0] + [0.1] * model.model_order)

# Run filter
state_estimates, error_covariances = ekf.run(
    dataset=dataset,
    initial_state=initial_state,
    initial_covariance=initial_covariance,
)

# Extract SOC estimates
soc_estimates = state_estimates[:, 0]
```

**Example 2: SOC estimation using PF**

```python
import numpy as np
from pybatteryid.utilities import load_model_from_file
from pybatteryse.statespace import StateSpace
from pybatteryse.filters import ParticleFilter

model = load_model_from_file('path/to/model.npy')
dataset = {
    'current_values': current_values,
    'voltage_values': voltage_values,
    'temperature_values': temperature_values,
}

# Build state-space representation
ss = StateSpace(model, state_components=['s', 'overpotentials'])

# Initialize PF
pf = ParticleFilter(
    statespace=ss,
    num_particles=20,
    eta_bounds=(-4, -1),        # Uniform bounds for current-sensor bias
    variance_eta_y_e=1e-3,      # Measurement noise variance
)

# Run filter ('auto' initialises particles uniformly over the SOC domain)
state_estimates, _ = pf.run(
    dataset=dataset,
    initial_particles='auto',
)

soc_estimates = state_estimates[:, 0]
```

#### 2. SOC observability analysis

SOC observability can be quantified using finite-horizon observability Gramians, which measure the accumulated sensitivity of the terminal voltage to a unit SOC perturbation over a fixed time horizon. An example is provided in `examples/3_soc_observability_analysis.ipynb`.

```python
from pybatteryid.utilities import load_model_from_file
from pybatteryse.statespace import StateSpace
from pybatteryse.utilities import compute_soc_observability_contributions, simulate_state_trajectory
from pybatteryse.plotter import plot_soc_vs_gramian

model = load_model_from_file('path/to/model.npy')
ss = StateSpace(model, state_components=['s', 'overpotentials'])

state_trajectory = simulate_state_trajectory(ss, dataset)
gramian_contributions = compute_soc_observability_contributions(ss, state_trajectory, dataset)

plot_soc_vs_gramian(
    [(state_trajectory[:, 0], gramian_contributions['total'])],
)
```

#### 3. Ageing-aware model estimation

Joint estimation of battery capacity and ageing-related model parameters can be performed by extending the state vector with the parameters to be tracked. These *free parameters* follow random-walk dynamics and are estimated alongside the physical states using EKF. An example is provided in `examples/4_ageing_aware_model_estimation_with_ekf.ipynb`.

```python
import numpy as np
from pybatteryid.utilities import load_model_from_file
from pybatteryse.statespace import StateSpace
from pybatteryse.filters import ExtendedKalmanFilter

model = load_model_from_file('path/to/model.npy')
dataset = {
    'current_values': current_values,
    'voltage_values': voltage_values,
    'temperature_values': temperature_values,
}

# Augment state with free parameters and capacity
ss = StateSpace(model, state_components=['s', 'overpotentials', 'theta_3', 'theta_6', 'capacity'])

ekf = ExtendedKalmanFilter(
    statespace=ss,
    variance_eta_u=1e-3,
    variance_eta_y_e=1e-3,
    variance_eta_theta=1e-12,       # Process noise for theta random walks
    variance_eta_capacity=0.1,      # Process noise for capacity random walk
)

# Initial state: [soc, overpotentials..., theta_3, theta_6, capacity]
initial_state = np.array([0.5, 0.0, theta_3_mean, theta_6_mean, capacity_nominal])
initial_covariance = np.zeros((len(initial_state), len(initial_state)))

state_estimates, _ = ekf.run(
    dataset=dataset,
    initial_state=initial_state,
    initial_covariance=initial_covariance,
)

soc_est      = state_estimates[:, 0]
theta_3_est  = state_estimates[:, 2]
theta_6_est  = state_estimates[:, 3]
capacity_est = state_estimates[:, 4]
```

## Examples

The `examples/` folder contains four Jupyter notebooks along with the datasets required to run them:

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_soc_estimation_with_ekf.ipynb` | SOC estimation for NMC and LFP batteries using LTI and LPV models with EKF |
| 2 | `2_soc_estimation_with_pf.ipynb` | SOC estimation for NMC and LFP batteries using LTI and LPV models with PF |
| 3 | `3_soc_observability_analysis.ipynb` | SOC observability analysis via finite-horizon observability Gramians |
| 4 | `4_ageing_aware_model_estimation_with_ekf.ipynb` | Recursive joint estimation of capacity and ageing parameters with EKF |

Datasets are stored under `examples/data/` in three sub-folders: `nmc_soc_estimation/`, `lfp_soc_estimation/`, and `nmc_ageing_estimation/`.
