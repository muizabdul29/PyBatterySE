"""Utilities concerning model state-space representation."""

import re

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoefficientTerm:
    """A single additive term in a model coefficient.

    A model coefficient is expressed as a sum of terms, where each term
    is the product of a scalar parameter and one or more basis functions.
    For example:

        b_0(k) = 2 * s(k) * log[s](k) + 4 * T(k) * 1/s(k)

    is represented as two CoefficientTerm instances — one per summand.
    The first would be
    ``CoefficientTerm(parameter=2.0, basis_function_strings=["s(k)", "log[s](k)"])``.

    Attributes:
        parameter: Scalar multiplier for the term.
        basis_function_strings: String identifiers of the basis functions
            whose product forms the term. An empty list denotes a constant
            term (i.e., just the parameter itself).
    """

    parameter: float
    basis_function_strings: list[str] = field(default_factory=list)


# Maps a coefficient name (e.g., "b_0") to its list of additive terms.
Coefficients = dict[str, list[CoefficientTerm]]


def extract_model_coefficients(model_terms, model_estimate, output_symbol='v', input_symbol='i'):
    """Extract p-dependent model coefficients."""

    coefficients: dict[str, list[CoefficientTerm]] = {}

    for term, parameter in zip(model_terms, model_estimate):
        result = re.search(f'({output_symbol}|{input_symbol})\\(k(-(\\d+))?\\)', term)
        if result is None:
            raise ValueError('Invalid model terms.')

        # Define coefficient
        result_groups = result.groups()
        coefficient_variable = 'a' if result_groups[0] == output_symbol else 'b'
        coefficient_delay = result_groups[2] if result_groups[2] is not None else 0

        # Remove input/output terms plus the time indices
        coefficient_term = re.sub(f'({output_symbol}|{input_symbol})?\\(k(-(\\d+))?\\)', '', term)
        # We split the term into a list of individual
        # basis function strings
        coefficient_term_bfs = coefficient_term.strip('×').split('×')

        # Define coefficient key, e.g., b_0, b_1, ...
        coefficient_key = f'{coefficient_variable}_{coefficient_delay}'
        if coefficient_key not in coefficients:
            coefficients[coefficient_key] = []
        #
        coefficient_parameter = -parameter if coefficient_variable == 'a' else parameter
        coefficients[coefficient_key].append(CoefficientTerm(coefficient_parameter,
                                                             coefficient_term_bfs))

    return coefficients


def evaluate_coefficient(coefficient_terms: list[CoefficientTerm],
                         signal_trajectories: dict,
                         time_instant: int):
    """Evaluate a coefficient at a certain time instant."""

    result = []
    for coefficient_term in coefficient_terms:
        #
        term_value = coefficient_term.parameter
        for bf_string in coefficient_term.basis_function_strings:
            if bf_string == '':
                continue
            #
            if time_instant == 0:
                key = f'{bf_string}(k)'
            elif time_instant > 0:
                key = f'{bf_string}(k+{time_instant})'
            else:
                key = f'{bf_string}(k{time_instant})'  # negative sign is part of time_instant
            #
            term_value *= signal_trajectories[key][0]
        result.append(term_value)

    return np.sum(result)
