"""Utilities concerning model state-space representation."""

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class CoefficientTerm:
    """
    Represents a term in the model coefficient. Note that the
    coefficient is made up of several terms, e.g.,
    b_0(k) = 2 * s(k) × log[s](k) + 4 * T(k) × 1/s(k) is made up
    of two terms.
    """
    parameter: float
    basis_function_strings: list[str]


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
            term_value *= signal_trajectories[f'{bf_string}(k)'][time_instant]
        result.append(term_value)

    return np.sum(result)


def update_model_parameters(coefficients: Coefficients, model_estimate: list[float]) -> None:
    """Update coefficient parameters in-place from a flat model_estimate vector.

    Iterates coefficients.items() in insertion order (a_1, a_2, ..., b_0, b_1, ...)
    which matches the canonical parameter order guaranteed by extract_model_coefficients.
    a-term parameters are stored with negated sign (convention from extract_model_coefficients).

    Raises:
        ValueError: If len(model_estimate) does not match the total subterm count.
    """
    all_keyed_subterms = [
        (subterm, coefficient_key.startswith('a'))
        for coefficient_key, subterms in coefficients.items()
        for subterm in subterms
    ]
    if len(model_estimate) != len(all_keyed_subterms):
        raise ValueError(
            f"Parameter count mismatch: expected {len(all_keyed_subterms)} parameters "
            f"but got {len(model_estimate)}."
        )
    for (subterm, is_a_coefficient), new_parameter in zip(all_keyed_subterms, model_estimate):
        subterm.parameter = -new_parameter if is_a_coefficient else new_parameter
