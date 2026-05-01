"""Utility functions for working with state-space representations."""

import copy

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markup import escape

from pybatteryid.dataclasses import Model

from .coefficient import extract_model_coefficients
from .statespace import StateSpace, SocSource


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def print_statespace_equations(statespace: StateSpace) -> None:
    """Print the expanded state-space equations for a StateSpace instance.

    Renders the A, B, C, D matrices as ASCII-art bracketed blocks with
    dashed separators between per-component blocks (overpotentials, 's',
    'capacity', 'theta_i'). For model_order > 4 the overpotential block is
    elided with ⋮, ⋯, ⋱ so the display stays compact.

    The measurement equation is written as
        V_k = EMF(s_k, T_k) + C · x_k + D(p_k) · u_k
    because EMF is a nonlinear function of SOC and temperature; it is not
    absorbed into C.

    Rich is used only for the panel frame and title label; the matrices
    themselves are plain text.
    """
    state_components = statespace.state_components
    n = statespace.model_order
    elide = n > 4

    # Battery capacity is written with a _k subscript only when it is a
    # random-walk state ('capacity' in state_components). Otherwise it is
    # a fixed model parameter and carries no time index.
    capacity_symbol = "ℚ_k" if 'capacity' in state_components else "ℚ"

    # Visible row/column indices inside the overpotential block.
    # None marks an elision row/column (⋮ / ⋯ / ⋱).
    op_visible: list[int | None] = (
        [1, 2, None, n] if elide else list(range(1, n + 1))
    )
    op_visible_n = len(op_visible)

    # Per-component visible block width (rows == columns for scalar blocks).
    def _block_width(component: str) -> int:
        return op_visible_n if component == 'overpotentials' else 1

    # Render a labeled ASCII matrix with dashed block separators.
    def _render_matrix(rows: list[list[str]],
                       row_separators: set[int],
                       col_separators: set[int],
                       label: str) -> list[str]:
        n_cols = len(rows[0])
        col_widths = [max(len(rows[r][c]) for r in range(len(rows)))
                      for c in range(n_cols)]

        def _format_row_cells(row: list[str]) -> str:
            parts: list[str] = []
            for c, cell in enumerate(row):
                parts.append(cell.rjust(col_widths[c]))
                if c < n_cols - 1:
                    parts.append(" ¦ " if c in col_separators else "  ")
            return "".join(parts)

        inner_width = len(_format_row_cells(rows[0]))

        def _dashed_row() -> str:
            chars = ["-"] * inner_width
            offset = 0
            for c in range(n_cols):
                offset += col_widths[c]
                if c < n_cols - 1:
                    if c in col_separators:
                        chars[offset + 1] = "+"
                        offset += 3
                    else:
                        offset += 2
            return "".join(chars)

        body_lines: list[str] = ["┌ " + " " * inner_width + " ┐"]
        for r, row in enumerate(rows):
            body_lines.append("│ " + _format_row_cells(row) + " │")
            if r in row_separators and r < len(rows) - 1:
                body_lines.append("│ " + _dashed_row() + " │")
        body_lines.append("└ " + " " * inner_width + " ┘")

        prefix = f"{label} = "
        pad = " " * len(prefix)
        center = len(body_lines) // 2
        return [(prefix if i == center else pad) + line
                for i, line in enumerate(body_lines)]

    # Column separator positions: after each block boundary except the last.
    col_separators: set[int] = set()
    col_offset = 0
    for component in state_components[:-1]:
        col_offset += _block_width(component)
        col_separators.add(col_offset - 1)

    # Build a row for a scalar block: a single `1` at its own column,
    # zeros elsewhere. `self_column` is the absolute visible-column index
    # of that block's diagonal entry.
    def _scalar_row(self_column: int) -> list[str]:
        row: list[str] = []
        abs_col = 0
        for ccomp in state_components:
            width = _block_width(ccomp)
            for k in range(width):
                row.append("1" if abs_col + k == self_column else "0")
            abs_col += width
        return row

    # Build the op-block rows for A (companion form, possibly elided).
    def _op_matrix_a_rows() -> list[list[str]]:
        rows: list[list[str]] = []
        for vr in op_visible:
            row: list[str] = []
            for ccomp in state_components:
                if ccomp == 'overpotentials':
                    for vc in op_visible:
                        if vr is None and vc is None:
                            row.append("⋱")
                        elif vr is None:
                            row.append("⋮")
                        elif vc is None:
                            row.append("⋯")
                        elif vc == 1:
                            row.append(f"-a_{vr}(p_k)")
                        elif vc == vr + 1:
                            row.append("1")
                        else:
                            row.append("0")
                else:
                    row.append(" " if vr is None else "0")
            rows.append(row)
        return rows

    # Assemble A: one block of rows per component in declared order.
    matrix_a_rows: list[list[str]] = []
    matrix_a_row_separators: set[int] = set()
    abs_col_by_block: dict[str, int] = {}
    col_cursor = 0
    for component in state_components:
        abs_col_by_block[component] = col_cursor
        col_cursor += _block_width(component)

    for bi, component in enumerate(state_components):
        if component == 'overpotentials':
            matrix_a_rows.extend(_op_matrix_a_rows())
        else:
            matrix_a_rows.append(_scalar_row(abs_col_by_block[component]))
        if bi < len(state_components) - 1:
            matrix_a_row_separators.add(len(matrix_a_rows) - 1)

    # Build B: one column. op-block rows carry b_i(p_k) - a_i(p_k)·b_0(p_k);
    # 's' carries Δt/ℚ; capacity/theta_i carry 0.
    matrix_b_rows: list[list[str]] = []
    matrix_b_row_separators: set[int] = set()
    for bi, component in enumerate(state_components):
        if component == 'overpotentials':
            for vr in op_visible:
                if vr is None:
                    matrix_b_rows.append(["⋮"])
                else:
                    matrix_b_rows.append([f"b_{vr}(p_k) - a_{vr}(p_k)·b_0(p_k)"])
        elif component == 's':
            matrix_b_rows.append([f"Δt / {capacity_symbol}"])
        elif component == 'capacity' or component.startswith('theta_'):
            matrix_b_rows.append(["0"])
        if bi < len(state_components) - 1:
            matrix_b_row_separators.add(len(matrix_b_rows) - 1)

    # Build C: one row. A `1` at the overpotentials-block's first column,
    # zeros everywhere else (including the 's' column — EMF lives outside).
    matrix_c_row: list[str] = []
    matrix_c_col_separators: set[int] = set()
    abs_col = 0
    op_first_col = abs_col_by_block['overpotentials']
    for bi, component in enumerate(state_components):
        width = _block_width(component)
        for k in range(width):
            matrix_c_row.append("1" if abs_col + k == op_first_col else "0")
        abs_col += width
        if bi < len(state_components) - 1:
            matrix_c_col_separators.add(abs_col - 1)

    # D: scalar b_0(p_k).
    matrix_d_rows: list[list[str]] = [["b_0(p_k)"]]

    # x_k composition.
    x_k_parts: list[str] = []
    for component in state_components:
        if component == 's':
            x_k_parts.append("s_k")
        elif component == 'overpotentials':
            if n <= 4:
                x_k_parts.extend(f"x_{i},k" for i in range(1, n + 1))
            else:
                x_k_parts.extend(["x_1,k", "x_2,k", "...", f"x_{n},k"])
        elif component == 'capacity':
            x_k_parts.append("ℚ_k")
        elif component.startswith('theta_'):
            idx = component[len('theta_'):]
            x_k_parts.append(f"θ_{idx},k")

    header_lines = [
        "x_{k+1} = A(p_k) · x_k + B(p_k) · u_k",
        "V_k     = EMF(s_k, T_k) + C · x_k + D(p_k) · u_k",
        "",
        "where",
        "",
        f"x_k = [{', '.join(x_k_parts)}]   (dim = {statespace.state_dimension})",
        "p_k = [s_k, i_k, T_k]",
        "u_k = i_k",
        "",
    ]

    # Render each matrix.
    matrix_lines: list[str] = []
    matrix_lines.extend(_render_matrix(
        matrix_a_rows, matrix_a_row_separators, col_separators, "A(p_k)"
    ))
    matrix_lines.append("")
    matrix_lines.extend(_render_matrix(
        matrix_b_rows, matrix_b_row_separators, set(), "B(p_k)"
    ))
    matrix_lines.append("")
    matrix_lines.extend(_render_matrix(
        [matrix_c_row], set(), matrix_c_col_separators, "C"
    ))
    matrix_lines.append("")
    matrix_lines.extend(_render_matrix(
        matrix_d_rows, set(), set(), "D(p_k)"
    ))

    body = "\n".join([""] + header_lines + matrix_lines)

    Console().print(Panel(
        Text(body),
        title="[bold]State-space equations[/bold]",
        expand=False,
    ))


# pylint: disable-next=too-many-locals
def print_input_output_coefficients(model: Model):
    """Print input-output (LPV) model coefficients as a rich table."""
    coefficients = extract_model_coefficients(
        model.model_terms, model.model_estimate
    )

    console = Console()
    table = Table(title="Model Coefficients")
    table.add_column("Coefficient equations")
    table.add_column("Parameter values")

    subs = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    theta_idx = 1
    for coeff_name, terms in coefficients.items():
        if "_" in coeff_name:
            base, idx = coeff_name.split("_", 1)
            display_name = f"{base}{idx.translate(subs)}(p)"
        else:
            base = coeff_name
            display_name = f"{coeff_name}(p)"

        negate = base == "a"

        summands = []
        values = []
        for term in terms:
            theta = f"θ{str(theta_idx).translate(subs)}"
            basis_strs = [s for s in term.basis_function_strings if s]
            if basis_strs:
                summand = f"{theta}·{'·'.join(basis_strs)}"
            else:
                summand = theta
            summands.append(summand)
            value = -term.parameter if negate else term.parameter
            values.append(f"{theta} = {value:.4g}")
            theta_idx += 1

        if not summands:
            expression = "0"
        elif negate:
            expression = f"-{summands[0]}"
            for s in summands[1:]:
                expression += f"\n  - {s}"
        else:
            expression = summands[0]
            for s in summands[1:]:
                expression += f"\n  + {s}"

        equation = f"{display_name} = {expression}"
        values_str = "\n".join(values) if values else "—"
        table.add_row(escape(equation), escape(values_str))

    console.print(table)


# pylint: disable=R0913, R0914, R0917
def compute_soc_observability_contributions(
    statespace: StateSpace,
    state_values: np.ndarray,
    current_values: float | np.ndarray,
    horizon: int = 100,
    temperature_values: float | np.ndarray | None = None,
) -> dict:
    """Decompose the SOC-diagonal of a finite-horizon observability
    Gramian into contributions from three measurement-equation sources.

    Single-point or trajectory analysis based on the shape of
    ``state_values``:
        - 1-D (shape ``(state_dimension,)``): single linearization point;
          ``current_values`` and ``temperature_values`` must be scalars (or
          ``None`` for ``temperature_values``). Returns a dict of scalars.
        - 2-D (shape ``(T, state_dimension)``): trajectory;
          ``current_values`` is a length-T array, ``temperature_values`` is
          a length-T array or ``None``. Returns a dict of length-T arrays.

    For each linearization point, computes
    ``W_i = sum_{k=0}^{horizon} ((C_i @ A^k)[soc_offset])^2`` for three
    partial measurement Jacobians ``C_i``, with ``A`` (= ``df_dx``) held
    constant at the lin point. This is the standard local observability
    formulation; an LPV-time-varying analogue would require a known future
    input trajectory.

    Sources:
        emf:         only ``dV_OCV/ds`` at ``soc_offset``.
        b0_direct:   only ``(db_0/ds) * u`` at ``soc_offset``.
        op_dynamics: only ``e_1^T`` on the overpotentials block. SOC enters
                     future measurements via the SOC column of ``A``
                     propagating perturbations into overpotentials.

    Cross terms between the three sources mean their percentages don't sum
    to 100; the gap reflects cross-term contributions and can be negative
    when sources partially cancel.

    Calls ``statespace.update_model_from_state(...)`` at every linearization
    point so the coefficients reflect any theta / capacity entries in the
    state vector.

    Parameters
    ----------
    statespace : StateSpace
        State-space model. Must have ``'s'`` in ``state_components``.
    state_values : np.ndarray
        State vector or trajectory; see shape rules above.
    current_values : float or np.ndarray
        Input current at the lin point (scalar) or trajectory (1-D array).
    horizon : int, default 100
        Number of forward steps included in the Gramian sum (k = 0..horizon).
    temperature_values : float, np.ndarray, or None
        Same conventions as ``StateSpace.linearize``, broadcast along the
        trajectory in the 2-D case.

    Returns
    -------
    dict with keys 'total', 'emf', 'b0_direct', 'op_dynamics', 'cross_term'.
    Values are scalars (single-point case) or length-T arrays (trajectory
    case). ``cross_term`` is ``total - (emf + b0_direct + op_dynamics)`` --
    the gap captures pairwise cross terms between the three sources and
    can be negative when contributions partially cancel. Percentages are
    not provided -- compute them from these values (e.g.
    ``100 * out['emf'] / out['total']``).

    Raises
    ------
    ValueError
        If SOC is not in ``state_components``, ``horizon < 1``,
        ``state_values`` is not 1-D or 2-D, or input arrays are shorter
        than the state trajectory.
    """
    # pylint: disable=protected-access
    if statespace._soc_source is not SocSource.STATE:
        raise ValueError(
            "compute_soc_observability_contributions requires 's' in "
            "state_components; SOC observability is undefined when SOC "
            "is supplied exogenously."
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    state_values = np.asarray(state_values, dtype=float)
    if state_values.ndim not in (1, 2):
        raise ValueError(
            f"state_values must be 1-D (single point) or 2-D (trajectory), "
            f"got shape {state_values.shape}."
        )

    # Normalize to the trajectory shape so the body has a single code path.
    is_single_point = state_values.ndim == 1
    if is_single_point:
        states = state_values.reshape(1, -1)
        currents = np.array([float(current_values)], dtype=float)
        temperatures = (
            None if temperature_values is None
            else np.array([float(temperature_values)], dtype=float)
        )
    else:
        states = state_values
        currents = np.asarray(current_values, dtype=float)
        temperatures = (
            None if temperature_values is None
            else np.asarray(temperature_values, dtype=float)
        )

    num_timesteps = states.shape[0]
    if currents.shape[0] < num_timesteps:
        raise ValueError(
            f"current_values has length {currents.shape[0]} but "
            f"state_values has {num_timesteps} entries; need at least "
            f"as many."
        )
    if temperatures is not None and temperatures.shape[0] < num_timesteps:
        raise ValueError(
            f"temperature_values has length {temperatures.shape[0]} but "
            f"state_values has {num_timesteps} entries."
        )

    # Work on a copy of the statespace to avoid side effects on the caller's
    # instance: each iteration calls update_model_from_state(state_k), which
    # mutates model_estimate, battery_capacity, and subterm.parameter fields.
    statespace = _copy_statespace(statespace)

    soc_offset = statespace._soc_offset
    op_offset = statespace._overpotentials_offset
    state_dim = statespace.state_dimension

    keys = ('total', 'emf', 'b0_direct', 'op_dynamics', 'cross_term')
    out = {key: np.full(num_timesteps, np.nan) for key in keys}

    iterator = range(num_timesteps)

    for k in iterator:
        state_k = states[k]
        current_k = float(currents[k])
        temperature_k = (
            None if temperatures is None else float(temperatures[k])
        )

        statespace.update_model_from_state(state_k)
        soc_scalar = float(state_k[soc_offset])

        lin = statespace.linearize(
            state_k, current_k,
            temperature_value=temperature_k,
        )
        matrix_a = lin.df_dx

        # Split the SOC entry of dh_dx into its EMF and b_0-direct parts.
        # linearize stores their sum (demf_ds + db0_ds * u) in dh_dx[soc_offset];
        # we re-derive the split via _soc_derivatives.
        _, _, db0_ds, demf_ds = statespace._soc_derivatives(
            soc_scalar=soc_scalar,
            current_value=current_k,
            temperature_value=temperature_k,
        )

        # Three partial C row-vectors with the same shape as dh_dx.
        c_emf = np.zeros(state_dim, dtype=float)
        c_emf[soc_offset] = demf_ds

        c_b0_direct = np.zeros(state_dim, dtype=float)
        c_b0_direct[soc_offset] = db0_ds * current_k

        c_op_dynamics = np.zeros(state_dim, dtype=float)
        c_op_dynamics[op_offset] = 1.0

        # lin.dh_dx equals c_emf + c_b0_direct + c_op_dynamics PLUS any
        # theta-in-b_0 entries (cross-cutting thetas). Using lin.dh_dx for
        # the total includes those theta contributions.

        # Inline Gramian-diagonal computation: for each partial C row,
        # propagate v_k = v_{k-1} @ A and accumulate (v_k[soc_offset])^2.
        for label, c_row in (
            ('emf', c_emf),
            ('b0_direct', c_b0_direct),
            ('op_dynamics', c_op_dynamics),
            ('total', lin.dh_dx),
        ):
            v = c_row.copy()
            w = float(v[soc_offset]) ** 2
            for _ in range(1, horizon + 1):
                v = v @ matrix_a
                w += float(v[soc_offset]) ** 2
            out[label][k] = w

        # Cross-term contribution: gap between the full-C Gramian and the
        # sum of self-contributions. Captures pairwise cross terms between
        # the three sources plus any contribution from theta-in-b_0 entries
        # of dh_dx (which are present in the total but not in the three
        # partial C's). Can be negative when sources partially cancel.
        out['cross_term'][k] = (
            out['total'][k]
            - out['emf'][k] - out['b0_direct'][k] - out['op_dynamics'][k]
        )

    if is_single_point:
        return {key: float(out[key][0]) for key in keys}
    return out


def simulate_state_trajectory(
    statespace: StateSpace,
    initial_state: np.ndarray,
    current_values: np.ndarray,
    temperature_values: np.ndarray | None = None,
    soc_values: np.ndarray | None = None,
) -> np.ndarray:
    """Forward-simulate the deterministic state trajectory under the model.

    Iterates ``statespace.evaluate_next_state`` over the provided input
    series, producing the model's noise-free state evolution. Useful for
    generating reference / 'true' state trajectories given an input
    current sequence -- e.g. to compare against an EKF estimate.

    Conventions match ``ExtendedKalmanFilter.run``: the returned array
    has shape ``(T, state_dimension)`` where ``T = len(current_values)``,
    ``state_values[0] = initial_state``, and ``state_values[k]`` is
    obtained by applying ``evaluate_next_state`` to ``state_values[k-1]``
    with input ``current_values[k-1]`` (so ``current_values[-1]`` is
    unused, matching the EKF convention).

    Operates on a local copy of ``statespace`` so the caller's instance
    is not mutated. ``update_model_from_state(initial_state)`` is called
    on the copy at the start so that any ``theta_*`` / ``capacity``
    entries in ``initial_state`` are reflected in the coefficients used
    throughout the simulation. Random-walk states (theta, capacity) keep
    their initial values for the whole trajectory; their effect on
    coefficients is therefore fixed at the initial-state values.

    Parameters
    ----------
    statespace : StateSpace
        State-space model (not mutated).
    initial_state : np.ndarray
        Shape ``(state_dimension,)``. Block layout matches
        ``state_components``.
    current_values : np.ndarray
        Length-T input current series.
    temperature_values : np.ndarray or None, default None
        Length-T temperature series for temperature-dependent models.
        Pass ``None`` for temperature-independent models.
    soc_values : np.ndarray or None, default None
        Length-T exogenous SOC series. Required when SOC source is
        ``SocSource.EXOGENOUS``; ignored when SOC is in state.

    Returns
    -------
    np.ndarray of shape ``(T, state_dimension)``.

    Raises
    ------
    ValueError
        If ``current_values`` is empty, the optional input arrays are
        shorter than ``current_values``, or the SOC source is
        ``EXOGENOUS`` but ``soc_values`` is None.
    """
    initial_state = np.asarray(initial_state, dtype=float)
    current_values = np.asarray(current_values, dtype=float)
    num_timesteps = current_values.shape[0]
    if num_timesteps < 1:
        raise ValueError(
            f"current_values must have at least 1 entry, got {num_timesteps}."
        )

    if temperature_values is not None:
        temperature_values = np.asarray(temperature_values, dtype=float)
        if temperature_values.shape[0] < num_timesteps:
            raise ValueError(
                f"temperature_values has length {temperature_values.shape[0]} "
                f"but current_values has {num_timesteps} entries."
            )

    # pylint: disable=protected-access
    if statespace._soc_source is SocSource.EXOGENOUS and soc_values is None:
        raise ValueError(
            "soc_values is required when the statespace uses "
            "SocSource.EXOGENOUS (i.e. 's' is not in state_components)."
        )
    if soc_values is not None:
        soc_values = np.asarray(soc_values, dtype=float)
        if soc_values.shape[0] < num_timesteps:
            raise ValueError(
                f"soc_values has length {soc_values.shape[0]} "
                f"but current_values has {num_timesteps} entries."
            )

    statespace = _copy_statespace(statespace)
    statespace.update_model_from_state(initial_state)

    state_values = np.empty((num_timesteps, statespace.state_dimension), dtype=float)
    state_values[0] = initial_state

    for k in range(1, num_timesteps):
        state_values[k] = statespace.evaluate_next_state(
            state_values[k - 1],
            float(current_values[k - 1]),
            temperature_value=(
                None if temperature_values is None
                else float(temperature_values[k - 1])
            ),
            soc_value=(
                None if soc_values is None
                else float(soc_values[k - 1])
            ),
        )

    return state_values


def _copy_statespace(statespace: StateSpace) -> StateSpace:
    """Return a copy of ``statespace`` whose mutation surface is independent
    of the original.

    Shallow-copies the ``StateSpace`` shell, then deep-copies the mutable
    parts that ``update_model_from_state`` writes to:
        - ``model_estimate`` (numpy array): independent copy.
        - subterm objects in ``coefficients``: shallow-copied so their
          ``.parameter`` fields are independent. ``coefficients`` dict is
          rebuilt to point at the new subterms.
        - ``_theta_info``: rebuilt so the (subterm, key) tuples reference
          the new subterms.

    The rest (``emf_function``, ``basis_functions``, layout caches,
    ``_soc_domain_min/max``) is shared with the original, since these are
    immutable from the perspective of mutation paths in
    ``update_model_from_state``. Subterm basis-function attributes are
    stateless and remain shared.
    """
    # pylint: disable=protected-access
    new = copy.copy(statespace)
    new.model_estimate = statespace.model_estimate.copy()
    old_to_new_subterm = {}
    new_coefficients = {}
    for key, subterms in statespace.coefficients.items():
        new_subterms = []
        for subterm in subterms:
            new_subterm = copy.copy(subterm)
            old_to_new_subterm[id(subterm)] = new_subterm
            new_subterms.append(new_subterm)
        new_coefficients[key] = new_subterms
    new.coefficients = new_coefficients
    new._theta_info = [
        (old_to_new_subterm[id(subterm)], key)
        for subterm, key in statespace._theta_info
    ]
    return new
