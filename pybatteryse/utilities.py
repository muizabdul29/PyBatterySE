"""Utility functions for working with state-space representations."""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .statespace import StateSpace


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
