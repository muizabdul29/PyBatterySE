"""
A collection of utilities for battery plots.
"""

from typing import Tuple

import numpy as np

from pybatteryid.plotter import plot_custom, plt


# pylint: disable=R0913, R0917
def plot_time_vs_soc(time_soc_data_tuples: list[Tuple], figsize: Tuple[int, int]=(6, 2),
                     legends: list[str]|None=None, linestyles: list[str]|None=None,
                     linewidth: float|None=None, units: Tuple[str, str]=('sec', '-'),
                     xlims: Tuple[int, int]|None=None, ylims: Tuple[int, int]|None=None,
                     colors: list[str]|None=None, yscale: str='linear'):
    """A shortcut for time vs. soc plots."""
    #
    plot_custom(time_soc_data_tuples, figsize, xlabel=f'Time [{units[0]}]',
        ylabel=f'SOC [{units[1]}]', legends=legends, linestyles=linestyles,
        linewidth=linewidth, xlims=xlims, ylims=ylims, colors=colors, yscale=yscale)


def plot_time_vs_parameter(time_soc_data_tuples: list[Tuple], figsize: Tuple[int, int]=(6, 2),
                           legends: list[str]|None=None, linestyles: list[str]|None=None,
                           linewidth: float|None=None, units: Tuple[str, str]=('sec', '-'),
                           xlims: Tuple[int, int]|None=None, ylims: Tuple[int, int]|None=None,
                           colors: list[str]|None=None):
    """A shortcut for time vs. parameter plots."""
    #
    plot_custom(time_soc_data_tuples, figsize, xlabel=f'Time [{units[0]}]',
        ylabel=f'Parameter [{units[1]}]', legends=legends, linestyles=linestyles,
        linewidth=linewidth, xlims=xlims, ylims=ylims, colors=colors)


def plot_soc_vs_gramian(soc_gramian_data_tuples: list[tuple], figsize: tuple[int, int] = (6, 2),
                        legends: list[str] | None = None, n_bins: int = 200,
                        units: Tuple[str, str] = ('-', '$V^2$'),
                        ylims: tuple[float, float] | None = None,
                        xlims: tuple[float, float] | None = None,
                        colors: list[str] | None = None,
                        title: str | None = None,
                        xaxis_reverse: bool = False) -> None:
    _, ax = plt.subplots(figsize=figsize)

    for idx, (soc, contributions) in enumerate(soc_gramian_data_tuples):
        kwargs = dict(alpha=0.9, label=legends[idx] if legends is not None else None)
        if colors is not None:
            kwargs['color'] = colors[idx % len(colors)]

        soc_bins = np.linspace(soc.min(), soc.max(), n_bins + 1)
        soc_centers = 0.5 * (soc_bins[:-1] + soc_bins[1:])
        bin_width = (soc_bins[1] - soc_bins[0]) * 0.4
        bin_indices = np.clip(np.digitize(soc, soc_bins) - 1, 0, n_bins - 1)

        ymins = np.full(n_bins, np.nan)
        ymaxs = np.full(n_bins, np.nan)
        for i in np.unique(bin_indices):
            mask = bin_indices == i
            ymins[i] = contributions[mask].min()
            ymaxs[i] = contributions[mask].max()

        ax.bar(soc_centers, height=ymaxs - ymins, bottom=ymins, width=bin_width, **kwargs)

    ax.set_xlabel(f"SOC [{units[0]}]")
    ax.set_ylabel(f"Gramian [{units[1]}]")
    if xlims is not None:
        ax.set_xlim(*xlims)
    if xaxis_reverse:
        ax.invert_xaxis()
    if ylims is not None:
        ax.set_ylim(*ylims)
    if title is not None:
        ax.set_title(title)
    if legends is not None:
        ax.legend()
    ax.grid(ls='--', alpha=0.5, lw=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
