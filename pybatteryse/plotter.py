"""
A collection of utilities for battery plots.
"""

from typing import Tuple

from pybatteryid.plotter import plot_custom


# pylint: disable=R0913, R0917
def plot_time_vs_soc(time_soc_data_tuples: list[Tuple], figsize: Tuple[int, int]=(6, 2),
                     legends: list[str]|None=None, linestyles: list[str]|None=None,
                     linewidth: float|None=None, units: Tuple[str, str]=('sec', '-'),
                     xlims: Tuple[int, int]|None=None, ylims: Tuple[int, int]|None=None,
                     colors: list[str]|None=None):
    """A shortcut for time vs. soc plots."""
    #
    plot_custom(time_soc_data_tuples, figsize, xlabel=f'Time [{units[0]}]',
        ylabel=f'SOC [{units[1]}]', legends=legends, linestyles=linestyles,
        linewidth=linewidth, xlims=xlims, ylims=ylims, colors=colors)


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
