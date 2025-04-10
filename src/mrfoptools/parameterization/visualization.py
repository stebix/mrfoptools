"""
Tooling to visualize the parameterizations and their evolution.

@Author: Jannik Stebani 2025
"""
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axis import Axis

from mrfoptools.parameterization.parameterization import ControlPoints


def _identity(x: Any) -> Any:
    return x


def plot_control_points(
    controlpoints: ControlPoints,
    ax: Axis | None = None,
    postprocess_y: Callable | None = None
) -> Axis:
    """
    Plot controlpoints as x-y scatter plot with vertical lines.

    Parameters
    ----------
    controlpoints : ControlPoints
        Control points to be plotted.
    ax : Axis, optional
        Matplotlib axis to plot the control points.
        Default is None, i.e. current axis is deduced from outer scope.
    postprocess_y : Callable, optional
        Postprocessing function for y-values.
        Must be a function of the form `f(y: ArrayLike) -> ArrayLike`.
        Default is None, i.e. no postprocessing.

    Returns
    -------
    ax : Axis
        Matplotlib axis with the control points plotted.
    """
    ax = ax or plt.gca()
    postprocess_y = _identity if postprocess_y is None else postprocess_y
    ax.plot(
        controlpoints.x, postprocess_y(controlpoints.y),
        marker='x', color='k', ls='')
    ax.vlines(
        controlpoints.x, ymin=0, ymax=postprocess_y(controlpoints.y),
        alpha=0.4, ls='dotted', color='k')
    return ax