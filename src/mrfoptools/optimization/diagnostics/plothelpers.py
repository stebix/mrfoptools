"""
PLotting utilities specialized for in-progress diagnostics and evaluation.

@Author: Jannik Stebani 2025
"""
import logging
import attrs
import matplotlib.pyplot as plt
import numpy as np
import jax

from collections.abc import Mapping
from matplotlib.axes import Axes
from matplotlib.figure import Figure

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

ArrayLike = np.ndarray | jax.Array


class Plotter:
    """
    Plotting utility for tensorboard logging.

    Parameters
    ----------
    figsize : tuple[int, int], optional
        Figure size in inches, by default (6, 4)

    title : str, optional
        Title of the plot, by default ''

    xlabel : str, optional
        Label of the x-axis, by default ''

    ylabel : str, optional
        Label of the y-axis, by default ''

    label : str, optional
        Label of the plotted data, by default ''

    legend : bool, optional
        Show the legend, by default False

    grid : bool, optional
        Show the grid on the canvas, by default False

    ylim : tuple[float, float] | None, optional
        Y-axis limits, by default None, i.e. automatic scaling

    xlim : tuple[float, float] | None, optional
        X-axis limits, by default None, i.e. automatic scaling

    dpi : int, optional
        Dots per inch, by default 100
    """
    def __init__(
        self,
        figsize: tuple[int, int] = (6, 4),
        baseline_data: ArrayLike | None = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        legend: bool = False,
        grid: bool = False,
        ylim: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        dpi: int = 100,
        baseline_data_plot_kwargs: Mapping | None = None,
        data_plot_kwargs: Mapping | None = None,
    ) -> None:
        self.figsize = figsize
        self.baseline_data = baseline_data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend
        self.grid = grid
        self.ylim = ylim
        self.xlim = xlim
        self.dpi = dpi
        self.baseline_data_plot_kwargs = baseline_data_plot_kwargs or {}
        self.data_plot_kwargs = data_plot_kwargs or {}


    def generate(
        self,
        data: ArrayLike,
    ) -> tuple[Figure, Axes]:
        """
        Generate a plot with the given data.
        Data should be a 1D or 2D array.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        if self.baseline_data is not None:
            self.plot(ax, self.baseline_data, self.baseline_data_plot_kwargs)
        self.plot(ax, data, self.data_plot_kwargs)
        self.configure(
            ax, self.title, self.xlabel, self.ylabel,
            self.legend, self.grid, self.ylim, self.xlim)
        return (fig, ax)


    @staticmethod
    def plot(ax: Axes, data: ArrayLike, kwargs: Mapping) -> None:
        if data.ndim == 1:
            ax.plot(data, **kwargs)
        elif data.ndim == 2:
            for linedata in data:
                ax.plot(linedata, **kwargs)
        else:
            msg = f'Can not plot data with more than 2 dimensions: got shape {data.shape}'
            logger.error(msg)


    @staticmethod
    def configure(
        ax: Axes,
        title: str,
        xlabel: str,
        ylabel: str,
        legend: bool,
        grid: bool,
        ylim: tuple[float, float] | None,
        xlim: tuple[float, float] | None,
    ) -> None:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(grid)
        if legend:
            ax.legend()
        if ylim:
            ax.set_ylim(ylim)
        if xlim:
            ax.set_xlim(xlim)
