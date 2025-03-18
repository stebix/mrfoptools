"""
PLotting utilities specialized for in-progress diagnostics and evaluation.

@Author: Jannik Stebani 2025
"""
import attrs
import matplotlib.pyplot as plt
import numpy as np
import jax
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        label: str = '',
        legend: bool = False,
        grid: bool = False,
        ylim: tuple[float, float] | None = None,
        xlim: tuple[float, float] | None = None,
        dpi: int = 100,
    ) -> None:
        self.figsize = figsize
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.label = label
        self.legend = legend
        self.grid = grid
        self.ylim = ylim
        self.xlim = xlim
        self.dpi = dpi

    def generate(
        self,
        y: ArrayLike,
    ) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(y, label=self.label)
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.grid(self.grid)
        if self.legend:
            ax.legend()
        if self.ylim:
            ax.set_ylim(self.ylim)
        if self.xlim:
            ax.set_xlim(self.xlim)
        return (fig, ax)


@attrs.define
class PlotSettings:
    title: str = ''
    xlabel: str = ''
    ylabel: str = ''
    legend: bool = False
    grid: bool = False
    ylim: tuple[float, float] = None


def configure(plot: tuple[Figure, Axes], settings: PlotSettings) -> None:
    """Configure a matplotlib plot with the given settings.
    """
    fig, ax = plot
    ax.set_title(settings.title)
    ax.set_xlabel(settings.xlabel)
    ax.set_ylabel(settings.ylabel)
    ax.grid(settings.grid)
    if settings.legend:
        ax.legend()
    if settings.ylim:
        ax.set_ylim(settings.ylim)