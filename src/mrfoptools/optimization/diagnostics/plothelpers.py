"""
PLotting utilities specialized for in-progress diagnostics and evaluation.

@Author: Jannik Stebani 2025
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import jax

from collections import OrderedDict
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
        data: ArrayLike
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


    @classmethod
    def create_flipangle_plotter(
        cls,
        baseline_data: ArrayLike | None = None,
        is_radians: bool = False,
        NR: int = 1000,
        **kwargs) -> 'Plotter':
        """
        Create a plotter for flip angle trains.
        Convenience method that sets sensible defaults for flip angle plots.

        Parameters
        ----------
        baseline_data : ArrayLike, optional
            Baseline data to plot, by default None

        is_radians : bool, optional
            Whether the default baseline flip angles are in radians.
            If data is in radians, it will be converted to degrees.
            By default False, i.e. degrees.

        NR : int, optional
            Number of repetitions in the flip angle train.
            Value is used to fix the x-axis limits.
            By default 1000.

        **kwargs
            Additional keyword arguments to pass to the plotter
            initializer function.
        """
        processor = np.rad2deg if is_radians else lambda x: x
        baseline_data = processor(np.asarray(baseline_data)) if baseline_data is not None else None
        xlim = (0, NR)
        base_kwargs = {
            'title' : 'Flipangle Snapshot',
            'xlabel' : 'NR index',
            'ylabel' : 'FA [deg]',
            'legend' : False,
            'baseline_data' : baseline_data,
            'grid' : True,
            'ylim' : (0, 94),
            'xlim' : xlim,
            'baseline_data_plot_kwargs' : {'label': 'initial',
                                           'ls' : 'dotted',
                                           'color' : 'black',
                                           'alpha' : 0.5}
        }
        final_kwargs = base_kwargs | kwargs
        return cls(**final_kwargs)
    

    @classmethod
    def create_signal_plotter(
        cls,
        baseline_data: ArrayLike | None = None,
        NR: int = 1000,
        **kwargs
    ) -> 'Plotter':
        """
        Create a plotter for signal vectors.
        Convenience method that sets sensible defaults for signal plots.

        Parameters
        ----------
        baseline_data : ArrayLike, optional
            Baseline data to plot, by default None.

        NR : int, optional
            Number of elements in the signal vector.
            Value is used to fix the x-axis limits.
            By default 1000.

        **kwargs
            Additional keyword arguments to pass to the plotter
            initializer function.
        """
        base_kwargs = {
            'title' : 'Signal Snapshot',
            'xlabel' : 'NR index',
            'ylabel' : 'Signal [a.u.]',
            'legend' : False,
            'baseline_data' : baseline_data,
            'grid' : True,
            'ylim' : (-0.5, 0.5),
            'xlim' : (0, NR),
            'baseline_data_plot_kwargs' : {'label': 'initial',
                                           'ls' : 'dotted',
                                           'alpha' : 0.5}
        }
        final_kwargs = base_kwargs | kwargs
        return cls(**final_kwargs)




class CachingPlotter(Plotter):
    """
    Plotter that recalls plot items from cache by comparing input data
    to previous plot data via Python object identity.
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
        cache_size: int = 3,
    ) -> None:
        super().__init__(
            figsize, baseline_data, title, xlabel, ylabel, legend, grid, ylim, xlim, dpi,
            baseline_data_plot_kwargs, data_plot_kwargs)
        
        self._cache: OrderedDict[int, Figure] = OrderedDict()
        self._cache_size = cache_size
    

    def generate(self, data: ArrayLike) -> tuple[Figure, Axes]:
        data_id = id(data)
        
        try:
            (fig, ax) = self._cache[data_id]
        except KeyError:
            if len(self._cache) >= self._cache_size:
                (fig, _) = self._cache.popitem(last=False)
                plt.close(fig)

            (fig, ax) = super().generate(data)
            self._cache[data_id] = (fig, ax)

        return (fig, ax)
