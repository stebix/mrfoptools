"""
Reporting module for CLI and graphical output
of benchmarking results.

@Author: Jannik Stebani 2025
"""
import numpy as np

from collections.abc import Sequence

import rich.console
import rich.table
import matplotlib.colors

from mrfoptools.testtooling.testtooling import BenchmarkingMeasurement

def aggregate(
    timings: list[BenchmarkingMeasurement],
    scale: float = 1e3
) -> tuple[tuple[str, ...], np.ndarray]:
    """
    Aggregate timings from multiple implementations
    into a 2D runtime array.

    Parameters
    ----------
    timings : list[BenchmarkingMeasurement]
        List of benchmarking measurements.
    scale : float, optional
        Scaling factor for runtimes. Defaults to 1e3,
        i.e. scaling from seconds to milliseconds.

    Returns
    -------
    tuple[tuple[str, ...], np.ndarray]
        Tuple of implementation IDs and 2D runtime array
        of shape `(n_implementations, n_repeats)`.
    """
    runtimes: list[float] = []
    IDs: list[str] = []

    for timing in timings:
        runtimes.append(timing.runtimes)
        IDs.append(timing.ID)

    return (tuple(IDs), np.array(runtimes) * scale)
    

def metrics(
    runtimes: np.ndarray
) -> np.ndarray:
    """
    Compute mean and standard deviation of runtimes.
    Runtimes array is expected to have shape `(n_implementations, n_repeats)`.
    """
    return np.array(
        [np.mean(runtimes, axis=1),
         np.std(runtimes, ddof=1, axis=1),
         np.min(runtimes, axis=1),
         np.max(runtimes, axis=1)]
    ).T


def get_rich_rgb(value: Sequence[float]) -> str:
    """Convert float RGB spec into rich-compatible string spec."""
    r, g, b, _ = value  
    return f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 



def generate_rows(
    metrics: np.ndarray,
    IDs: tuple[str, ...]
) -> list[list[str]]:
    """
    Generate rich-compatible table rows from computed metrics array of shape
    `(n_implementations, n_metrics)` and corrsponding implementation IDs.
    """
    mins = np.min(metrics, axis=0)
    maxs = np.max(metrics, axis=0)

    mean_min = mins[0]
    mean_max = maxs[0]

    colornorm = matplotlib.colors.Normalize(vmin=mean_min, vmax=mean_max)
    colormap = matplotlib.cm.get_cmap('RdYlGn_r')

    rows: list[list[str]] = []

    for ID, row_data in zip(IDs, metrics):
        mean, std, min_, max_ = row_data
        mean_rgb = get_rich_rgb(colormap(colornorm(mean)))
        rows.append(
            [ID,
             f'[{mean_rgb}]{mean:.5f}[/{mean_rgb}]',
             f'{std:.5f}',
             f'{min_:.5f}',
             f'{max_:.5f}',
             f'{mean/mean_min:.2f}']
        )
    return rows


def display_report(
    benchmark_measurements: list[BenchmarkingMeasurement],
    header: str
) -> None:
    """
    Display benchmarking report in a rich table.
    """
    IDs, runtimes = aggregate(benchmark_measurements)
    metrics_ = metrics(runtimes)
    rows = generate_rows(metrics_, IDs)

    table = rich.table.Table(title=header)

    table.add_column('Implementation', justify='right', style='cyan')
    table.add_column('Avg (ms)', justify='center', style='magenta')
    table.add_column('StDev (ms)', justify='center', style='magenta')
    table.add_column('Min (ms)', justify='center', style='magenta')
    table.add_column('Max (ms)', justify='center', style='magenta')
    table.add_column('Relative', justify='center', style='magenta')

    for row in rows:
        table.add_row(*row)

    console = rich.console.Console()
    console.print(table)
