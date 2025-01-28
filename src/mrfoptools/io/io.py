import pathlib
import dataclasses
import numpy as np

from collections.abc import Mapping
from typing import Any

import zarr



@dataclasses.dataclass
class OptimizationBag:
    settings: Mapping[str, Any]
    results: Mapping[str, np.ndarray]
    histories: Mapping[str, np.ndarray] 



def store_optimization_bag(
        path: str | pathlib.Path,
        bag: OptimizationBag
) -> pathlib.Path:
    """
    Stpre optimimzation settings and results in a zarr file.
    """
    zarr_file = zarr.convenience.open(path)
    # Optimization configuration data
    for k, v in bag.settings.items():
        zarr_file.attrs[k] = v

    results_group = zarr_file.create_group('results')

    for k, v in bag.results.items():
        results_group.create_dataset(name=k, data=v)

    histories_group = zarr_file.create_group('histories')

    for k, v in bag.histories.items():
        histories_group.create_dataset(name=k, data=v)

    return path