import pathlib

import zarr

from collections.abc import Mapping
import dataclasses


@dataclasses.dataclass
class OptimizationBag:
    settings: Mapping
    results: Mapping
    histories: Mapping 



def store_optimization_bag(
        path: str | pathlib.Path,
        bag: OptimizationBag
) -> pathlib.Path:
    """
    Stpre optimimzation settings and results in a zarr file.
    """
    zarr_file = zarr.convenience.open()