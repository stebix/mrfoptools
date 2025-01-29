import pathlib
import dataclasses
import numpy as np

from collections.abc import Mapping, Sequence
from typing import Any

import zarr



@dataclasses.dataclass
class OptimizationBag:
    settings: Mapping[str, Any]
    results: Mapping[str, np.ndarray]
    histories: Mapping[str, np.ndarray] 


def _store_bag_to_group(
        group: zarr.hierarchy.Group,
        bag: OptimizationBag
) -> None:
    """
    Store optimization bag in a zarr group.
    """
    for k, v in bag.settings.items():
        group.attrs[k] = v

    results_group = group.create_group('results')

    for k, v in bag.results.items():
        results_group.create_dataset(name=k, data=v)

    histories_group = group.create_group('histories')

    for k, v in bag.histories.items():
        histories_group.create_dataset(name=k, data=v)



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


def store_optimization_bags(
        path: str | pathlib.Path,
        bags: Sequence[OptimizationBag] | Mapping[str, OptimizationBag]
) -> pathlib.Path:
    """
    Store multiple optimization bags in a single zarr file.
    Typical use case: Runs with very tight coupling/relationship, e.g.
    repeated runs over different random seeds.

    Parameters
    ----------

    path : str | pathlib.Path
        Path to the zarr file to be created.

    bags : Sequence[OptimizationBag] | Mapping[str, OptimizationBag]
        Sequence of optimization bags or a mapping of bags with keys.
        If a mapping is provided, the keys are used as group names in the zarr file.
        If a sequence is provided, the groups are named 'bag-1', 'bag-2', etc.

    Returns
    -------

    pathlib.Path
        Path to the created zarr file.
    """
    if not isinstance(bags, Mapping):
        bags = {f'bag-{i}': bag for i, bag in enumerate(bags, start=1)}

    zarr_file = zarr.convenience.open(path)
    for name, bag in bags.items():
        group = zarr_file.create_group(name)
        _store_bag_to_group(group, bag)

    return path

    


def load_optimization_bag(
        path: str | pathlib.Path
) -> OptimizationBag:
    """
    Load stored optimization settings and results from a zarr file.
    """
    zarr_file = zarr.convenience.open(path, mode='r')
    settings = {k : v for k, v in zarr_file.attrs.items()}
    results = {k : v[...] for k, v in zarr_file['results'].items()}
    histories = {k : v[...] for k, v in zarr_file['histories'].items()}
    return OptimizationBag(settings=settings, results=results, histories=histories)