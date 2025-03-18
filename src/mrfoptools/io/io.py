import pathlib
import dataclasses
import numpy as np

from collections.abc import Mapping, Sequence
from typing import Any

import zarr


def expand_to_repr(m: Mapping[str, Sequence]) -> str:
    """
    Expand a mapping from strings to sequences (expected to be
    large lists or arrays) to a usable string representation
    that displays shapes instead of full contents.
    """
    key_to_shape_or_length: dict[str, int | tuple[int, ...]] = {}
    for k, v in m.items():
        try:
            shape_or_length = v.shape if len(v.shape) > 1 else v.shape[0]
            type_ = 'Array'
        except AttributeError:
            shape_or_length = len(v)
            type_ = 'List'

        key_to_shape_or_length[k] = f'{type_}({shape_or_length})'

    return ', '.join((f'\'{k}\'->{v}'
                      for k, v in key_to_shape_or_length.items()))


@dataclasses.dataclass
class OptimizationBag:
    settings: Mapping[str, Any]
    results: Mapping[str, np.ndarray] = dataclasses.field(repr=False)
    histories: Mapping[str, np.ndarray] = dataclasses.field(repr=False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s = ''.join((s, f'settings={self.settings}'))
        s = ', '.join((s, f'results={{{expand_to_repr(self.results)}}}'))
        s = ', '.join((s, f'histories={{{expand_to_repr(self.histories)}}})'))
        return s

def _store_bag_to_group(
        group: zarr.Group,
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