import datetime
import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, Union, NamedTuple
from pathlib import Path
from numbers import Number

import jax
import numpy as np
import zarr

from attrs import define, field

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

PathLike = str | Path
Array = jax.Array | np.ndarray

ArrayLike = np.ndarray | Sequence[Number]
# DataMapping intended to reference nested dictionaries of numpy arrays
# or sequences of numbers that can be written to zarr arrays directly.
DataMapping: TypeAlias = Mapping[str, Union['DataMapping', ArrayLike]]

# Jsonifiable intended to reference data types that can be serialized to JSON.
# This is utilized for metadata mappings that are stored as attributes in zarr groups.
Jsonifiable = str | int | float | bool | None | Mapping[str, 'Jsonifiable'] | Sequence['Jsonifiable']
MetadataMapping: TypeAlias = Mapping[str, Union['MetadataMapping', Jsonifiable]]


class Metadata:
    creation_time: datetime.datetime
    git_hash: str
    git_branch: str
    optimization_bag_version: str


@define
class Initializations:
    fa: Array
    tr: Array
    seed: int | None


@define
class OptimizationBag:
    protocol: Mapping[str, Any]
    hyperparameters: dict[str, Any]
    histories: Mapping[str, Any] = field(repr=False)
    initializations: Mapping[str, Any] = field(repr=False)
    results: Mapping[str, Any] = field(repr=False)
    metadata: Metadata = field(repr=False)



def datamappings_are_equal(
    mapping_a: DataMapping,
    mapping_b: DataMapping
) -> bool:
    """
    Compare two data mappings for equality.
    """
    if mapping_a.keys() != mapping_b.keys():
        return False

    for key in mapping_a.keys():
        value_a = mapping_a[key]
        value_b = mapping_b[key]

        if isinstance(value_a, Mapping):
            if not datamappings_are_equal(value_a, value_b):
                return False
        else:
            if not np.allclose(value_a, value_b):
                return False

    return True


def store_datalike(
    group: zarr.Group,
    mapping: DataMapping
) -> None:
    """
    Store a mapping of data-like objects to a Zarr store.
    """
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            subgroup = group.create_group(key, overwrite=True)
            store_datalike(subgroup, value)
            continue

        array = group.create_array(key, shape=value.shape, dtype=value.dtype, overwrite=True)
        array[...] = value
        logger.info(f'Successfully stored array with shape \'{array.shape}\' to {group}')
    return None


def store_metadatalike(
    group: zarr.Group,
    mapping: MetadataMapping
) -> None:
    """
    Store a metadata mapping (i.e. mainly 'scalar' key - value pairs,
    not substantial arrays) to a Zarr store.
    """
    group.attrs.update(mapping)
    logger.info(f'Successfully stored metadata mapping to {group}')
    return None


class GroupKeys(NamedTuple):
    array_keys: set[str]
    subgroup_keys: set[str]

def retrieve_keys(
    group: zarr.Group
) -> tuple[set[str], set[str]]:
    """
    Retrieve all keys of a `zarr.Group` object and sort into array keys and subgroup keys.

    Parameters
    ----------
    group : zarr.Group
        The group to retrieve keys from.

    Returns
    -------
    GroupKeys
        A named tuple with the keys sorted into array keys and
    """
    array_keys = set(group.array_keys())
    subgroup_keys = set(group.keys()) - array_keys
    return GroupKeys(array_keys=array_keys, subgroup_keys=subgroup_keys)


ArrayMapping = Mapping[str, Union[np.ndarray, 'ArrayMapping']]  


def load_group(
    group: zarr.Group
) -> DataMapping:
    """
    Eagerly load the content - arrays and subgroups - of a `zarr.Group` object
    into a dictionary.
    """
    keys = retrieve_keys(group)
    data = {
        key : group[key][...]
        for key in keys.array_keys
    }
    subdata = {key : load_group(group[key]) for key in keys.subgroup_keys}
    return data | subdata



def store_optimization_bag(
    bag: OptimizationBag,
    path: PathLike,
    overwrite: bool = False
) -> None:
    """
    Store an `OptimizationBag` object to a Zarr store."
    """
    if not isinstance(path, Path):
        path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(f'File already exists at \'{path}\'')

    store = zarr.storage.LocalStore(path)
    root = zarr.group(store=store, overwrite=overwrite)

    # Store the protocol
    protocol_group = root.create_group('protocol', overwrite=overwrite)
    store_metadatalike(protocol_group, bag.protocol)
    # Store the hyperparameters
    hyperparameters_group = root.create_group('hyperparameters', overwrite=overwrite)
    store_metadatalike(hyperparameters_group, bag.hyperparameters)
    # Store the histories
    histories_group = root.create_group('histories', overwrite=overwrite)
    store_datalike(histories_group, bag.histories)
    # Store the initializations
    initializations_group = root.create_group('initializations', overwrite=overwrite)
    store_datalike(initializations_group, bag.initializations)
    # Store the results
    results_group = root.create_group('results', overwrite=overwrite)
    store_datalike(results_group, bag.results)
    # Store the metadata
    metadata_group = root.create_group('metadata', overwrite=overwrite)
    store_metadatalike(metadata_group, bag.metadata)

    logger.info(f'Successfully stored OptimizationBag to \'{path}\'')
    return None