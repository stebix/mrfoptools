import logging

from collections.abc import Mapping
from typing import Any

import numpy as np
import jax
import zarr

from mrfoptools.io.mappingtypes import DataMapping, MetadataMapping, GroupKeys

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


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
        logger.debug(f'Successfully stored array with shape \'{array.shape}\' to {group}')
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
    logger.debug(f'Successfully stored metadata mapping to {group}')
    return None


def store(
    group: zarr.Group,
    mapping: Mapping[str, Any],
    *,
    overwrite: bool = False
) -> None:
    """
    Store a mixed mapping to a zarr group.
    """
    array_like_types = (np.ndarray, jax.Array)

    for key, value in mapping.items():

        if isinstance(value, Mapping):
            subgroup = group.create_group(key, overwrite=overwrite)
            store(subgroup, value)
            continue

        if isinstance(value, array_like_types):
            array = group.create_array(
                key, shape=value.shape, dtype=value.dtype, overwrite=overwrite
            )
            array[...] = value
            logger.debug(f'Successfully stored array with shape \'{array.shape}\' to {group}')
            continue

        group.attrs[key] = value
    
    logger.debug(f'Successfully stored mapping mapping to \'{group}\'')
    return None



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


def load(
    group: zarr.Group
) -> DataMapping:
    """
    Eagerly load the content - arrays and subgroups - of a `zarr.Group` object
    into a dictionary.
    """
    keys = retrieve_keys(group)
    metadata = dict(group.attrs)
    data = {
        key : group[key][...]
        for key in keys.array_keys
    }
    subdata = {key : load(group[key]) for key in keys.subgroup_keys}
    return metadata | data | subdata