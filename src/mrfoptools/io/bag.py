import datetime
import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias, Union
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



protocol = {
    'NR' : 1000,
    'phases' : np.linspace(0, 2 * np.pi, 100),
    'TE' : 0.01,
    'n_controlpoints' : 10,
}

hyperparameters = {
    'learning_rate' : 0.01,
    'n_iterations' : 1000,
    'n_samples' : 100,
    'bathtub_parameters' : {
        'alpha' : 0.1,
        'beta' : 0.3,
        'gamma' : 0.7,
        'delta' : 2.5
    }
}

NITER = 500
NR = 1000

histories = {
    'fa' : np.zeros((NITER, NR)),
    'tr' : np.zeros((NITER, NR)),
    'cost' : np.zeros(NITER),
    'irregularly_logged_gradient' : {
        'iterations' : np.array([1, 250, 300, 499]),                        
        'values' : np.zeros((4, NR))
    }
}


ArrayLike = np.ndarray | Sequence[Number]
DataMapping: TypeAlias = Mapping[str, Union['DataMapping', ArrayLike]]

Jsonifiable = str | int | float | bool | None | Mapping[str, 'Jsonifiable'] | Sequence['Jsonifiable']
MetadataMapping: TypeAlias = Mapping[str, Union['MetadataMapping', Jsonifiable]]


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

