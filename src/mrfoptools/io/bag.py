import datetime
import logging
from collections.abc import Mapping
from typing import Any
from pathlib import Path

import jax
import numpy as np
import zarr

from attrs import define, field

import mrfoptools.io.zarrinterface as zarrinterface

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



def store_optimization_bag(
    bag: OptimizationBag,
    path: PathLike,
    *,
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
    zarrinterface.store(protocol_group, bag.protocol)
    # Store the hyperparameters
    hyperparameters_group = root.create_group('hyperparameters', overwrite=overwrite)
    zarrinterface.store(hyperparameters_group, bag.hyperparameters)
    # Store the histories
    histories_group = root.create_group('histories', overwrite=overwrite)
    zarrinterface.store(histories_group, bag.histories)
    # Store the initializations
    initializations_group = root.create_group('initializations', overwrite=overwrite)
    zarrinterface.store(initializations_group, bag.initializations)
    # Store the results
    results_group = root.create_group('results', overwrite=overwrite)
    zarrinterface.store(results_group, bag.results)
    # Store the metadata
    metadata_group = root.create_group('metadata', overwrite=overwrite)
    zarrinterface.store(metadata_group, bag.metadata)

    logger.info(f'Successfully stored OptimizationBag to \'{path}\'')
    return None


def load_optimization_bag(
    path: PathLike
) -> OptimizationBag:
    """
    Load an `OptimizationBag` object from a Zarr store.
    """
    path = Path(path) if not isinstance(path, Path) else path
    store = zarr.storage.LocalStore(path)
    root = zarr.group(store=store)

    protocol = zarrinterface.load(root['protocol'])
    hyperparameters = zarrinterface.load(root['hyperparameters'])
    histories = zarrinterface.load(root['histories'])
    initializations = zarrinterface.load(root['initializations'])
    results = zarrinterface.load(root['results'])
    metadata = zarrinterface.load(root['metadata'])

    bag = OptimizationBag(
        protocol=protocol,
        hyperparameters=hyperparameters,
        histories=histories,
        initializations=initializations,
        results=results,
        metadata=metadata
    )
    return bag