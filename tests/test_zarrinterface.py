"""
test lower-level read and write functions for zarr arrays.
"""
import datetime
import numpy as np
import jax.numpy as jnp
import zarr
import zoneinfo

import pytest

import mrfoptools.io.zarrinterface as zarrinterface
import mrfoptools.io.mappingtypes as mappingtypes

@pytest.fixture
def metadata() -> dict:
    metadata = {
        'timestamp' : (datetime.
                       datetime.
                       now(tz=zoneinfo.ZoneInfo('Europe/Berlin')).
                       isoformat()
                       ),
        'author' : 'testuser',
        'git-hash' : '1234567890abcdef',
        'git-branch' : 'main',
    }
    return metadata


@pytest.fixture(params=(np, jnp))
def data(request) -> dict:
    xnp = request.param
    NR = 100
    NSTEPS = 10
    data = {
        'test_array_1' : xnp.zeros((NSTEPS, NR)),
        'test_array_2' : xnp.full((NSTEPS, NR), fill_value=25),
        'test_complex_array' : xnp.full((10, 10), fill_value=1+1j),
        'test_irregular_array' : {
            'sampling_steps' : xnp.arange(33),
            'sampling_values' : xnp.full((33, 15), fill_value=1337)
        }
    }
    return data




def test_store_and_load_metadata_mapping(tmp_path, metadata):
    filepath = tmp_path / 'test-metadata.zarr'
    store = zarr.storage.LocalStore(filepath)
    root = zarr.create_group(store=store, overwrite=False, path='test-metadata')
    zarrinterface.store(root, metadata)

    loaded_metadata = zarrinterface.load_group(root)
    assert metadata == loaded_metadata


def test_store_and_load_data_mapping(tmp_path, data):
    """
    Store pure data mapping (str key to np.ndarray or jax array value)
    to zarr store and load it back in.
    """
    filepath = tmp_path / 'test-metadata.zarr'
    store = zarr.storage.LocalStore(filepath)
    root = zarr.create_group(store=store, overwrite=False, path='test-metadata')
    zarrinterface.store(root, data)
    loaded_data = zarrinterface.load(root)
    assert mappingtypes.datamappings_are_equal(data, loaded_data)


def test_store_and_load_mixed_mapping(tmp_path, data, metadata):
    filepath = tmp_path / 'test-metadata.zarr'
    store = zarr.storage.LocalStore(filepath)
    root = zarr.create_group(store=store, overwrite=False, path='test-metadata')
    mapping = {
        'data' : data,
        'metadata' : metadata
    }
    zarrinterface.store(root, mapping)
    loaded_mapping = zarrinterface.load(root)

    assert mapping['metadata'] == loaded_mapping['metadata']
    assert mappingtypes.datamappings_are_equal(mapping['data'], loaded_mapping['data'])