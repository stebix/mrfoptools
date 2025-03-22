"""
test lower-level read and write functions for zarr arrays.
"""
import zarr

import mrfoptools.io.zarrinterface as zarrinterface
import mrfoptools.io.mappingtypes as mappingtypes



def test_store_and_load_metadata_mapping(tmp_path, metadata):
    filepath = tmp_path / 'test-metadata.zarr'
    store = zarr.storage.LocalStore(filepath)
    root = zarr.create_group(store=store, overwrite=False, path='test-metadata')
    zarrinterface.store(root, metadata)

    loaded_metadata = zarrinterface.load(root)
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