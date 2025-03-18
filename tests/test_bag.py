import numpy as np
import zarr

from mrfoptools.io.bag import store_datalike, store_metadatalike, load_group, datamappings_are_equal


def test_store_datalike(tmp_path):
    test_dir = tmp_path / 'test-1'
    test_dir.mkdir()

    NR = 100
    NSTEPS = 10

    array_path = test_dir / 'array.zarr'
    test_group_name = 'test_group'
    metadata_group_name = 'metadata'
    # set up zarr array structure on file system store
    root = zarr.create_group(array_path, overwrite=False)
    test_group = root.create_group(test_group_name, overwrite=False)
    metadata_group = root.create_group(metadata_group_name, overwrite=False)
    # set up mock data
    data = {
        'test_array_1' : np.zeros((NSTEPS, NR)),
        'test_array_2' : np.full((NSTEPS, NR), fill_value=25),
        'test_irregular_array' : {
            'sampling_steps' : np.arange(33),
            'sampling_values' : np.full((33, 15), fill_value=1337)
        }
    }
    metadata = {
        'test_metadata_1' : 'test',
        'parameter' : 1000,
        'learning_rate' : 0.01,
        'in_depth_settings' : {
            'n_iterations' : 1000,
            'n_samples' : 100,
            'bathtub_parameters' : {
                'alpha' : 0.1,
                'beta' : 0.3,
                'gamma' : 0.7,
                'delta' : 2.5
            }
        }
    }
    store_datalike(test_group, data)
    store_metadatalike(metadata_group, metadata)
    # load data back in
    retrieved_data = load_group(test_group)
    retrieved_metadata = dict(metadata_group.attrs)
    # check equivalence
    assert datamappings_are_equal(data, retrieved_data)
    assert metadata == retrieved_metadata