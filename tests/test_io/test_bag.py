import numpy as np
import zarr

from mrfoptools.io.bag import (OptimizationBag,
                               store_optimization_bag,
                               )
from mrfoptools.io.zarrinterface import load_group, store_datalike, store_metadatalike
from mrfoptools.io.mappingtypes import datamappings_are_equal

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


def test_store_bag(tmp_path):
    test_dir = tmp_path / 'test-1'
    test_dir.mkdir()
    test_path = test_dir / 'test.zarr'

    NR = 100
    NSTEPS = 10
    data = {
        'test_array_1' : np.zeros((NSTEPS, NR)),
        'test_array_2' : np.full((NSTEPS, NR), fill_value=25),
        'test_irregular_array' : {
            'sampling_steps' : np.arange(33),
            'sampling_values' : np.full((33, 15), fill_value=1337)
        }
    }
    protocol = {
        'sequence' : 'fisp',
        'NR' : 1000,
        'test' : 'nein-koenig-nein',
        'T1' : [1, 2, 3, 4, 5, 6],
        'T2' : [10, 20, 30, 40, 50, 60]
    }
    initializations = {
        'flipangles' : 'yun-base',
        'tr' : 'constant',
        'seed' : 1234
    }
    hyperparameter = {
        'optimization' : 'gradient_descent',
        'optimizer' : 'adam',
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
    metadata = {
        'timestamp' : '2021-09-10T12:00:00',
        'author' : 'testuser',
        'git-hash' : '1234567890abcdef',
        'git-branch' : 'main',
    }

    bag = OptimizationBag(
        protocol=protocol,
        hyperparameters=hyperparameter,
        histories=data,
        initializations=initializations,
        results={},
        metadata=metadata
    )

    store_optimization_bag(bag, test_path)