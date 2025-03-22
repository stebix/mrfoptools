"""
Test storage of multiple optimization bags.
"""
from mrfoptools.io.bag import OptimizationBag, store_optimization_bag, load_optimization_bag
from mrfoptools.io.mappingtypes import datamappings_are_equal

def test_store_and_load_bag(
        tmp_path,
        data,
        hyperparameters,
        metadata,
        initialization,
        results,
        protocol
):
    """
    Test storage and loading of an optimization bag.
    """
    fpath = tmp_path / 'test-bag.zarr'
    bag = OptimizationBag(
        histories=data,
        hyperparameters=hyperparameters,
        metadata=metadata,
        initializations=initialization,
        results=results,
        protocol=protocol
    )
    store_optimization_bag(bag, fpath)
    loaded_bag = load_optimization_bag(fpath)

    assert datamappings_are_equal(bag.histories, loaded_bag.histories)
    assert bag.hyperparameters == loaded_bag.hyperparameters
    assert bag.metadata == loaded_bag.metadata
    assert bag.results == loaded_bag.results
    assert datamappings_are_equal(bag.protocol, loaded_bag.protocol)
    assert datamappings_are_equal(bag.initializations, loaded_bag.initializations)