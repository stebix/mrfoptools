import numpy as np
import zarr
import pytest
import zarr.convenience

from collections.abc import Mapping


from mrfoptools.io.io import (
    _store_bag_to_group,
    store_optimization_bag,
    load_optimization_bag,
    store_optimization_bags,
    OptimizationBag
)


def mappings_are_equal(map_a: Mapping[str, np.ndarray], map_b: Mapping[str, np.ndarray]) -> bool:
    """
    Quick-compare two mappings of numpy arrays.
    """
    if map_a.keys() != map_b.keys():
        return False
    for k in map_a.keys():
        if not np.allclose(map_a[k], map_b[k]):
            return False
    return True


def bags_are_equal(bag_a: OptimizationBag, bag_b: OptimizationBag) -> bool:
    """
    Quick-compare two optimization bags for equality.
    """
    if bag_a.settings != bag_b.settings:
        return False
    if not mappings_are_equal(bag_a.results, bag_b.results):
        return False
    if not mappings_are_equal(bag_a.histories, bag_b.histories):
        return False
    return True


def make_mock_optimization_bag(**kwargs) -> OptimizationBag:
    """Create filled optimization bag with mock data."""
    alpha = kwargs.get('alpha', 1)
    name = kwargs.get('name', 'picard-jean-luc')
    beta = kwargs.get('beta', [1, 2, 3])

    default_results = {'r-ones' : np.full(shape=3, fill_value=1),
                       'r-tens' : np.full(shape=3, fill_value=10)}
    default_histories = {'h-fives' : np.full(shape=3, fill_value=5),
                         'h-sevens' : np.full(shape=3, fill_value=7)}
    
    results = kwargs.get('results', default_results)
    histories = kwargs.get('histories', default_histories)

    bag = OptimizationBag(
        settings={'alpha' : alpha, 'beta' : beta, 'name' : name},
        results=results,
        histories=histories
    )
    return bag


@pytest.fixture
def stored_optimization_bag_path(tmp_path):
    """
    Create stored optimization bag at temporary location.
    Also works as implicit test for `store_optimization_bag`.
    """
    storage_path = tmp_path / 'test.zarr'
    bag = make_mock_optimization_bag()
    p = store_optimization_bag(storage_path, bag)
    return p


def test_load_stored_optimization_bag(stored_optimization_bag_path):
    bag = load_optimization_bag(stored_optimization_bag_path)

    settings = bag.settings
    expected_settings: dict = {'alpha' : 1, 'beta' : [1, 2, 3], 'name' : 'picard-jean-luc'}
    assert settings == expected_settings

    assert np.allclose(bag.results['r-ones'], np.full(shape=3, fill_value=1))
    assert np.allclose(bag.results['r-tens'], np.full(shape=3, fill_value=10))
    assert np.allclose(bag.histories['h-fives'], np.full(shape=3, fill_value=5))
    assert np.allclose(bag.histories['h-sevens'], np.full(shape=3, fill_value=7))


def test_store_optimization_bag_to_group(tmp_path):
    path = tmp_path / 'test.zarr'
    bag = make_mock_optimization_bag()

    zarr_file = zarr.convenience.open(path)
    test_group = zarr_file.create_group('test_group')
    _store_bag_to_group(test_group, bag)

    reloaded_bag = load_optimization_bag(path / 'test_group')

    assert bags_are_equal(bag, reloaded_bag)


def test_store_optimization_bags(tmp_path):
    """
    Test storage of multiple optimization bags.
    """
    bag_a = make_mock_optimization_bag()
    bag_b = make_mock_optimization_bag(
        alpha=2, name='riker-william-thomas', beta=[1, 7, 0, 1],
        results={'r-fives' : np.full(shape=4, fill_value=5),
                 'r-nines' : np.full(shape=4, fill_value=9)},
        histories={'h-threes' : np.full(shape=4, fill_value=3),
                   'h-elevens' : np.full(shape=4, fill_value=11)}
    )

    path = tmp_path / 'test.zarr'
    store_optimization_bags(path, {'repeat-a' : bag_a, 'repeat-b' : bag_b})

    reloaded_bag_a = load_optimization_bag(path / 'repeat-a')
    reloaded_bag_b = load_optimization_bag(path / 'repeat-b')

    assert bags_are_equal(bag_a, reloaded_bag_a)
    assert bags_are_equal(bag_b, reloaded_bag_b)