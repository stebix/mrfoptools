import numpy as np
import pytest

from mrfoptools.io.io import (
    store_optimization_bag,
    load_optimization_bag,
    OptimizationBag
)


def make_mock_optimization_bag() -> OptimizationBag:
    """Create filled optimization bag with mock data."""
    bag = OptimizationBag(
        settings={'alpha' : 1, 'beta' : [1, 2, 3], 'name' : 'picard-jean-luc'},
        results={'r-ones' : np.full(shape=3, fill_value=1),
                 'r-tens' : np.full(shape=3, fill_value=10)},
        histories={'h-fives' : np.full(shape=3, fill_value=5),
                   'h-sevens' : np.full(shape=3, fill_value=7)}
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


