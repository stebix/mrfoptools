import numpy as np

from mrfoptools.optimization.costgrad import NumpyCostGradTuple, combine_cost_grad_mappings


def test_combine_cost_grad_mappings():
    """
    Test combining cost-grad mappings.
    """
    cost_grad_mapping_1 = {
        'cost_a': NumpyCostGradTuple(cost=1.0, grad=np.array([1.0, 2.0])),
        'cost_b': NumpyCostGradTuple(cost=10.0, grad=np.array([5.0, 4.0]))
    }
    cost_grad_mapping_2 = {
        'cost_a': NumpyCostGradTuple(cost=2.0, grad=np.array([10.0, 20.0])),
        'cost_b': NumpyCostGradTuple(cost=20.0, grad=np.array([4.0, 3.0]))
    }
    cost_grad_mapping_3 = {
        'cost_a': NumpyCostGradTuple(cost=3.0, grad=np.array([100.0, 200.0])),
        'cost_b': NumpyCostGradTuple(cost=30.0, grad=np.array([3.0, 2.0]))
    }
    mappings = [cost_grad_mapping_1, cost_grad_mapping_2, cost_grad_mapping_3]
    combined_mapping = combine_cost_grad_mappings(mappings=mappings)

    costs = combined_mapping['costs']
    grads = combined_mapping['grads']

    assert costs.keys() == {'cost_a', 'cost_b'}
    assert grads.keys() == {'cost_a', 'cost_b'}

    costs_a = costs['cost_a']
    grads_a = grads['cost_a']

    assert np.allclose(costs_a, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(grads_a, np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]))

    costs_b = costs['cost_b']
    grads_b = grads['cost_b']

    assert np.allclose(costs_b, np.array([10.0, 20.0, 30.0]))
    assert np.allclose(grads_b, np.array([[5.0, 4.0], [4.0, 3.0], [3.0, 2.0]]))