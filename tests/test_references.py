import numpy as np

import mrfoptools.optimization.diagnostics.references as refcs

class Test_compute_reference_costs:
    def test_smoke_invocation(self):
        """
        Check if the function can be invoked without errors
        and computes the costs correctly.
        """
        def forward(a, b, x): return a * x + b
        def linear_cost(y): return np.abs(y)
        def square_cost(y): return y**2
        costfuncs = {
            'linear': linear_cost,
            'square': square_cost
        }
        reference_specs = {
            'ref1': (1, 0, 1),
            'ref2': (1, 1, 2)
        }
        refcosts = refcs.compute_reference_costs(
            forward,
            costfuncs,
            reference_specs
        )
        assert refcosts[refcs.NamePair('linear', 'ref1')] == 1
        assert refcosts[refcs.NamePair('linear', 'ref2')] == 3
        assert refcosts[refcs.NamePair('square', 'ref1')] == 1
        assert refcosts[refcs.NamePair('square', 'ref2')] == 9


class Test_ReferenceCost:

    def test_smoke_creation(self):
        """
        Check if the class can be instantiated without errors.
        """
        refcost = refcs.ReferenceCost(costname='linear', refname='ref1', value=1)
        assert refcost.costname == 'linear'
        assert refcost.refname == 'ref1'
        assert refcost.value == 1
        print(refcost)


    def test_creation_from_mapping(self):
        def forward(a, b, x): return a * x + b
        def linear_cost(y): return np.abs(y)
        def square_cost(y): return y**2
        costfuncs = {
            'linear': linear_cost,
            'square': square_cost
        }
        reference_specs = {
            'ref1': (1, 0, 1),
            'ref2': (1, 1, 2)
        }
        refcosts = refcs.compute_reference_costs(
            forward,
            costfuncs,
            reference_specs
        )
        refcosts = refcs.ReferenceCost.from_mapping(refcosts)

        assert isinstance(refcosts, list)
        assert len(refcosts) == 4
        assert all(isinstance(refcost, refcs.ReferenceCost) for refcost in refcosts)