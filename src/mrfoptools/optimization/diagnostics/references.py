"""
References are utilized to compare the fitness of an optimization
against certain reference patterns.

Typical reference patterns are e.g.:

- Constant flip angle pattern
- Yun flip angle pattern (sinusoidal pattern)

@Author: Jannik Stebani 2025
"""
import enum
from collections.abc import Callable, Mapping
from numbers import Number
from typing import NamedTuple

import attrs

from mrfoptools.optimization.costgrad import CostContainer


class NamePair(NamedTuple):
    costname: str
    refname: str


class CostValueRange(enum.Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'


def compute_reference_costs(
    forward: Callable,
    costfuncs: Mapping[str, Callable],
    reference_specs: Mapping[str, tuple]
) -> dict[NamePair, float]:
    """
    Compute the reference costs for the given cost functions and
    reference specifications.

    Parameters
    ----------

    forward : Callable
        Forward model function.
        Outputs must be digestible by the cost functions.

    costfuncs : Mapping[str, Callable]
        Mapping of cost function names to cost function implementations.

    reference_specs : Mapping[str, tuple]
        Mapping of reference names to reference specifications.
        Each reference specification is a tuple of arguments to the
        forward function.

    Returns
    -------

    reference_costs : dict[NamePair, float]
        Mapping of cost-name reference-name pairs to the computed costs.
    """
    reference_costs: dict[str, float] = {}
    for costname, costfunc in costfuncs.items():
        for refname, refspec in reference_specs.items():
            cost = costfunc(forward(*refspec))
            reference_costs[NamePair(costname, refname)] = cost
    return reference_costs



@attrs.define
class ReferenceCost:
    """
    Container for reference cost values.

    Parameters
    ----------
    costname : str
        Name of the reference cost function. This ID string is used
        to match reference costs with transient costs downstream.

    refname : str
        Name of the reference parameters or pattern that
        were used to compute the reference cost. This is typically
        a certain initialization (e.g. Yun or Cao or constant pattern)

    value : Number
        Scalar value of the reference cost computed with the reference
        parameters or pattern.

    range : diag.CostValueRange
        Range of the cost value. Determines whether the cost value
        is negative (always and unbounded) or positive.
        Usage is for downstream relative gain computation.
    """
    costname: str
    refname: str
    value: Number
    range: CostValueRange = CostValueRange.POSITIVE

    @classmethod
    def from_mapping(cls, mapping: Mapping[NamePair, float]):
        return [
            cls(costname=pair.costname, refname=pair.refname, value=value)
            for pair, value in mapping.items()
        ]


class RelativeGainEvaluator:
    """
    Compute relative gains of cost functions with respect to reference costs.

    Relative gains are computed by matching reference costs from the initializer method
    via their `costname` to supplied transient cost values in the
    `transient_costs` argument of the `__call__` method.

    Note: The `transient_costs` argument is expected to provide a cost value for 
          each reference cost in the initializer method. If a base cost is not found in
          `transient_costs`, a `KeyError` is raised.
    """
    def __init__(
        self,
        *reference_costs: ReferenceCost,
        prefix: str = 'relative_gain'
    ) -> None:
        self.reference_costs: list[ReferenceCost] = reference_costs
        self.prefix: str = prefix

    def __call__(
        self,
        transient_costs: Mapping[str, CostContainer]
    ) -> dict[str, float]:

        relative_gains: dict[str, float] = {}
        for base_cost in self.base_costs:
            logname = f'{self.prefix}/{base_cost.costname}-{base_cost.refname}'
            cost_container = transient_costs[base_cost.costname]
            if base_cost.range is CostValueRange.POSITIVE:
                relative_gain = base_cost.value / cost_container.cost
            elif base_cost.range is CostValueRange.NEGATIVE:
                relative_gain = cost_container.cost / base_cost.value
            else:
                raise ValueError(f'Invalid cost value range: {base_cost.range}')

            relative_gains[logname] = relative_gain
        return relative_gains

    def __str__(self):
        return f'{self.__class__.__name__}(reference_costs=\'{self.reference_costs}\')'

    def __repr__(self):
        return str(self)


