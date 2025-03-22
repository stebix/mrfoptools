"""
Tooling for cost value and gradient computation and casting.

@author: Jannik Stebani 2025
"""
import typing

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import jax
import numpy as np


class CostGradTuple(typing.NamedTuple):
    """
    Container for cost value and gradient vector.
    Typically received from `jax.value_and_grad` or similar.
    """
    cost: jax.Array
    grad: jax.Array
    gradnorm: jax.Array | None = None


class NumpyCostGradTuple(typing.NamedTuple):
    """
    Container for cost value and gradient vector as numpy ndarray.
    """
    cost: np.ndarray | float
    grad: np.ndarray
    gradnorm: np.ndarray | float | None = None


class CostContainer(typing.Protocol):
    """
    Container protocol to annotate objects (e.g. dataclass or namedtuple)
    that minimally carry a scalar cost value.
    """
    cost: float


class GradientContainer(typing.Protocol):
    """
    Container protocol to annotate objects (e.g. dataclass or namedtuple)
    that minimally carry a gradient vector value.

    The gradient attribute is expected to be an array-like object
    that implements the numpy array interface.
    """
    grad: jax.Array | np.ndarray
    gradnorm: jax.Array | np.ndarray | float | None


def cast_cg_tuple_to_numpy(
    cg_tuple: CostGradTuple,
    compute_gradnorm: bool = True
) -> NumpyCostGradTuple:
    """
    Cast jax-based `CostGradTuple` to its numpy variant.
    
    Note: For async operations, this is blocking and may impede Python run-ahead exec.
    """
    # Use np.asarray to hope for efficient no-copy cast with shared memory buffer.
    cost = np.asarray(cg_tuple.cost)
    grad = np.asarray(cg_tuple.grad)
    gradnorm = np.linalg.norm(grad) if compute_gradnorm else None
    return NumpyCostGradTuple(cost=cost, grad=grad, gradnorm=gradnorm)



def cost_grad_builder(
    functions: Mapping[str, Callable[..., tuple[jax.Array, jax.Array]]]
) -> Callable[..., dict[str, CostGradTuple]]:
    """
    Build a joint cost value and gradient vector evaluation function from an
    arbitrary number of named input functions.
    
    The input functions are expected to return individual cost and gradient
    values. Usually, the input functions are produced via `jax.value_and_grad`
    from upstream forward functions.
    
    Output is a dictionary relating the name to the cost-and-gradient tuple.
    
    TODO: Determine runtime cost of CostGradTuple casting.
    """
    def _joint_cost_grad_fun(*args) -> dict[str, CostGradTuple]:
        return {name : CostGradTuple(*func(*args)) for name, func in functions.items()}
    return _joint_cost_grad_fun


def cast_to_numpy(
    cost_grad_mapping: Mapping[str, CostGradTuple],
    compute_gradnorm: bool = True
) -> dict[str, NumpyCostGradTuple]:
    """
    Cast mapping from jax-based values to numpy-based values.
    
    Usage example: Cast gradient and cost values prior to logging with tensorboard.
    """
    return {k : cast_cg_tuple_to_numpy(v, compute_gradnorm) for k, v in cost_grad_mapping.items()}


def recast_values_to_numpy(
    mapping: Mapping
) -> dict:
    """
    Cast all array-like values in a mapping to numpy arrays.
    Automatically handles nested mappings.
    """
    recast_mapping = {}
    for k, v in mapping.items():
        if isinstance(v, Mapping):
            recast_mapping[k] = recast_values_to_numpy(v)
        elif isinstance(v, (list, tuple, jax.Array)):
            recast_mapping[k] = np.asarray(v)
    return recast_mapping


def combine_cost_grad_mappings(
    mappings: Sequence[Mapping[str, NumpyCostGradTuple]]
) -> dict[str, dict[str, np.ndarray]]:
    """
    Combine a sequence of cost-and-gradient mappings into a mapping
    that has separate mappings for costs and gradients.
    The separate mappings relate from criterion name to an array of cost or gradient values.
    """
    combined = {'costs': defaultdict(list), 'grads': defaultdict(list)}
    for mapping in mappings:
        for costname, cg_tuple in mapping.items():
            combined['costs'][costname].append(cg_tuple.cost)
            combined['grads'][costname].append(cg_tuple.grad)

    return recast_values_to_numpy(combined)