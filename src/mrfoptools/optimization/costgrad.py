"""
Tooling for cost value and gradient computation and casting.

@author: Jannik Stebani 2025
"""
from collections.abc import Callable, Mapping
from typing import NamedTuple
import jax
import numpy as np


class CostGradTuple(NamedTuple):
    """
    Container for cost value and gradient vector.
    Typically received from `jax.value_and_grad` or similar.
    """
    cost: jax.Array
    grad: jax.Array
    gradnorm: jax.Array | None = None


class NumpyCostGradTuple(NamedTuple):
    """
    Container for cost value and gradient vector as numpy ndarray.
    """
    cost: np.ndarray | float
    grad: np.ndarray
    gradnorm: np.ndarray | float | None = None


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