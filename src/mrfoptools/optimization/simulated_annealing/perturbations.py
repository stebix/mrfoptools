"""
Functions to perturb parameterizations (i.e. control point collections)
of canonical optimization variables (TR arrays/patterns, FA arrays/patterns).

Naming rationale:

- `global` perturbation functions:
        *All* points are perturbed jointly by drawing from a scaled
        normal distribution.

- `indexed` perturbation functions:
        Only a single, indexed point is perturbed by drawing
        from a scaled normal distribution.

- `fixededge` perturbation functions:
        The edge values of the parameterization are fixed and
        remain unchanged during the perturbation.

- `varedge` perturbation functions:
        The edge values of the parameterization are variable.
        Everything can be perturbed.

@Author: Jannik Stebani 2025
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp

from mrfoptools.parameterization.parameterization import ControlPoints, EdgeMode


def min_clamp_index(i: int, min: int) -> int:
    return jax.lax.cond(
        pred=i < min,
        true_fun=lambda i: min,
        false_fun=lambda i: i,
        operand=i
    )

def max_clamp_index(i: int, max: int) -> int:
    return jax.lax.cond(
        pred=i > max,
        true_fun=lambda i: max,
        false_fun=lambda i: i,
        operand=i
    )

def clamp_index(i: int, min: int, max: int) -> int:
    return max_clamp_index(min_clamp_index(i, min), max)


def legacy_perturb(
    index: int,
    parameters: jax.Array,
    bounds: tuple[float, float],
    relscale: float,
    key: jax.Array,
    *,
    edgemode: EdgeMode = EdgeMode.FLOATING
) -> jax.Array:
    """
    Perturb the coordinate at the given index of the parameter array.

    Parameters
    ----------
    index : int
        Index of the parameter to be perturbed.
    parameters : jax.Array
        Parameterization value array to be perturbed.
        Expected to be of shape ``n_controlpoints``.
    bounds : tuple[float, float]
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
    relscale : float
        Scaling factor for the perturbation.
    key : jax.Array
        Random key for reproducible randomness.
    edgemode : EdgeMode, optional
        Mode for edge handling during perturbation.
        Defaults to EdgeMode.FLOATING.
    
    Returns
    -------
    perturbed_parameters : jax.Array
        Perturbed parameter array.
    """    
    if edgemode is EdgeMode.FIXED:
        # sanitize index to inner region
        index = clamp_index(index, min=1, max=len(parameters)-2)
    
    extent: float = bounds[1] - bounds[0]
    update = (  parameters.at[index].get(mode='clip')
              + jax.random.normal(key) * relscale * extent)
    parameters_perturbed = parameters.at[index].set(update, mode='clip')
    parameters_perturbed = jnp.clip(parameters_perturbed, min=bounds[0], max=bounds[1])
    return parameters_perturbed


class PerturbedControlPoints(NamedTuple):
    """
    Pair of unperturbed and subsequent
    perturbed control points or parameterizations.
    """
    initial: ControlPoints
    perturbed: ControlPoints



def perturb_gaussian(
    index: int,
    parameters: jax.Array,
    bounds: tuple[float, float],
    relscale: float,
    key: jax.Array,
) -> jax.Array:
    """
    Perturb the coordinate at the given index of the parameter array.

    Parameters
    ----------
    index : int
        Index of the parameter to be perturbed.
    parameters : jax.Array
        Parameterization value array to be perturbed.
        Expected to be of shape ``n_controlpoints``.
    bounds : tuple[float, float]
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
    relscale : float
        Scaling factor for the perturbation.
    key : jax.Array
        Random key for reproducible randomness.
    
    Returns
    -------
    perturbed_parameters : jax.Array
        Perturbed parameter array.
    """        
    extent: float = bounds[1] - bounds[0]
    update = (  parameters.at[index].get(mode='clip')
              + jax.random.normal(key) * relscale * extent)
    update = jnp.clip(update, min=bounds[0], max=bounds[1])
    parameters_perturbed = parameters.at[index].set(update, mode='clip')
    return parameters_perturbed