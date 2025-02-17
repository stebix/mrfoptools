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
import jax
import jax.numpy as jnp

def generate_global_fixededge_perturbation(
    parameters: jax.Array,
    bounds: tuple[float, float],
    scale: float,
    key: jax.Array
) -> jax.Array:
    """
    Generate global (i.e. all points are perturbed jointly)
    perturbation for parameter array with *fixed* edge values,

    Parameters
    ----------
    parameters : jax.Array
        Parameterization value array to be perturbed.
        First and last entry remain fixed.
    bounds : tuple[float, float]
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
    scale : float
        Scaling factor for the perturbation.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    perturbed_parameters : jax.Array
        Perturbed parameter array.
    """
    shape = parameters.shape[0] - 2
    delta = jax.random.normal(key, shape=shape) * scale
    inner = parameters[1:-1] + delta
    parameters_perturbed = jnp.clip(
        parameters.at[1:-1].set(inner), *bounds
    )
    return parameters_perturbed


def generate_global_varedge_perturbation(
    parameters: jax.Array,
    bounds: tuple[float, float],
    scale: float,
    key: jax.Array,
) -> jax.Array:
    """
    Generate global (i.e. all points are perturbed jointly)
    perturbation for parameter array with *variable* edge values,

    Parameters
    ----------
    parameters : jax.Array
        Parameterization value array to be perturbed.
        All elements can be modified.
    bounds : tuple[float, float]
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
    scale : float
        Scaling factor for the perturbation.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    perturbed_parameters : jax.Array
        Perturbed parameter array.
    """
    shape = parameters.shape[0]
    delta = jax.random.normal(key, shape=shape) * scale
    parameters_perturbed = jnp.clip(
        parameters + delta, *bounds
    )
    return parameters_perturbed


def generate_indexed_fixededge_perturbation(
    index: int,
    parameters: jax.Array,
    bounds: tuple[float, float],
    scale: float,
    key: jax.Array
) -> jax.Array:
    """
    Generate a indexed (i.e. only a single point is perturbed)
    perturbation for parameter array with *fixed* edge values.

    If the index is out of bounds, it is clipped to the
    mutation-enabled inner region.

    Parameters
    ----------
    index : int
        Index of the parameter to be perturbed.
        Must be in the range [1, len(parameters) - 1],
        otherwise it is clipped to the inner region.
    parameters : jax.Array
        Parameterization value array to be perturbed.
        First and last entry remain fixed.
    bounds : tuple[float, float]
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
    scale : float
        Scaling factor for the perturbation.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    perturbed_parameters : jax.Array
        Perturbed parameter array.
    """

    # TODO: Possible microoptimization to pre-check
    #       the correct index range.
    # Otherwise clip to inner region: i.e. perturbation preserves
    # the edge values
    # [  a  ,  b  ,  c  ,  d  ]
    #    ^                 ^
    #   fix               fix
    # indices outside get clipped to inner region
    if index not in range(1, parameters.shape[0] - 1):
        index = 1 if index < 1 else parameters.shape[0] - 2
        
    update = jnp.clip(
        parameters[index] + jax.random.normal(key) * scale,
        *bounds
    )
    parameters_perturbed = parameters.at[index].set(update, mode='clip')
    return parameters_perturbed


def generate_indexed_varedge_perturbation(
    index: int,
    parameters: jax.Array,
    bounds: tuple[float, float],
    scale: float,
    key: jax.Array
) -> jax.Array:
    """
    Generate a indexed (i.e. only a single point is perturbed)
    perturbation for parameter array with *variable* edge values.

    If the index is out of bounds, it is clipped to the
    mutation-enabled inner region.

    Parameters
    ----------
    index : int
        Index of the parameter to be perturbed.
        Must be in the range [1, len(parameters) - 1],
        otherwise it is clipped to the inner region.
    parameters : jax.Array
        Parameterization value array to be perturbed.
    bounds : tuple[float, float]
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
    scale : float
        Scaling factor for the perturbation.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    perturbed_parameters : jax.Array
        Perturbed parameter array.
    """
    # TODO: Possible microoptimization to pre-check
    #       the correct index range.
    # Otherwise clip to inner region: i.e. perturbation gets
    # clipped inside parameters values
    # [  a  ,  b  ,  c  ,  d  ]
    #    ^                 ^
    #   var               var
    update = jnp.clip(
        parameters[index] + jax.random.normal(key) * scale,
        *bounds
    )
    parameters_perturbed = parameters.at[index].set(update, mode='clip')
    return parameters_perturbed