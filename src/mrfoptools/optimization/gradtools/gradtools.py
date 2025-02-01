"""
Tooling for gradient-wise diagnostics, visualization and surgery.

@author: Jannik Stebani 2025
"""
import itertools
import numpy as np
import jax
import jax.numpy as jnp

from collections.abc import Mapping


from mrfoptools.optimization.costgrad import NumpyCostGradTuple


def compute_gradient_cosine_similarities(
    cost_grad_mapping: Mapping[str, NumpyCostGradTuple]
) -> dict[str, float]:
    """
    Compute gradient cosine similiarity of all unique pairs of gradients.

    Parameters
    ----------

    cost_grad_mapping: Mapping[str, NumpyCostGradTuple]
        Mapping from criterion string name to cost-and-gradient tuples.

    Returns
    -------

    dict[str, float]
        Mapping from criterion string name to cosine similarity.
    """
    cosine_similiarities: dict[str, float] = {}
    assert len(cost_grad_mapping) >= 2, f'requiring >= 2 gradients but got {len(cost_grad_mapping)}'
    for (name_a, name_b) in itertools.combinations(cost_grad_mapping.keys(), r=2):
        cg_tpl_a = cost_grad_mapping[name_a]
        cg_tpl_b = cost_grad_mapping[name_b]
        cossim = np.dot(cg_tpl_a.grad, cg_tpl_b.grad) / (cg_tpl_a.gradnorm * cg_tpl_b.gradnorm)
        joint_name: str = '-'.join((name_a, name_b))
        cosine_similiarities[joint_name] = cossim
    return cosine_similiarities



def compute_gradient_magnitude_similarities(
    cost_grad_mapping: Mapping[str, NumpyCostGradTuple]
) -> dict[str, float]:
    """
    Compute gradient magnitude similiarity of all unique pairs of gradients.

    Parameters
    ----------

    cost_grad_mapping: Mapping[str, NumpyCostGradTuple]
        Mapping from criterion string name to cost-and-gradient tuples.

    Returns
    -------

    dict[str, float]
        Mapping from criterion string name to magnitude similarity.
    """
    magnitude_similarities: dict[str, float] = {}
    assert len(cost_grad_mapping) >= 2, f'requiring >= 2 gradients but got {len(cost_grad_mapping)}'
    for (name_a, name_b) in itertools.combinations(cost_grad_mapping.keys(), r=2):
        cg_tpl_a = cost_grad_mapping[name_a]
        cg_tpl_b = cost_grad_mapping[name_b]
        magnitude_sim = 2 * cg_tpl_a.gradnorm * cg_tpl_b.gradnorm / (cg_tpl_a.gradnorm ** 2 + cg_tpl_b.gradnorm ** 2)
        joint_name: str = '-'.join((name_a, name_b))
        magnitude_similarities[joint_name] = magnitude_sim
    return magnitude_similarities



def compute_conFIG_gradient(gradients: jax.Array) -> jax.Array:
    """
    Compute the deconflicted and rescaled conFIG gradient from a set of gradients.

    Paper: https://arxiv.org/abs/2408.11104

    Parameters
    ----------

    gradients: jax.Array
        Gradients to compute the conFIG gradient from.
        Shape (n_criterion, n_parameters)

    Returns
    -------

    jax.Array
        "Optimal" conFIG gradient of shape (n_parameters,)
    """
    weights = jnp.ones(shape=gradients.shape[0], dtype=jnp.float32)
    unit_gradients = gradients / jnp.linalg.norm(gradients, ord=2, axis=1, keepdims=True)
    optimal_direction, *_ = jnp.linalg.lstsq(unit_gradients, weights)
    
    # rescale length
    unit_optimal_direction = optimal_direction / jnp.linalg.norm(optimal_direction, ord=2)
    factors = jnp.sum(
        jnp.stack([jnp.dot(g, unit_optimal_direction) for g in gradients])
    )
    return factors * optimal_direction