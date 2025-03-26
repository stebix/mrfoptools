"""
Functions for computing, normalizing, and regularizing Gramian matrices.

Partly based on the implementation provided by the paper https://arxiv.org/pdf/2406.16232

@Author: Jannik Stebani 2025
"""
from typing import TypeAlias

import jax
import jax.numpy as jnp

Array: TypeAlias = jax.Array


def compute_gramian(matrix: Array) -> Array:
    """
    Compute the Gramian matrix the vectors contained in `matrix`.
    """
    return matrix @ matrix.T


def compute_normalized_gramian(
        matrix: Array,
        eps: float
    ) -> Array:
    """
    Compute the normalized Gramian matrix of a matrix.
    """
    (left_unitary_matrix,
     singular_values,
     right_unitary_matrix) = jnp.linalg.svd(matrix, full_matrices=False)
    max_singular_value = jnp.max(singular_values)

    if max_singular_value < eps:
        scaled_singular_values = jnp.zeros_like(singular_values)
    else:
        scaled_singular_values = singular_values / max_singular_value

    normalized_gramian = (
        left_unitary_matrix @ jnp.diag(scaled_singular_values**2) @ left_unitary_matrix.T
    )
    return normalized_gramian


def regularize(gramian: Array, eps: float) -> Array:
    """
    Regularize the Gramian matrix.
    """
    return gramian + eps * jnp.eye(gramian.shape[0])


def compute_normalized_regularized_gramian(
    matrix: Array,
    norm_eps: float,
    reg_eps: float
    ) -> Array:
    """
    Compute the normalized and regularized Gramian matrix of a matrix.
    """
    normalized_gramian = compute_normalized_gramian(matrix, eps=norm_eps)
    return regularize(normalized_gramian, eps=reg_eps)