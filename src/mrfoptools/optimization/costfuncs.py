"""
IMplement different cost dunctions for the 
optimization of the MR Fingerprinting sequence.

@author: Jannik Stebani 2025
"""
from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array



def orthogonality_criterion(signals: Array) -> Array:
    """
    Cost function for the total pairwise orthogonality of the signals.
    Note: Signals are assumed to be normalized.

    Parameters
    ----------

    signals : Array
        Signals array of the shape (n_species, n_shots).

    Returns
    -------

    cost : Array
        Orthogonality criterion.
    """
    n_species, _ = signals.shape
    return jnp.linalg.norm(jnp.eye(n_species) - signals @ jnp.conjugate(signals.T))


def mean_signal_criterion(signals: Array) -> Array:
    """
    Cost function evaluating the mean absolute value of the signals.
    Negative sign to make it a cost function, since we
    want to maximize the signals.

    Signals should be a 2D array of shape (n_species, n_shots).
    """
    return -jnp.mean(jnp.abs(signals))


def fa_total_variation_criterion(fa: Array) -> Array:
    """
    Cost function evaluating the total variation of the flip angles.
    Smooth flip angle sequences should have a lower cost.

    Flip angles should be a 1D array of length `n_shots`.
    """
    return jnp.linalg.norm(fa[:-1] - fa[1:], ord=2)



def minimum_average_criterion(signals: Array) -> Array:
    """
    Compute inverse if minimum average signal of the
    tissue species wise signals.

    Parameters
    ----------
    signals : Array
        Signals with shape: ``(n_species, n_tr)``.

    Notes
    -----
    Quite similar to the mean signal criterion, but emphasizes
    that even the smallest signal should be large. In principle,
    mean_signal_criterion could promote solutions where one species
    is fully suppressed while others are maximized.
    """
    return 1 / jnp.min(jnp.linalg.norm(signals, ord=2, axis=1))


def create(spec: Sequence[dict[str, float | str]]) -> Array:
    """
    Programmatically create a sequence of cost functions from a sequence
    of specification dictionaries that contain the function name and
    a weighting factor.
    """
    raise NotImplementedError('not yet')

