"""
IMplement different cost dunctions for the 
optimization of the MR Fingerprinting sequence.

@author: Jannik Stebani 2025
"""
import functools
from collections.abc import Sequence, Callable

import jax
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


def bathtub_loss(
    x: Array,
    radius: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    gamma: float = 5.0
    ) -> Array:
    """
    Specialized bathtub loss function with flat valley and smooth transition
    to quadratic loss outside the valley.
    
    Parameters
    ----------
    x : Array
        Input data, i.e. the free variable.

    radius : float, optional
        Width of the flat valley, by default 1.0.

    alpha : float, optional
        Weight of the inner quadratic loss, by default 0.1.
        Controls steepness of the valley edges in the
        transition zones. This also affects the gradients
        in the transition zones.

    beta : float, optional
        Weight of the outer quadratic loss, by default 0.5.
        This is the standard quadratic loss outside the valley.
        Controls overall steepness of the loss function.

    gamma : float, optional
        Controls the interpolation between the inner and outer
        loss functions. Higher values effect transitions
        zones with higher curvature, in turn effecting the
        gradients in the transition zones.
        Default is 5.0.

    Returns
    -------
    Array
        Loss function value.
    """
    # Small quadratic loss near zero
    inner_loss = alpha * (x**2)
    # Standard quadratic loss outside
    outer_loss = beta * (x**2)
    # Smooth transition using sigmoid
    weight = jax.nn.sigmoid((jnp.abs(x) - radius) * gamma)
    # Blend the two losses
    return weight * outer_loss + (1 - weight) * inner_loss



def construct_bathtub_loss(
    radius: float,
    alpha: float,
    beta: float,
    gamma: float
) -> Callable[[jax.Array], jax.Array]:
    """
    Fix a bathtub loss function with given parameters.

    Parameters
    ----------
    x : Array
        Input data, i.e. the free variable.

    radius : float, optional
        Width of the flat valley, by default 1.0.

    alpha : float, optional
        Weight of the inner quadratic loss, by default 0.1.
        Controls steepness of the valley edges in the
        transition zones. This also affects the gradients
        in the transition zones.

    beta : float, optional
        Weight of the outer quadratic loss, by default 0.5.
        This is the standard quadratic loss outside the valley.
        Controls overall steepness of the loss function.

    gamma : float, optional
        Controls the interpolation between the inner and outer
        loss functions. Higher values effect transitions
        zones with higher curvature, in turn effecting the
        gradients in the transition zones.
        Default is 5.0.

    Returns
    -------
    Callable
        Single-argument bathtub loss function.
    """
    return functools.partial(
        bathtub_loss,
        radius=radius, alpha=alpha, beta=beta, gamma=gamma
    )