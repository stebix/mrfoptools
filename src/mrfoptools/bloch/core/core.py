"""
Implement of core functions for the Bloch isochromat simulation.

Based on PyTorch implementation by Tom Griesler.

@Author: Jannik Stebani 2025
"""
import jax
import jax.numpy as jnp


def q_operator(alpha: float, phi: float) -> jax.Array:
    """
    Calculate the Bloch excitation matrix for the given flip angle `alpha`
    and phase `phi`.

    Arguments
    ---------

    alpha : float
        The flip angle in radians.

    phi : float
        The phase in radians.

    Returns
    -------

    q : jax.Array
        The Bloch excitation matrix.
    """
    sinphi = jnp.sin(phi)
    cosphi = jnp.cos(phi)
    sinalpha = jnp.sin(alpha)
    cosalpha = jnp.cos(alpha)

    M1 = jnp.array(
        [[cosphi, sinphi, 0.0],
         [-sinphi, cosphi, 0.0],
         [0.0, 0.0, 1.0]]
    )
    M2 = jnp.array(
        [[1.0, 0.0, 0.0],
         [0.0, cosalpha, sinalpha],
         [ 0.0, -sinalpha, cosalpha]]
    )
    M3 = jnp.array(
        [[cosphi, -sinphi, 0.0],
         [sinphi, cosphi, 0.0],
         [0.0, 0.0, 1.0]]
    )
    return M1 @ M2 @ M3


def r_operator(T1: float, T2: float, dt: float) -> jax.Array:
    """
    Compute the relaxation operator for the given T1, T2 and time step dt.

    Parameters
    ---------

    T1 : float
        The T1 relaxation time in ms.

    T2 : float
        The T2 relaxation time in ms.

    dt : float
        The time step in ms.

    Returns
    -------

    r : jax.Array
        The relaxation operator.
    """
    E1 = jnp.exp(-dt / T1)
    E2 = jnp.exp(-dt / T2)
    r = jnp.array(
        [[E2, 0.0, 0.0],
         [0.0, E2, 0.0],
         [0.0, 0.0, E1]]
    )
    return r


def b_operator(T1: float, dt: float) -> jax.Array:
    """
    Compute the longitudianl relaxation operator for the given T1
    and time step dt.

    Parameters
    ----------

    T1 : float
        The T1 relaxation time in ms.

    dt : float
        The time step in ms.

    Returns
    -------

    b : jax.Array
        The longitudinal relaxation operator.
    """
    return (1 - jnp.exp(-dt / T1)) * jnp.array([[0.0], [0.0], [1.0]])


def g_operator(beta: float) -> jax.Array:
    """
    Compute the rotation matrix around the z axis with angle `beta`.
    Intended use for spin dephasing simulation.

    Parameters
    ----------

    beta : float
        The rotation angle in radians.

    Returns
    -------

    g : jax.Array
        The rotation matrix.
    """
    sinbeta = jnp.sin(beta)
    cosbeta = jnp.cos(beta)
    return jnp.array(
        [[cosbeta, sinbeta, 0.0],
         [-sinbeta, cosbeta, 0.0],
         [0.0, 0.0, 1.0]]
    )


def inversion_operator(inversion_efficiency: float) -> jax.Array:
    """
    Compute the inversion operator for the given inversion efficiency.

    Parameters
    ----------

    inversion_efficiency : float
        The inversion efficiency.

    Returns
    -------

    jax.Array
        The inversion operator.
    """
    return jnp.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, -inversion_efficiency]]
    )


projection_operator = jnp.array([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0]])

