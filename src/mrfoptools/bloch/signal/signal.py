"""
Implement signal formation simulations tooling.

@Author Jannik Stebani 2025
"""
# ruff: noqa: F401 
import functools
import jax
import jax.numpy as jnp
import numpy as np

import mrfoptools.bloch.core.core as bloch


def propagate_isochromat(
        i: int,
        carry: tuple[jax.Array, jax.Array],
        *,
        fa: jax.Array,
        phases: jax.Array,
        TR: jax.Array,
        beta: float,
        T1: float,
        T2: float,
        isochromat_magnetization: float,
        TE: float,
        r_TE: jax.Array,
        b_TE: jax.Array,
        P: jax.Array
) -> jax.Array:
    """
    Propagate state of a single isochromat.

    Parameters
    ---------

    i : int
        jax.lax.fori_loop index.

    carry : tuple[jax.Array, jax.Array]
        Tuple of the current state of the magnetization and the signal.
        Propagates through the fori_loop.

    fa : jax.Array
        Flip angles in radians.

    phases : jax.Array
        Phase angles in radians.

    TR : jax.Array
        Repetition times.

    beta : float
        Dephasing angle in radians.

    T1 : float
        Longitudinal relaxation time.

    T2 : float
        Transversal relaxation time.

    isochromat_magnetization : float
        Magnetization per isochromat.
        Usually `M0 / n_isochromats`.

    TE : float
        Echo time.

    r_TE : jax.Array
        Relaxation operator for the echo time.

    b_TE : jax.Array
        Longitudinal relaxation operator for the echo time.

    P : jax.Array
        Projection operator.

    Returns
    -------

    tuple[jax.Array, jax.Array]
        Tuple of the new state of the isochromat magnetization and the signal.
    """
    (m, signal) = carry
    q = bloch.q_operator(fa[i], phases[i])

    m_new = r_TE @ q @ m + isochromat_magnetization * b_TE
    m_transversal = (P @ m_new) * jnp.exp(-1j * phases[i])

    # record magnetization state
    signal = signal.at[i].set(m_transversal[0, 0] + 1j * m_transversal[1, 0])

    # propagate state further: relaxation during TR - TE and gradient dephasing
    m_new = (  bloch.g_operator(beta) @ bloch.r_operator(T1=T1, T2=T2, dt=TR[i]-TE) @ m_new
             + isochromat_magnetization * bloch.b_operator(T1=T1, dt=TR[i]-TE))

    return (m_new, signal)



def compute_signal_isochromat(
        beta: float,
        T1: float,
        T2: float,
        fa: jax.Array,
        TR: jax.Array,
        phases: jax.Array,
        isochromat_magnetization: float,
        TI: float,
        TE: float,
        inversion_efficiency: float = 1.0
) -> jax.Array:
    """
    Compute full signal evolution for a single isochromat.
    Isochromat-wise inhomogeneities are repsected via beta dephasing parameter.
    """
    r_TE = bloch.r_operator(T1, T2, TE)
    b_TE = bloch.b_operator(T1, TE)
    inversion_op = bloch.inversion_operator(inversion_efficiency)

    P = bloch.projection_operator

    # prepare initial isochromat magnetization state vector
    m = jnp.array([[0.0], [0.0], [isochromat_magnetization]])
    # begin with initial inversion of the magnetization
    r_TI = bloch.r_operator(T1, T2, TI)
    b_TI = bloch.b_operator(T1, TI)

    m = r_TI @ inversion_op @ m + isochromat_magnetization * b_TI

    signal = jnp.zeros(shape=fa.shape, dtype=jnp.complex64)

    propagate = functools.partial(
        propagate_isochromat,
        fa=fa, phases=phases, TR=TR, beta=beta, T1=T1, T2=T2,
        isochromat_magnetization=isochromat_magnetization,
        TE=TE, r_TE=r_TE, b_TE=b_TE, P=P
    )

    # signal evolution
    init_val = (m, signal)
    lower = 0
    upper = len(fa)
    m, signal = jax.lax.fori_loop(lower, upper, propagate, init_val)
   
    return signal


def compute_signal(
        T1: float,
        T2: float,
        fa: jax.Array,
        TR: jax.Array,
        phases: jax.Array,
        M0: float,
        TI: float,
        TE: float,
        n_isochromats: int,
        inversion_efficiency: float = 1.0,
        dephasing: int = 1
) -> jax.Array:
    """
    Compute signal evolution for a given set of parameters.
    """
    isochromat_magnetization = M0 / n_isochromats
    betas = jnp.linspace(0, dephasing * jnp.pi, num=n_isochromats)

    compute_signal_isochromats = functools.partial(
        compute_signal_isochromat,
        T1=T1, T2=T2, fa=fa, TR=TR, phases=phases,
        isochromat_magnetization=isochromat_magnetization,
        TI=TI, TE=TE, inversion_efficiency=inversion_efficiency
    )
    compute_signal_isochromats = jax.jit(jax.vmap(compute_signal_isochromats, in_axes=0))

    signals = compute_signal_isochromats(betas)

    return jnp.sum(signals, axis=0)