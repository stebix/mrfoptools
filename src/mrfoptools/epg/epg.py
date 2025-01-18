from enum import Enum

import jax
import jax.numpy as jnp

import numpy as np

Array = jax.Array | np.ndarray

# port the q_epg function from the signalmodel_epg.py file to JAX
def q_epg(alpha: float, phi: float | None = None) -> Array:
    """Compute EPG excitation matrix for a given flip angle alpha and phase phi.

    Parameters
    ----------

    alpha : float
        Flip angle in radians.

    phi : float, optional
        Phase in radians. Default is pi/2.

    Returns
    -------

    q : array
        EPG excitation matrix.
    """
    phi = phi or jnp.pi / 2

    c1 = jnp.stack(
        [jnp.cos(alpha / 2) ** 2,
         jnp.exp(2 * 1j * phi) * jnp.sin(alpha / 2) ** 2,
         -1j * jnp.exp(1j*phi) * jnp.sin(alpha)]
    )
    c2 = jnp.stack(
        [jnp.exp(-2 * 1j * phi) * jnp.sin(alpha / 2) ** 2, 
         jnp.cos(alpha / 2) ** 2, 
         1j * jnp.exp(-1j * phi) * jnp.sin(alpha)]
    )
    c3 = jnp.stack(
        [-1j/2 * jnp.exp(-1j * phi) * jnp.sin(alpha), 
          1j/2 * jnp.exp(1j * phi) * jnp.sin(alpha), 
         jnp.cos(alpha)]
    )
    return jnp.conjugate(jnp.stack([c1, c2, c3]))


def r_epg(T1: float, T2: float, dt: float) -> Array:
    """Compute EPG relaxation matrix for a given T1, T2 and time step dt.

    Parameters
    ----------

    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    dt : float
        Time step in seconds.

    Returns
    -------

    r : array
        EPG relaxation matrix.
    """
    E1 = jnp.exp(-dt / T1)
    E2 = jnp.exp(-dt / T2)
    return jnp.stack(
        [jnp.stack([E2, 0.0, 0.0]),
         jnp.stack([0.0, E2, 0.0]),
         jnp.stack([0.0, 0.0, E1])]
    )


def dr_dT1_epg(T1: float, dt: float) -> Array:
    """Compute the derivative of the EPG relaxation matrix with respect to T1.

    Parameters
    ----------

    T1 : float
        Longitudinal relaxation time in seconds.

    dt : float
        Time step in seconds.

    Returns
    -------

    dr_dT1 : array
        Derivative of the EPG relaxation matrix with respect to T1.
    """
    E1 = jnp.exp(-dt / T1)
    return jnp.stack(
        [jnp.stack([0.0, 0.0, 0.0]),
         jnp.stack([0.0, 0.0, 0.0]),
         jnp.stack([0.0, 0.0, dt / T1**2 * E1])]
    )


def dr_dT2_epg(T2: float, dt: float) -> Array:
    """Compute the derivative of the EPG relaxation matrix with respect to T2.

    Parameters
    ----------

    T2 : float
        Transverse relaxation time in seconds.

    dt : float
        Time step in seconds.

    Returns
    -------

    dr_dT2 : array
        Derivative of the EPG relaxation matrix with respect to T2.
    """
    E2 = jnp.exp(-dt / T2)
    return jnp.stack(
        [jnp.stack([0.0, 0.0, 0.0]),
         jnp.stack([0.0, 0.0, 0.0]),
         jnp.stack([0.0, 0.0, dt / T2**2 * E2])]
    )



def b_epg(T1: float, dt: float) -> Array:
    """
    Calculate the EPG longitudinal relaxation term.

    Parameters
    ----------

    T1 : float
        Longitudinal relaxation time in seconds.

    dt : float
        Time step in seconds.

    Returns
    -------

    b : array
        EPG longitudinal relaxation term.
    """
    return 1 - jnp.exp(-dt / T1)


def epg_unit_grad(omega: Array) -> Array:
    """
    Apply the unit gradient operator to the EPG state.

    Parameters
    ----------

    omega : array
        EPG state.

    Returns
    -------

    omega_new : array
        EPG state after applying the unit gradient operator.
    """
    # expand omega by one column
    # first row: F+ perform right shift
    # second row: F- perform left shift
    # third row: Z0 perform no shift
    omega = jnp.concatenate([omega, jnp.zeros(shape=(3, 1))], axis=-1)
    omega_new = jnp.array(
        [
            [jnp.conjugate(omega[1, 0]), *omega[0, :-1]],
            [*omega[1, 1:], 0.0],
            [*omega[2, :]]
        ]
    )
    return omega_new


def grad_shift(omega: Array, dk: int) -> Array:
    """
    Apply gradient shift operator to the EPG state.

    Parameters
    ----------

    omega : array
        EPG state.

    dk : int
        Number of gradient twists.

    Returns
    -------

    omega_new : array
        EPG state after applying the gradient shift operator.


    Notes
    -----

    Function implementation heavily inspired by:
    https://github.com/imr-framework/epg
    """
    n = jnp.shape(omega)[1]
    if dk == 0:
        omega_new = omega
    else:
        if n > 1:
            # build one large state vector ranging from largest +Fz to smalles -Fz
            f = jnp.hstack((jnp.fliplr(omega[0, :][jnp.newaxis, :]), omega[1, 1:][jnp.newaxis, :], jnp.zeros((1, dk))))
            z = jnp.hstack((omega[2, :][jnp.newaxis, :], jnp.zeros((1, dk))))            
            fp = jnp.array([jnp.conjugate(f[0, n+dk-1]), *f[0, 0:n+dk-1][::-1]])
            fm = jnp.hstack((f[0, n+dk-1:][jnp.newaxis, :], jnp.zeros((1, dk))))
        else:
            # n = 1:  This happens if pulse sequence starts with nonzero transverse components
            #         and no RF pulse at t = 0 -- that is, the gradient happens first
            fp = jnp.hstack((np.zeros((1, dk)), jnp.array([[omega[0, 0]]])))
            fm = jnp.zeros((1, dk + 1))
            z = jnp.hstack((jnp.array([[omega[2,0]]]), np.zeros((1, dk))))
    omega_new = jnp.vstack((fp, fm, z))
    return omega_new


def inversion(inversion_efficiency: float) -> Array:
    """
    Compute the inversion operator.

    Parameters
    ----------

    inversion_efficiency : float
        Inversion efficiency.

    Returns
    -------

    inversion_op : array
        Inversion efficiency operator.
    """
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -inversion_efficiency]
        ]
    )


def preparation_T2_op(T2: float,  T2_preptime: float) -> Array:
    """
    Compute the T2 preparation operator.

    Parameters
    ----------

    T2 : float
        T2 relaxation time in seconds.

    T2_preptime : float
        T2 preparation time in seconds.

    Returns
    -------

    T2_op : array
        T2 preparation operator.
    """
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, - T2_preptime / T2]
        ]
    )


def dT2_preparation_dT2(T2: float, T2_preptime: float) -> Array:
    """
    Compute the derivative of the T2 preparation operator with respect to T2.

    Parameters
    ----------

    T2 : float
        T2 relaxation time in seconds.

    T2_preptime : float
        T2 preparation time in seconds.

    Returns
    -------

    dT2_preparation_dT2 : array
        Derivative of the T2 preparation operator with respect to T2.
    """
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, T2_preptime / T2**2 * jnp.exp(-T2_preptime / T2)]
        ]
    )



class PreparationType(Enum):
    NONE = 0
    INVERSION = 1
    T2_PREPARATION = 2


def compute_signal(
        T1: float,
        T2: float,
        M0: float,
        fa: Array,
        preparations: list[PreparationType],
        n_blocks: int,
        n_shots: int,
        TR: Array,
        phases: Array,
        TI: Array,
        T2_preptimes: Array,
        TE: float,
        inversion_efficiency: float = 1.0,
        delta_B1: float = 1.0,
) -> Array:
    """
    Compute the MR signal for a given set of parameters.

    Expected static arguments: {n_blocks, n_shots,}
    """
    fa = jnp.deg2rad(fa)
    phases = jnp.deg2rad(phases)

    r_TE = r_epg(T1, T2, TE)
    b_TE = b_epg(T1, TE)
    inv_op = inversion(inversion_efficiency)
    # Build initial state matrix.
    omega = jnp.array([[0.0], [0.0], [M0]])
    # Build array to store signal.
    signal = jnp.zeros((len(fa)*len(preparations)), dtype=jnp.complex64)

    for blk_idx in range(n_blocks):

        preparation = preparations[blk_idx]

        if preparation is PreparationType.INVERSION:
            omega = r_epg(T1, T2, TI[blk_idx]) @ inv_op @ omega
            omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_epg(T1, TI[blk_idx]))
        
        elif preparation is PreparationType.T2_PREPARATION:
            omega = preparation_T2_op(T2, T2_preptimes[blk_idx]) @ omega
        
        else:
            # is no preparation
            pass

        for shot_idx in range(n_shots):

            glob_idx = blk_idx * len(fa) + shot_idx

            # Excitation matrix
            q_n = q_epg(delta_B1 * fa[glob_idx], phases[glob_idx])

            # Update state matrix (excitation and relaxation during TE)
            omega = r_TE @ q_n @ omega
            omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_TE)

            # Compute and store MR signal
            signal = signal.at[glob_idx].set(omega[0, 0]) * jnp.exp(1j * phases[glob_idx])

            # Update state matrix (relaxation during TR - TE, gradient dephasing)
            omega = grad_shift(r_epg(T1, T2, TR[glob_idx] - TE) @ omega, dk=1)
            omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_epg(T1, TR[glob_idx] - TE))

    return signal