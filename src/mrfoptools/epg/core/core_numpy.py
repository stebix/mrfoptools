"""
Implements core EPG simulation tooling and functions.

Partial jax port of Tom Grieslers pytorch EPG code.

@Author: Jannik Stebani 2025
"""
from types import ModuleType

import numpy as np

Array = np.ndarray
np: ModuleType = np


# port the q_epg function from the signalmodel_epg.py file to JAX
def q_epg(alpha: float, phi: float = np.pi / 2) -> Array:
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
    # q = np.zeros()
    c1 = np.array(
        (np.cos(alpha / 2) ** 2,
         np.exp(2 * 1j * phi) * np.sin(alpha / 2) ** 2,
         -1j * np.exp(1j*phi) * np.sin(alpha))
    )
    c2 = np.array(
        [np.exp(-2 * 1j * phi) * np.sin(alpha / 2) ** 2, 
         np.cos(alpha / 2) ** 2, 
         1j * np.exp(-1j * phi) * np.sin(alpha)]
    )
    c3 = np.array(
        [-1j/2 * np.exp(-1j * phi) * np.sin(alpha), 
          1j/2 * np.exp(1j * phi) * np.sin(alpha), 
         np.cos(alpha)]
    )
    return np.conjugate(np.stack((c1, c2, c3)))



def q_alt(alpha: float, phi: float = np.pi / 2) -> Array:
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
    q = np.zeros(shape=(3, 3), dtype=np.complex64)
    q[0, :] = ((
         np.cos(alpha / 2) ** 2,
         np.exp(2 * 1j * phi) * np.sin(alpha / 2) ** 2,
         -1j * np.exp(1j*phi) * np.sin(alpha)
    ))
    q[1, :] = ((
         np.exp(-2 * 1j * phi) * np.sin(alpha / 2) ** 2, 
         np.cos(alpha / 2) ** 2, 
         1j * np.exp(-1j * phi) * np.sin(alpha)
    ))
    q[2, :] = np.array((
         -1j/2 * np.exp(-1j * phi) * np.sin(alpha), 
         1j/2 * np.exp(1j * phi) * np.sin(alpha), 
         np.cos(alpha)
    ))
    return np.conjugate(q)





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
    E1 = np.exp(-dt / T1)
    E2 = np.exp(-dt / T2)
    return np.array(
        ((E2, 0.0, 0.0),
         (0.0, E2, 0.0),
         (0.0, 0.0, E1))
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
    E1 = np.exp(-dt / T1)
    return np.stack(
        [np.stack([0.0, 0.0, 0.0]),
         np.stack([0.0, 0.0, 0.0]),
         np.stack([0.0, 0.0, dt / T1**2 * E1])]
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
    E2 = np.exp(-dt / T2)
    return np.stack(
        [np.stack([0.0, 0.0, 0.0]),
         np.stack([0.0, 0.0, 0.0]),
         np.stack([0.0, 0.0, dt / T2**2 * E2])]
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
    return 1 - np.exp(-dt / T1)


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
    omega = np.concatenate([omega, np.zeros(shape=(3, 1))], axis=-1)
    omega_new = np.array(
        [
            [np.conjugate(omega[1, 0]), *omega[0, :-1]],
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
    n = np.shape(omega)[1]
    if dk == 0:
        omega_new = omega
    else:
        if n > 1:
            # build one large state vector ranging from largest +Fz to smalles -Fz
            f = np.hstack((np.fliplr(omega[0, :][np.newaxis, :]), omega[1, 1:][np.newaxis, :], np.zeros((1, dk))))
            z = np.hstack((omega[2, :][np.newaxis, :], np.zeros((1, dk))))            
            fp = np.array([np.conjugate(f[0, n+dk-1]), *f[0, 0:n+dk-1][::-1]])
            fm = np.hstack((f[0, n+dk-1:][np.newaxis, :], np.zeros((1, dk))))
        else:
            # n = 1:  This happens if pulse sequence starts with nonzero transverse components
            #         and no RF pulse at t = 0 -- that is, the gradient happens first
            fp = np.hstack((np.zeros((1, dk)), np.array([[omega[0, 0]]])))
            fm = np.zeros((1, dk + 1))
            z = np.hstack((np.array([[omega[2,0]]]), np.zeros((1, dk))))
    omega_new = np.vstack((fp, fm, z))
    return omega_new


def unit_grad_shift_allocating(omega: Array) -> Array:
    """
    Apply an unit gradient shift to the EPG state.

    This is the allocating version that expands the state matrix on every
    gradient application.
    
    Parameters
    ----------

    omega : Array
        EPG state matrix of shape (3, n).

    Returns
    -------

    omega_new : Array
        EPG state matrix after applying the unit gradient shift.
        New shape is (3, n+1).
    """
    omega = np.hstack((omega, np.zeros((3, 1), dtype=omega.dtype)))
    omega[0, 1:] = omega[0, :-1]
    omega[1, :-1] = omega[1, 1:]
    omega[1, -1] = 0.0
    omega[0, 0] = np.conjugate(omega[1, 0])
    return omega


def unit_grad_shift_static(omega: Array) -> Array:
    """
    Apply an unit gradient shift to the EPG state.

    This is the static version that does not expand the state matrix on every
    gradient application but expects a large matrix preallocated to be able to contain
    all EPG states of the simulation.

    Parameters
    ----------

    omega : Array
        EPG state matrix of shape (3, N).

    Returns
    -------

    omega : Array
        EPG state matrix after applying the unit gradient shift.
        Shape is (3, N).
    """
    omega[0, 1:] = omega[0, :-1]
    omega[1, -1] = 0.0
    omega[1, :-1] = omega[1, 1:]
    omega[0, 0] = np.conjugate(omega[1, 0])
    return omega



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
    return np.array(
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
    return np.array(
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
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, T2_preptime / T2**2 * np.exp(-T2_preptime / T2)]
        ]
    )
