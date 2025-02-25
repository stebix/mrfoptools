import os
import numpy as np
import numba

import mrfoptools.epg.core.core_numpy as npycore
from mrfoptools.epg.signal.preparations import PreparationType


USE_FASTMATH = os.environ.get('NUMBA_FASTMATH', '0') == '1'
NUMBA_JIT = os.environ.get('NUMBA_JIT', '0') == '1'


Array = np.ndarray


def prepare_equilibrium_omega(M0: float, max_states: int) -> Array:
    """
    Create large static EPG state matrix for equilibrium state.
    """
    dtype = np.complex64
    omega = np.hstack(
        (np.array([[0.0], [0.0], [M0]], dtype=dtype),
         np.zeros((3, max_states-1), dtype=dtype))
    )
    return omega


def prepare_inversion_omega(T1: float,
                            T2: float,
                            M0: float,
                            max_states: int,
                            inversion_operator: Array,
                            TI: float
    ) -> Array:
    """
    Generate large static initial EPG magnetization state matrix for inversion preparation.
    """
    omega = npycore.r_epg(T1, T2, TI) @ inversion_operator @ prepare_equilibrium_omega(M0, max_states)
    omega[2, 0] = omega[2, 0] + M0 * npycore.b_epg(T1, TI)
    return omega


def prepare_T2_omega(T2: float,
                     T2_prep_time: float,
                     M0: float,
                     max_states: int
    ) -> Array:
    """
    Generate large static initial EPG magnetization state matrix for T2 preparation.
    """
    omega = npycore.preparation_T2_op(T2, T2_prep_time) @ prepare_equilibrium_omega(M0, max_states)
    return omega


def prepare_omega(preparation: PreparationType,
                  max_states: int,
                  T1: float,
                  T2: float,
                  M0: float,
                  inversion_operator: Array,
                  TI: float,
                  T2_prep_time: float
    ) -> Array:
    """
    Generate large static initial EPG magnetization state matrix.

    Parameters
    ----------

    preparation : PreparationType
        Preparation type.

    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    M0 : float
        Equilibrium magnetization.

    max_states : int
        Maximum number of states to preallocate.
        Resulting static state matrix has shape (3, max_states)

    Returns
    -------

    omega : Array
        Initial EPG state matrix.
    """
    if preparation is PreparationType.NONE:
        return prepare_equilibrium_omega(M0, max_states)

    elif preparation is PreparationType.INVERSION:
        return  prepare_inversion_omega(T1, T2, M0, max_states, inversion_operator, TI)
    
    elif preparation is PreparationType.T2_PREPARATION:
        return prepare_T2_omega(T2, T2_prep_time, M0, max_states)
    
    else:
        msg = f'Invalid preparation type: {preparation}'
        raise ValueError(msg)


if NUMBA_JIT:
    q_epg = numba.njit(npycore.q_alt, fastmath=USE_FASTMATH)
    r_epg = numba.njit(npycore.r_epg, fastmath=USE_FASTMATH)
    b_epg = numba.njit(npycore.b_epg, fastmath=USE_FASTMATH)
    unit_grad_shift_static = numba.njit(npycore.unit_grad_shift_static, fastmath=USE_FASTMATH)
else:
    q_epg = npycore.q_alt
    r_epg = npycore.r_epg
    b_epg = npycore.b_epg
    unit_grad_shift_static = npycore.unit_grad_shift_static

def compute_signal_numpynaive(
    omega: Array,
    T1: float,
    T2: float,
    M0: float,
    fa: Array,
    phases: Array,
    TR: Array,
    TE: float,
    b_TE: float,
    r_TE: Array
) -> Array:

    signal = np.zeros(len(fa), dtype=np.complex64)

    for i, (alpha, phi, tr) in enumerate(zip(fa, phases, TR)):
        q = q_epg(alpha, phi)

        omega = r_TE @ q @ omega
        omega[2, 0] = omega[2, 0] + M0 * b_TE

        signal[i] = omega[0, 0] * np.exp(1j * phi)

        omega = unit_grad_shift_static(
            r_epg(T1, T2, tr - TE) @ omega
        )
        omega[2, 0] = omega[2, 0] + M0 * b_epg(T1, tr - TE)

    return signal
