import os
import numpy as np
import numba

import mrfoptools.epg.core.numpy as npycore

USE_FASTMATH = os.environ.get('NUMBA_FASTMATH', '0') == '1'
NUMBA_JIT = os.environ.get('NUMBA_JIT', '0') == '1'

Array = np.ndarray


if True:
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
