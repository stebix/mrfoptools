"""
High level interface to generate signal tensors for simulation of sequences.

@author: Jannik Stebani 2025
"""
import functools
from collections.abc import Callable
import warnings
import logging

import numpy as np

import mrfoptools.epg.core.numpy as npycore
import mrfoptools.epg.signal.numpy as npysig
import mrfoptools.epg.signal.numpy.preparations as npyprep

DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)

# TODO: This is awkward - Remove this when simple and coherent envrionment specification for joint
#       usage of numba, torch and jax is established. 
try:
    import numba as nb
except ImportError:
    msg: str = 'Numba not found. Using uncompiled numpy fallback implementation.'
    warnings.warn(msg)
    logger.warning(msg)    
    import mrfoptools.testtooling.nbmock as nb


Array = np.ndarray

USE_FASTMATH: bool = True

# JIT compile the functions used here -> numba does not recursively compile
compute_signal_optimized = nb.njit(
    npysig.compute_signal_numpynaive, fastmath=USE_FASTMATH
)
b_epg = nb.njit(npycore.b_epg, fastmath=USE_FASTMATH)
r_epg = nb.njit(npycore.r_epg, fastmath=USE_FASTMATH)
inversion = nb.njit(npycore.inversion, fastmath=USE_FASTMATH)

# should be jitted
prepare_inversion_omega = npyprep.prepare_inversion_omega


@nb.njit(fastmath=USE_FASTMATH)
def _simulate_fisp(
        fa: Array,
        TR: Array,
        T1: float,
        T2: float,
        M0: float,
        phases: Array,
        TE: float,
        TI: float,
        inversion_efficiency: float,
        max_states: int
) -> Array:
    """
    Simulate default FISP sequence
    """
    b_TE = b_epg(T1=T1, dt=TE)
    r_TE = r_epg(T1=T1, T2=T2, dt=TE)
    inv_op = inversion(inversion_efficiency)

    omega = prepare_inversion_omega(T1=T1,
                                    T2=T2,
                                    M0=M0,
                                    TI=TI,
                                    inversion_operator=inv_op,
                                    max_states=max_states)
    
    # Compute the signal for the current (T1, T2) pair
    s = compute_signal_optimized(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa,
        phases=phases,
        TR=TR,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )
    return s


def simulate_fisp(
        fa: Array,
        TR: Array,
        T1: Array,
        T2: Array,
        M0: float,
        phases: Array,
        TI: float,
        TE: float,
        max_states: int,
        inversion_efficiency: float = 1.0,
        delta_B1: float = 1.0
) -> Array:
    """
    Simulate the default FISP MRF sequence.
    Protocol: Inversion at the beginning and subsequent 
              application of the flip angle pattern

    Automatic vmapping across the T1 and T2 dimensions.
    """
    fa = fa * delta_B1

    # First partially apply all constant arguments
    # to allow correct vmapping
    n_tr = len(fa)
    n_species = len(T1)
    assert n_species == len(T2)

    signals = np.zeros(shape=(n_species, n_tr), dtype=np.complex64)

    for i in nb.prange(n_species):
        signals[i] = _simulate_fisp(
            T1=T1[i], T2=T2[i],
            fa=fa, TR=TR,
            M0=M0,
            phases=phases,
            TE=TE,
            TI=TI,
            inversion_efficiency=inversion_efficiency,
            max_states=max_states
        )

    return signals


def specialize_simulate_fisp(
    T1: Array,
    T2: Array,
    M0: float,
    phases: Array,
    TI: float,
    TE: float,
    max_states: int,
    inversion_efficiency: float = 1.0,
    delta_B1: float = 1.0
) -> Callable[[Array, Array], Array]:
    """
    Specialize ``simulate_fisp`` into a two-parameter function
    of the canonical optimization variables ``fa`` and ``TR``.
    """
    kwargs = {'T1' : T1, 'T2' : T2, 'M0' : M0, 'phases': phases,
              'TI' : TI, 'TE' : TE, 'max_states' : max_states,
              'inversion_efficiency' : inversion_efficiency,
              'delta_B1' : delta_B1
             }
    func = functools.partial(simulate_fisp, **kwargs)
    func.__doc__ = """
    Simulate FISP sequence with presets.
    
    Parameters
    ----------
    fa : jax.Array
        Flip angle train in radians.
        
    TR : jax.Array
        Repetition times.
    
    Returns
    -------
    
    signals : jax.Array
        Signals with shape ``(n_species, n_tr)``
    """
    return func