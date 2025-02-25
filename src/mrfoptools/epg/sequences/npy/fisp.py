"""
High level interface to generate signal tensors for simulation of sequences.

@author: Jannik Stebani 2025
"""
import functools
from collections.abc import Callable

import numpy as np

import mrfoptools.epg.core.numpy as epg
import mrfoptools.epg.signal.numpy as epgsig

Array = np.ndarray

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
    b_TE = epg.b_epg(T1=T1, dt=TE)
    r_TE = epg.r_epg(T1=T1, T2=T2, dt=TE)
    inv_op = epg.inversion(inversion_efficiency)

    omega = epgsig.prepare_inversion_omega(T1=T1,
                                           T2=T2,
                                           M0=M0,
                                           TI=TI,
                                           inversion_operator=inv_op,
                                           max_states=max_states)
    
    # Compute the signal for the current (T1, T2) pair
    s = epgsig.compute_signal_optimized(
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
    simulate_fisp_vmap = functools.partial(
        _simulate_fisp,
        fa,
        TR,
        M0=M0,
        phases=phases,
        TE=TE,
        TI=TI,
        inversion_efficiency=inversion_efficiency,
        max_states=max_states
    )
    simulate_fisp_vmap = jax.vmap(simulate_fisp_vmap, in_axes=(0, 0))
    simulate_fisp_vmap = jax.jit(simulate_fisp_vmap)

    return simulate_fisp_vmap(T1, T2)


def specialize_simulate_fisp(
    T1: jax.Array,
    T2: jax.Array,
    M0: float,
    phases: jax.Array,
    TI: float,
    TE: float,
    max_states: int,
    inversion_efficiency: float = 1.0,
    delta_B1: float = 1.0
) -> Callable[[jax.Array, jax.Array], jax.Array]:
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