"""
WIP buidling blocks for more dynamic assembly of optimization problems
inside of other scripts or notebooks.

@Author: Jannik Stebani 2025
"""
import functools
from collections.abc import Callable

import jax

import mrfoptools.epg.core as epg
from mrfoptools.epg.signal.signal import (compute_signal_optimized,
                                          prepare_inversion_omega)

def forward(
        T1: float,
        T2: float,
        fa: jax.Array,
        TR: jax.Array,
        M0: float,
        phases: jax.Array,
        TE: float,
        TI: float,
        inversion_efficiency: float,
        max_states: int
    ) -> jax.Array:
    """
    Forward model for basic FISP sequence with initial inversion.
    """        
    b_TE = epg.b_epg(T1=T1, dt=TE)
    r_TE = epg.r_epg(T1=T1, T2=T2, dt=TE)
    inv_op = epg.inversion(inversion_efficiency)
    omega = prepare_inversion_omega(T1=T1, T2=T2,
                                    M0=M0,
                                    TI=TI,
                                    inversion_operator=inv_op,
                                    max_states=max_states)
    return compute_signal_optimized(
        omega, T1, T2, M0, fa, phases, TR, TE, b_TE, r_TE
    )


def build_specialized_forward(
        TR: jax.Array,
        M0: float,
        phases: jax.Array,
        TE: float,
        TI: float,
        inversion_efficiency: float,
        max_states: int,
        func: Callable = forward
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Convenience function to specialize the forwards model function
    for a given set of parameters that are usually constant for a given
    optimization problem.
    Specialization entails fixing all quasi-constant parameters and
    vectorizing the function over the relaxometric parameters.

    Dynamic are:
     - relaxometric parameters T1 and T2 : these are vmapped over
     - flip angles : this is the optimization variable
    """
    # Fix all pseudo-constant parameters
    forward = functools.partial(func,
                                TR=TR, M0=M0, phases=phases, TE=TE,
                                TI=TI, inversion_efficiency=inversion_efficiency,
                                max_states=max_states)
    # Vectorize over T1 and T2 such that we can pass them as arrays
    forward = jax.vmap(forward, in_axes=(0, 0, None))
    return forward

