"""
Generate the MR with EPG but use specially parameterized flip angle trains
to reduce the dimensionality of the optimization problem.

@author: Jannik Stebani 2025
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp

from mrfoptools.epg.signal import compute_signal_optimized

def house_of_nicolouse(values: jax.Array,
                       segment_size: int = 100) -> jax.Array:
    result = []
    for v_start, v_stop in zip(values[:-1], values[1:]):
        result.append(jnp.linspace(v_start, v_stop, num=segment_size))
    return jnp.concatenate(result)


def compute_signal(
    omega: jax.Array,
    T1: float,
    T2: float,
    M0: float,
    fa_parameters: jax.Array,
    fa_generator: Callable[[jax.Array], jax.Array],
    phases: jax.Array,
    TR: jax.Array,
    TE: float,
    b_TE: float,
    r_TE: jax.Array) -> jax.Array:
    """
    Compute MR signal.
    Uses alternative code path by generating/expanding actual flip angle vector
    from parameters using the function `fa_generator`.
    """
    fa = fa_generator(fa_parameters)
    return compute_signal_optimized(
        omega, T1, T2, M0, fa, phases, TR, TE, b_TE, r_TE
    )