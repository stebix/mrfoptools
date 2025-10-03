"""
Signal computation functions for EPG simulations.

This module provides different implementations for the comoputation of the MR signal.
The rationale behind the different implementations is to experiment with different
state matrix representations (state matrix is often termed omega or Q) and
different jax/XLA compilation strategies.

State Matrix Representations
----------------------------

The state matrix is a 3xN matrix that contains the EPG states of the simulation.
During a unit gradient application, the state matrix is updated by shifting the
states of the first row to the right and the second row to the left.
Icplementations can be divided into two categories:

Dynamic:    The state matrix is expanded on every gradient application. This can
            incur a performance penalty due to the allocation of
            new memory on every gradient application.

Static: The state matrix is preallocated to contain all states of the simulation.
        This requires more memory upfront and requires to know the maximum number
        of states that will be generated during the simulation.

Watch out for hints in the function names that indicate the representation used.


Compilation Strategies
----------------------

JAX/XLA compilation can be used to speed up the computation of the MR signal.
However, the evolution of the state matrix can not be parallelized due to the
sequential nature of the EPG simulation.

Using native Python looping constructs and jax.jit functionality can lead to
large compilation times due to loop unrolling and thus compilation of large
programs (for long sequences with long flip angle trains).

This can be mitigated by either compiling only the inner fucntion of the state
evolution loop or by using more advanced lax.fori_loop constructs.

Watch out for hints in the function names that indicate the compilation strategy used.

@author: Jannik Stebani 2025
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

import mrfoptools.epg.core.jax as epgjax

Array = jax.Array | np.ndarray


JAX_JIT_COMPILE = True

if JAX_JIT_COMPILE:
    q = jax.jit(epgjax.q_epg)
    r = jax.jit(epgjax.r_epg)
    b = jax.jit(epgjax.b_epg)
    unit_grad_shift_static = jax.jit(epgjax.unit_grad_shift_static)
else:
    q = epgjax.q_epg
    r = epgjax.r_epg
    b = epgjax.b_epg
    unit_grad_shift_static = epgjax.unit_grad_shift_static


@jax.jit
def advance_state(
        omega: Array,
        alpha: float,
        phi: float,
        T1: float,
        T2: float,
        M0: float,
        dt: float,
        TE: float,
        r_TE: Array,
        b_TE: float
    ) -> tuple[Array, complex]:
    """
    Compute the change in EPG state and MR signal for a given flip angle and phase.

    Parameters
    ----------

    omega : Array
        Initial EPG state matrix (static version with maximum width)
        Expected to be of shape (3, N)

    alpha : float
        Flip angle in radians.

    phi : float
        Phase in radians.

    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    M0 : float
        Equilibrium magnetization.

    dt : float
        Time step in seconds.

    TE : float
        Echo time in seconds.

    r_TE : Array
        EPG relaxation matrix for TE.

    b_TE : float
        Longituinal relaxation term during TE.

    Returns
    -------

    (omega, signal) : tuple[Array, complex]
        Updated EPG state matrix and MR signal.
    """
    # Excitation matrix for flip angle alpha and phase phi of protocol
    q_n = q(alpha, phi)

    # Update state matrix (excitation and relaxation during TE)
    omega = r_TE @ q_n @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_TE)

    # Compute and store MR signal
    signal = omega[0, 0] * jnp.exp(1j * phi)

    # Update state matrix (relaxation during TR - TE, gradient dephasing)
    omega = unit_grad_shift_static(r(T1, T2, dt - TE) @ omega)
    omega = omega.at[2, 0].set(omega[2, 0] + M0 * b(T1, dt - TE))

    return (omega, signal)



def compute_signal_inner_compiled(
        omega: Array,
        T1: float,
        T2: float,
        M0: float,
        fa: Array,
        phases: Array,
        TR: Array,
        TE: float,
        b_TE: float,
        r_TE: Array) -> Array:
    """
    Compute MR signal using a pure Python loop with an inner compiled function
    using the staic state matrix representation.

    Parameters
    ----------

    omega : Array
        Initial EPG state matrix (static version with maximum width)
        Expected to be of shape (3, N)

    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    M0 : float
        Equilibrium magnetization.

    fa : Array
        Flip angles in radians.

    phases : Array
        FLip pulse phase offsets in radians.

    TR : Array
        Repetition times in seconds.
    
    TE : float
        Echo time in seconds.

    b_TE : float
        Precomputed longituinal relaxation term during TE.

    r_TE : Array
        Precomputed EPG relaxation matrix for TE.

    Returns
    -------

    signal : Array
        MR signal.
    """
    signal = jnp.zeros(len(fa), dtype=jnp.complex64)
    for i, (alpha, phi, tr) in enumerate(zip(fa, phases, TR)):
        omega, s = advance_state(omega, alpha, phi, T1, T2, M0, tr, TE, r_TE, b_TE)
        signal = signal.at[i].set(s)
    return signal




def compute_signal_jaxnaive(
        omega: Array,
        T1: float,
        T2: float,
        M0: float,
        fa: Array,
        phases: Array,
        TR: Array,
        TE: float,
        b_TE: float,
        r_TE: Array) -> Array:
    """
    Compute raw signal from initial static state matrix.

    Note: This function uses the naive Python looping strategy and may incur
          large compilation times for long sequences due to loop unrolling
          with `len(fa) == len(TR) == len(phases)` iterations.

    Parameters
    ----------

    omega : Array
        Initial EPG state matrix (static version with maximum width)
        Expected to be of shape (3, N)
    
    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    M0 : float
        Equilibrium magnetization.

    fa : Array
        Flip angles in radians.

    phases : Array
        FLip pulse phase offsets in radians.
    
    TR : Array
        Repetition times in seconds.

    TE : float
        Echo time in seconds.

    b_TE : float
        Longituinal relaxation term during TE.

    r_TE : Array
        EPG relaxation matrix for TE.
    
    Returns
    -------

    signal : Array
        MR signal.
    """
    signal = jnp.zeros(len(fa), dtype=jnp.complex64)
    for i, (alpha, phi, tr) in enumerate(zip(fa, phases, TR)):
        # Excitation matrix for flip angle alpha and phase phi of protocol
        q_n = q(alpha, phi)

        # Update state matrix (excitation and relaxation during TE)
        omega = r_TE @ q_n @ omega
        omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_TE)

        # Compute and store MR signal
        signal = signal.at[i].set(omega[0, 0] * jnp.exp(1j * phi))

        # Update state matrix (relaxation during TR - TE, gradient dephasing)
        omega = unit_grad_shift_static(r(T1, T2, tr - TE) @ omega)
        omega = omega.at[2, 0].set(omega[2, 0] + M0 * b(T1, tr - TE))

    return signal



def propagate_state_helper(
        i: int,
        carry: tuple[Array, Array],
        *,
        fa: Array,
        phases: Array,
        TR: Array,
        T1: float,
        T2: float,
        M0: float,
        TE: float,
        r_TE: Array,
        b_TE: float
    ) -> tuple[Array, complex]:
    """
    Helper function to compute state and signal for single time step
    inside a lax.fori_loop.

    Note: Function uses large static (3, N) state matrix representation.
          See usage of `unit_grad_shift_static`

    Parameters
    ----------

    i : int
        `jax.lax.fori_loop` loop index.

    carry : tuple[Array, Array]
        Tuple containing the current state matrix `omega` (shape (3, N))
        and the MR signal array (shape (len(fa))).

    Subsequent arguments are keyword-only and should be bound
    semi-statically using `functools.partial` to ensure the correct
    function signature for `jax.lax.fori_loop`.
    """

    omega, signal = carry

    # Excitation matrix for flip angle alpha and phase phi of protocol
    q_n = q(fa[i], phases[i])

    # Update state matrix (excitation and relaxation during TE)
    omega = r_TE @ q_n @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_TE)

    # Compute and store MR signal
    signal = signal.at[i].set(omega[0, 0] * jnp.exp(1j * phases[i]))

    # Update state matrix (relaxation during TR - TE, gradient dephasing)
    omega = unit_grad_shift_static(r(T1, T2, TR[i] - TE) @ omega)
    omega = omega.at[2, 0].set(omega[2, 0] + M0 * b(T1, TR[i] - TE))

    return (omega, signal)


def compute_signal_optimized(
        omega: jax.Array,
        T1: float,
        T2: float,
        M0: float,
        fa: jax.Array,
        phases: jax.Array,
        TR: jax.Array,
        TE: float,
        b_TE: float,
        r_TE: jax.Array) -> Array:
    """
    Compute raw signal from initial state matrix.

    Note: This is currently the most optimized version of the signal computation
          using the `lax.fori_loop` construct and a static state matrix representation.

    For the most optimized version, the function should be compiled using `jax.jit`.
    Array arguments should be passed as `jnp.array` and not `np.array`.

    Parameters
    ----------

    omega : Array
        Initial EPG state matrix (static version with maximum width)
        Expected to be of shape (3, N)
    
    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    M0 : float
        Equilibrium magnetization.

    fa : Array
        Flip angles in radians.

    phases : Array
        FLip pulse phase offsets in radians.
    
    TR : Array
        Repetition times in seconds.

    TE : float
        Echo time in seconds.

    b_TE : float
        Longituinal relaxation term during TE.

    r_TE : Array
        EPG relaxation matrix for TE.
    
    Returns
    -------

    signal : Array
        MR signal.
    """
    signal = jnp.zeros(len(fa), dtype=jnp.complex64)

    kwargs = {
        'fa' : fa,
        'phases' : phases,
        'TR' : TR,
        'T1' : T1,
        'T2' : T2,
        'M0' : M0,
        'TE' : TE,
        'r_TE' : r_TE,
        'b_TE' : b_TE
    }

    propagate = functools.partial(propagate_state_helper, **kwargs)
    # propagate = jax.jit(propagate)
    init_val = (omega, signal)
    lower = 0
    upper = len(fa)
    omega, signal = jax.lax.fori_loop(lower, upper, propagate, init_val)

    return signal


from typing import NamedTuple


class StateSignalVector(NamedTuple):
    """
    Named tuple to hold the state matrix and the signal vector.
    This is used to return both the state matrix and the signal vector
    from the `compute_signal_and_state` function.
    """
    state: jax.Array
    signal: jax.Array


def compute_signal_and_state(
    omega: jax.Array,
    T1: float,
    T2: float,
    M0: float,
    fa: jax.Array,
    phases: jax.Array,
    TR: jax.Array,
    TE: float,
    b_TE: float,
    r_TE: jax.Array
) -> StateSignalVector:
    """
    Compute raw signal from initial state matrix.

    Note: This is currently the most optimized version of the signal computation
          using the `lax.fori_loop` construct and a static state matrix representation.

    For the most optimized version, the function should be compiled using `jax.jit`.
    Array arguments should be passed as `jnp.array` and not `np.array`.

    Parameters
    ----------

    omega : Array
        Initial EPG state matrix (static version with maximum width)
        Expected to be of shape (3, N)
    
    T1 : float
        Longitudinal relaxation time in seconds.

    T2 : float
        Transverse relaxation time in seconds.

    M0 : float
        Equilibrium magnetization.

    fa : Array
        Flip angles in radians.

    phases : Array
        FLip pulse phase offsets in radians.
    
    TR : Array
        Repetition times in seconds.

    TE : float
        Echo time in seconds.

    b_TE : float
        Longituinal relaxation term during TE.

    r_TE : Array
        EPG relaxation matrix for TE.
    
    Returns
    -------

    signal : Array
        MR signal.
    """
    signal = jnp.zeros(len(fa), dtype=jnp.complex64)

    kwargs = {
        'fa' : fa,
        'phases' : phases,
        'TR' : TR,
        'T1' : T1,
        'T2' : T2,
        'M0' : M0,
        'TE' : TE,
        'r_TE' : r_TE,
        'b_TE' : b_TE
    }

    propagate = functools.partial(propagate_state_helper, **kwargs)
    # propagate = jax.jit(propagate)
    init_val = (omega, signal)
    lower = 0
    upper = len(fa)
    omega, signal = jax.lax.fori_loop(lower, upper, propagate, init_val)

    return StateSignalVector(state=omega, signal=signal)



'''
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
            signal = signal.at[glob_idx].set(omega[0, 0] * jnp.exp(1j * phases[glob_idx]))

            # Update state matrix (relaxation during TR - TE, gradient dephasing)
            omega = epg.unit_grad_shift_allocating(r_epg(T1, T2, TR[glob_idx] - TE) @ omega)
            omega = omega.at[2, 0].set(omega[2, 0] + M0 * b_epg(T1, TR[glob_idx] - TE))

    return signal

'''