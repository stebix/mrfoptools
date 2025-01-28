"""
Optimize the MRF sequence.

@author: Jannik Stebani 2025
"""
import functools
import jax
import jax.numpy as jnp
import tqdm
import optax

from jax import Array

import mrfoptools.epg.core as epg
import mrfoptools.optimization.costfuncs as costfuncs
from mrfoptools.epg.signal.signal import PreparationType, prepare_inversion_omega
from mrfoptools.epg.signal.signal import compute_signal_optimized

def optimize(
        T1: Array,
        T2: Array,
        M0: float,
        step_size: float,
        max_iterations: float,
        intial_fa: Array,
        TR: Array,
        phases: Array,
        preparation: PreparationType,
        TI: float,
        TE: float,
        inversion_efficiency: float = 1.0,
        delta_B1: float = 1.0
) -> Array:
    """
    Optimize with simple gradient descent.
    """
    # Initialize the flip angles
    fa = intial_fa.copy()
    # Initialize the cost function
    function = jax.jit(costfuncs.legacy_orthogonality_criterion,
                       static_argnames=('preparation', 'inversion_efficiency', 'delta_B1'))
    
    # Initialize the gradient
    # grad = jax.grad(function, argnums=3)
    grad_and_loss = jax.value_and_grad(function, argnums=3)

    fa_history = []
    loss_history = []

    # Perform the optimization

    for iteration in tqdm.trange(max_iterations):
        fa_history.append(fa)
        # Compute the gradient
        value, gradient = grad_and_loss(
            T1, T2, M0, fa, TR, phases, preparation, TI, TE, inversion_efficiency, delta_B1
        )
        # Update the flip angles
        fa = fa - step_size * gradient
        loss_history.append(value)

    return (fa_history, loss_history)


def optimize2(
        T1: Array,
        T2: Array,
        M0: float,
        step_size: float,
        max_iterations: float,
        intial_fa: Array,
        TR: Array,
        phases: Array,
        TI: float,
        TE: float,
        inversion_efficiency: float = 1.0,
) -> Array:
    """
    Optimize with simple gradient descent.
    """
    # Initialize the flip angles
    fa = intial_fa.copy()

    MAX_STATES: int = 1000

    lower = jnp.deg2rad(1.0)
    upper = jnp.deg2rad(90.0)
    

    def forward(T1, T2, fa, M0, phases, TR, TE, TI, inversion_efficiency):
        b_TE = epg.b_epg(T1=T1, dt=TE)
        r_TE = epg.r_epg(T1=T1, T2=T2, dt=TE)
        inv_op = epg.inversion(inversion_efficiency)
        omega = prepare_inversion_omega(T1=T1, T2=T2,
                                        M0=M0,
                                        TI=TI,
                                        inversion_operator=inv_op,
                                        max_states=MAX_STATES)
        return compute_signal_optimized(
            omega, T1, T2, M0, fa, phases, TR, TE, b_TE, r_TE
        )
    
    forward = functools.partial(forward,
                                M0=M0, phases=phases, TR=TR, TE=TE,
                                TI=TI, inversion_efficiency=inversion_efficiency)

    forward = jax.vmap(forward, in_axes=(0, 0, None))

    def step(T1, T2, fa):
        signals = forward(T1, T2, fa)

        #ortho_cost = costfuncs.orthogonality_criterion(signals)
        #signal_cost = - jnp.sum(jnp.linalg.norm(signals, ord=2, axis=1))

        mabs_cost = - jnp.mean(jnp.abs(signals))

        smoothness_cost = jnp.linalg.norm(fa[1:] - fa[:-1], ord=2)
        return 50 * mabs_cost +  smoothness_cost
    
    # Initialize the gradient
    value_and_grad = jax.jit(jax.value_and_grad(step, argnums=2))

    fa_history = []
    loss_history = []

    # Perform the optimization

    optimizer = optax.adam(learning_rate=step_size)
    optim_state = optimizer.init(fa)

    for iteration in tqdm.trange(max_iterations):
        fa_history.append(fa)
        # Compute the gradient
        value, gradient = value_and_grad(T1, T2, fa)

        update, optim_state = optimizer.update(gradient, optim_state, fa)
        fa = optax.apply_updates(fa, update)

        fa = optax.projections.projection_box(fa, lower=lower, upper=upper)
        
        loss_history.append(value)

    return (fa_history, loss_history)


def optimize3(
        T1: Array,
        T2: Array,
        M0: float,
        step_size: float,
        max_iterations: float,
        intial_fa: Array,
        TR: Array,
        phases: Array,
        TI: float,
        TE: float,
        inversion_efficiency: float,
        min_fa: float,
        max_fa: float
) -> dict[str, Array]:
    """
    Optimize with simple gradient descent.
    """
    pass