"""
Optimize the MRF sequence.

@author: Jannik Stebani 2025
"""
import jax
import jax.numpy as jnp
import numpy as np
import tqdm


from jax import Array
from jax.typing import ArrayLike

from mrfoptools.epg.signal import PreparationType

import mrfoptools.optimization.costfuncs as costfuncs

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
    grad = jax.grad(function, argnums=3)
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
    grad = jax.grad(function, argnums=3)
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