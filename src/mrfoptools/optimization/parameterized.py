"""
Optimize parameterized flip angle trains.
"""
import functools
import jax
import jax.numpy as jnp

import optax

import tqdm
import tqdm.auto

from mrfoptools.optimization.costfuncs import orthogonality_criterion
import mrfoptools.epg.signal.parameterized as sigparam 
import mrfoptools.epg.core as epg
import mrfoptools.epg.signal.signal as epgsig


def optimize(
        T1: jax.Array,
        T2: jax.Array,
        M0: float,
        step_size: float,
        max_iterations: float,
        initial_parameters: jax.Array,
        TR: jax.Array,
        phases: jax.Array,
        TI: float,
        TE: float,
        inversion_efficiency: float = 1.0,
        lower: float = 5.0,
        upper: float = 75.0
    ) -> jax.Array:
    """
    Optimize parameterized flip angle trains.
    """
    initial_parameters = initial_parameters.copy()
    MAX_STATES: int = 1000

    # initial_parameters = jnp.deg2rad(initial_parameters)

    lower = jnp.deg2rad(lower)
    upper = jnp.deg2rad(upper)


    def forward(T1: float, T2: float,
                initial_parameters: jax.Array,
                M0: float,
                expander: callable,
                TR: jax.Array,
                phases: jax.Array,
                TI: float,
                TE: float,
                inversion_efficiency: float):

        b_TE = epg.b_epg(T1=T1, dt=TE)
        r_TE = epg.r_epg(T1=T1, T2=T2, dt=TE)
        inv_op = epg.inversion(inversion_efficiency)
        omega = epgsig.prepare_inversion_omega(T1=T1, T2=T2,
                                               M0=M0,
                                               TI=TI,
                                               inversion_operator=inv_op,
                                               max_states=MAX_STATES)
        return sigparam.compute_signal(
            omega, T1, T2, M0, initial_parameters, expander, phases, TR, TE, b_TE, r_TE
        )
    
    forward = functools.partial(
        forward,
        M0=M0,
        expander=sigparam.house_of_nicolouse,
        TR=TR, phases=phases, TI=TI, TE=TE,
        inversion_efficiency=inversion_efficiency
    )

    forward = jax.vmap(forward, in_axes=(0, 0, None))


    def iteration(T1, T2, parameters):
        signals = forward(T1, T2, parameters)
        return orthogonality_criterion(signals)
    
    iteration = jax.jit(iteration)
    
    value_and_grad = jax.value_and_grad(iteration, argnums=2)

    losshist = []
    paramhist = [initial_parameters]

    p = initial_parameters

    optimizer = optax.adam(learning_rate=step_size)
    optimizer_state = optimizer.init(initial_parameters)
    
    for _ in tqdm.auto.trange(max_iterations):
        loss, gradient = value_and_grad(T1, T2, p)

        updates, optimizer_state = optimizer.update(gradient, optimizer_state, p)
        p = optax.apply_updates(p, updates)

        # p = p - step_size * gradient

        p = optax.projections.projection_box(p, lower=lower, upper=upper)

        paramhist.append(p)
        losshist.append(loss)

    return (paramhist, losshist)




