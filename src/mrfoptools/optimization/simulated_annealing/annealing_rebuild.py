import datetime

from collections.abc import Callable, Sequence
from typing import Any
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import tqdm.auto as tqdm

from mrfoptools.optimization.simulated_annealing.perturbations import perturb_gaussian
from mrfoptools.optimization.simulated_annealing.cooling import exponential_cooling
from mrfoptools.optimization.costfuncs import minimum_average_criterion

from mrfoptools.parameterization.expansion import expand

from mrfoptools.epg.sequences.jax.fisp import specialize_simulate_fisp

from mrfoptools.optimization.simulated_annealing.variables import (
    Variable, Designation, ParameterType, ParameterAxis,
    are_ordered_correctly,
    get_parameters
)


float32 = np.float32



def metropolis_probability(cost: float, cost_prev: float, temp: float) -> jax.Array:
    """
    Acceptance probability for the Metropolis criterion in simulated annealing.
    """
    return jnp.min(jnp.array([1, jnp.exp(-(cost - cost_prev) / temp)]))


def create_cost_function(
    simulate: Callable[[jax.Array, jax.Array], jax.Array],
    nreq : int,
    weights: tuple[float, float],
    *,
    fa_bounds: tuple[float32, float32] = (jnp.deg2rad(1.0), jnp.deg2rad(90.0)),
    tr_bounds: tuple[float32, float32] = (5, 250)
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Create the cost function that directly ingests the parameter arrays
    and expands towards full FA and TR patterns internally.

    Parameters
    ----------

    simulate : Callable
        Forward simulation function. Must be specialized to have
        signature `simulate(fa, tr)` and return the signals
        tensor of shape `(n_species, n_tr)`.

    nreq : int
        Number of requested points for the interpolation.
        This is the temporal dimension in the full pattern.

    weights : tuple[float, float]
        Weights for the time and signal cost terms.

    fa_bounds : tuple[float32, float32], optional
        Min-max bounds for the flip angles in radians.
        Values of the spline-interpolated pattern
        are clipped to these bounds.

    tr_bounds : tuple[float32, float32], optional
        Min-max bounds for the repetition times.
        Values of the spline-interpolated pattern
        are clipped to these bounds.
        Unit must be compatible to the other use
        time units in the simulation.
    """
    w_time, w_signal = weights

    def cost_function(fa: jax.Array, tr: jax.Array) -> jax.Array:
        signals = simulate(fa, tr)
        return w_time * jnp.sqrt(jnp.sum(tr)) + w_signal / minimum_average_criterion(signals)

    fa_expander = partial(expand, nreq=nreq, method='cubic', extrap=False, bounds=fa_bounds)
    tr_expander = partial(expand, nreq=nreq, method='cubic', extrap=False, bounds=tr_bounds)

    def _adaptor(
        fa_x: jax.Array,
        fa_y: jax.Array,
        tr_x: jax.Array,
        tr_y: jax.Array
    ) -> jax.Array:
        """
        Adapt the cost function to enable calling with separate
        control point parameter arrays.
        """
        fa = fa_expander(fa_x, fa_y)
        tr = tr_expander(tr_x, tr_y)
        return cost_function(fa, tr)
    
    return _adaptor



def anneal(
    fa_x: Variable,
    fa_y: Variable,
    tr_x: Variable,
    tr_y: Variable,
    simulate: Callable[[jax.Array, jax.Array], jax.Array],
) -> Any:
    
    key = 1

    variables: Sequence[Variable] = [fa_x, fa_y, tr_x, tr_y]

    if not are_ordered_correctly(variables):
        raise ValueError('ordering mismatch for variables')
    
    fa_interp_bounds: tuple[float32, float32] = (jnp.deg2rad(1.0), jnp.deg2rad(90.0))
    tr_interp_bounds: tuple[float32, float32] = (5.0, 500.0)

    cost_func = create_cost_function(
        simulate=None,
        nreq=1000,
        weights=(1.0, 1.0),
        fa_bounds=fa_interp_bounds,
        tr_bounds=tr_interp_bounds
    )

    cost = cost_func(*variables)

    # setup
    temp_initial: float = 1.0
    max_iter: int = 100
    n_controlpoints: int = 10
    R: float = 0.00001
    r: float = R ** (1 / max_iter) # noqa: F841

    key: jax.Array = jax.random.key(seed=1701) 

    cost = cost_func(*get_parameters(variables))

    total = max_iter * len(variables) * np.sum((len(p) for p in get_parameters(variables)))
    pbar = tqdm.tqdm(desc='total annealing progress', unit='it', total=total)    

    for n_iter in range(max_iter):

        temp = exponential_cooling(temp_initial, 0.5, n_iter)

        for i, variable in enumerate(variables):

            for index in variable.indices:
                
                # split the key for next iteration, perturbation and acceptance
                key, ptrb_key, acct_key = jax.random.split(key, num=3)

                perturbed_parameters = perturb_gaussian(
                    index=index,
                    parameters=variable.parameters,
                    bounds=variable.bounds,
                    relscale=variable.relscale,
                    key=ptrb_key
                )
                if variable.sort:
                    perturbed_parameters = jnp.sort(perturbed_parameters)
                # correct ordering is paramount here since we stuff this directly in the 
                # cost function that internally expand the control points into the full
                # specifications via spline interpolation
                args = tuple(
                    variables[j].parameters
                    if j != i else perturbed_parameters
                    for j in range(len(variables))
                )

                candidate_cost = cost_func(*args)
                acc_prob = metropolis_probability(candidate_cost, cost, temp)
                
                pbar.update()

                if jax.random.uniform(acct_key) < acc_prob:
                    # accept the new parameters
                    variables[i] = variable.with_new_parameters(perturbed_parameters)
                    cost = candidate_cost

                    # optimize this away possibly
                    pbar.set_postfix_str(f'accept! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')

                else:
                    # reject the new parameters
                    pbar.set_postfix_str(f'reject! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pbar.write(f'finished @ {now}')
    return variables              






def main():
    INIT_FA = 50
    INIT_TR = 12
    NR = 1000
    M0 = 1.0
    TE = 2.2
    TI = 20

    max_states = 750
    inversion_efficiency = 1.0


    fa = jnp.deg2rad(jnp.full(fill_value=INIT_FA, shape=NR, dtype=jnp.float32))
    tr = jnp.full(fill_value=INIT_TR, shape=NR, dtype=jnp.float32)
    phases = jnp.full_like(fa, fill_value=jnp.pi/2)

    T1 = jnp.array([1500, 2000, 2500, 3000])
    T2 = jnp.array([500, 500, 600, 700])

    simulate_fisp = specialize_simulate_fisp(
        T1=T1,
        T2=T2,
        M0=M0,
        phases=phases,
        TI=TI,
        TE=TE,
        max_states=max_states,
        inversion_efficiency=inversion_efficiency   
    )

    n_controlpoints: int = 10

    fa_controlpoint_bounds = (jnp.deg2rad(1.0), jnp.deg2rad(90.0))
    tr_controlpoint_bounds = (5.0, 500.0)
    x_bounds = (0.0, 1000.0)
    
    fa_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.X),
        sort=True
    )
    tr_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.X),
        sort=True
    )
    fa_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_FA),
        bounds=fa_controlpoint_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.Y),
        sort=False
    )
    tr_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_TR),
        bounds=tr_controlpoint_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.TR, axis=ParameterAxis.Y),
        sort=False
    )

    result = anneal()



