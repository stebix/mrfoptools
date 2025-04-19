import datetime
import warnings

from collections.abc import Callable, Sequence
from typing import Any
from functools import partial
from copy import deepcopy

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


def precompute_total_iterations(
    variables: Sequence[Variable],
    max_iter: int
) -> int:
    """
    Precompute the total number of iterations for the annealing process.
    This is the product of the number of variables, the number of indices
    and the number of iterations.
    """
    pcount = np.sum(np.fromiter((len(p) for p in get_parameters(variables)), dtype=int))
    return max_iter * len(variables) * pcount


def precheck_variables_1D(
    variables: Sequence[Variable]
) -> None:
    for variable in variables:
        p = variable.parameters
        if p.ndim != 1:
            warnings.warn(
                f'Variable {variable.designation} has {p.ndim} '
                f'dimensions, expected 1D array.'
            )
    


def anneal_modern(
    key: jax.Array,
    variables: Sequence[Variable],
    cost_func: Callable[[*tuple[jax.Array, ...]], jax.Array],
    max_iter: int,
    temp_initial: float,
    cooling_func: Callable[[float, float, int], float],
) -> Any:
    pass

    if not are_ordered_correctly(variables):
        raise ValueError('ordering mismatch for variables')
    
    precheck_variables_1D(variables)
    total = precompute_total_iterations(variables, max_iter)
    pbar = tqdm.tqdm(desc='total annealing progress', unit='it', total=total)    

    cost = cost_func(*get_parameters(variables))
    temp = temp_initial
    r = 0.01 ** (1 / max_iter) # noqa: F841

    cooling_scheme = partial(cooling_func, r=r)

    accept_counter = 0
    reject_counter = 0
    temp_history: list[float] = []

    cost_history: list[float] = []
    candidate_cost_history: list[float] = []
    variables_history: list[list[Variable]] = []

    for n_iter in range(max_iter):

        for varindex, variable in enumerate(variables):

            for paramindex in variable.indices:

                perturbed_parameters = perturb_gaussian(
                    index=paramindex,
                    parameters=variable.parameters,
                    bounds=variable.bounds,
                    relscale=temp/variable.relscale,
                    key=key
                )

                if variable.sort:
                    perturbed_parameters = jnp.sort(perturbed_parameters)
                # correct ordering is paramount here since we stuff this directly in the 
                # cost function that internally expand the control points into the full
                # specifications via spline interpolation
                args = tuple(
                    variables[j].parameters
                    if j != varindex else perturbed_parameters
                    for j in range(len(variables))
                )
                candidate_cost = cost_func(*args)

                candidate_cost_history.append(candidate_cost)
                cost_history.append(cost)
                variables_history.append(
                    deepcopy(variables)
                )

                acc_prob = metropolis_probability(candidate_cost, cost, temp)

                if jax.random.uniform(key) < acc_prob:
                    # accept the new parameters
                    variables[varindex] = variable.with_new_parameters(perturbed_parameters)
                    cost = candidate_cost
                    # optimize this away possibly
                    pbar.set_postfix_str(f'accept! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')
                    accept_counter += 1
                else:
                    # reject the new parameters
                    pbar.set_postfix_str(f'reject! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')
                    reject_counter += 1

                pbar.update()

        temp_history.append(temp)
        temp = cooling_scheme(temp_init=temp_initial, n_iter=n_iter)

    results = {
        'variables': variables,
        'temp_history': temp_history,
        'cost_history': cost_history,
        'candidate_cost_history': candidate_cost_history,
        'variables_history': variables_history,
        'accept_counter': accept_counter,
        'reject_counter': reject_counter,
    }
    return results


def anneal_abstract(
    key: jax.Array,
    variables: Sequence[Variable],
    cost_func: Callable[[*tuple[jax.Array, ...]], jax.Array],
    acceptance_func: Callable[[float, float, float], bool],

) -> None:
    pass



def main():
    INIT_FA = 50.0
    INIT_TR = 12.0
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
    tr_controlpoint_bounds = (10.0, 250.0)
    x_bounds = (0.0, 1000.0)
    
    fa_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=8.633,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.X),
        sort=True
    )
    tr_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=882.0545,
        designation=Designation(type=ParameterType.TR, axis=ParameterAxis.X),
        sort=True
    )
    fa_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=jnp.deg2rad(INIT_FA)),
        bounds=fa_controlpoint_bounds,
        relscale=1.8557,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.Y),
        sort=False
    )
    tr_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_TR),
        bounds=tr_controlpoint_bounds,
        relscale=195.0854,
        designation=Designation(type=ParameterType.TR, axis=ParameterAxis.Y),
        sort=False
    )

    @jax.jit
    def costfunc(
        fa_x, fa_y, tr_x, tr_y
    ):
        nreq = 1000
        w_time = 1.0
        w_signal = 1.0
        
        fa = expand(fa_x, fa_y, nreq=nreq, extrap=False, bounds=(jnp.deg2rad(1.0), jnp.deg2rad(90.0)))
        tr = expand(tr_x, tr_y, nreq=nreq, extrap=False, bounds=(6.0, 500.0))
        signals = simulate_fisp(fa, tr)
        return w_time * jnp.sqrt(jnp.sum(tr)) + w_signal / minimum_average_criterion(signals)
    

    results = anneal_modern(
        key=jax.random.key(seed=145383),
        variables=[fa_x, fa_y, tr_x, tr_y],
        cost_func=costfunc,
        max_iter=250,
        temp_initial=5.0,
        cooling_func=exponential_cooling
    )

    print('acceptance ratio:', results[1] / (results[1] + results[2]))
    print('final temperature:', results[0][-1])


if __name__ == '__main__':
    main()



