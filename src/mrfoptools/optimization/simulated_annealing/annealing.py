"""
Actual optimization algorithm for simulated annealing.

@Author: Jannik Stebani 2025
"""
from collections.abc import Mapping, Callable
from enum import Enum

import jax
import jax.numpy as jnp
from tqdm.autonotebook import tqdm, trange

from mrfoptools.optimization.simulated_annealing.calibration import VariableType
from mrfoptools.parameterization.parameterization import ControlPoints

from mrfoptools.parameterization.parameterization import initialize_parameterization
from mrfoptools.parameterization.expansion import expand

from mrfoptools.optimization.simulated_annealing.perturbations import legacy_perturb

from mrfoptools.epg.sequences.jax.fisp import specialize_simulate_fisp
from mrfoptools.optimization.costfuncs import minimum_average_criterion
from mrfoptools.optimization.simulated_annealing.cooling import exponential_cooling



def metropolis_probability(cost: float, cost_prev: float, temp: float) -> jax.Array:
    """
    Acceptance probability for the Metropolis criterion in simulated annealing.
    """
    return jnp.min(jnp.array([1, jnp.exp(-(cost - cost_prev) / temp)]))


class ParameterType(Enum):
    """Optimization variable selection."""
    FA = 'fa'
    TR = 'tr'


class ParameterAxis(Enum):
    X = 'x'
    Y = 'y'


def rebuild_scale_factors(
        scale_factors: Mapping[VariableType, float]
    ) -> dict[tuple[ParameterType, ParameterAxis], float]:
    return {
        (ParameterType.FA, ParameterAxis.X): scale_factors[VariableType.FA_XCOORDS],
        (ParameterType.FA, ParameterAxis.Y): scale_factors[VariableType.FA_YCOORDS],
        (ParameterType.TR, ParameterAxis.X): scale_factors[VariableType.TR_XCOORDS],
        (ParameterType.TR, ParameterAxis.Y): scale_factors[VariableType.TR_YCOORDS],
    }

def rebuild_absbounds(
    absbounds: Mapping[VariableType, tuple[float, float]]
) -> dict[tuple[ParameterType, ParameterAxis], tuple[float, float]]:
    return {
        (ParameterType.FA, ParameterAxis.X): absbounds[VariableType.FA_XCOORDS],
        (ParameterType.FA, ParameterAxis.Y): absbounds[VariableType.FA_YCOORDS],
        (ParameterType.TR, ParameterAxis.X): absbounds[VariableType.TR_XCOORDS],
        (ParameterType.TR, ParameterAxis.Y): absbounds[VariableType.TR_YCOORDS],
    }


def generate_cost_function(
    simulate: Callable[[jax.Array, jax.Array], jax.Array],
    weights: tuple[float, float]
) -> Callable:
    
    w_time, w_signal = weights
    
    def cost_function(fa: jax.Array, TR: jax.Array) -> jax.Array:
        signals = simulate(fa, TR)
        return w_time * jnp.sqrt(jnp.sum(TR)) + w_signal / minimum_average_criterion(signals)
    
    return cost_function



def autoexpand_arguments(
    function: Callable[[jax.Array, jax.Array], jax.Array],
    nreq: int,
    *,
    fa_bounds: tuple[float, float] = (jnp.deg2rad(1), jnp.deg2rad(90)),
    tr_bounds: tuple[float, float] = (5.0, 1250.0)
) -> Callable[[ControlPoints, ControlPoints], jax.Array]:
    """
    Wrapper function to automatically expand control points arguments
    to full patterns before passing them to the function.
    Function are typically forward simulation functions or cost functions
    that

    Parameters
    ----------
    cost_function : Callable[[jax.Array, jax.Array], jax.Array]
        Cost function ingesting two arguments: the full-sized FA
        and TR patterns.
    nreq : int
        Number of requested evaluation points for the expansion,
        i.e. the length of the full-sized patterns.

    fa_bounds : tuple[float, float], optional
        Bounds/limits for the expanded FA pattern. Values
        produced by the spline expansion exceeding these bounds
        are clipped. Default is (jnp.deg2rad(1), jnp.deg2rad(90)).

    tr_bounds : tuple[float, float], optional
        Bounds/limits for the expanded TR pattern. Values
        produced by the spline expansion exceeding these bounds
        are clipped. Default is (5.0, 1250.0).

    Returns
    -------
    Callable[[ControlPoints, ControlPoints], jax.Array]
        Wrapped function that automatically expands the control points
        before passing them to the original function.
    """
    def wrapped_function(fa: ControlPoints, tr: ControlPoints) -> jax.Array:
        fa_expanded = expand(xc=fa.x, yc=fa.y, nreq=nreq, bounds=fa_bounds)
        tr_expanded = expand(xc=tr.x, yc=tr.y, nreq=nreq, bounds=tr_bounds)
        return function(fa_expanded, tr_expanded)
    
    return wrapped_function




def adaptive_non_isotropic(
    scale_factors: Mapping[VariableType, float],
    cost_function: Callable[[jax.Array, jax.Array], jax.Array],
    fa, tr
) -> dict[VariableType, ControlPoints]:
    

    absbounds: dict[VariableType, tuple[float, float]] = {
        VariableType.FA_XCOORDS : (0.0, 1000.0),
        VariableType.FA_YCOORDS : (jnp.deg2rad(1.0), jnp.deg2rad(90.0)),
        VariableType.TR_XCOORDS : (0.0, 1000.0),
        VariableType.TR_YCOORDS : (5.0, 1250.0)
    }
    
    temp_initial: float = 1.0
    n_iterations: int = 10000

    n_controlpoints: int = 10

    R: float = 0.00001
    r: float = R ** (1 / n_iterations) # noqa F841

    key = jax.random.key(1215615)

    ptype_controlpoints_pairs = (
        (ParameterType.FA, fa),
        (ParameterType.TR, tr)
    )

    _scale_factors = rebuild_scale_factors(scale_factors)
    _absbounds = rebuild_absbounds(absbounds)

    # intitial cost
    cost = cost_function(fa, tr)

    total = n_iterations * len(ptype_controlpoints_pairs) * 2 * n_controlpoints

    pbar = tqdm(total=total, desc='total anneal progress')

    costhist = []
    controlhist = []

    for iter in trange(n_iterations, desc='temperature schedule'):

        temp = exponential_cooling(temp_initial, 0.5, iter)

        for ptype, controlpoints in ptype_controlpoints_pairs:

            paxis_parameter_pairs = (
                (ParameterAxis.X, controlpoints.x),
                (ParameterAxis.Y, controlpoints.y)
            )

            for paxis, parameters in paxis_parameter_pairs:
                
                if paxis == ParameterAxis.X:
                    continue

                relscale = _scale_factors[(ptype, paxis)]
                bounds = _absbounds[(ptype, paxis)]

                if paxis == ParameterAxis.Y:
                    indices = range(n_controlpoints)
                else:
                    # do not modify x coordinates at the edges
                    # extrapolation with splines can be problematic
                    indices = range(1, n_controlpoints-1)

                for index in indices:

                    key, subkey_p, subkey_a = jax.random.split(key, num=3)

                    perturbed_parameters = legacy_perturb(index, parameters, bounds, temp/relscale, subkey_p)
                    
                    # rebuild control points
                    perturbed_controlpoints = ControlPoints(
                        x=perturbed_parameters if paxis == ParameterAxis.X else controlpoints.x,
                        y=perturbed_parameters if paxis == ParameterAxis.Y else controlpoints.y,
                        edge_mode=controlpoints.edge_mode
                    )
                    
                    fa_candidate = perturbed_controlpoints if ptype == ParameterType.FA else fa
                    tr_candidate = perturbed_controlpoints if ptype == ParameterType.TR else tr

                    # TODO: cost function wrapper with expansion to full fa and tr pattern
                    # handeled by cost_function
                    candidate_cost = cost_function(fa_candidate, tr_candidate)

                    if jnp.isnan(candidate_cost):
                        print(fa_candidate)
                        print(tr_candidate)
                        

                    acc_prob = metropolis_probability(candidate_cost, cost, temp)

                    pbar.update()

                    if jax.random.uniform(subkey_a) < acc_prob:
                        fa = fa_candidate
                        tr = tr_candidate
                        cost = candidate_cost
                        pbar.set_postfix_str(f'accept! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')
                    else:
                        pbar.set_postfix_str(f'reject! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')
                    

                    ptype_controlpoints_pairs = (
                       (ParameterType.FA, fa),
                       (ParameterType.TR, tr)
                    )

                    costhist.append(cost)
                    controlhist.append((fa, tr))

                    if temp < 1e-6:
                        return costhist, controlhist



def main():

    INIT_FA = 50
    INIT_TR = 12
    NR = 1000
    M0 = 1.0
    TE = 1
    TI = 20

    max_states = 1000
    inversion_efficiency = 1.0

    min_fa = jnp.deg2rad(1) # noqa F841
    max_fa = jnp.deg2rad(90) # noqa F841

    fa = jnp.deg2rad(jnp.full(fill_value=INIT_FA, shape=NR, dtype=jnp.float32))
    phases = jnp.full_like(fa, fill_value=jnp.pi/2)
    tr = jnp.full(fill_value=INIT_TR, shape=NR, dtype=jnp.float32)

    T1 = jnp.array([1500, 2000, 2500, 3000])
    T2 = jnp.array([500, 500, 600, 700])

    spec_kwargs = {
        'T1' : T1,
        'T2' : T2,
        'M0' : M0,
        'phases' : phases,
        'TI' : TI,
        'TE' : TE,
        'max_states' : max_states,
        'inversion_efficiency' : inversion_efficiency
    }

    simfisp = jax.jit(specialize_simulate_fisp(**spec_kwargs))

    cost_func = generate_cost_function(simfisp, (1.0, 1.0))
    cost_func = autoexpand_arguments(cost_func, NR)


    key = jax.random.key(1215615)
    fa_init_key, tr_init_key = jax.random.split(key, num=2)

    n_controlpoints = 10

    fa_con_param_kwargs = {
        'n_controlpoints' : n_controlpoints,
        'x_scale' : 1.0,
        'y_scale' : 0.25,
        'y_offset' : jnp.deg2rad(49),
        'extent' : NR,
        'key' : fa_init_key
    }
    fa = initialize_parameterization(**fa_con_param_kwargs)

    tr_con_param_kwargs = {
        'n_controlpoints' : n_controlpoints,
        'x_scale' : 1.0,
        'y_scale' : 1.0,
        'y_offset' : 12,
        'extent' : NR,
        'key' : tr_init_key
    }
    tr = initialize_parameterization(**tr_con_param_kwargs)


    scale_factors = {
        VariableType.FA_XCOORDS: 1e-3 * 10,
        VariableType.FA_YCOORDS: 2/jnp.pi * 0.25,
        VariableType.TR_XCOORDS: 1e-3 * 10,
        VariableType.TR_YCOORDS: 1/7 * 2,
    }

    print(fa)
    print(tr)



    adaptive_non_isotropic(scale_factors, cost_func, fa, tr)

if __name__ == "__main__":
    main()