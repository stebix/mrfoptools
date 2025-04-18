"""
Calibration tooling for simulated annealing optimization.

@Author: Jannik Stebani 2025
"""
import enum
import json
import datetime

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Hashable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import tqdm.auto as tqdm

from mrfoptools.parameterization.expansion import expand_fa, expand_tr, expand
from mrfoptools.parameterization.parameterization import (
    initialize_parameterization, ControlPoints)
from mrfoptools.optimization.simulated_annealing.perturbations import (
    PerturbedControlPoints,
    legacy_perturb,
    perturb_gaussian,
)

from mrfoptools.optimization.simulated_annealing.variables import (
    Designation, Variable,
    regenerate_randomized,
    regenerate_with_relscale,
    get_parameters
)
import mrfoptools.optimization.simulated_annealing.utils as utils

import line_profiler


@line_profiler.profile
def _sample_perturbation_costs(
    designation: Designation,
    variables: Sequence[Variable],
    sample_count: int,
    cost_func: Callable[[*tuple[jax.Array, ...]], jax.Array],
    key: jax.Array,
    *,
    pbar_kwargs: Mapping | None = None,
) -> jax.Array:
    """
    Compute a vector of `sample_count` cost differences generated
    by sequentially perturbing the designated variable from the list of variables.

    Note: It is the responsibility of the `cost_func` to process the 
          raw variable parameter arrays. This means usually expanding
          the control point coordinates to full patterns via interpolation.
          The order of the variables in the list must match the order
          of the cost function signature.
    """
    mapping: dict[Designation, Variable] = {
        variable.designation: variable
        for variable in variables
    }

    n_variables: int = len(variables)

    if designation not in mapping:
        raise ValueError(f'variable with designation {designation} not found')

    pbar_kwargs = pbar_kwargs or {}
    wrapped_sample_counter = tqdm.trange(sample_count, **pbar_kwargs)

    costs: list[jax.Array] = []
    # In addition to the keys for re-generating the variables 
    # we need a new key for the next iteration and two for the perturbation
    key_count = n_variables + 3

    for _ in wrapped_sample_counter:

        key, *genkeys, idxkey, prtbkey = jax.random.split(key, num=key_count)
        # random reinit all variables
        mapping = {
            variable.designation : regenerate_randomized(genkeys[i], variable)
            for i, variable in enumerate(variables)
        }

        cost_pre = cost_func(*get_parameters(mapping.values()))

        # perturb the variable that is selected via the designation
        variable = mapping[designation]
        index = jax.random.choice(idxkey, variable.indices)
        perturbed_parameters = perturb_gaussian(
            index=index,
            parameters=variable.parameters,
            bounds=variable.bounds,
            relscale=variable.relscale,
            key=prtbkey
        )
        if variable.sort:
            perturbed_parameters = jnp.sort(perturbed_parameters)

        mapping[designation] = variable.with_new_parameters(perturbed_parameters)

        # evaluate cost and add to cache
        cost_post = cost_func(*get_parameters(mapping.values()))
        costs.append(jnp.abs(cost_post - cost_pre))

    return jnp.array(costs)



def sample_perturbation_costs(
    key: jax.Array,
    variables: Sequence[Variable],
    sample_count: int,
    cost_func: Callable[[*tuple[jax.Array, ...]], jax.Array],
    *,
    pbar_kwargs: Mapping | None = None,
) -> Mapping[Designation, jax.Array]:
    """
    Compute the sample perturbation costs for all given variables.

    It is the responsibility of the `cost_func` to process the
    raw variable parameter arrays in the order corresponding to the
    variables.

    It is also the responsibility of the caller to ensure that the
    variables have the desired `relscale` for which the perturbation
    costs are sampled for every variable.
    """
    perturbation_costs: dict[Designation, jax.Array] = {}
    pbar_kwargs = pbar_kwargs or {}
    wrapped_variables = tqdm.tqdm(variables, **pbar_kwargs)
    for variable in wrapped_variables:
        designation = variable.designation
        wrapped_variables.set_postfix_str(
            f'current type={designation.type} axis={designation.axis}'
        )
        costs = _sample_perturbation_costs(
            designation=designation,
            variables=variables,
            sample_count=sample_count,
            cost_func=cost_func,
            key=key,
            pbar_kwargs={'leave' : False}
        )
        perturbation_costs[designation] = costs

    return perturbation_costs




def sweep_sample_perturbations_costs(
    key: jax.Array,
    variables: Sequence[Variable],
    j_max: int,
    sample_count: int,
    cost_func: Callable[[*tuple[jax.Array, ...]], jax.Array],
    *,
    pbar_kwargs: Mapping | None = None,
) -> dict[Designation, dict[float, jax.Array]]:
    """
    Sample the perturbation cost differences for a range of scale factors.

    Parameters
    ----------
    key : jax.Array
        Random key for reproducible randomness.

    variables : Sequence[Variable]
        List of variables to sample the perturbation costs for.
        Note that the ordering of the variables determines the
        order of arguments passed to the cost function.
        It ois the callers responsibility to ensure that the variables
        are ordered fitting to the cost function signature.

    j_max : int
        Maximum scale factor to sample.
        Actual relative scale factors are computed as
        `w * exp(-2 * j)`, where `w` is a constant (default `0.03`)
        and `j` is incremented from `1` to `j_max + 1`.

    sample_count : int
        Number of perturbations to compute for each scale factor.
        Typically, this is `500`.

    cost_func : Callable[[*tuple[jax.Array, ...]], jax.Array]
        Cost function to evaluate the the state vector.
        Must accept the raw `jax.Array` parameter arrays of the
        variables in the order they are given in the `variables`.
    """
    w: float = 0.03
    pbar_kwargs = pbar_kwargs or {}
    # nested mapping with scale factor to designation tuple to costs
    scale2costs: dict[float, dict[Designation, jax.Array]] = {}
    wrapped_j_values = tqdm.trange(1, j_max + 1, **pbar_kwargs)
    for j in wrapped_j_values:
        relscale = float(w * jnp.exp(-2 * j))
        variables = regenerate_with_relscale(variables, relscale=w)
        key, subkey = jax.random.split(key)
        scale2costs[relscale] = sample_perturbation_costs(
            key=subkey,
            variables=variables,
            sample_count=sample_count,
            cost_func=cost_func,
            pbar_kwargs={'leave' : False}
        )
    # nested mapping with designation tuple to scale factor to costs
    variable2cost: dict[Designation, dict[float, jax.Array]] = utils.transpose(scale2costs)
    return variable2cost


def compute_median_costs(
    sweep_result: dict[Designation, dict[float, jax.Array]]
) -> dict[Designation, dict[float, float]]:
    """
    Compute the median costs for each designation from the sweep result.
    """
    return {
        designation : {
            scale : float(jnp.median(costs))
            for scale, costs in scale2cost.items()}
        for designation, scale2cost in sweep_result.items()
    }
    

def compute_scale_factors(
    median_costs: dict[Designation, dict[float, float]],
) -> dict[Designation, jax.Array]:
    """
    Compute the fit of the median costs to find the scale factors.
    """
    scale_factors: dict[Designation, float] = {}
    for designation, scale2mediancost in median_costs.items():
        x = np.array(sorted(scale2mediancost.keys()))
        y = np.array([scale2mediancost[scale] for scale in x])
        # compute scale factors as slope for linear function y = m*x
        scale_factors[designation] = jnp.sum(x*y) / jnp.sum(x**2)
    return scale_factors



def store_perturbation_costs_sampling_sweep(
    sweep_result: dict[Designation, dict[float, jax.Array]],
    target: str | Path,
    variables: Sequence[Variable] | None = None,
    additional_metadata: Mapping[Hashable, Any] | None = None,
    *,
    overwrite: bool = False
) -> None:
    additional_metadata = additional_metadata or {}
    variables = [v._asdict() for v in variables] if variables else None
    metadata = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'variables': variables,
        **additional_metadata,
    }
    outdata = {
        'metadata': metadata,
        'data' : utils.recursive_cast(sweep_result, value_cast_types=(np.ndarray, jax.Array)),
    }
    mode = 'w' if overwrite else 'x'
    with open(target, mode) as f:
        json.dump(outdata, f, indent=4)











class VariableType(enum.Enum):
    """Perturbation-wise variable selection"""
    # Could not decide on capitalization rule :()
    FA_XCOORDS = 'fa_xcoords'
    FA_YCOORDS = 'fa_ycoords'
    TR_XCOORDS = 'tr_xcoords'
    TR_YCOORDS = 'tr_ycoords'


def sample_perturbation_costs_legacy(
    sample_count: int,
    variable: VariableType,
    relscale: float,
    bounds_specification: Mapping[VariableType, tuple[float, float]],
    cost_func: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array,
    n_controlpoints: int,
    *,
    NR: int = 1000,
    pbar_kwargs: Mapping | None = None
) -> tuple[list[PerturbedControlPoints], jax.Array]:
    
    # TODO: Check if necessary
    # Restrict parameterization to this hypercube
    fa_bounds = bounds_specification[VariableType.FA_YCOORDS]
    tr_bounds = bounds_specification[VariableType.TR_YCOORDS]
    fa_x_bounds = bounds_specification[VariableType.FA_XCOORDS]
    tr_x_bounds = bounds_specification[VariableType.TR_XCOORDS]

    # cache holds the (pre, post) pairs of perturbed
    # control points/parameterizations
    cache: list[PerturbedControlPoints] = []
    costs: list[jax.Array] = []

    defaults = {'leave' : False}
    pbar_kwargs = defaults | (pbar_kwargs or {})
    
    for i in tqdm.trange(sample_count, **pbar_kwargs):
        
        (key, fa_init_key, tr_init_key, index_key, perturb_key) = jax.random.split(key, num=5) 
        
        # Generate FA part of state vector
        fa_con_param_kwargs = {
            'n_controlpoints' : n_controlpoints,
            'x_scale' : 1.0,
            'y_scale' : 0.25,
            'y_offset' : jnp.deg2rad(49),
            'extent' : NR,
            'key' : fa_init_key
        }
        fa_con_pre = initialize_parameterization(**fa_con_param_kwargs)
        fa_pre = expand_fa(xc=fa_con_pre.x, fa_cons=fa_con_pre.y, n_tr=NR)

        # Generate TR part of state vector
        tr_con_param_kwargs = {
            'n_controlpoints' : n_controlpoints,
            'x_scale' : 1.0,
            'y_scale' : 0.25,
            'y_offset' : 12,
            'extent' : NR,
            'key' : tr_init_key
        }
        tr_con_pre = initialize_parameterization(**tr_con_param_kwargs)
        tr_pre = expand_tr(xc=tr_con_pre.x, tr_cons=tr_con_pre.y, n_tr=NR)
        
        # For variable pulse, we perturb the control points of the flip angle train
        # and keep the repetition time control points constant
        # For the variable time, it's obviously the other way round.
        if variable is VariableType.FA_XCOORDS:
            # x coordinates of the FA control points are fixed at edge values
            # thus indices are in the restricted range [1, n_controlpoints-1]
            index = jax.random.randint(index_key, shape=1, minval=1, maxval=n_controlpoints-1)
            fa_x_perturbed = legacy_perturb(
                index=index, bounds=fa_x_bounds, parameters=fa_con_pre.x, relscale=relscale, key=perturb_key
            )
            fa_con_perturbed = ControlPoints(fa_x_perturbed, fa_con_pre.y, edge_mode=fa_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand(xc=fa_con_perturbed.x, yc=fa_con_perturbed.y, nreq=NR)
            tr_post = tr_pre
            
        elif variable is VariableType.FA_YCOORDS:
            # for y coordinates, we can perturb all control points
            # thus the indices are in the full range [0, n_controlpoints]
            index = jax.random.randint(index_key, shape=1, minval=0, maxval=n_controlpoints)
            fa_y_perturbed = legacy_perturb(
                index=index, bounds=fa_bounds, parameters=fa_con_pre.y, relscale=relscale, key=perturb_key
            )
            fa_con_perturbed = ControlPoints(fa_con_pre.x, fa_y_perturbed, edge_mode=fa_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand(xc=fa_con_perturbed.x, yc=fa_con_perturbed.y, nreq=NR)
            tr_post = tr_pre

        elif variable is VariableType.TR_XCOORDS:
            index = jax.random.randint(index_key, shape=1, minval=1, maxval=n_controlpoints-1)
            tr_x_perturbed = legacy_perturb(
                index=index, bounds=tr_x_bounds, parameters=tr_con_pre.x, relscale=relscale, key=perturb_key
            )
            tr_con_perturbed = ControlPoints(tr_x_perturbed, tr_con_pre.y, edge_mode=tr_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(tr_con_pre, tr_con_perturbed))
            tr_post = expand(xc=tr_con_perturbed.x, yc=tr_con_perturbed.y, nreq=NR)
            fa_post = fa_pre
        
        elif variable is VariableType.TR_YCOORDS:
            index = jax.random.randint(index_key, shape=1, minval=0, maxval=n_controlpoints)
            tr_y_perturbed = legacy_perturb(
                index=index, bounds=tr_bounds, parameters=tr_con_pre.y, relscale=relscale, key=perturb_key
            )
            tr_con_perturbed = ControlPoints(tr_con_pre.x, tr_y_perturbed, edge_mode=tr_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(tr_con_pre, tr_con_perturbed))
            tr_post = expand(xc=tr_con_perturbed.x, yc=tr_con_perturbed.y, nreq=NR)
            fa_post = fa_pre
        
        else:
            raise RuntimeError('wat')

            
        c_pre = cost_func(fa_pre, tr_pre)
        c_post = cost_func(fa_post, tr_post)
        
        cdiff = jnp.abs(c_pre - c_post)
        costs.append(cdiff)
        
    return (cache, jnp.array(costs))




def compute_median_difference_legacy(
    sample_count: int,
    variable: VariableType,
    relscale: float,
    bounds_specification: Mapping[VariableType, tuple[float, float]],
    cost_func: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array,
    n_controlpoints: int,
    pbar_kwargs: Mapping | None = None
) -> float:
    """
    Compute median cost difference for between
    pairs of initial and perturbed parameterizations.

    Parameters
    ----------
    sample_count : int
        Number of perturbations to compute.
    variable : Variable
        Selector for the variable to perturb.
    relscale : float
        Relative scale factors for the perturbation of the control points.
    bounds_specification : Mapping[Variable, tuple[float, float]]
        Bounds specification for the control points.
    cost_func : Callable[[jax.Array, jax.Array], jax.Array]
        Cost function to evaluate the perturbed state vectors.
        Must accept two arguments: the full-siized FA and TR
        state vectors.
    key : jax.Array
        Random key for reproducible randomness.
    n_controlpoints : int
        Number of control points in the parameterization.
    pbar_kwargs : Mapping, optional
        Keyword arguments for the tqdm progress bar indicating
        the progress of the perturbation cost sampling.
        Default is None.

    Returns
    -------
    median_cost_diff : float
        Median cost difference between initial and perturbed
        parameterizations.
    """
    _, costs = sample_perturbation_costs_legacy(
        sample_count, variable, relscale, bounds_specification, cost_func,
        key, n_controlpoints, pbar_kwargs=pbar_kwargs
    )
    return float(jnp.median(costs))    



def find_scale_factors_legacy(
    j_max: int,
    bounds_specification: Mapping[VariableType, tuple[float, float]],
    cost_func: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array,
    sample_count: int = 500,
) -> jax.Array:
    """
    Compute scale factors by least squares fitting.
    """
    w: float = 0.03
    n_controlpoints = 9
    median_differences: dict[VariableType, list[tuple[float, float]]] = {}

    wrapped_variables = tqdm.tqdm(bounds_specification.keys(), desc='variables', leave=False)

    for variable in wrapped_variables:
        wrapped_variables.set_postfix_str(f'current: {variable}')
        mdiffs: list[tuple[float, float]] = []
        wrapped_js = tqdm.trange(1, j_max + 1, leave=False)
        for j in wrapped_js:
            wrapped_js.set_postfix_str(f'current j={j}')
            relscale = w * jnp.exp(-2 * j)
            
            key, subkey = jax.random.split(key)
            
            mdiff = compute_median_difference_legacy(
                sample_count, variable, relscale, bounds_specification,
                cost_func, subkey, n_controlpoints
            )
            mdiffs.append((relscale, mdiff))
            
        median_differences[variable] = mdiffs
        
    return median_differences

