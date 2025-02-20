"""
Calibration tooling for simulated annealing optimization.

@Author: Jannik Stebani 2025
"""
import enum
from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
from tqdm.autonotebook import tqdm, trange

from mrfoptools.parameterization.expansion import expand_fa, expand_tr, expand
from mrfoptools.parameterization.parameterization import (initialize_parameterization,
                                                          ControlPoints)
from mrfoptools.optimization.simulated_annealing.perturbations import (
    PerturbedControlPoints,
    perturb
)

class VariableType(enum.Enum):
    """Perturbation-wise variable selection"""
    # Could not decide on capitalization rule :()
    FA_XCOORDS = 'fa_xcoords'
    FA_YCOORDS = 'fa_ycoords'
    TR_XCOORDS = 'tr_xcoords'
    TR_YCOORDS = 'tr_ycoords'



def sample_perturbation_costs(
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
    
    for i in tqdm(range(sample_count), **pbar_kwargs):
        
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
            index = jnp.random.randint(index_key, minval=1, maxval=n_controlpoints-1)
            fa_x_perturbed = perturb(
                index=index, bounds=fa_x_bounds, parameters=fa_con_pre.x, relscale=relscale, key=perturb_key
            )
            fa_con_perturbed = ControlPoints(fa_x_perturbed, fa_con_pre.y, edge_mode=fa_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand(xc=fa_con_perturbed.x, yc=fa_con_perturbed.y, nreq=NR)
            tr_post = tr_pre
            
        elif variable is VariableType.FA_YCOORDS:
            # for y coordinates, we can perturb all control points
            # thus the indices are in the full range [0, n_controlpoints]
            index = jnp.random.randint(index_key, minval=0, maxval=n_controlpoints)
            fa_y_perturbed = perturb(
                index=index, bounds=fa_bounds, parameters=fa_con_pre.y, relscale=relscale, key=perturb_key
            )
            fa_con_perturbed = ControlPoints(fa_con_pre.x, fa_y_perturbed, edge_mode=fa_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand(xc=fa_con_perturbed.x, yc=fa_con_perturbed.y, nreq=NR)
            tr_post = tr_pre

        elif variable is VariableType.TR_XCOORDS:
            index = jnp.random.randint(index_key, minval=1, maxval=n_controlpoints-1)
            tr_x_perturbed = perturb(
                index=index, bounds=tr_x_bounds, parameters=tr_con_pre.x, relscale=relscale, key=perturb_key
            )
            tr_con_perturbed = ControlPoints(tr_x_perturbed, tr_con_pre.y, edge_mode=tr_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(tr_con_pre, tr_con_perturbed))
            tr_post = expand(xc=tr_con_perturbed.x, yc=tr_con_perturbed.y, nreq=NR)
            fa_post = fa_pre
        
        elif variable is VariableType.TR_YCOORDS:
            index = jnp.random.randint(index_key, minval=0, maxval=n_controlpoints)
            tr_y_perturbed = perturb(
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




def compute_median_difference(
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
    _, costs = sample_perturbation_costs(
        sample_count, variable, relscale, bounds_specification, cost_func,
        key, n_controlpoints, pbar_kwargs=pbar_kwargs
    )
    return float(jnp.median(costs))    



def find_scale_factors(
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

    wrapped_variables = tqdm(VariableType)

    for variable in wrapped_variables:
        wrapped_variables.set_postfix_str(f'current: {variable}')
        mdiffs: list[tuple[float, float]] = []
        wrapped_js = trange(1, j_max + 1, leave=False)
        for j in wrapped_js:
            wrapped_js.set_postfix_str(f'current j={j}')
            relscale = w * jnp.exp(-2 * j)
            
            key, subkey = jax.random.split(key)
            
            mdiff = compute_median_difference(
                sample_count, variable, relscale, bounds_specification,
                cost_func, subkey, n_controlpoints
            )
            mdiffs.append((relscale, mdiff))
            
        median_differences[variable] = mdiffs
        
    return median_differences