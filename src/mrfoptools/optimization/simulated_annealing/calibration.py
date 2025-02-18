"""
Calibration tooling for simulated annealing optimization.

@Author: Jannik Stebani 2025
"""
import enum
from collections.abc import Callable

import jax
import jax.numpy as jnp
import tqdm

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



def compute_perturbation_costs(
    N: int,
    n_controlpoints: int,
    scale: float,
    bounds: tuple[float, float],
    cost_func: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array,
    variable: VariableType
) -> tuple[list[PerturbedControlPoints], jax.Array]:
    
    NR: int = 1000

    key, subkey = jax.random.split(key, num=2)

    indices = jax.random.randint(
        key=subkey, shape=N, minval=0, maxval=n_controlpoints
    )

    # TODO: Check if necessary
    # Restrict parameterization to this hypercube

    fa_bounds = (jnp.deg2rad(0), jnp.deg2rad(90))
    tr_bounds = (5.0, 1000.0)
    x_bounds = (0, 1000)


    # cache holds the (pre, post) pairs of perturbed
    # control points/parameterizations
    cache: list[PerturbedControlPoints] = []
    costs: list[jax.Array] = []
    
    for index in tqdm.tqdm(indices):
        
        (key, fa_init_key, tr_init_key, perturb_key) = jax.random.split(key, num=4) 
        
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
            fa_x_perturbed = perturb(
                index=index, bounds=x_bounds, parameters=fa_con_pre.x, relscale=scale, key=perturb_key
            )
            fa_con_perturbed = ControlPoints(fa_x_perturbed, fa_con_pre.y, edge_mode=fa_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand(xc=fa_con_perturbed.x, yc=fa_con_perturbed.y, nreq=NR)
            tr_post = tr_pre
            
        elif variable is VariableType.FA_YCOORDS:
            fa_y_perturbed = perturb(
                index=index, bounds=fa_bounds, parameters=fa_con_pre.y, relscale=scale, key=perturb_key
            )
            fa_con_perturbed = ControlPoints(fa_con_pre.x, fa_y_perturbed, edge_mode=fa_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand(xc=fa_con_perturbed.x, yc=fa_con_perturbed.y, nreq=NR)
            tr_post = tr_pre

        elif variable is VariableType.TR_XCOORDS:
            tr_x_perturbed = perturb(
                index=index, bounds=x_bounds, parameters=tr_con_pre.x, relscale=scale, key=perturb_key
            )
            tr_con_perturbed = ControlPoints(tr_x_perturbed, tr_con_pre.y, edge_mode=tr_con_pre.edge_mode)
            cache.append(PerturbedControlPoints(tr_con_pre, tr_con_perturbed))
            tr_post = expand(xc=tr_con_perturbed.x, yc=tr_con_perturbed.y, nreq=NR)
            fa_post = fa_pre
        
        elif variable is VariableType.TR_YCOORDS:
            tr_y_perturbed = perturb(
                index=index, bounds=tr_bounds, parameters=tr_con_pre.y, relscale=scale, key=perturb_key
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
    N: int,
    n_controlpoints: int,
    variable: VariableType,
    scale: float,
    bounds: tuple[float, float],
    cost_func: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array
) -> float:
    """
    Compute median cost difference for between
    pairs of initial and perturbed parameterizations.

    Parameters
    ----------
    N : int
        Number of perturbations to compute.
    n_controlpoints : int
        Number of control points in the parameterization.
    variable : Variable
        Selector for the variable to perturb.
    scale : float
        Scale factors for the perturbation of the control points.
    cost_func : Callable[[jax.Array, jax.Array], jax.Array]
        Cost function to evaluate the perturbed state vectors.
        Must accept two arguments: the full-siized FA and TR
        state vectors.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    median_cost_diff : float
        Median cost difference between initial and perturbed
        parameterizations.
    """
    _, costs = compute_perturbation_costs(
        N, n_controlpoints, scale, bounds, cost_func, key, variable
    )
    return float(jnp.median(costs))    



def find_scale_factors(
    j_max: int,
    fa_scale: tuple[float, float],
    key: jax.Array,
    N: int = 500,
) -> jax.Array:
    """
    Compute scale factors by least squares fitting.
    """
    pass