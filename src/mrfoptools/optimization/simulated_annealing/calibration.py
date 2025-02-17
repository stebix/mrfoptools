"""
Calibration tooling for simulated annealing optimization.

@AuthorL Jannik Stebani 2025
"""
import enum
from collections.abc import Callable

import jax
import jax.numpy as jnp
import tqdm

from mrfoptools.parameterization.expansion import expand_fa, expand_tr
from mrfoptools.parameterization.parameterization import (initialize_bounds,
                                                          initialize_parameterization)
from mrfoptools.optimization.simulated_annealing.perturbations import (
    generate_indexed_varedge_perturbation,
    PerturbedControlPoints
)

class Variable(enum.Enum):
    """Perturbation-wise variable selection"""
    # Could not decide on capitalization rule :()
    PULSE = 'pulse'
    TIME = 'time'


def compute_perturbation_costs(
    N: int,
    n_controlpoints: int,
    variable: Variable,
    fa_scale: tuple[float, float],
    tr_scale: tuple[float, float],
    cost_func: Callable[[jax.Array, jax.Array], jax.Array],
    key: jax.Array
) -> tuple[list, list]:
    """
    Compute cost difference for between
    pairs of initial and perturbed parameterizations.

    TODO: Expose scales as parameters???
    
    Parameters
    ----------
    N : int
        Number of perturbations to compute.
    n_controlpoints : int
        Number of control points in the parameterization.
    variable : Variable
        Selector for the variable to perturb.
        Can be either `Variable.PULSE` or `Variable.TIME`.
    fa_scale : tuple[float, float]
        Scale factors for the perturbation of the flip angle control points.
        First element is for the x-coordinate, second for the y-coordinate.
    tr_scale : tuple[float, float]
        Scale factors for the perturbation of the repetition time control points.
        First element is for the x-coordinate, second for the y-coordinate.
    cost_func : Callable[[jax.Array, jax.Array], jax.Array]
        Cost function to evaluate the perturbed state vectors.
        Must accept two arguments: the full-siized FA and TR
        state vectors.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    (cache, cost) : tuple[list[PerturbedControlPoints], jax.Array]
        Tuple consisting of pairs of initial and perturbed control points
        and the cost differences for each perturbation.
    """
    NR: int = 1000
    # Note: These scales are the `meat` and stochastically influence
    # cost difference of the initial and perturbed control point parameters 
    fa_scales = jnp.array(fa_scale)
    tr_scales = jnp.array(tr_scale)
    # Randomized selection of the control point components:
    # Can be control point x-coordinate or y-coordinate
    key, subkey = jax.random.split(key, num=2)
    indices = jax.random.randint(
        key=subkey, shape=N, minval=0, maxval=2*n_controlpoints
    )
    # Restrict parameterization to this hypercube
    fa_cons_bounds = initialize_bounds(
        n_controlpoints, x_bounds=(0, 1000), y_bounds=(jnp.deg2rad(0), jnp.deg2rad(90))
    )    
    tr_cons_bounds = initialize_bounds(
        n_controlpoints, x_bounds=(0, 1000), y_bounds=(5.0, 1000.0)
    )
    # cache holds the (pre, post) pairs of perturbed
    # control points/parameterizations
    cache: list[PerturbedControlPoints] = []
    costs: list[jax.Array] = []
    
    for index in tqdm.tqdm(indices):
        
        (key,
         fa_init_key, tr_init_key,
         fa_perturb_key, tr_perturb_key) = jax.random.split(key, num=5) 
        
        # Generate FA part of state vector
        fa_con_param_kwargs = {
            'n_controlpoints' : n_controlpoints,
            'x_scale' : 1.0,
            'y_scale' : 0.25,
            'y_offset' : jnp.deg2rad(49),
            'n_tr' : NR,
            'key' : fa_init_key
        }
        fa_con_pre = initialize_parameterization(**fa_con_param_kwargs)
        fa_pre = expand_fa(xc=fa_con_pre[0, :], fa_cons=fa_con_pre[1, :], n_tr=NR)
        # Generate TR part of state vector
        tr_con_param_kwargs = {
            'n_controlpoints' : n_controlpoints,
            'x_scale' : 1.0,
            'y_scale' : 0.25,
            'y_offset' : 12,
            'n_tr' : NR,
            'key' : tr_init_key
        }
        tr_con_pre = initialize_parameterization(**tr_con_param_kwargs)
        tr_pre = expand_tr(xc=tr_con_pre[0, :], tr_cons=tr_con_pre[1, :], n_tr=NR)
        
        # For variable pulse, we perturb the control points of the flip angle train
        # and keep the repetition time control points constant
        # For the variable time, it's obviously the other way round.
        if variable is Variable.PULSE:
            fa_con_perturbed = generate_indexed_varedge_perturbation(
                index=index, bounds=fa_cons_bounds, parameters=fa_con_pre, scales=fa_scales, key=fa_perturb_key
            )
            cache.append(PerturbedControlPoints(fa_con_pre, fa_con_perturbed))
            fa_post = expand_fa(xc=fa_con_perturbed[0, :], fa_cons=fa_con_perturbed[1, :], n_tr=NR)
            tr_post = tr_pre
            
        elif variable is Variable.TIME:
            tr_con_perturbed = generate_indexed_varedge_perturbation(
                index=index, bounds=tr_cons_bounds, parameters=tr_con_pre, scales=tr_scales, key=tr_perturb_key
            )
            cache.append(PerturbedControlPoints(tr_con_pre, tr_con_perturbed))
            tr_post = expand_tr(xc=tr_con_perturbed[0, :], tr_cons=tr_con_perturbed[1, :], n_tr=NR)
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
    variable: Variable,
    fa_scale: tuple[float, float],
    tr_scale: tuple[float, float],
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
        Can be either `Variable.PULSE` or `Variable.TIME`.
    fa_scale : tuple[float, float]
        Scale factors for the perturbation of the flip angle control points.
        First element is for the x-coordinate, second for the y-coordinate.
    tr_scale : tuple[float, float]
        Scale factors for the perturbation of the repetition time control points.
        First element is for the x-coordinate, second for the y-coordinate.
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
        N, n_controlpoints, variable, fa_scale, tr_scale, cost_func, key
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
    n_controlpoints: int = 8
    w: float = 0.03

    variable = Variable('pulse')
    
    def cost_func(fa: jax.Array, tr: jax.Array) -> jax.Array:
        raise NotImplementedError

    for j in range(1, j_max + 1):
        scale = w * jnp.exp(- 2 * j)
        median_difference = compute_median_difference(
            N, n_controlpoints=n_controlpoints, variable=variable,
            fa_scale=fa_scale, tr_scale=(1.0, 1.0),
            cost_func=cost_func, key=key
        )
    