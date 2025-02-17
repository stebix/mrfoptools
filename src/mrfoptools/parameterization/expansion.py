"""
Functions to expand prameterizations (i.e. control point collections)
of canonical optimization variables (TR arrays/patterns, FA arrays/patterns)
into full ``n_tr`` sized patterns via (spline) interpolation.

@Author: Jannik Stebani 2025
"""
import jax
import jax.numpy as jnp
import interpax

def expand_alpha(
    xc: jax.Array,
    alphas: jax.Array,
    n_tr: int,
    *,
    method: str = 'cubic',
    extrap: bool = True
) -> jax.Array:
    """
    Cubic interpolation of flip angle train from support points.
    Interpolated values are postprocessed by taking absolute values
    to fold negative values to positive half.

    Parameters
    ----------
    xc : jax.Array of shape `(n_controlpoints,)`
        x-coordinates of interpolation control points.    
    alphas : jax.Array of shape `(n_controlpoints,)`
        Flip angle amplitude values of interpolation
        control points.
        Expected unit is radians.
    n_tr : int
        Repetition number count, i.e. total length
        of the sequence.
    extrap : bool, optional
        Enable extrapolation of the interpolation.
        Default is True.
    """
    # Interpolation requested at these points
    xq: jax.Array = jnp.arange(n_tr)
    return jnp.abs(interpax.interp1d(xq=xq, x=xc, f=alphas, method=method, extrap=extrap))


def expand_tr(
    xc: jax.Array,
    trs: jax.Array,
    n_tr: int,
    *,
    method: str = 'cubic',
    extrap: bool = True
) -> jax.Array:
    """
    Cubic interpolation of repetition timings TR from support points.
    Interpolated values are postprocessed by taking absolute values
    to fold negative values to positive half.
    
    Parameters
    ----------
    xc : jax.Array of shape `(n_controlpoints,)`
        x-coordinates of interpolation control points.    
    trs : jax.Array of shape `(n_controlpoints,)`
        Repetition time values of interpolation
        control points.
        Unit should be coherent with outer time convention.
    n_tr : int
        Repetition number count, i.e. total length
        of the sequence.
    extrap : bool, optional
        Enable extrapolation of the interpolation.
        Default is True.
    """
    # Interpolation requested at these points
    xq: jax.Array = jnp.arange(n_tr)
    return jnp.abs(interpax.interp1d(xq=xq, x=xc, f=trs, method=method, extrap=extrap))