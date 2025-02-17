"""
Functions to expand prameterizations (i.e. control point collections)
of canonical optimization variables (TR arrays/patterns, FA arrays/patterns)
into full ``n_tr`` sized patterns via (spline) interpolation.

@Author: Jannik Stebani 2025
"""
import jax
import jax.numpy as jnp
import interpax


def expand(
    xc: jax.Array,
    yc: jax.Array,
    nreq: int,
    *,
    method: str = 'cubic',
    extrap: bool = True,
    bounds: tuple[float | None, float | None] = (None, None)
) -> jax.Array:
    """
    Cubic interpolation of control points `(xc, yc)` at `arange(0, nreq)`.
    Interpolated values are postprocessed by clipping values
    to interval specified by ``bounds``.

    Parameters
    ----------
    xc : jax.Array of shape `(n_controlpoints,)`
        x-coordinates of interpolation control points.    
    yc : jax.Array of shape `(n_controlpoints,)`
        y-coordinate values of interpolation
        control points.
    nreq : int
        Requestes evaluation points.
    method : str, optional
        Spline interpolation method.
        See ``interpax.interp1d`` for available methods.
        Default is 'cubic'.
    extrap : bool, optional
        Enable extrapolation of the interpolation.
        Default is True.
    bounds : tuple[float | None, float | None], optional
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
        Default is ``(None, None``, i.e. no clipping.
    """
    # Interpolation requested at these points
    xq: jax.Array = jnp.arange(nreq)
    expanded = jnp.clip(
        interpax.interp1d(xq=xq, x=xc, f=yc, method=method, extrap=extrap),
        min=bounds[0], max=bounds[1]
    )
    return expanded


def expand_fa(
    xc: jax.Array,
    fa_cons: jax.Array,
    n_tr: int,
    *,
    method: str = 'cubic',
    extrap: bool = True,
    bounds: tuple[float, float] = (0.0, jnp.pi/2)
) -> jax.Array:
    """
    Cubic interpolation of flip angle train from support points.
    Interpolated values are postprocessed by clipping values
    to interval specified by ``bounds``.

    Parameters
    ----------
    xc : jax.Array of shape `(n_controlpoints,)`
        x-coordinates of interpolation control points.    
    fa_cons : jax.Array of shape `(n_controlpoints,)`
        Flip angle y-coordinate values of interpolation
        control points.
        Typical unit is radians (due to assumption in
        ``bounds``, if not: change accordingly)
    n_tr : int
        Repetition number count, i.e. total length
        of the sequence.
    method : str, optional
        Spline interpolation method.
        See ``interpax.interp1d`` for available methods.
        Default is 'cubic'.
    extrap : bool, optional
        Enable extrapolation of the interpolation.
        Default is True.
    bounds : tuple[float, float], optional
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
        Default is ``(0.0, pi/2)``.
    """
    # Interpolation requested at these points
    xq: jax.Array = jnp.arange(n_tr)
    expanded_fa = jnp.clip(
        interpax.interp1d(xq=xq, x=xc, f=fa_cons, method=method, extrap=extrap),
        min=bounds[0], max=bounds[1]
    )
    return expanded_fa


def expand_tr(
    xc: jax.Array,
    tr_cons: jax.Array,
    n_tr: int,
    *,
    method: str = 'cubic',
    extrap: bool = True,
    bounds: tuple[float, float | None] = (5.0, None)
) -> jax.Array:
    """
    Cubic interpolation of repetition timings TR from support points.
    Interpolated values are postprocessed by clipping values
    to interval specified by ``bounds``.
    
    Parameters
    ----------
    xc : jax.Array of shape `(n_controlpoints,)`
        x-coordinates of interpolation control points.    
    tr_cons : jax.Array of shape `(n_controlpoints,)`
        Repetition time values of interpolation
        control points.
        Unit should be coherent with outer time convention.
        Typical unit is milliseconds (due to assumption in
        ``bounds``, if not: change accordingly)
    n_tr : int
        Repetition number count, i.e. total length
        of the sequence.
    method : str, optional
        Spline interpolation method.
        See ``interpax.interp1d`` for available methods.
        Default is 'cubic'.
    extrap : bool, optional
        Enable extrapolation of the interpolation.
        Default is True.
    bounds : tuple[float, float | None], optional
        Lower and upper bounds for the parameter values.
        Excursion beyond these bounds is clipped.
        Default is ``(5.0, None)``.
    """
    # Interpolation requested at these points
    xq: jax.Array = jnp.arange(n_tr)
    expanded_tr = jnp.clip(
        interpax.interp1d(xq=xq, x=xc, f=tr_cons, method=method, extrap=extrap),
        min=bounds[0], max=bounds[1]
    )
    return expanded_tr