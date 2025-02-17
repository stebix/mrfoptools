"""
Implement parameterizations of typical optimization variables
like the TR array and the FA array to reduce the number of
degrees of freedom in the optimization problem.

Full TR and FA patterns can be create via the 
``expand_...`` functions that create linearly or
cubically interpolated patterns from control amplitudes
and control points

@Author: Jannik Stebani 2025
"""
from typing import NamedTuple
from enum import Enum

import jax
import jax.numpy as jnp


class EdgeMode(Enum):
    """
    Enumeration of edge modes, i.e. modification
    possibilitites for the control points.
    """
    FREE = 'free'
    X_FIXED = 'x_fixed'
    Y_FIXED = 'y_fixed'
    FIXED = 'fixed'
    


class ControlPoints(NamedTuple):
    """
    Control points for a parameterization.

    Attributes
    ----------

    x : jax.Array
        x-coordinates of the control points.

    y : jax.Array
        y-coordinates of the control points.

    edge_mode : EdgeMode
        Edge mode for the control point boundaries.
    """
    x: jax.Array
    y: jax.Array
    edge_mode: EdgeMode


def initialize_parameterization_legacy(
    n_controlpoints: int,
    x_scale: float,
    y_scale: float,
    y_offset: float,
    n_tr: float,
    key: jax.Array,
    edge_mode: str | EdgeMode = EdgeMode.FREE
) -> ControlPoints:
    """
    Initialize a parameterized representation.

    NOTE: Legacy version with separate x and y arrays.
    """
    edge_mode = EdgeMode(edge_mode) if isinstance(edge_mode, str) else edge_mode
    subkey_y, subkey_x = jax.random.split(key, num=2)
    y = (  jax.random.normal(subkey_y, shape=n_controlpoints) * y_scale
         + y_offset)
    x = (  jnp.linspace(0, n_tr, num=n_controlpoints)
         + jax.random.normal(subkey_x, shape=n_controlpoints) * x_scale)
    return ControlPoints(x, y, edge_mode)


def initialize_parameterization(
    n_controlpoints: int,
    x_scale: float,
    y_scale: float,
    y_offset: float,
    n_tr: float,
    key: jax.Array
) -> jax.Array:
    """
    Initialize a parameterized representation.

    Parameterization via control points array:
    [  xc_0  ,  xc_1  ,  ...  ,  xc_n_controlpoints  ] -> x_scale
    [  yc_0  ,  yc_1  ,  ...  ,  yc_n_controlpoints  ] -> y_scale, y_offset

    Parameters
    ----------
    n_controlpoints : int
        Number of control points.
    x_scale : float
        Scaling factor for the x-coordinates.
    y_scale : float
        Scaling factor for the y-coordinates.
    y_offset : float
        Offset for the y-coordinates.
    n_tr : float
        Number of TR points, i.e. extent of the
        x-coordinates.
    key : jax.Array
        Random key for reproducible randomness.

    Returns
    -------
    control_points : jax.Array
        Control points array of shape ``(2, n_controlpoints)``.
    """
    subkey_y, subkey_x = jax.random.split(key, num=2)
    y = (  jax.random.normal(subkey_y, shape=n_controlpoints) * y_scale
         + y_offset)
    x = (  jnp.linspace(0, n_tr, num=n_controlpoints)
         + jax.random.normal(subkey_x, shape=n_controlpoints) * x_scale)
    return jnp.stack((x, y), axis=0)


