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


def initialize_parameterization(
    n_controlpoints: int,
    amplitude_scale: float,
    amplitude_offset: float,
    points_scale: float,
    n_tr: float,
    key: jax.Array,
    edge_mode: str | EdgeMode = EdgeMode.FREE
) -> ControlPoints:
    """
    Initialize a parameterized representation.
    """
    edge_mode = EdgeMode(edge_mode) if isinstance(edge_mode, str) else edge_mode
    subkey_amp, subkey_pts = jax.random.split(key, num=2)
    amplitudes = (  jax.random.normal(subkey_amp, shape=n_controlpoints) * amplitude_scale
                  + amplitude_offset)
    points = (  jnp.linspace(0, n_tr, num=n_controlpoints)
              + jax.random.normal(subkey_pts, shape=n_controlpoints) * points_scale)
    return ControlPoints(amplitudes, points, edge_mode)


