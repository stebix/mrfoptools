"""
Variables for simulated annealing optimization.

Variables encapsulate the raw parameters (control point coordinates) and supporting
information like:

    - indices: which parameter elements are perturbed
    - bounds: the [min, max] bounds for the parameters
    - relscale: the relative scale for perturbations
    - designation: the type and axis signifiers of the parameters
    - sort: whether the parameters should be sorted before being passed to the cost function

Further tooling encompasses generation of randomized variables, where the
randomization of the parameters is done uniformly in the space defined by the bounds.
"""
from enum import Enum
from typing import NamedTuple
from collections.abc import Sequence
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

float32 = np.float32

class EdgeMode(Enum):
    """
    Enumeration of edge modes, i.e. modification
    possibilitites for the control points.
    """
    FIXED = 'fixed'
    FLOATING = 'floating'


class ParameterType(Enum):
    """Optimization variable selection."""
    FA = 'fa'
    TR = 'tr'

class ParameterAxis(Enum):
    X = 'x'
    Y = 'y'


class Designation(NamedTuple):
    type: ParameterType
    axis: ParameterAxis

    @classmethod
    def from_strings(cls, type_str: str, axis_str: str, /) -> 'Designation':
        type = ParameterType(type_str.lower())
        axis = ParameterAxis(axis_str.lower())
        return cls(type=type, axis=axis)

    def to_string(self) -> str:
        """Stable conversion to string representation."""
        return f'{self.type.value},{self.axis.value}'


class Variable(NamedTuple):
    """
    Specifies a full variable for simulated annealing optimization.

    Attributes
    ----------
    indices : jax.Array
        Indices of the parameters that are perturbed during optimization.
        E.g. can be only inner indices for fixed edge control points.

    parameters : jax.Array
        Parameter values of the variable.
        E.g. x- or y-coordinates of controlpoints.

    bounds : tuple[float32, float32]
        Bounds for the parameters.
        E.g. min and max values for the control points.

    relscale : float32
        Relative scale scalar. Utilized during generation of
        perturbations.

    designation : Designation
        Designation metadata for the variable.

    sort : bool
        Whether the variable should be sorted or not.
        If True, the parameters are sorted in ascending order
        before being passed to the cost function.
        This is useful for variables that are expected to be
        monotonically increasing (e.g. x-coordinates of control points).
        If False, the parameters are passed to the cost function
        in the order they are given.

    Methods
    -------
    with_new_parameters(parameters: jax.Array) -> Variable
        Create a new Variable with the same indices, bounds, relscale,
        designation and sort, but with new parameters.
    """
    indices: jax.Array
    parameters: jax.Array
    bounds: tuple[float32, float32]
    relscale: float32
    designation: Designation
    sort: bool = False

    def with_new_parameters(self, parameters: jax.Array) -> 'Variable':
        """
        Create a new Variable with the same indices, bounds, relscale,
        designation and sort, but with new parameters.
        """
        return Variable(
            indices=self.indices,
            parameters=parameters,
            bounds=self.bounds,
            relscale=self.relscale,
            designation=self.designation,
            sort=self.sort
        )
    
    @classmethod
    def create_with_fixed_edges(
        cls,
        parameters: jax.Array,
        bounds: tuple[float32, float32],
        relscale: float32,
        designation: Designation,
        sort: bool = False
    ) -> 'Variable':
        """
        Create a variable with constant control point parameter
        values at the edges.

        The indices are automatically set to the inner indices to
        accomplish this.
        """
        indices = jnp.arange(1, len(parameters) - 1)
        return cls(
            indices=indices,
            parameters=parameters,
            bounds=bounds,
            relscale=relscale,
            designation=designation,
            sort=sort
        )
    
    @classmethod
    def create_with_floating_edges(
        cls,
        parameters: jax.Array,
        bounds: tuple[float32, float32],
        relscale: float32,
        designation: Designation,
        sort: bool = False
    ) -> 'Variable':
        """
        Create a variable with floating control point parameter
        values at the edges.

        The indices are automatically set to the full indices to
        accomplish this.
        """
        indices = jnp.arange(0, len(parameters))
        return cls(
            indices=indices,
            parameters=parameters,
            bounds=bounds,
            relscale=relscale,
            designation=designation,
            sort=sort
        )


def regenerate_with_relscale(
    variables: Sequence[Variable],
    relscale: float32
) -> list[Variable]:
    """
    Regenerate the variables with the given relative scale.
    """
    return [
        Variable(
            indices=variable.indices,
            parameters=variable.parameters,
            bounds=variable.bounds,
            relscale=relscale,
            designation=variable.designation,
            sort=variable.sort
        )
        for variable in variables
    ]


def generate_random_variable_fixed(
    key: jax.Array,
    n_controlpoints: int,
    bounds: tuple[float32, float32],
    relscale: float32,
    designation: Designation,
    sort: bool
) -> Variable:
    """
    Generate a `Variable` object uniformly in the space defined by the bounds
    using fixed edges: parameter values at the edges are expected to be immutable.
    """
    parameters = jax.random.uniform(
        key=key, shape=n_controlpoints, minval=bounds[0], maxval=bounds[1]
    )
    # first and last control points are static on the bounds
    parameters = parameters.at[0].set(bounds[0])
    parameters = parameters.at[-1].set(bounds[1])
    # only the inner control points are mutable: restrict indices
    indices = jnp.arange(1, n_controlpoints - 1)
    return Variable(
        indices=indices,
        parameters=parameters,
        bounds=bounds,
        relscale=relscale,
        designation=designation,
        sort=sort
    )


def generate_random_variable_floating(
    key: jax.Array,
    n_controlpoints: int,
    bounds: tuple[float32, float32],
    relscale: float32,
    designation: Designation,
    sort: bool
) -> Variable:
    """
    Generate a `Variable` object uniformly in the space defined by the bounds
    using floating edges: parameter values at the edges are expected to be mutable.
    """
    parameters = jax.random.uniform(
        key=key, shape=n_controlpoints, minval=bounds[0], maxval=bounds[1]
    )
    indices = jnp.arange(0, n_controlpoints)
    return Variable(
        indices=indices,
        parameters=parameters,
        bounds=bounds,
        relscale=relscale,
        designation=designation,
        sort=sort
    )


def regenerate_randomized(key: jax.Array, variable: Variable) -> Variable:
    """
    Re-generation of the variable with randomized parameter array.
    The new parameter array is generated uniformly in the space defined by
    the bounds of the variable.
    Indices, bounds, relscale and designation are reused.
    """
    indices = variable.indices
    randomized_parameters = jax.random.uniform(
        key=key, shape=indices.shape,
        minval=variable.bounds[0], maxval=variable.bounds[1]
    )
    parameters = variable.parameters.at[indices].set(randomized_parameters)
    if variable.sort:
        parameters = jnp.sort(parameters)
    return variable.with_new_parameters(parameters)



def generate_randomized_variable(
    key: jax.Array,
    n_controlpoints: int,
    bounds: tuple[float32, float32],
    relscale: float32,
    designation: Designation,
    sort: bool,
    edge_mode: EdgeMode
) -> Variable:
    """
    Generate a `Variable` object uniformly in the space defined by the bounds
    using the specified edge mode.
    
    Can be used to uniformly generate sample variables for determination of the
    median error difference during calibration of the simulated annealing core.
    """
    if edge_mode is EdgeMode.FLOATING:
        variable = generate_random_variable_floating(
            key, n_controlpoints, bounds, relscale, designation, sort
        )
    elif edge_mode is EdgeMode.FIXED:
        variable = generate_random_variable_fixed(
            key, n_controlpoints, bounds, relscale, designation, sort
        )
    else:
        raise ValueError(f'invalid edge mode: \'{edge_mode}\'')
    return variable


def are_ordered_correctly(
    variables: Sequence[Variable],
    type_order: tuple[ParameterType, ParameterType] = (ParameterType.FA, ParameterType.TR),
    axis_order: tuple[ParameterAxis, ParameterAxis] = (ParameterAxis.X, ParameterAxis.Y),
) -> bool:
    """
    Check that the variables in the given sequence
    are ordered correctly.
    Can be used to enusre fit of variable order to cost function
    signature.
    """
    expected_designations = tuple(
        Designation(type=type, axis=axis)
        for type, axis in product(type_order, axis_order)
    )
    if len(variables) != len(expected_designations):
        return False
    
    for variable, expected_designation in zip(variables, expected_designations):
        if variable.designation != expected_designation:
            return False
        
    return True


def get_parameters(
    variables: Sequence[Variable]
) -> tuple[jax.Array, ...]:
    """
    Get the parameters of the variables in the given sequence.
    """
    return tuple(variable.parameters for variable in variables)
