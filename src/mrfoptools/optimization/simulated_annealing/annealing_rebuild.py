import datetime
import json

from collections.abc import Callable, Sequence, Mapping
from enum import Enum
from typing import Any, NamedTuple, Hashable, TypeAlias
from itertools import product
from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import tqdm

from mrfoptools.optimization.simulated_annealing.perturbations import perturb_v2
from mrfoptools.optimization.simulated_annealing.cooling import exponential_cooling
from mrfoptools.optimization.costfuncs import minimum_average_criterion

from mrfoptools.parameterization.expansion import expand

from mrfoptools.epg.sequences.jax.fisp import specialize_simulate_fisp

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
    Random re-generation of the variable.
    The re-generation affects the parameter array only
    and reuses the same indices, bounds, relscale and designation.
    """
    parameters = jax.random.uniform(
        key=key, shape=variable.parameters.shape,
        minval=variable.bounds[0], maxval=variable.bounds[1]
    )
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



def metropolis_probability(cost: float, cost_prev: float, temp: float) -> jax.Array:
    """
    Acceptance probability for the Metropolis criterion in simulated annealing.
    """
    return jnp.min(jnp.array([1, jnp.exp(-(cost - cost_prev) / temp)]))


def create_cost_function(
    simulate: Callable[[jax.Array, jax.Array], jax.Array],
    nreq : int,
    weights: tuple[float, float],
    *,
    fa_bounds: tuple[float32, float32] = (jnp.deg2rad(1.0), jnp.deg2rad(90.0)),
    tr_bounds: tuple[float32, float32] = (5, 250)
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Create the cost function that directly ingests the parameter arrays
    and expands towards full FA and TR patterns internally.

    Parameters
    ----------

    simulate : Callable
        Forward simulation function. Must be specialized to have
        signature `simulate(fa, tr)` and return the signals
        tensor of shape `(n_species, n_tr)`.

    nreq : int
        Number of requested points for the interpolation.
        This is the temporal dimension in the full pattern.

    weights : tuple[float, float]
        Weights for the time and signal cost terms.

    fa_bounds : tuple[float32, float32], optional
        Min-max bounds for the flip angles in radians.
        Values of the spline-interpolated pattern
        are clipped to these bounds.

    tr_bounds : tuple[float32, float32], optional
        Min-max bounds for the repetition times.
        Values of the spline-interpolated pattern
        are clipped to these bounds.
        Unit must be compatible to the other use
        time units in the simulation.
    """
    w_time, w_signal = weights

    def cost_function(fa: jax.Array, tr: jax.Array) -> jax.Array:
        signals = simulate(fa, tr)
        return w_time * jnp.sqrt(jnp.sum(tr)) + w_signal / minimum_average_criterion(signals)

    fa_expander = partial(expand, nreq=nreq, method='cubic', extrap=False, bounds=fa_bounds)
    tr_expander = partial(expand, nreq=nreq, method='cubic', extrap=False, bounds=tr_bounds)

    def _adaptor(
        fa_x: jax.Array,
        fa_y: jax.Array,
        tr_x: jax.Array,
        tr_y: jax.Array
    ) -> jax.Array:
        """
        Adapt the cost function to enable calling with separate
        control point parameter arrays.
        """
        fa = fa_expander(fa_x, fa_y)
        tr = tr_expander(tr_x, tr_y)
        return cost_function(fa, tr)
    
    return _adaptor



def anneal(
    fa_x: Variable,
    fa_y: Variable,
    tr_x: Variable,
    tr_y: Variable,
    simulate: Callable[[jax.Array, jax.Array], jax.Array],
) -> Any:
    
    key = 1

    variables: Sequence[Variable] = [fa_x, fa_y, tr_x, tr_y]

    if not are_ordered_correctly(variables):
        raise ValueError('ordering mismatch for variables')
    
    fa_interp_bounds: tuple[float32, float32] = (jnp.deg2rad(1.0), jnp.deg2rad(90.0))
    tr_interp_bounds: tuple[float32, float32] = (5.0, 500.0)

    cost_func = create_cost_function(
        simulate=None,
        nreq=1000,
        weights=(1.0, 1.0),
        fa_bounds=fa_interp_bounds,
        tr_bounds=tr_interp_bounds
    )

    cost = cost_func(*variables)

    # setup
    temp_initial: float = 1.0
    max_iter: int = 100
    n_controlpoints: int = 10
    R: float = 0.00001
    r: float = R ** (1 / max_iter) # noqa: F841

    key: jax.Array = jax.random.key(seed=1701) 

    cost = cost_func(*get_parameters(variables))

    total = max_iter * len(variables) * np.sum((len(p) for p in get_parameters(variables)))
    pbar = tqdm.tqdm(desc='total annealing progress', unit='it', total=total)    

    for n_iter in range(max_iter):

        temp = exponential_cooling(temp_initial, 0.5, n_iter)

        for i, variable in enumerate(variables):

            for index in variable.indices:
                
                # split the key for next iteration, perturbation and acceptance
                key, ptrb_key, acct_key = jax.random.split(key, num=3)

                perturbed_parameters = perturb_v2(
                    index=index,
                    parameters=variable.parameters,
                    bounds=variable.bounds,
                    relscale=variable.relscale,
                    key=ptrb_key
                )
                if variable.sort:
                    perturbed_parameters = jnp.sort(perturbed_parameters)
                # correct ordering is paramount here since we stuff this directly in the 
                # cost function that internally expand the control points into the full
                # specifications via spline interpolation
                args = tuple(
                    variables[j].parameters
                    if j != i else perturbed_parameters
                    for j in range(len(variables))
                )

                candidate_cost = cost_func(*args)
                acc_prob = metropolis_probability(candidate_cost, cost, temp)
                
                pbar.update()

                if jax.random.uniform(acct_key) < acc_prob:
                    # accept the new parameters
                    variables[i] = variable.with_new_parameters(perturbed_parameters)
                    cost = candidate_cost

                    # optimize this away possibly
                    pbar.set_postfix_str(f'accept! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')

                else:
                    # reject the new parameters
                    pbar.set_postfix_str(f'reject! @ prob: {acc_prob:.3f} | cost {cost:.2f} | candcost: {candidate_cost:.2f} | temp: {temp:.6f}')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pbar.write(f'finished @ {now}')
    return variables              




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
        perturbed_parameters = perturb_v2(
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


InnerKey: TypeAlias = Hashable
OuterKey: TypeAlias = Hashable

def transpose(
    d: Mapping[OuterKey, Mapping[InnerKey, Any]]
) -> Mapping[InnerKey, Mapping[OuterKey, Any]]:
    """
    Transpose a singularly nested mapping.
    """
    transposed_mapping: dict[InnerKey, dict[OuterKey, Any]] = {}
    for outer_key, inner_mapping in d.items():
        for inner_key, value in inner_mapping.items():
            if inner_key not in transposed_mapping:
                transposed_mapping[inner_key] = {}
            transposed_mapping[inner_key][outer_key] = value
    return transposed_mapping


def recursive_cast(
    mapping: Mapping,
    key_cast_types: tuple[type, ...] = (Designation,),
    value_cast_types: tuple[type, ...] = (np.ndarray, jax.Array),
) -> dict:
    """
    Recursively cast all values of `cast_types` in the mapping.
    The values of type `value_cast_types` must support the `tolist()` method.
    The keys of type `key_cast_types` are cast to their `asdict()` representation.
    """
    recast = {}
    for k, v in mapping.items():
        if isinstance(k, key_cast_types):
            k = k.to_string()
        if isinstance(v, Mapping):
            recast[k] = recursive_cast(v, value_cast_types)
        elif isinstance(v, value_cast_types):
            recast[k] = v.tolist()
        else:
            recast[k] = v
    return recast



def sweep_sample_perturbations_costs(
    key: jax.Array,
    variables: Sequence[Variable],
    j_max: int,
    sample_count: int,
    cost_func: Callable[[*tuple[jax.Array, ...]], jax.Array],
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

    # nested mapping with scale factor to designation tuple to costs
    scale2costs: dict[float, dict[Designation, jax.Array]] = {}
    for j in range(1, j_max + 1):
        relscale = float(w * jnp.exp(-2 * j))
        variables = regenerate_with_relscale(variables, relscale=w)
        key, subkey = jax.random.split(key)
        scale2costs[relscale] = sample_perturbation_costs(
            key=subkey,
            variables=variables,
            sample_count=sample_count,
            cost_func=cost_func
        )
    # nested mapping with designation tuple to scale factor to costs
    variable2cost: dict[Designation, dict[float, jax.Array]] = transpose(scale2costs)
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
) -> Any:
    """
    Compute the fit of the median costs to find the scale factors.
    """
    scale_factors: dict[Designation, float] = {}
    for designation, scale2mediancost in median_costs.items():
        x = np.array(sorted(scale2mediancost.keys()))
        y = np.array([scale2mediancost[scale] for scale in x])
        # fit the data with a linear least squares fit
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
        'data' : recursive_cast(sweep_result, value_cast_types=(np.ndarray, jax.Array)),
    }
    mode = 'w' if overwrite else 'x'
    with open(target, mode) as f:
        json.dump(outdata, f, indent=4)



def main():
    INIT_FA = 50
    INIT_TR = 12
    NR = 1000
    M0 = 1.0
    TE = 2.2
    TI = 20

    max_states = 750
    inversion_efficiency = 1.0


    fa = jnp.deg2rad(jnp.full(fill_value=INIT_FA, shape=NR, dtype=jnp.float32))
    tr = jnp.full(fill_value=INIT_TR, shape=NR, dtype=jnp.float32)
    phases = jnp.full_like(fa, fill_value=jnp.pi/2)

    T1 = jnp.array([1500, 2000, 2500, 3000])
    T2 = jnp.array([500, 500, 600, 700])

    simulate_fisp = specialize_simulate_fisp(
        T1=T1,
        T2=T2,
        M0=M0,
        phases=phases,
        TI=TI,
        TE=TE,
        max_states=max_states,
        inversion_efficiency=inversion_efficiency   
    )

    n_controlpoints: int = 10

    fa_controlpoint_bounds = (jnp.deg2rad(1.0), jnp.deg2rad(90.0))
    tr_controlpoint_bounds = (5.0, 500.0)
    x_bounds = (0.0, 1000.0)
    
    fa_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.X),
        sort=True
    )
    tr_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.X),
        sort=True
    )
    fa_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_FA),
        bounds=fa_controlpoint_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.Y),
        sort=False
    )
    tr_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_TR),
        bounds=tr_controlpoint_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.TR, axis=ParameterAxis.Y),
        sort=False
    )

    result = anneal()



