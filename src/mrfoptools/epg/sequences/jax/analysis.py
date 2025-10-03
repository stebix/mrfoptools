"""
Code for analysis and evaluation of simulated EPG sequences in JAX.


Overview:

This module is concerned with the analysis of EPG sequences and their simulated signals.

WE thus define several data structures of the `Section` type that are quite analogous to the
`Segment` type in the `pfisp2` module, but with augmented information about the absolute timepoints
and the signal outputs of the sequence simulation.

Thus:
    - `Segment`-like PyTree structure that holds information for the core sequence simulation
    - `Section`-like attrs-classes that hold semnatic information (signals, timepoints, parameters)
      for ex-post analysis and plotting of the sequence simulation results.

"""
from typing import Any, NamedTuple, TypeAlias, Protocol, Literal
from collections.abc import Sequence

import attrs
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from attrs import define
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import mrfoptools.epg.sequences.jax.pfisp2 as pfisp


Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.Array | np.ndarray


class RelaxationParameters(NamedTuple):
    T1: float | ArrayLike
    T2: float | ArrayLike
    M0: float | ArrayLike
    
    @classmethod
    def from_provider(
        cls,
        provider: Any,
        index: int | None = None
    ) -> 'RelaxationParameters':
        """
        Create instance from abitrary provider objects with the necessary information.
        Via the 'index', a certain relaxometric species can be selected in case
        the information provider object contains relaxometric (T1, T2) information
        for multiple species.
        """
        index = index if index is not None else slice(None)
        return cls(T1=provider.T1[index], T2=provider.T2[index], M0=provider.M0[index])



@define
class SignalSection:
    """
    Semantic section of muti-species signal output in a sequence simulation.
    """
    timepoints: ArrayLike
    signal: ArrayLike
    parameters: RelaxationParameters


@define
class PreparationSection:
    """
    Semantic segment of a preparation module in a sequence simulation.
    Holds the (start, stop) timepoints of the preparation segment.
    """
    timepoints: ArrayLike
    

@define
class InversionSection:
    """
    Semantic section of an inversion module in a sequence simulation.
    Holds the (start, stop) timepoints of the inversion segment.
    """
    timepoints: ArrayLike


class RelaxationParametersProvider(Protocol):
    """
    Protocol for objects that provide relaxometric parameters
    (T1, T2, M0) for a multitude species.
    """
    T1: ArrayLike
    T2: ArrayLike
    M0: ArrayLike


def expand_to_semantic_signal_segments(
    signals: ArrayLike,
    timepoints: ArrayLike,
    parameters: RelaxationParametersProvider
) -> list[SignalSection]:
    """
    Create semantic signal segments for a pre-split (n_species, temporal_segment) signals
    tensor by splitting into `SignalSegments` that hold the segment-wise information:
        - segment-wise temporal information
        - segment-wise signal information (single signal, aka only one species)
        - species-specific relaxometric information
    """
    segments: list[SignalSection] = []
    for species_index, signal in enumerate(signals):
        sigseg = SignalSection(
            timepoints=timepoints,
            signal=signal,
            parameters=RelaxationParameters.from_provider(parameters, index=species_index)
        )
        segments.append(sigseg)
        
    return segments
        

@define
class SequenceSignals:
    """Signals a segmented sequence."""
    segments: list[SignalSection | PreparationSection | InversionSection]



def transduce_to_sections(
    signals: ArrayLike,
    sequence: Sequence[pfisp.SequenceSegment],
    parameters: RelaxationParametersProvider
):
    """
    Transduce a simulation output (i.e. a signals array of layout (n_species, n_tr))
    into a list of semantic segments that hold the information of the sequence.
    """
    if not signals.ndim == 2:
        raise ValueError(
            f'expected 2D (n_species, n_tr) signals tensor, but got shape {signals.shape}'
        )
    semantic_sections: list[list[SignalSection] | PreparationSection | InversionSection] = []
    time_offset: float = 0
    index_offset: int = 0
    for element in sequence.elements:
        if isinstance(element, pfisp.ModulationSegment):
            timepoints = jnp.cumsum(element.TR) + time_offset
            n_tr: int = len(element.fa)
            assert n_tr == len(timepoints), f'timepoints ({len(timepoints)}) and fa ({n_tr}) length mismatch'
            # subselect the signals along the temporal dimension and expand
            # into semantic segments for every relaxometric species
            section = SignalSection(
                timepoints=timepoints,
                signal=signals[:, index_offset:index_offset+n_tr],
                parameters=RelaxationParameters.from_provider(parameters)
            )
            semantic_sections.append(section)
            time_offset += timepoints[-1]
            index_offset += n_tr
            
        elif isinstance(element, pfisp.InversionModule):
            module_duration = element.TI
            segment = InversionSection(
                timepoints=jnp.array([time_offset, time_offset+module_duration]),
            )
            time_offset += module_duration
            semantic_sections.append(segment)
        
        elif isinstance(element, pfisp.PreparationModule):
            module_duration = element.delay + element.preptime
            segment = PreparationSection(
                timepoints=jnp.array([time_offset, time_offset+module_duration]),
            )
            time_offset += module_duration
            semantic_sections.append(segment)
            
    return semantic_sections


def get_complex_component(
    data: ArrayLike,
    complex_component: Literal['real', 'imag'] = 'imag'
) -> ArrayLike:
    """
    Retrieve the indicated complex component of the data structure.
    Returns the data itself if the input data does not define the requested
    component as an attribute.
    """
    if complex_component not in {'real', 'imag'}:
        raise ValueError(
            f'invalid complex component \'{complex_component}\':'
            f' must be either \'real\' or \'imag\''\
    )
    try:
        return getattr(data, complex_component)
    except AttributeError:
        pass
    return data


def create_signals_analysisplot(
    signals: jax.Array,
    sequence: pfisp.PFISP,
    statics: pfisp.Statics,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    complex_component: Literal['real', 'imag'] = 'imag'
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    for seg, signal in enumerate(signals):
        
        #element = sequence.elements[sigindex]
        
        component = getattr(signal, complex_component)
        ax.plot(component)
    
    if figsize:
        fig.set_size_inches(figsize)
    return (fig, ax)




def compute_encoding_strength(signals: Array):
    absolute_differences = jnp.abs(
        signals[np.newaxis, :, :] - signals[:, np.newaxis, :]
    )
    return jnp.mean(absolute_differences)



def evaluate_T2_encoding(
    signals: Array,
    parameters: RelaxationParameters,
):
    # group by similar T1 species
    T1_groups = jnp.unique(parameters.T1)

    for T1 in T1_groups:
        # select signals for the current T1 group
        mask = jnp.isclose(parameters.T1, T1)
        signals_T1 = signals[mask]

        # compute encoding strength for the current T1 group
        encoding_strength = compute_encoding_strength(signals_T1)

        # do something with the encoding strength, e.g., store it
        print(f"Encoding strength for T1={T1}: {encoding_strength}")



def evaluate_T1_encoding(
    signals: Array,
    parameters: RelaxationParameters,
):
    pass






@attrs.define
class Range:
    """Represents a range with lower and upper bounds."""
    lower: float = attrs.field()
    upper: float = attrs.field()
    
    def __attrs_post_init__(self):
        if self.lower > self.upper:
            raise ValueError(
                f'violation of (lower) {self.lower} < {self.upper} (upper) bound for range'
            )
        
    @classmethod
    def from_array(cls, array: ArrayLike) -> 'Range':
        return cls(
            lower=float(np.min(array)),
            upper=float(np.max(array))
        )


@attrs.define
class ComplexRange:
    """
    Represents a dynamic range of signals array-like data that contains
    complex components (real and imaginary parts).
    """
    imag: Range
    real: Range
    
    @property
    def lower(self) -> float:
        return min(self.imag.lower, self.real.lower)
    
    @property
    def upper(self) -> float:
        return max(self.imag.upper, self.real.upper)
    
    @classmethod
    def from_array(cls, array: ArrayLike) -> 'ComplexRange':
        """
        Create a ComplexRange instance from an array-like data structure
        that contains complex components (real and imaginary parts).
        If the data is not complex, the imaginary part is set to zero.
        """
        real_range = Range.from_array(array.real) if hasattr(array, 'real') else Range.from_array(array)
        imag_range = Range.from_array(array.imag) if hasattr(array, 'imag') else Range(0.0, 0.0)
        return cls(real=real_range, imag=imag_range)


class Grouping(NamedTuple):
    unique_value: float
    indices: np.ndarray
    values: np.ndarray


def compute_fuzzy_grouping_by_uniques(
    dictionary: ArrayLike,
    parameter_column_index: int,
    *,
    atol: float = 1.5
) -> list[Grouping]:
    """
    Group rows of a 2D array-like structure by the unqiue values in the specified
    ``parameter_column_index`` column.
    Grouping is 'fuzzy' in the sense that it allows for a tolerance level (``atol``)
    when comparing the unique values in the parameter column.
    """
    if dictionary.ndim != 2:
        raise ValueError(
            f'expecting 2D parameter array, but got ndim = {dictionary.ndim}'
        )
    parameter_column = dictionary[:, parameter_column_index]
    uniques = np.unique(parameter_column)
    # we want to check that the dictionary elements grouped with a certain
    # unique value in the parameter column are not selected multiple times
    cumulative_mask = np.full_like(parameter_column, fill_value=False, dtype=bool)
    groupings: list[tuple[float, np.ndarray]] = []
    for unique_value in uniques:
        mask = np.isclose(parameter_column, unique_value, atol=atol)

        if np.any(mask & cumulative_mask):
            overlap = mask & cumulative_mask
            raise ValueError(
                f'multi-selection of grouped elements for unique value {unique_value} '
                f'at row indices {np.nonzero(overlap)}'
            )
        else:
            cumulative_mask = cumulative_mask | mask

        grouping = Grouping(
            unique_value=unique_value,
            indices=np.nonzero(mask),
            values=dictionary[mask]
        )
        groupings.append(grouping)

    return groupings



def group_by_uniques(
    dictionary: ArrayLike,
    data: ArrayLike,
    parameter_column_index: int
) -> tuple[float, ArrayLike, ArrayLike]:
    """
    Group the dictionary and data arrays by the unique values in the
    specified ``parameter_column_index`` column of the dictionary.

    Parameters
    ----------
    dictionary : ArrayLike
        A 2D array-like structure where each row represents a set of parameters.

    data : ArrayLike
        A 2D array-like structure where each row corresponds to a signal or data point

    parameter_column_index : int
        The index of the column in the dictionary that contains the parameter values
        used for grouping.

    Returns
    -------
    tuple[float, ArrayLike, ArrayLike]
        A tuple containing:
        - uniques: An array of unique parameter values from the specified column.
        - grouped_dictionary: A 2D array-like structure where each row corresponds to
          the parameters grouped by the unique values.
        - grouped_data: A 2D array-like structure where each row corresponds to the
          data points grouped by the unique values.
    """
    if data.ndim != 2:
        raise ValueError(
            f'expecting 2D data array, but got ndim = {data.ndim}'
        )

    groupings = compute_fuzzy_grouping_by_uniques(
        dictionary=dictionary,
        parameter_column_index=parameter_column_index
    )

    uniques: list[float] = []
    grouped_dictionary: list[ArrayLike] = []
    grouped_data: list[ArrayLike] = []

    for grouping in groupings:
        uniques.append(grouping.unique_value)
        grouped_dictionary.append(grouping.values)
        grouped_data.append(data[grouping.indices])
    
    grouped_dictionary = jnp.stack(grouped_dictionary, axis=0)
    grouped_data = jnp.stack(grouped_data, axis=0)
    uniques = jnp.array(uniques)

    return (uniques, grouped_dictionary, grouped_data)

