"""
Implement timing data classes for analysis of EPG sequences in JAX.
"""
import copy
from collections.abc import Sequence
from typing import NamedTuple, TypeAlias

import numpy as np

from jax.typing import ArrayLike
from attrs import define

import mrfoptools.epg.sequences.jax.pfisp2 as pfisp
from mrfoptools.epg.sequences.jax.analysis import group_by_uniques, RelaxationParameters

@define
class InversionModuleTiming:
    """
    Timing data for the inversion module in EPG sequences.
    """
    points: ArrayLike
    TI: float
    total_duration: float

    @classmethod
    def create_from_offset(
        cls,
        offset: float,
        TI: float,
    ) -> 'InversionModuleTiming':
        points = offset + np.array([0, TI])
        return cls(points=points, TI=TI, total_duration=TI)


@define
class PreparationModuleTiming:
    """
    Timing data for the preparation module in EPG sequences.
    """
    points: ArrayLike
    delay: float
    preptime: float
    total_duration: float

    @classmethod
    def create_from_offset(
        cls,
        offset: float,
        delay: float,
        preptime: float,
    ) -> 'PreparationModuleTiming':
        points = offset + np.array([0, delay, delay + preptime])
        return cls(points=points, delay=delay, preptime=preptime, total_duration=preptime + delay)
        
        

@define
class ModulationSegmentTiming:
    """
    Timing data for the modulation segment in EPG sequences.
    """
    points: ArrayLike
    total_duration: float

    @classmethod
    def create_from_offset(
        cls,
        offset: float,
        TR: ArrayLike,
    ) -> 'ModulationSegmentTiming':
        points = offset + np.cumulative_sum(TR)
        return cls(points=points, total_duration=np.sum(TR))



Timing: TypeAlias = InversionModuleTiming | PreparationModuleTiming | ModulationSegmentTiming


class TimedSignals(NamedTuple):
    """
    Container for signal information of typical layout (n_species, n_tr)
    with associated temporal points.
    """
    t: ArrayLike
    signals: ArrayLike


@define
class SignalSection:
    """
    Container for a section of signal information
    with associated temporal information.
    """
    timepoints: ArrayLike
    signals: ArrayLike
    parameters: pfisp.RelaxationParameters

    def group_signals_by(self, parameter_index: int) -> 'SignalSection':
        dictionary = np.stack([self.parameters.T1, self.parameters.T2], axis=1)
        _, grpd_dictionary, grpd_signals = group_by_uniques(
            dictionary=dictionary,
            data=self.signals,
            parameter_column_index=parameter_index
        )
        parameters = RelaxationParameters(
            T1=grpd_dictionary[..., 0],
            T2=grpd_dictionary[..., 1],
            M0=self.parameters.M0,
        )
        return SignalSection(
            timepoints=copy.deepcopy(self.timepoints),
            signals=grpd_signals,
            parameters=parameters
        )
        


@define
class SignalsSections:
    signals: ArrayLike
    timings: list[ModulationSegmentTiming]
    parameters: pfisp.RelaxationParameters

    @classmethod
    def create_from(
        cls,
        sequence: Sequence[pfisp.SequenceSegment],
        signals: ArrayLike,
        parameters: pfisp.RelaxationParameters,
    ) -> 'SignalsSections':
        """
        Create the signals sections from the multi-species signals array
        and the sequence given as segments.
        """
        timings = [
            timing for timing in parse(sequence)
            if isinstance(timing, ModulationSegmentTiming)
        ]
        return cls(
            signals=signals,
            timings=timings,
            parameters=parameters
        )


    def split(self) -> list[SignalSection]:
        """
        Split the signals into the timed segments as defined by the
        sequence definition.
        """
        result: list[SignalSection] = []
        offset: int = 0
        for timing in self.timings:
            n_tr = len(timing.points)
            section = SignalSection(
                timepoints=timing.points,
                signals=self.signals[:, offset:offset + n_tr],
                parameters=self.parameters
            )
            result.append(section)
            offset += n_tr
        return result



def bulk_group_signals_by(
    sections: list[SignalSection],
    parameter_index: int = 0
) -> list[SignalSection]:
    """
    Group the signals in each section by the specified parameter index.
    This is useful for analyzing multiple species with different T1, T2 values.
    """
    return [section.group_signals_by(parameter_index) for section in sections]


def parse(sequence: Sequence[pfisp.SequenceSegment]) -> list[Timing]:
    """
    Parse the sequence into a list of timing objects.
    """
    timings: list = []
    offset: float = 0.0
    
    for segment in sequence:
        if isinstance(segment, pfisp.InversionModule):
            timing = InversionModuleTiming.create_from_offset(
                offset=offset,
                TI=segment.TI
            )
            timings.append(timing)
        elif isinstance(segment, pfisp.PreparationModule):
            timing = PreparationModuleTiming.create_from_offset(
                offset=offset,
                delay=segment.delay,
                preptime=segment.preptime
            )
            timings.append(timing)
        elif isinstance(segment, pfisp.ModulationSegment):
            timing = ModulationSegmentTiming.create_from_offset(
                offset=offset,
                TR=segment.TR
            )
            timings.append(timing)
        else:
            raise ValueError(f'Unknown segment type: {type(segment)}')
        offset += timing.total_duration
    
    return timings