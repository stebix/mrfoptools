"""
Tooling to covneniently create initializations for the 
flip angle and repetition time optimization problem.

@author: Jannik Stebani 2025
"""
from collections.abc import Sequence
from numbers import Number
from typing import Literal, NamedTuple
from pathlib import Path

import numpy as np


CAO_FA_PATTERN_PATH: Path = Path(__file__).parent / 'assets/fa_cao.npy'
CAO_TR_PATTERN_PATH: Path = Path(__file__).parent / 'assets/tr_cao.npy'


class Pattern(NamedTuple):
    flip_angles: np.ndarray
    repetition_times: np.ndarray


def create_sinusoidal_pattern(
        amplitudes: Sequence[Number],
        segment_size: int,
) -> np.ndarray:
    """
    Create a Yun-like sinusoidal flip angle pattern.
    Pattern consists of multiple sinus segments with specified amplitudes.
    Total length is `len(amplitudes) * segment_size`.

    Parameters
    ----------

    amplitudes: Sequence[Number]
        Amplitudes of the sinusoidal segments.
        Unit (e.g. radians or degrees) of amplitudes determines
        the unit of the output pattern.

    segment_size: int
        Number of samples per segment.

    Returns
    -------

    np.ndarray
        Sinusoidal flip angle pattern.
    """
    segments = []
    for amplitude in amplitudes:
        segments.append(
            amplitude * np.sin(np.linspace(0, np.pi, segment_size))
        )
    return np.concatenate(segments, axis=-1)


def load_cao_pattern(type_: Literal['fa', 'tr', 'both'] = 'fa') -> np.ndarray | Pattern:
    """
    Load the predefined Cao pattern.
    Flip angle and repetition time patterns are available.
    """
    if type_ == 'fa':
        return np.load(CAO_FA_PATTERN_PATH)
    elif type_ == 'tr':
        return np.load(CAO_TR_PATTERN_PATH)

    return Pattern(
        flip_angles=np.load(CAO_FA_PATTERN_PATH),
        repetition_times=np.load(CAO_TR_PATTERN_PATH)
    )
