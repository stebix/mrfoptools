"""
Tooling to covneniently create initializations for the 
flip angle and repetition time optimization problem.

Patterns directly sourced or inspired by:
Cao Pattern:     Xaiozhi Cao et al. https://doi.org/10.1002/mrm.29194
Yun Pattern:     Yun Jiang et al.   https://doi.org/10.1002/mrm.25559

@author: Jannik Stebani 2025
"""
from collections.abc import Sequence
from numbers import Number
from typing import Literal, NamedTuple
from pathlib import Path

import numpy as np


CAO_FA_PATTERN_PATH: Path = Path(__file__).parent / 'assets/fa_cao.npy'
CAO_TR_PATTERN_PATH: Path = Path(__file__).parent / 'assets/tr_cao.npy'

YUN_FA_PATTERN_PATH: Path = Path(__file__).parent / 'assets/fa_yun.npy'
YUN_TR_PATTERN_PATH: Path = Path(__file__).parent / 'assets/tr_yun.npy'


class Pattern(NamedTuple):
    flip_angles: np.ndarray
    repetition_times: np.ndarray


def create_sinusoidal_pattern(
        amplitudes: Sequence[Number],
        segment_size: int,
) -> np.ndarray:
    """
    Create a Yun-like sinusoidal pattern.
    Pattern consists of multiple sinus segments with specified amplitudes.
    Total length is `len(amplitudes) * segment_size`.

    Parameters
    ----------

    amplitudes: Sequence[Number]
        Amplitudes of the sinusoidal segments.
        Unit (e.g. radians or degrees for flip angles or milliseconds for repetition times)
        of amplitudes determines the unit of the output pattern.

    segment_size: int
        Number of samples per segment.

    Returns
    -------

    np.ndarray
        Resulting sinusoidal-shaped pattern.
    """
    segments = []
    for amplitude in amplitudes:
        segments.append(
            amplitude * np.sin(np.linspace(0, np.pi, segment_size))
        )
    return np.concatenate(segments, axis=-1)


def create_constant_pattern(amplitude: Number, length: int) -> np.ndarray:
    """
    Create a constant pattern.
    Input unit of the amplitude determines the unit of the output pattern.
    """
    dtype: type = np.float32
    return np.full(shape=length, fill_value=amplitude, dtype=dtype)



def load_yun_pattern(type_: Literal['fa', 'tr', 'both'] = 'fa') -> np.ndarray | Pattern:
    """
    Load the predefined Yun pattern.
    Flip angle and repetition time patterns are available.
    """
    if type_ == 'fa':
        return np.load(YUN_FA_PATTERN_PATH)
    elif type_ == 'tr':
        return np.load(YUN_TR_PATTERN_PATH)

    return Pattern(
        flip_angles=np.load(YUN_FA_PATTERN_PATH),
        repetition_times=np.load(YUN_TR_PATTERN_PATH)
    )



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
