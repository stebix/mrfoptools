"""
Tooling to covneniently create initializations for the 
flip angle and repetition time optimization problem.

Patterns directly sourced or inspired by:
Cao Pattern:     Xaiozhi Cao et al. https://doi.org/10.1002/mrm.29194
Yun Pattern:     Yun Jiang et al.   https://doi.org/10.1002/mrm.25559

@author: Jannik Stebani 2025
"""
import enum

from collections.abc import Sequence
from numbers import Number
from typing import Literal, NamedTuple
from pathlib import Path

import numpy as np


CAO_FA_PATTERN_PATH: Path = Path(__file__).parent / 'assets/fa_cao.npy'
CAO_TR_PATTERN_PATH: Path = Path(__file__).parent / 'assets/tr_cao.npy'

YUN_FA_PATTERN_PATH: Path = Path(__file__).parent / 'assets/fa_yun.npy'
YUN_TR_PATTERN_PATH: Path = Path(__file__).parent / 'assets/tr_yun.npy'

# Raw Yun pattern amplitudes in degrees: Can be used to create pattern
# without the zero-parts in between the segments.
YUN_FA_PATTERN_AMPLITUDES: np.ndarray = np.array([35, 43, 70, 45, 27])

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



def load_yun_pattern(
        style: Literal['canonical', 'tight'] = 'canonical',
        element: Literal['fa', 'tr', 'both'] = 'fa') -> np.ndarray | Pattern:
    """
    Load the predefined Yun pattern (e.g. flip angle and/or repetition times)
    The FA pattern can be loaded in canonical (small segments where FA = 0 in between)
    or tight (no zero segments) style.

    Parameters
    ----------
    style: Literal['canonical', 'tight']
        Style of the pattern.

    element: Literal['fa', 'tr', 'both']
        Element(s) of the pattern to load.

    Returns
    -------
    np.ndarray | Pattern
        Loaded pattern(s).
    """
    dtype: type = np.float32
    segment_size: int = 200

    if element == 'fa' and style == 'canonical':
        return np.load(YUN_FA_PATTERN_PATH).astype(dtype)
    elif element == 'fa' and style == 'tight':
        return create_sinusoidal_pattern(
            YUN_FA_PATTERN_AMPLITUDES, segment_size=segment_size
        ).astype(dtype)
    elif element == 'tr':
        return np.load(YUN_TR_PATTERN_PATH).astype(dtype)
    elif element == 'both' and style == 'canonical':
        return Pattern(
            flip_angles=np.load(YUN_FA_PATTERN_PATH).astype(dtype),
            repetition_times=np.load(YUN_TR_PATTERN_PATH).astype(dtype)
        )
    elif element == 'both' and style == 'tight':
        return Pattern(
            flip_angles=create_sinusoidal_pattern(
                YUN_FA_PATTERN_AMPLITUDES, segment_size=segment_size
            ).astype(dtype),
            repetition_times=np.load(YUN_TR_PATTERN_PATH).astype(dtype)
        )
    else:
        raise ValueError('Invalid combination of style and element.')



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


class InitializationType(enum.Enum):
    CONSTANT = 'constant'
    CONSTANT_PERTURBED = 'constant_perturbed'
    YUN_CANONICAL = 'yun_canonical'
    YUN_TIGHT = 'yun_tight'
    ARBITRARY = 'arbitrary'

