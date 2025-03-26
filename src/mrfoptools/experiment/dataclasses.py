"""
Encapsulate and soft-validate the input parameters utilized for
a typical gradient-based optimization experiment.

@Author: Jannik Stebani 2025
"""
import attrs

from typing import TypeAlias
from collections.abc import Sequence
from numbers import Number

import jax
import numpy as np

Array: TypeAlias = jax.Array | np.ndarray

@attrs.define
class Protocol:
    M0: float
    T1: Sequence[Number] | Array
    T2: Sequence[Number] | Array
    TE: float
    TI: float
    phase: float
    inversion_efficiency: float
    max_states: int
    NR: int


@attrs.define
class Initialization:
    fa: Array
    tr: Array
    seed: int


@attrs.define
class Hyperparameter:
    step_size: float
    max_iterations: int
    min_fa: float
    max_fa: float