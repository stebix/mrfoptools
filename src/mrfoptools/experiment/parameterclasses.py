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

from mrfoptools.initialization.initialization import InitializationType

Array: TypeAlias = jax.Array | np.ndarray

@attrs.define
class Protocol:
    """
    Core protocol parameters for the MR Fingerprinting optimization
    experiment.
    The species are generated from the flat combinations of the
    T1 and T2 candidates.
    """
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
class Initializations:
    fa: Array
    tr: Array
    phases: Array
    seed: int
    type_: InitializationType
    fa_value: float | None = None
    tr_value: float | None = None

    def __attrs_post_init__(self) -> None:
        if self.type_ in (InitializationType.CONSTANT, InitializationType.CONSTANT_PERTURBED):
            if self.fa_value is None:
                raise ValueError('`fa_value` must be specified for constant initializations.')
            if self.tr_value is None:
                raise ValueError('`tr_value` must be specified for constant initializations.')


@attrs.define
class BathtubLossParameters:
    radius: float
    alpha: float
    beta: float
    gamma: float


@attrs.define
class Hyperparameter:
    step_size: float
    max_iterations: int
    min_fa: float
    max_fa: float
    bathtub_loss_parameters: BathtubLossParameters