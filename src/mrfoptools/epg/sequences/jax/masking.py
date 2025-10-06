import enum
import jax
import equinox as eqx

from typing import Any, TypeAlias

from mrfoptools.epg.sequences.jax import pfisp2 as pfisp

Node: TypeAlias = Any


class SequenceVariable(enum.Enum):
    """
    MRI sequence elements that can be interpreted as optimization variables.
    """
    REPTIMES = 'reptimes'
    FLIPANGLES = 'flipangles'
    DELAYS = 'delays'
    PREPTIMES = 'preptimes'
    INVTIMES = 'invtimes'


def collect_reptimes_leaves(sequence: pfisp.Sequence) -> list[Node]:
    """Collects the repetition time array leaves from a pfisp.Sequence."""
    tr_leaves: list[Node] = []
    for element in sequence.elements:
        if isinstance(element, pfisp.ModulationSegment):
            tr_leaves.append(element.TR)
    return tr_leaves

def collect_flipangle_leaves(sequence: pfisp.Sequence) -> list[Node]:
    """Collects the flip angle array leaves from a pfisp.Sequence."""
    fa_leaves: list[Node] = []
    for element in sequence.elements:
        if isinstance(element, pfisp.ModulationSegment):
            fa_leaves.append(element.fa)
    return fa_leaves

def collect_delay_leaves(sequence: pfisp.Sequence) -> list[Node]:
    """Collects the delay time leaves from a pfisp.Sequence."""
    delay_leaves: list[Node] = []
    for element in sequence.elements:
        if isinstance(element, pfisp.PreparationModule):
            delay_leaves.append(element.delay)
    return delay_leaves

def collect_preptime_leaves(sequence: pfisp.Sequence) -> list[Node]:
    """Collects the preparation time leaves from a pfisp.Sequence."""
    preptime_leaves: list[Node] = []
    for element in sequence.elements:
        if isinstance(element, pfisp.PreparationModule):
            preptime_leaves.append(element.preptime)
    return preptime_leaves

def collect_invtime_leaves(sequence: pfisp.Sequence) -> list[Node]:
    """Collects the inversion time leaves from a pfisp.Sequence."""
    invtime_leaves: list[Node] = []
    for element in sequence.elements:
        if isinstance(element, pfisp.InversionModule):
            invtime_leaves.append(element.TI)
    return invtime_leaves


def make_variable_mask(
    sequence: pfisp.Sequence,
    variables: set[SequenceVariable]
) -> pfisp.Sequence:
    """
    Create a PyTree mask for the given sequence, marking the specified variables as True.
    This is typically used to filter optimizable parameters in downstream routines.
    """
    mask_generators = {
        SequenceVariable.REPTIMES : collect_reptimes_leaves,
        SequenceVariable.FLIPANGLES : collect_flipangle_leaves,
        SequenceVariable.DELAYS : collect_delay_leaves,
        SequenceVariable.PREPTIMES : collect_preptime_leaves,
        SequenceVariable.INVTIMES : collect_invtime_leaves
    }
    optmask = jax.tree.map(lambda _: False, sequence)

    for variable in variables:
        mask_generator = mask_generators[variable]
        optmask = eqx.tree_at(
            mask_generator,
            pytree=optmask,
            replace_fn=lambda _: True
        )

    return optmask
