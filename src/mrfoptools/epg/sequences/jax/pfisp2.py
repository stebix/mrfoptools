"""
Optimize for a FISP sequence with preparation pulses.
"""
import itertools

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, NamedTuple, TypeAlias


import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import mrfoptools.epg.core.jax as epgjax
import mrfoptools.epg.signal.jax as epgsig
import mrfoptools.epg.signal.jax.preparations as jaxprep

Array: TypeAlias= jax.Array


class Transients(NamedTuple):
    """
    Transient parameters of the propagation simulation.
    """
    fa: Array
    TR: Array
    phases: Array



class Statics(NamedTuple):
    """
    Static parameters of the propagation simulation.
    """
    T1: float
    T2: float
    M0: float
    TE: float
    r_TE: Array
    b_TE: float


class Parameters(NamedTuple):
    transients: Transients
    statics: Statics


class StateSignal(NamedTuple):
    """
    Represents the state and signal of the EPG sequence.
    
    Attributes
    ----------
    
    state : array
        EPG state, i.e. the omega matrix.
        
    signal : array
        Signal computed from the EPG state.
    """
    state: Array
    signal: Array



def _propagate(
    carry: tuple[Array, Statics],
    transients: tuple[float, float, float]
) -> tuple[tuple[Array, Statics], complex]:
    """
    Propagate the EPG state through one transient step with
    a given the transients values (flip angle, TR time, phase).
    """
    omega, statics = carry
    alpha, TR, phi = transients
    omega = statics.r_TE @ epgjax.q_epg(alpha, phi) @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + statics.M0 * statics.b_TE)
    signal = omega[0, 0] * jnp.exp(1j * phi)
    omega = epgjax.unit_grad_shift_static(
        epgjax.r_epg(statics.T1, statics.T2, TR - statics.TE) @ omega 
    )
    omega = omega.at[2, 0].set(
        omega[2, 0] + statics.M0 * epgjax.b_epg(statics.T1, TR - statics.TE)
    )
    return ((omega, statics), signal)


def propagate(
    omega: Array,
    parameters: Parameters
) -> StateSignal:
    carry = (omega, parameters.statics)
    (omega, _), signal = jax.lax.scan(
        f=_propagate,
        init=carry,
        xs=parameters.transients
    )
    return StateSignal(state=omega, signal=signal)


class SequenceSegment(eqx.Module):
    pass

    def __call__(self, omega: Array, statics: Statics) -> StateSignal:
        """
        Propagate the EPG state through this segment and compute
        the corresponding signal.
        
        Parameters
        ----------
        omega : Array
            The EPG state matrix.
        
        statics : Statics
            Static parameters of the sequence.
        
        Returns
        -------
        StateSignal
            The propagated state and computed signal.
        """
        raise NotImplementedError('Subclasses must implement __call__ method')



class ModulationSegment(SequenceSegment):
    """
    Represents segment in the sequence where the magnetization is modulated
    via a regular sequence of RF pulses.
    """
    fa: Array
    TR: Array
    phases: Array

    def __call__(self, omega: Array, statics: Statics) -> StateSignal:
        parameters = Parameters(
            transients=Transients(
                fa=self.fa,
                TR=self.TR,
                phases=self.phases
            ),
            statics=statics
        )
        return propagate(omega, parameters)



class PreparationModule(SequenceSegment):
    """
    Represents segment in the sequence where the magnetization is prepared
    via a preparation pulse.
    """
    delay: float
    preptime: float

    def __call__(self, omega: Array, statics: Statics) -> StateSignal:
        dtype = jnp.complex64
        # free relaxation during the delay time
        delay_operator = epgjax.r_epg(
            T1=statics.T1,
            T2=statics.T2,
            dt=self.delay
        )
        # T1 recovery
        b_delay = epgjax.b_epg(
            T1=statics.T1,
            dt=self.delay
        )
        omega = delay_operator @ omega
        omega = omega.at[2, 0].set(omega[2, 0] + statics.M0 * b_delay)
        # NOTE: here we do the Max Gram assumption to nuke the (0, 1) rows of the EPG
        #       omega state matrix. Thus we also only dampen the (0, 2) row. If we move
        #       to a more general model, we will need to dampen all rows.
        omega = omega.at[2, :].set(
            omega[2, :] * jnp.exp(-self.preptime / statics.T2)
        )
        omega = omega.at[0, :].set(0.0)  # Nuke the first row to zero
        omega = omega.at[1, :].set(0.0)  # Nuke even moar
        return StateSignal(state=omega, signal=jnp.array([], dtype=dtype))



class InversionModule(SequenceSegment):
    """
    Represents segment in the sequence where the magnetization is inverted
    via an inversion pulse.
    """
    inversion_efficiency: float
    TI: float

    def __call__(self, omega: Array, statics: Statics) -> StateSignal:
        dtype = jnp.complex64
        inversion_operator = epgjax.inversion(
            inversion_efficiency=self.inversion_efficiency
        )
        omega = epgjax.r_epg(statics.T1,
                             statics.T2,
                             self.TI) @ inversion_operator @ omega
        omega = omega.at[2, 0].set(
            omega[2, 0] + statics.M0 * epgjax.b_epg(statics.T1, self.TI)
        )
        return StateSignal(state=omega, signal=jnp.array([], dtype=dtype))




class PFISP(eqx.Module):
    elements: list[ModulationSegment | PreparationModule | InversionModule]
    

class Transients(NamedTuple):
    """
    Transient parameters of the propagation simulation.
    """
    fa: Array
    TR: Array
    phases: Array






class Statics(NamedTuple):
    """
    Static parameters of the propagation simulation.
    Can hold single parameter set or multiple sets for
    multiple (T1, T2) pairs.
    """
    T1: float | Array
    T2: float | Array
    M0: float
    TE: float
    r_TE: Array
    b_TE: float | Array

    @classmethod
    def from_sequence(
        cls,
        T1: Sequence[float],
        T2: Sequence[float],
        M0: float,
        TE: float
    ) -> 'Statics':
        """
        Create a Statics instance for multiple (T1, T2) species.
        """
        b_TE = []
        r_TE = []
        for T1value, T2value in zip(T1, T2, strict=True):
            b_TE.append(epgjax.b_epg(T1=T1value, dt=TE))
            r_TE.append(epgjax.r_epg(T1=T1value, T2=T2value, dt=TE))
        
        return cls(
            T1=jnp.array(T1),
            T2=jnp.array(T2),
            M0=M0,
            TE=TE,
            b_TE=jnp.stack(b_TE, axis=0),
            r_TE=jnp.stack(r_TE, axis=0)
        )


def simulate_scan(
    pfisp: PFISP,
    omega: Array,
    statics: Statics,
) -> StateSignal:
    """
    Simulate the (prepared) FISP sequence consisting of differing modules
    and segments using the `jax.lax.scan` primitive.
    """
    def _propagate(
        carry: tuple[Array, list, Statics],
        i: int
    ):
        omega, elements, statics = carry
        # select the current segment/module of the sequence
        # and simulate its effect on the EPG state
        statesig = elements[i](omega, statics)
        carry = (statesig.state, statics)
        return (carry, statesig.signal)
    
    carry = (omega, pfisp.elements, statics)
    omega, signal = jax.lax.scan(
        f=_propagate,
        init=carry,
        xs=jnp.arange(len(pfisp.elements))
    )
    return StateSignal(state=omega, signal=signal)



def simulate_pyloop(
    pfisp: PFISP,
    omega: Array,
    statics: Statics,
) -> StateSignal:
    """
    Simulate the (prepared) FISP sequence consisting of differing modules
    and segments using a python loop construct.
    """
    signal: list[Array] = []
    
    for element in pfisp.elements:
        (omega, signalsection) = element(omega, statics)
        signal.append(signalsection)

    return StateSignal(state=omega, signal=jnp.concatenate(signal, axis=0))




def evaluate_sequence(
    sequence: PFISP,
    signals: Array,
    statics: Statics
):
    pass


class RelaxationParameters(Protocol):
    """
    Protocol for relaxation parameters.
    """
    T1: Array
    T2: Array



