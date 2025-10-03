"""
Optimize for a FISP sequence with preparation pulses.
"""
import itertools

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, NamedTuple, TypeAlias


import jax
import jax.numpy as jnp
import equinox as eqx

import mrfoptools.epg.core.jax as epgjax
import mrfoptools.epg.signal.jax as epgsig
import mrfoptools.epg.signal.jax.preparations as jaxprep

Array: TypeAlias= jax.Array


class Element(Protocol):
    pass


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



class ModulationSegment(eqx.Module):
    """
    Represents segment in the sequence where the magnetization is modulated
    via a regular sequence of RF pulses.
    """
    fa: Array
    TR: Array
    phases: Array



class PreparationModule(eqx.Module):
    """
    Represents segment in the sequence where the magnetization is prepared
    via a preparation pulse.
    """
    delay: float
    preptime: float


class InversionModule(eqx.Module):
    """
    Represents segment in the sequence where the magnetization is inverted
    via an inversion pulse.
    """
    inversion_efficiency: float
    TI: float


class EchoInfo(NamedTuple):
    TE: float
    r_TE: Array
    b_TE: float

class RelaxometricParameters(NamedTuple):
    T1: float
    T2: float
    M0: float



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


def _propagate_legacy(
    carry: tuple[Array, Statics],
    alpha: float,
    TR: float,
    phi: float,
) -> tuple[tuple[Array, Statics], complex]:
    """
    Propagate the EPG state through one transient step with
    a given flip angle, TR time and phase.
    """
    omega, statics = carry
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
    


def synthesize_parameters_from(
    segment: ModulationSegment,
    relaxometric_params: RelaxometricParameters,
    echoinfo: EchoInfo,
) -> Parameters:
    """
    Synthesize the parameters tuple from the disparate information parts
    segment, relaxometric parameters and echo information.
    """
    transients = Transients(
        fa=segment.fa,
        TR=segment.TR,
        phases=segment.phases
    )
    statics = Statics(
        T1=relaxometric_params.T1,
        T2=relaxometric_params.T2,
        M0=relaxometric_params.M0,
        TE=echoinfo.TE,
        r_TE=epgjax.r_epg(relaxometric_params.T1, relaxometric_params.T2, echoinfo.TE),
        b_TE=epgjax.b_epg(relaxometric_params.T1, echoinfo.TE)
    )
    return Parameters(transients=transients, statics=statics)



def simulate_modulation(
    omega: Array,
    element: ModulationSegment,
    statics: Statics,
) -> StateSignal:
    """
    Simulate the effect of a modulation segment on the EPG state
    and compute the resulting signal.
    """
    parameters = Parameters(
        transients=Transients(
            fa=element.fa,
            TR=element.TR,
            phases=element.phases
        ),
        statics=statics
    )
    statesig = propagate(omega, parameters)
    return statesig



def simulate_preparation(
    omega: Array,
    element: PreparationModule,
    statics: Statics,
) -> StateSignal:
    """
    Simulate the effect of a T2 preparation module on the EPG state.
    """
    dtype = jnp.complex64
    # free relaxation during the delay time
    delay_operator = epgjax.r_epg(
        T1=statics.T1,
        T2=statics.T2,
        dt=element.delay
    )
    # T1 recovery
    b_delay = epgjax.b_epg(
        T1=statics.T1,
        dt=element.delay
    )
    omega = delay_operator @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + statics.M0 * b_delay)
    # NOTE: here we do the Max Gram assumption to nuke the (0, 1) rows of the EPG
    #       omega state matrix. Thus we also only dampen the (0, 2) row. If we move
    #       to a more general model, we will need to dampen all rows.
    omega = omega.at[2, :].set(
        omega[2, :] * jnp.exp(-element.preptime / statics.T2)
    )
    omega = omega.at[0, :].set(0.0)  # Nuke the first row to zero
    omega = omega.at[1, :].set(0.0)  # Nuke even moar
    return StateSignal(state=omega, signal=jnp.array([], dtype=dtype))



def simulate_inversion(
    omega: Array,
    element: InversionModule,
    statics: Statics,
) -> StateSignal:
    """
    Simulate the effect of an inversion module on the EPG state.
    """
    dtype = jnp.complex64
    inversion_operator = epgjax.inversion(
        inversion_efficiency=element.inversion_efficiency
    )
    omega = epgjax.r_epg(statics.T1,
                         statics.T2,
                         element.TI) @ inversion_operator @ omega
    omega = omega.at[2, 0].set(
        omega[2, 0] + statics.M0 * epgjax.b_epg(statics.T1, element.TI)
    )
    return StateSignal(state=omega, signal=jnp.array([], dtype=dtype))


def get_simulate_func(
    element: ModulationSegment | PreparationModule | InversionModule
):
    if isinstance(element, ModulationSegment):
        return simulate_modulation
    elif isinstance(element, PreparationModule):
        return simulate_preparation
    elif isinstance(element, InversionModule):
        return simulate_inversion
    else:
        raise ValueError('Unknown sequence element type')


def simulate(
    pfisp: PFISP,
    omega: Array,
    statics: Statics,
) -> StateSignal:
    
    elements_and_funcs = tuple(
        (element, get_simulate_func(element)) for element in pfisp.elements
    )

    def _propagate(
        carry: tuple[Array, Statics],
        element_and_func: tuple[ModulationSegment | PreparationModule | InversionModule, callable]
    ):
        element, simulate_func = element_and_func
        statesig = simulate_func(omega, element, statics)
        carry = (statesig.state, statics)
        return (carry, statesig.signal)
    
    carry = (omega, statics)
    omega, signal = jax.lax.scan(
        f=_propagate,
        init=carry,
        xs=elements_and_funcs
    )
    return StateSignal(state=omega, signal=signal)



if __name__ == '__main__':

    fa = jnp.sin
    

    segment = ModulationSegment(
        fa=jnp.array([10.0, 20.0, 30.0]),
        TR=jnp.array([1000.0, 1000.0, 1000.0]),
        phases=jnp.array([0.0, 90.0, 180.0])
    )

    import jax.tree_util as jtu

    mask = jtu.tree_map(lambda x: False , segment)

    mask = eqx.tree_at(
        where=lambda x: x.fa,
        pytree=mask,
        replace=True
    )
    

    print('segment is:: ')
    print(segment)

    print('mask is:: ')
    print(mask)