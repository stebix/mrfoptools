"""
Optimize for a FISP sequence with preparation pulses.
"""
import jax.numpy as jnp

from attrs import define, field
from jax import Array

import mrfoptools.epg.core.jax as epgjax
import mrfoptools.epg.signal.jax as epgsig
import mrfoptools.epg.signal.jax.preparations as jaxprep


def simulate_prepped_fisp(
    fa: Array,
    TR: Array,
    preptimes: Array,
    T1: float,
    T2: float,
    M0: float,
    phases: Array,
    TE: float,
    TI: float,
    inversion_efficiency: float = 1.0,
    max_states: int = 1000,
) -> Array:
    
    b_TE = epgjax.b_epg(T1=T1, dt=TE)
    r_TE = epgjax.r_epg(T1=T1, T2=T2, dt=TE)
    inv_op = epgjax.inversion(inversion_efficiency)

    omega = jaxprep.prepare_inversion_omega(
        T1=T1,
        T2=T2,
        M0=M0,
        TI=TI,
        inversion_operator=inv_op,
        max_states=max_states,
    )

    # 3 * 96 first block
    first_block_start_index: int = 0
    first_block_end_index: int = 288
    fa_first_block = fa[first_block_start_index:first_block_end_index]
    TR_first_block = TR[first_block_start_index:first_block_end_index]
    phases_first_block = phases[first_block_start_index:first_block_end_index]
    omega, signal_first_block = epgsig.compute_signal_and_state(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa_first_block,
        phases=phases_first_block,
        TR=TR_first_block,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )

    prep_op_1 = epgjax.preparation_T2_op(
        T2=T2, T2_preptime=preptimes[0]
    )
    omega = prep_op_1 @ omega

    # 3 * 96 second block
    second_block_start_index: int = 288
    second_block_end_index: int = 576
    fa_second_block = fa[second_block_start_index:second_block_end_index]
    TR_second_block = TR[second_block_start_index:second_block_end_index]
    phases_second_block = phases[second_block_start_index:second_block_end_index]
    omega, signal_second_block = epgsig.compute_signal_and_state(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa_second_block,
        phases=phases_second_block,
        TR=TR_second_block,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )

    prep_op_2 = epgjax.preparation_T2_op(
        T2=T2, T2_preptime=preptimes[1]
    )
    omega = prep_op_2 @ omega

    # 2 * 96 second block
    third_block_start_index: int = 576
    third_block_end_index: int = 768
    fa_third_block = fa[third_block_start_index:third_block_end_index]
    TR_third_block = TR[third_block_start_index:third_block_end_index]
    phases_third_block = phases[third_block_start_index:third_block_end_index]
    omega, signal_third_block = epgsig.compute_signal_and_state(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa_third_block,
        phases=phases_third_block,
        TR=TR_third_block,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )
    signal = jnp.concatenate(
        [
            signal_first_block,
            signal_second_block,
            signal_third_block,
        ],
        axis=0
    )
    return signal



def simulate_prepped_fisp_with_delays(
    fa: Array,
    TR: Array,
    preptimes: Array,
    delays: Array,
    T1: float,
    T2: float,
    M0: float,
    phases: Array,
    TE: float,
    TI: float,
    inversion_efficiency: float = 1.0,
    max_states: int = 1000,
) -> Array:
    
    b_TE = epgjax.b_epg(T1=T1, dt=TE)
    r_TE = epgjax.r_epg(T1=T1, T2=T2, dt=TE)
    inv_op = epgjax.inversion(inversion_efficiency)

    omega = jaxprep.prepare_inversion_omega(
        T1=T1,
        T2=T2,
        M0=M0,
        TI=TI,
        inversion_operator=inv_op,
        max_states=max_states,
    )

    # 3 * 96 first block
    first_block_start_index: int = 0
    first_block_end_index: int = 288
    fa_first_block = fa[first_block_start_index:first_block_end_index]
    TR_first_block = TR[first_block_start_index:first_block_end_index]
    phases_first_block = phases[first_block_start_index:first_block_end_index]
    omega, signal_first_block = epgsig.compute_signal_and_state(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa_first_block,
        phases=phases_first_block,
        TR=TR_first_block,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )

    delay_op_1 = epgjax.r_epg(T1=T1, T2=T2, dt=delays[0])
    prep_op_1 = epgjax.preparation_T2_op(
        T2=T2, T2_preptime=preptimes[0]
    )
    # relaxation then preparation
    omega = delay_op_1 @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + M0 * epgjax.b_epg(T1, delays[0]))
    omega = omega * jnp.exp(-preptimes[0] / T2)
    omega = omega.at[0, :].set(0.0)  # Reset the first row to zero
    omega = omega.at[1, :].set(0.0)  # Reset the second row to zero


    # 3 * 96 second block
    second_block_start_index: int = 288
    second_block_end_index: int = 576
    fa_second_block = fa[second_block_start_index:second_block_end_index]
    TR_second_block = TR[second_block_start_index:second_block_end_index]
    phases_second_block = phases[second_block_start_index:second_block_end_index]
    omega, signal_second_block = epgsig.compute_signal_and_state(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa_second_block,
        phases=phases_second_block,
        TR=TR_second_block,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )

    delay_op_2 = epgjax.r_epg(T1=T1, T2=T2, dt=delays[1])
    prep_op_2 = epgjax.preparation_T2_op(
        T2=T2, T2_preptime=preptimes[1]
    )
    print(omega.shape)
    omega = delay_op_2 @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + M0 * epgjax.b_epg(T1, delays[1]))
    omega = omega * jnp.exp(-preptimes[1] / T2)

    omega = omega.at[0, :].set(0.0)  # Reset the first row to zero
    omega = omega.at[1, :].set(0.0)  # Reset the second row to zero

    # 2 * 96 second block
    third_block_start_index: int = 576
    third_block_end_index: int = 768
    fa_third_block = fa[third_block_start_index:third_block_end_index]
    TR_third_block = TR[third_block_start_index:third_block_end_index]
    phases_third_block = phases[third_block_start_index:third_block_end_index]
    omega, signal_third_block = epgsig.compute_signal_and_state(
        omega=omega,
        T1=T1,
        T2=T2,
        M0=M0,
        fa=fa_third_block,
        phases=phases_third_block,
        TR=TR_third_block,
        TE=TE,
        b_TE=b_TE,
        r_TE=r_TE
    )
    signal = jnp.concatenate(
        [
            signal_first_block,
            signal_second_block,
            signal_third_block,
        ],
        axis=0
    )
    print('hunter 27 actual')
    return signal



from dataclasses import dataclass
from typing import Protocol
from typing import NamedTuple

import equinox as eqx

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



class PreparationModule(Element):
    """
    Represents segment in the sequence where the magnetization is prepared
    via a preparation pulse.
    """
    delay: float
    preptime: float


class InversionModule(Element):
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


@dataclass
class PFISP:
    elements: list[Element]
    echoinfo: EchoInfo


def simulate_preparation(
    omega: Array,
    prep_module: PreparationModule,
    relaxometric_params: RelaxometricParameters,
) -> Array:
    """
    Simulate the effect of a T2 preparation module on the EPG state.
    """
    # free relaxation during the delay time
    delay_operator = epgjax.r_epg(
        T1=relaxometric_params.T1,
        T2=relaxometric_params.T2,
        dt=prep_module.delay
    )
    # T1 recovery
    b_delay = epgjax.b_epg(
        T1=relaxometric_params.T1,
        dt=prep_module.delay
    )
    omega = delay_operator @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + relaxometric_params.M0 * b_delay)
    # NOTE: here we do the Max Gram assumption to nuke the (0, 1) rows of the EPG
    #       omega state matrix. Thus we also only dampen the (0, 2) row. If we move
    #       to a more general model, we will need to dampen all rows.
    omega = omega.at[2, :].set(
        omega[2, :] * jnp.exp(-prep_module.preptime / relaxometric_params.T2)
    )
    omega = omega.at[0, :].set(0.0)  # Nuke the first row to zero
    omega = omega.at[1, :].set(0.0)  # Nuke even moar
    return omega


@define
class Transients:
    """
    Transient parameters of the propagation simulation.
    """
    fa: Array
    TR: Array
    phases: Array
    
    _n_tr: int | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        """
        Post-initialization to compute the number of TRs.
        """
        if self.fa.shape != self.TR.shape or self.fa.shape != self.phases.shape:
            raise ValueError(
                'fa, TR, and phases must have the same shape, but got shapes'
                f'fa = {self.fa.shape} | TR = {self.TR.shape} | phases = {self.phases.shape}.'
            )
        self._n_tr = self.fa.shape[0]




@dataclass
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

    def __post_init__(self):
        pass








def propagate(
    carry: Array,
    inputs
) -> tuple:
    pass


def simulate_modulation(
    omega: Array,
    segment: ModulationSegment,
    relaxometric_params: RelaxometricParameters,
    echoinfo: EchoInfo,
) -> StateSignal:
    """
    Simulate the effect of a modulation segment on the EPG state
    and compute the resulting signal.
    """
    pass
    





def simulate(
    pfisp: PFISP,
    omega: Array,
    relaxometric_params: RelaxometricParameters,
) -> StateSignal:
    pass





if __name__ == '__main__':
    

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