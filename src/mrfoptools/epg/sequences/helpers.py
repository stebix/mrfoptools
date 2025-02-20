"""
Helper tooling for rapid setup of an EPG simulation of a FISP
sequence.

@Author: Jannik Stebani 2025
"""
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from mrfoptools.epg.sequences.fisp import specialize_simulate_fisp


BASIC_CONFIGURATION: dict[str, float | jax.Array] = {
    'INIT_FA' : 50,
    'INIT_TR' : 12,
    'NR' : 1000,
    'M0' : 1.0,
    'TE' : 1,
    'TI' : 20,
    'max_states' : 1000,
    'inversion_efficiency' : 1.0,
    'min_fa' : jnp.deg2rad(1),
    'max_fa' : jnp.deg2rad(90),
    'T1' : jnp.array([1500, 2000, 2500, 3000]),
    'T2' : jnp.array([500, 500, 600, 700])
}


def get_basic_configuration(**kwargs) -> dict:
    """
    Just a sensible default configuration for a FISP sequence.
    Overrides of defaults of basic configuration via key-value kwargs.
    """
    conf: dict = BASIC_CONFIGURATION.copy() | kwargs
    fa = jnp.deg2rad(jnp.full(fill_value=conf['INIT_FA'], shape=conf['NR'], dtype=jnp.float32))
    phases = jnp.full_like(fa, fill_value=jnp.pi/2)
    tr = jnp.full(fill_value=conf['INIT_TR'], shape=conf['NR'], dtype=jnp.float32)

    configuration = {
        'fa' : fa,
        'tr' : tr,
        'T1' : conf['T1'],
        'T2' : conf['T2'],
        'M0' : conf['M0'],
        'phases' : phases,
        'TI' : conf['TI'],
        'TE' : conf['TI'],
        'max_states' : conf['max_states'],
        'inversion_efficiency' : conf['inversion_efficiency'],
        'min_fa' : conf['min_fa'],
        'max_fa' : conf['max_fa']
    }
    return configuration


def extract_specialization_kwargs(d: Mapping) -> dict:
    keys: set[str] = {'T1', 'T2', 'M0', 'phases', 'TI',
                      'TE', 'max_states', 'inversion_efficiency'}
    return {k: d[k] for k in keys}


def quickmake_simulate_fisp(**kwargs: Any) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Quick setup of a FISP simulation function.
    Simulation function ingests fully-fledged FA and TR pattern arrays.
    """
    do_jit: bool = kwargs.get('do_jit', True)

    basic_conf = get_basic_configuration(**kwargs)
    spec_conf = extract_specialization_kwargs(basic_conf)

    simulate_fisp = specialize_simulate_fisp(**spec_conf)

    if do_jit:
        return jax.jit(simulate_fisp)
    
    return simulate_fisp