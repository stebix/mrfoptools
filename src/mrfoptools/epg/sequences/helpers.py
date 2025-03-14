"""
Helper tooling for rapid setup of an EPG simulation of a FISP
sequence.

@Author: Jannik Stebani 2025
"""
import enum
from collections.abc import Callable, Mapping
from typing import Any

from types import ModuleType

import numpy as np

import jax
import jax.numpy as jnp


from mrfoptools.epg.sequences.jax.fisp import specialize_simulate_fisp

Array = jax.Array | np.ndarray

class Backend(enum.Enum):
    NUMPY = 'numpy'
    JAX = 'jax'

DEFAULT_BACKEND = Backend.JAX


def get_module(backend: Backend | str) -> ModuleType:
    backend = Backend(backend)
    if backend == Backend.JAX:
        return jnp
    elif backend == Backend.NUMPY:
        return np
    else:
        raise ValueError(f'Backend \'{backend}\' not supported.')


def make_species(n_species: int, xnp: ModuleType, *, dtype: type = np.float32) -> dict[str, np.ndarray]:
    T1_candidates = xnp.concatenate((
        xnp.array([1500, 2000, 2500, 3000]), # manually chosen base values
        xnp.linspace(50, 4000, max(0, n_species - 4)) # fill up with linearly spaced values
    )).astype(dtype)
    T2_candidates = xnp.concatenate((
        xnp.array([500, 500, 600, 700]), # manually chosen base values
        xnp.linspace(50, 4000, max(0, n_species - 4)) # fill up with linearly spaced values
    )).astype(dtype)
    return {'T1' : T1_candidates[:n_species], 'T2' : T2_candidates[:n_species]}



def _get_basic_configuration(
    backend: str | Backend | None,
    n_species: int = 4,
    dtype: str = 'float32'
    ) -> dict:
    xnp = get_module(backend) if backend else DEFAULT_BACKEND
    dtype = getattr(xnp, dtype)
    species = make_species(n_species, xnp, dtype=dtype)
    BASIC_CONFIGURATION: dict[str, float | Array] = {
        'INIT_FA' : 50.0,
        'INIT_TR' : 12.0,
        'NR' : 1000,
        'M0' : 1.0,
        'TE' : 1.0,
        'TI' : 20.0,
        'max_states' : 1000,
        'inversion_efficiency' : 1.0,
        'min_fa' : xnp.deg2rad(1).astype(dtype),
        'max_fa' : xnp.deg2rad(90).astype(dtype),
        **species,
        'dtype' : dtype
    }
    return BASIC_CONFIGURATION


def get_basic_configuration(
        backend: str | Backend | None = None,
        **kwargs
    ) -> dict:
    """
    Just a sensible default configuration for a FISP sequence.
    Overrides of defaults of basic configuration via key-value kwargs.
    """
    backend = Backend(backend) if backend is not None else DEFAULT_BACKEND
    xnp: ModuleType = get_module(backend)
    n_species = kwargs.pop('n_species', 4)
    conf: dict = _get_basic_configuration(backend, n_species=n_species).copy() | kwargs
    fa = xnp.deg2rad(xnp.full(fill_value=conf['INIT_FA'], shape=conf['NR'], dtype=xnp.float32))
    phases = xnp.full_like(fa, fill_value=xnp.pi/2)
    tr = xnp.full(fill_value=conf['INIT_TR'], shape=conf['NR'], dtype=xnp.float32)

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