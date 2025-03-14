"""
Benchmark runtime performance of backend implementations
for signal calculations with EPG.
"""
import numpy as np
import jax
import numba as nb

import mrfoptools.epg.signal.numpy as npysig
import mrfoptools.epg.signal.numpy.preparations as npysigprep

import mrfoptools.epg.signal.jax as jaxsig

import mrfoptools.epg.core.numpy as npycore

from mrfoptools.testtooling.testtooling import (JaxImplementation,
                                                NumpyImplementation,
                                                autocompile,
                                                benchmark,
                                                assert_pairwise_equivalence)

from mrfoptools.testtooling.reporting import display_report

from mrfoptools.initialization.initialization import load_yun_pattern


def init_helper() -> dict:
    """Setup for numpy-backend signal calculations."""
    pattern = load_yun_pattern(type_='both')
    fa = pattern.flip_angles
    tr = pattern.repetition_times
    phases = np.full_like(fa, 0.0)
    TI = 20
    TE = 1
    T1 = 1000
    T2 = 500
    M0 = 1.0
    inversion_efficiency = 1.0
    max_states = 500
    inversion_operator = npycore.inversion(inversion_efficiency)
    omega = npysigprep.prepare_inversion_omega(
        T1, T2, M0, max_states=max_states,
        inversion_operator=inversion_operator, TI=TI
    )
    dtype = np.complex64 # noqa F841
    parameters = {
        'omega' : omega,
        'T1' : T1,
        'T2' : T2,
        'M0' : M0,
        'fa' : fa,
        'TR' : tr,
        'phases' : phases,
        'TE' : TE,
        'b_TE' : npycore.b_epg(T1, TE),
        'r_TE' : npycore.r_epg(T1, T2, TE),
    }
    return parameters


def test_benchmark_epg_signal():
    numpyfunc = npysig.compute_signal_numpynaive
    jaxfunc = jax.jit(jaxsig.compute_signal_optimized)

    numpyimpl = NumpyImplementation(
        func=numpyfunc, ID='numpy'
    )
    jaximpl = JaxImplementation(
        func=jaxfunc, ID='jax_compiled'
    )

    kwargs = init_helper()

    implementations = autocompile([numpyimpl])
    implementations = implementations + [jaximpl]

    rt, rr = benchmark(implementations, kwargs=kwargs, repeats=100)

    display_report(rt, header='signal benchmarking test')

    assert_pairwise_equivalence(rr)



def test_benchmark_epg_signal_compileoptions():
    jaxfunc = jax.jit(jaxsig.compute_signal_optimized)
    numpyfunc = nb.jit(npysig.compute_signal_numpynaive,
                       nopython=True,
                       fastmath=True, parallel=False, nogil=True)

    numpyimpl = NumpyImplementation(
        func=numpyfunc, ID='numpy_compiled'
    )
    jaximpl = JaxImplementation(
        func=jaxfunc, ID='jax_compiled'
    )

    kwargs = init_helper()

    implementations = [numpyimpl, jaximpl]

    rt, rr = benchmark(implementations, kwargs=kwargs, repeats=100)

    display_report(rt, header='signal benchmarking test')

    assert_pairwise_equivalence(rr)


