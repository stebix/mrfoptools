"""
Test sequence numpy implementations.
"""
import numpy as np
import jax

from mrfoptools.epg.sequences.numpy.fisp import simulate_fisp
from mrfoptools.epg.sequences.jax.fisp import simulate_fisp as simulate_fisp_jax

from mrfoptools.epg.sequences.helpers import get_basic_configuration, extract_specialization_kwargs, quickmake_simulate_fisp

from mrfoptools.testtooling.testtooling import (NumpyImplementation,
                                                JaxImplementation,
                                                benchmark,
                                                autocompile)

from mrfoptools.testtooling.reporting import display_report


def test_simulate_fist_numpy_implementation():
    """
    Basic smoke test for the numpy implementation of the FISP sequence.
    """
    conf = get_basic_configuration(backend='jax', n_species=1)
    kwargs = extract_specialization_kwargs(conf)
    fa = conf.get('fa')
    TR = conf.get('tr')

    print('T1 aray shape:: ', conf.get('T1').shape)
    print('T2 aray shape:: ', conf.get('T2').shape)


    simfisp = quickmake_simulate_fisp(do_jit=True, n_species=2, max_states=500)

    # manual
    import time
    print('warmup')
    _ = simfisp(fa, TR).block_until_ready()

    print('manual bare benchmark')
    tstart = time.time()
    result = simfisp(fa, TR).block_until_ready()
    tend = time.time()

    print('Manual elapsed time:: ', tend - tstart)
    print(f'result shape:: {result.shape}')

    # jaxfunc = jax.jit(simulate_fisp_jax, static_argnames=kwargs.keys())

    kwargs = kwargs | {'fa': fa, 'TR': TR}


    # raise Exception('o7 saluting sir')

    numpyimpl = NumpyImplementation(simulate_fisp, ID='numpy')
    jaximpl = JaxImplementation(simfisp, ID='jax')

    #implementations = autocompile([numpyimpl]) + [jaximpl]

    implementations = autocompile([jaximpl])
    
    implementations = autocompile([numpyimpl])
    rt, rr = benchmark(implementations, kwargs=kwargs, repeats=25)

    display_report(rt, header='FISP sequence benchmarking test')
    
