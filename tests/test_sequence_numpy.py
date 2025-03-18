"""
Test sequence numpy implementations.
"""
import jax
import numpy as np

from mrfoptools.epg.sequences.numpy.fisp import simulate_fisp

from mrfoptools.epg.sequences.helpers import get_basic_configuration, extract_specialization_kwargs, quickmake_simulate_fisp

from mrfoptools.testtooling.testtooling import (NumpyImplementation,
                                                JaxImplementation,
                                                benchmark,
                                                autocompile)

from mrfoptools.testtooling.reporting import display_report

from collections.abc import Mapping

def recursive_to_numpy(
    mapping: Mapping
) -> Mapping:
    for key, value in mapping.items():
        if isinstance(value, Mapping):
            mapping[key] = recursive_to_numpy(value)
        elif isinstance(value, jax.Array):
            mapping[key] = np.array(value)
    return mapping


def test_simulate_fist_numpy_implementation():
    """
    Basic smoke test for the numpy implementation of the FISP sequence.
    """

    # TODO: When changing from the old enviroment `mrfopt` or `dip-X` to the new one
    #       `mrfoptools-uio` the test fails due to a BufferError in numba.
    #       When initializing the data with the numpy backend below, the test works again.
    #       Investigate the cause of the BufferError and re-check in old environment.
    conf = get_basic_configuration(backend='numpy', n_species=1)
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
    
