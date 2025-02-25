import numpy as np

import pytest


import mrfoptools.epg.core as jaxcore
import mrfoptools.contrib.signalmodel_epg as contrib
import mrfoptools.epg.core.core_numpy as npycore


from mrfoptools.testtooling.testtooling import (JaxImplementation,
                                                NumpyImplementation,
                                                TorchImplementation,
                                                benchmark,
                                                autocompile,
                                                assert_pairwise_equivalence)


from mrfoptools.testtooling.reporting import display_report


@pytest.mark.benchmark
def test_bench_q():

    dtype = np.complex64
    alpha = np.deg2rad(45).astype(dtype)
    phi = np.deg2rad(20).astype(dtype)

    numpyfunc = npycore.q_epg

    numpyfunc_alt = npycore.q_alt
    jaxfunc = jaxcore.q_epg
    torchfunc = contrib.q_epg

    numpyimpl = NumpyImplementation(numpyfunc, ID='numpy')
    numpyimpl_alt = NumpyImplementation(numpyfunc_alt, ID='numpy_alt')

    jaximpl = JaxImplementation(jaxfunc, ID='jax')
    torchimpl = TorchImplementation(torchfunc, ID='torch')

    implementations = autocompile([numpyimpl, numpyimpl_alt, jaximpl, torchimpl])

    rt, rr = benchmark(implementations, args=(alpha, phi), repeats=50)

    display_report(rt, header='q excitation benchmarking test')



@pytest.mark.benchmark
def test_bench_r():

    dtype = np.float32
    T1 = np.float32(1000).astype(dtype)
    T2 = np.float32(500).astype(dtype)
    dt = np.float32(0.01).astype(dtype)

    numpyfunc = npycore.r_epg

    numpyfunc_alt = npycore.r_epg
    jaxfunc = jaxcore.r_epg
    torchfunc = contrib.r_epg

    numpyimpl = NumpyImplementation(numpyfunc, ID='numpy')
    numpyimpl_alt = NumpyImplementation(numpyfunc_alt, ID='numpy_alt')

    jaximpl = JaxImplementation(jaxfunc, ID='jax')
    torchimpl = TorchImplementation(torchfunc, ID='torch')

    implementations = autocompile([numpyimpl, numpyimpl_alt, jaximpl, torchimpl])

    rt, rr = benchmark(implementations, args=(T1, T2, dt), repeats=50)

    display_report(rt, header='r relaxation benchmarking test')



@pytest.mark.benchmark
def test_bench_b():

    dtype = np.float32
    T1 = np.float32(1000).astype(dtype)
    dt = np.float32(0.01).astype(dtype)

    numpyfunc = npycore.b_epg

    jaxfunc = jaxcore.b_epg
    torchfunc = contrib.b_epg

    numpyimpl = NumpyImplementation(numpyfunc, ID='numpy')

    jaximpl = JaxImplementation(jaxfunc, ID='jax')
    torchimpl = TorchImplementation(torchfunc, ID='torch')

    implementations = autocompile([numpyimpl, jaximpl, torchimpl])

    rt, rr = benchmark(implementations, args=(T1, dt), repeats=50)

    display_report(rt, header='r relaxation benchmarking test')



@pytest.mark.benchmark
def test_bench_unit_grad_shift_allocating():
    dtype = np.complex64
    mx, my, mz = 0.5, 0.3, 0.2
    omega = np.array(
        [
            [mx],
            [my],
            [mz],
        ], dtype=dtype
    )
    numpyfunc = npycore.unit_grad_shift_allocating
    jaxfunc = jaxcore.unit_grad_shift_allocating
    torchfunc = contrib.epg_grad

    numpyimpl = NumpyImplementation(numpyfunc, ID='numpy')
    jaximpl = JaxImplementation(jaxfunc, ID='jax')
    torchimpl = TorchImplementation(torchfunc, ID='torch')
    implementations = autocompile([numpyimpl, jaximpl, torchimpl])

    rt, rr = benchmark(implementations, args=(omega,), repeats=50)
    display_report(rt, header='unit grad shift allocating benchmarking test')
    assert_pairwise_equivalence(rr)


@pytest.mark.benchmark
def test_bench_unit_grad_shift_static():
    mx, my, mz = 0.5, 0.3, 0.2
    omega_init = np.array(
        [
            [mx],
            [my],
            [mz],
        ], dtype=np.complex64
    )
    omega_static = np.hstack([omega_init, np.zeros((3, 25))])

    numpyfunc = npycore.unit_grad_shift_static
    jaxfunc = jaxcore.unit_grad_shift_static

    numpyimpl = NumpyImplementation(numpyfunc, ID='numpy')
    jaximpl = JaxImplementation(jaxfunc, ID='jax')
    implementations = autocompile([numpyimpl, jaximpl])

    rt, rr = benchmark(implementations, args=(omega_static,), repeats=50)
    display_report(rt, header='unit grad shift static benchmarking test')
    assert_pairwise_equivalence(rr)