"""
Equivalence tests for the core ported EPG code.

@author Jannik Stebani 2025
"""
import torch
import numpy as np
import jax
import jax.numpy as jnp

import pytest

import mrfoptools.contrib.signalmodel_epg as contrib

import mrfoptools.epg.core.jax as epgjax
import mrfoptools.epg.core.numpy as epgnp

# Design a payload class that encapsulates three function objects
# - one object for the torch implementation
# - one object for the jax implementation
# - on object for the compiled jax implementation
# The class should have a method that compares the outputs of the three implementations
# and raises an error if the outputs are not equal.

from mrfoptools.testtooling.testtooling import (JaxImplementation,
                                                NumpyImplementation,
                                                TorchImplementation,
                                                benchmark,
                                                autocompile)


from mrfoptools.testtooling.reporting import display_report


def test_autobench():

    dtype = np.complex64
    alpha = np.deg2rad(45).astype(dtype)
    phi = np.deg2rad(20).astype(dtype)

    numpyfunc = epgnp.q_epg

    numpyfunc_alt = epgnp.q_alt

    jaxfunc = epgjax.q_epg
    torchfunc = contrib.q_epg

    numpyimpl = NumpyImplementation(numpyfunc, ID='numpy')
    numpyimpl_alt = NumpyImplementation(numpyfunc_alt, ID='numpy_alt')

    jaximpl = JaxImplementation(jaxfunc, ID='jax')
    torchimpl = TorchImplementation(torchfunc, ID='torch')

    implementations = autocompile([numpyimpl, numpyimpl_alt, jaximpl, torchimpl])

    rt, rr = benchmark(implementations, args=(alpha, phi), repeats=50)

    display_report(rt, header='q excitation benchmarking test')


def test_numbaimpl_q_excitation():
    import numba as nb
    numpyimpl = epgnp.q_epg
    numpyimpl_compiled = nb.njit(epgnp.q_epg, fastmath=True)
    dtype = np.complex64
    alpha = np.deg2rad(45).astype(dtype)
    phi = np.deg2rad(20).astype(dtype)

    result_baseimpl = numpyimpl(alpha, phi)
    result_compiled = numpyimpl_compiled(alpha, phi)

    assert np.allclose(result_baseimpl, result_compiled, rtol=1e-5, atol=1e-5)


class Test_q_excitation:

    def test_q_excitation_equivalence(self):
        # input data
        alpha = np.deg2rad(45)
        phi = np.deg2rad(20)
        # functions
        jaximpl = epgjax.q_epg
        torchimpl = contrib.q_epg
        jaximpl_compiled = jax.jit(jaximpl)

        jaximpl_result = jaximpl(alpha, phi)
        torchimpl_result = torchimpl(
            torch.as_tensor(alpha), torch.as_tensor(phi)
        ).detach().resolve_conj().numpy()
        jaximpl_compiled_result = jaximpl_compiled(alpha, phi)

        assert np.allclose(jaximpl_result, torchimpl_result, rtol=1e-5, atol=1e-5), 'jax - torch mismatch'
        assert np.allclose(jaximpl_result, jaximpl_compiled_result, rtol=1e-5, atol=1e-5), 'jax - jax compiled mismatch'


def test_q_excitation_equivalence():
    alpha = np.deg2rad(45)
    phi = np.deg2rad(20)

    jaximp_result = epgjax.q_epg(alpha, phi)
    torchimp_result = contrib.q_epg(
        torch.as_tensor(alpha), torch.as_tensor(phi)
    ).detach().resolve_conj().numpy()

    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('T1', [1.0, 2.0, 3.0])
@pytest.mark.parametrize('T2', [0.1, 0.2, 0.3])
@pytest.mark.parametrize('dt', [0.01, 0.02, 0.03])
def test_r_relaxation_equivalence(T1, T2, dt):
    # functions
    jaximpl = epgjax.r_epg
    torchimpl = contrib.r_epg
    jaximpl_compiled = jax.jit(jaximpl)
    # results
    jaximpl_result = jaximpl(T1, T2, dt)
    jaximpl_compiled_result = jaximpl_compiled(T1, T2, dt)
    torchimpl_result = torchimpl(
        torch.as_tensor(T1), torch.as_tensor(T2), torch.as_tensor(dt)
    ).detach().numpy()

    assert np.allclose(jaximpl_result, torchimpl_result, rtol=1e-5, atol=1e-5), 'jax - torch mismatch'
    assert np.allclose(jaximpl_result, jaximpl_compiled_result, rtol=1e-5, atol=1e-5), 'jax - jax compiled mismatch'


@pytest.mark.parametrize('T1', [1.0, 2.0, 3.0])
@pytest.mark.parametrize('dt', [0.01, 0.02, 0.03])
def test_dr_dT1_equivalence(T1, dt):
    jaximp_result = epgjax.dr_dT1_epg(T1, dt)
    torchimp_result = contrib.dr_dt1_epg(
        torch.as_tensor(T1), torch.as_tensor(dt)
    ).detach().numpy()

    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


def test_apply_gradient():
    mx, my, mz = 0.5, 0.3, 0.2
    omega_jax = jnp.array(
        [
            [mx],
            [my],
            [mz],
        ]
    )
    omega_torch = torch.tensor([[mx], [my], [mz]])
    jaximp_result = epgjax.grad_shift(omega_jax, dk=1)
    torchimp_result = contrib.epg_grad(omega_torch).detach().numpy()
    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


def test_singular_apply_unit_gradient():
    """
    Test singular application of unit gradient is reproduced for jax and torch implementations.
    """
    # data
    mx, my, mz = 0.5, 0.3, 0.2
    omega_jax = jnp.array(
        [
            [mx],
            [my],
            [mz],
        ]
    )
    omega_torch = torch.tensor([[mx], [my], [mz]])
    # functions
    jaximpl = epgjax.unit_grad_shift_allocating
    torchimpl = contrib.epg_grad
    jaximpl_compiled = jax.jit(jaximpl)
    # results
    jaximpl_result = jaximpl(omega_jax)
    torchimpl_result = torchimpl(omega_torch).detach().numpy()
    jaximpl_compiled_result = jaximpl_compiled(omega_jax)

    assert np.allclose(jaximpl_result, torchimpl_result, rtol=1e-5, atol=1e-5), 'jax - torch mismatch'
    assert np.allclose(jaximpl_result, jaximpl_compiled_result, rtol=1e-5, atol=1e-5), 'jax - jax compiled mismatch'



@pytest.mark.parametrize('n', [2, 5, 15])
def test_multiple_apply_unit_gradient(n):
    """
    Test singular application of unit gradient is reproduced for jax and torch implementations.
    """
    mx, my, mz = 0.5, 0.3, 0.2
    omega_jax = jnp.array(
        [
            [mx],
            [my],
            [mz],
        ]
    )
    omega_torch = torch.tensor([[mx], [my], [mz]])
    omega_jax_compiled = omega_jax.copy()
    # functions
    jaximpl = epgjax.unit_grad_shift_allocating
    torchimpl = contrib.epg_grad
    jaximpl_compiled = jax.jit(jaximpl)
    # results
    for _ in range(n):
        omega_jax = jaximpl(omega_jax)
        omega_torch = torchimpl(omega_torch)
        omega_jax_compiled = jaximpl_compiled(omega_jax_compiled)

    assert np.allclose(omega_jax, omega_torch.detach().numpy(), rtol=1e-5, atol=1e-5), 'jax - torch mismatch'
    assert np.allclose(omega_jax, omega_jax_compiled, rtol=1e-5, atol=1e-5), 'jax - jax compiled mismatch'



class Test_unit_gradient_shift_static:

    def test_unit_gradient_shift_static(self):
        mx, my, mz = 0.5, 0.3, 0.2
        omega_init = jnp.array(
            [
                [mx],
                [my],
                [mz],
            ]
        )
        omega_static = jnp.hstack([omega_init, jnp.zeros((3, 3))])
        omega_dynamic = omega_init.copy()

        for _ in range(3):
            omega_static = epgjax.unit_grad_shift_static(omega_static)
            omega_dynamic = epgjax.unit_grad_shift_allocating(omega_dynamic)

        print(omega_static)

        print(omega_dynamic)
