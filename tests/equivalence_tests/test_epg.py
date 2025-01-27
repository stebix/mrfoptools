"""
Equivalence tests for the core ported EPG code.

@author Jannik Stebani 2025
"""
import torch
import numpy as np
import jax
import jax.numpy as jnp

import pytest

import mrfoptools.epg.core as epg
import mrfoptools.contrib.signalmodel_epg as contrib

# Design a payload class that encapsulates three function objects
# - one object for the torch implementation
# - one object for the jax implementation
# - on object for the compiled jax implementation
# The class should have a method that compares the outputs of the three implementations
# and raises an error if the outputs are not equal.


class Test_q_excitation:

    def test_q_excitation_equivalence(self):
        # input data
        alpha = np.deg2rad(45)
        phi = np.deg2rad(20)
        # functions
        jaximpl = epg.q_epg
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

    jaximp_result = epg.q_epg(alpha, phi)
    torchimp_result = contrib.q_epg(
        torch.as_tensor(alpha), torch.as_tensor(phi)
    ).detach().resolve_conj().numpy()

    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('T1', [1.0, 2.0, 3.0])
@pytest.mark.parametrize('T2', [0.1, 0.2, 0.3])
@pytest.mark.parametrize('dt', [0.01, 0.02, 0.03])
def test_r_relaxation_equivalence(T1, T2, dt):
    # functions
    jaximpl = epg.r_epg
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
    jaximp_result = epg.dr_dT1_epg(T1, dt)
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
    jaximp_result = epg.grad_shift(omega_jax, dk=1)
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
    jaximpl = epg.unit_grad_shift_allocating
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
    jaximpl = epg.unit_grad_shift_allocating
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
            omega_static = epg.unit_grad_shift_static(omega_static)
            omega_dynamic = epg.unit_grad_shift_allocating(omega_dynamic)

        print(omega_static)

        print(omega_dynamic)
