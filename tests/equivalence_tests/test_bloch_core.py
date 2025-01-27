"""
Test equivalence of Jax implementation and the contributed implementation of the Bloch core functions.

@Author: Jannik Stebani 2025
"""
import torch
import jax
import jax.numpy as jnp

import pytest

import mrfoptools.bloch.core.core as bloch
import mrfoptools.contrib.signalmodel_bloch as contrib

@pytest.mark.parametrize('alpha', [0.5, 1.5, jnp.pi])
@pytest.mark.parametrize('phi', [0.75, 1.25, 1.1*jnp.pi])
def test_q_operator_equivalence(alpha, phi):
    """Test equivalence for the q_operator, i.e. Bloch excitation matrix."""
    q_op_jax = bloch.q_operator(alpha, phi)
    q_op_torch = contrib.q_iso(
        torch.tensor(alpha), torch.tensor(phi)
    ).detach().resolve_conj().numpy()
    assert jnp.allclose(q_op_jax, q_op_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('dt', [0.01, 0.02, 0.03])
@pytest.mark.parametrize('T1', [1.0, 2.0, 3.0])
@pytest.mark.parametrize('T2', [0.1, 0.2, 0.3])
def test_r_operator_equivalence(T1, T2, dt):
    """Test equivalence for the r_operator, i.e. relaxation operator."""
    r_op_jax = bloch.r_operator(T1, T2, dt)
    r_op_torch = contrib.r_iso(
        torch.tensor(T1), torch.tensor(T2), torch.tensor(dt)
    ).detach().resolve_conj().numpy()
    assert jnp.allclose(r_op_jax, r_op_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('dt', [0.01, 0.02, 0.03])
@pytest.mark.parametrize('T1', [1.0, 2.0, 3.0])
def test_b_operator_quivalence(T1, dt):
    """
    Test equivalence for the b_operator, i.e. Bloch evolution matrix.
    """
    b_op_jax = bloch.b_operator(T1, dt)
    b_op_torch = contrib.b_iso(
        t1=torch.tensor(T1), t=torch.tensor(dt)
    ).detach().resolve_conj().numpy()
    assert jnp.allclose(b_op_jax, b_op_torch, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('beta', [0.5, 1.5, jnp.pi])
def test_g_operator_equivalence(beta):
    """
    Test equivalence for the g operator, i.e. Bloch rotation operator around
    the z axis (utilized in spin dephasing).
    """
    g_op_jax = bloch.g_operator(beta)
    g_op_torch = contrib.g(
        torch.tensor(beta)
    ).detach().resolve_conj().numpy()
    assert jnp.allclose(g_op_jax, g_op_torch, rtol=1e-5, atol=1e-5)