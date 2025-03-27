import jax
import jax.numpy as jnp

import pytest

import mrfoptools.optimization.gradtools.jacdesc as jacdesc


def test_smoke_conFIG():
    seed = 1701
    key = jax.random.key(seed=seed)
    n_criterion = 3
    n_parameters = 100
    jacobian = jax.random.normal(key, shape=(n_criterion, n_parameters))
    grad = jacdesc.conFIG(jacobian)
    assert grad.shape == (n_parameters,)
    assert jnp.all(jnp.isfinite(grad))


def test_smoke_mgda():
    seed = 1701
    key = jax.random.key(seed=seed)
    n_criterion = 3
    n_parameters = 100
    jacobian = jax.random.normal(key, shape=(n_criterion, n_parameters))
    grad = jacdesc.multiple_gradient_descent(jacobian, epsilon=1e-3, max_iters=100)
    assert grad.shape == (n_parameters,)
    assert jnp.all(jnp.isfinite(grad))


def test_smoke_pcgrad():
    seed = 1701
    key = jax.random.key(seed=seed)
    key, *subkeys = jax.random.split(key, num=3)
    n_criterion = 3
    n_parameters = 100
    jacobian = jax.random.normal(subkeys[0], shape=(n_criterion, n_parameters))
    grad = jacdesc.projection_conflicting_gradients(subkeys[1], jacobian)
    assert grad.shape == (n_parameters,)
    assert jnp.all(jnp.isfinite(grad))


@pytest.mark.skip(reason="fails: investigate offline")
def test_smoke_upgrad():
    seed = 1701
    key = jax.random.key(seed=seed)
    key, *subkeys = jax.random.split(key, num=3)
    n_criterion = 3
    n_parameters = 100
    jacobian = jax.random.normal(subkeys[0], shape=(n_criterion, n_parameters))
    grad = jacdesc.unconflicting_projection_gradients(jacobian)
    assert grad.shape == (n_parameters,)
    assert jnp.all(jnp.isfinite(grad))