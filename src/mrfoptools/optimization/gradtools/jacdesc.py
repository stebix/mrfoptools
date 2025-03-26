from typing import TypeAlias, Literal
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from qpsolvers import solve_qp

import mrfoptools.optimization.gradtools.gramutils as gramutils

Array: TypeAlias = jax.Array

def combine(matrix: Array, weights: Array) -> Array:
    """
    Combine the rows of a matrix according to the weights.
    """
    return weights @ matrix


def get_sum_weights(matrix: Array) -> Array:
    """
    Get the sum of the rows of a matrix.
    """
    return jnp.ones(shape=matrix.shape[0], dtype=jnp.float32)


def conFIG(
    jacobian: jax.Array,
    preference_vector: jax.Array | None = None,
) -> jax.Array:
    """
    Compute the conFIG gradient from a Jacobian matrix.
    
    Paper: https://arxiv.org/abs/2408.11104
    
    Parameters
    ----------
    
    jacobian: jax.Array
        Jacobian matrix of shape (n_criterion, n_parameters)
    
    Returns
    -------
    
    jax.Array
        "Optimal" conFIG gradient of shape (n_parameters,)
    """
    weights = preference_vector or get_sum_weights(jacobian)
    unit = jnp.nan_to_num(
        jacobian / jnp.linalg.norm(jacobian, axis=1, keepdims=True),
        nan=0.0
    )
    optimal_direction = jnp.linalg.pinv(unit) @ weights

    norm = jnp.linalg.norm(optimal_direction)
    if norm == 0.0:
        unit_optimal_direction = jnp.zeros_like(optimal_direction)
    else:
        unit_optimal_direction = optimal_direction / norm

    length = jnp.sum(jnp.dot(jacobian, unit_optimal_direction))

    return length * unit_optimal_direction




def solve_frank_wolfe(
    matrix: Array,
    max_iters: int,
    epsilon: float
) -> Array:
    """
    Solve convex problem with the Frank-Wolfe algorithm.
    """
    n_gradients: int = matrix.shape[0]
    gramian = gramutils.compute_gramian(matrix)
    alpha = jnp.full(shape=n_gradients, fill_value=1/n_gradients, dtype=jnp.float32)

    init_val = (0, alpha, jnp.inf)

    def proto_body_fun(
        carry: tuple,
        gramian: Array,
        n_gradients: int,
    ):
        (i, alpha, gamma) = carry    
        t = jnp.argmin(gramian @ alpha)
        e_t = jnp.zeros(shape=n_gradients, dtype=jnp.float32)
        e_t = e_t.at[t].set(1.0)
        a = alpha @ (gramian @ e_t)
        b = alpha @ (gramian @ alpha)
        c = e_t @ (gramian @ e_t)

        def gamma_fallthru_fun(a, b, c):
            return (b - a) / (b + c - 2 * a)
        
        gamma = jax.lax.cond(
            c <= a,
            lambda *_: 1.0,
            lambda a, b, c : jax.lax.cond(
                b <= a,
                lambda *_: 0.0,
                gamma_fallthru_fun,
                a, b, c
            ),
            a, b, c
        )
        
        alpha = (1 - gamma) * alpha + gamma * e_t

        retcarry = (i + 1, alpha, gamma)
        return retcarry
    
    custom_body_fun = jax.jit(partial(
        proto_body_fun,
        gramian=gramian,
        n_gradients=n_gradients
        )
    )

    def proto_cond_fun(carry: tuple, max_iters: int, epsilon: float) -> bool:
        (i, _, gamma) = carry
        continue_loop = jax.lax.cond(
            i < max_iters,
            true_fun=lambda _: jax.lax.cond(
                gamma < epsilon,
                true_fun=lambda _: False,
                false_fun=lambda _: True,
                operand=None
            ),
            false_fun=lambda _: False,
            operand=None
        )
        return continue_loop

    custom_cond_fun = partial(proto_cond_fun, max_iters=max_iters, epsilon=epsilon)    

    (_, alpha, _) = jax.lax.while_loop(custom_cond_fun, custom_body_fun, init_val)
    return alpha


def multiple_gradient_descent(
    jacobian: Array,
    epsilon: float,
    max_iters: int

) -> Array:
    """
    Compute the gradient aggregated with the `Multiple Gradient Descent`
    (MGDA) algorithm.

    Reference: https://inria.hal.science/inria-00389811v2/document
    """
    weights = solve_frank_wolfe(jacobian, max_iters, epsilon)
    return weights @ jacobian


def projection_conflicting_gradients(
    key: Array,
    jacobian: Array
) -> Array:
    """
    Compute the gradient aggregated with the `Projection of Conflicting Gradients`
    (PCG) algorithm.
    This function performs random projections of a gradient in the 
    orthogonal plane of the other gradients. Thus, a random key is required.

    Parameters
    ----------
    key: Array
        Random number generator key.

    jacobian: Array
        Jacobian matrix of shape `(n_criterion, n_parameters)`
        i.e. rows are gradients of the cost functions.

    References
    ----------
    https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf
    """
    gramian = gramutils.compute_gramian(jacobian)
    n_gradients: int = gramian.shape[0]

    weights = jnp.zeros(n_gradients, dtype=jnp.float32)

    # NOTE: The loops are unrolled and compiled as a full computation graph
    #       Implement via jax.lax.fori_loop if JIT compilation
    #       performance is an issue

    for i in range(n_gradients):
        key, subkey = jax.random.split(key)
        permutation = jax.random.permutation(key=subkey, x=n_gradients)

        transient_weights = jnp.zeros(n_gradients, dtype=jnp.float32)
        transient_weights = transient_weights.at[i].set(1.0)

        for j in permutation:
            if j == i:
                continue
            # inner product betwenn g^{pc} and g_j
            inner_product = gramian[j] @ transient_weights

            if inner_product < 0.0:
                # conflicting gradients
                transient_weights = transient_weights.at[j].set(
                    transient_weights[j] - inner_product / gramian[j, j]
                )

        weights = weights + transient_weights

    return weights @ jacobian



def to_float64_array(arr: Array) -> Array:
    return np.asarray(arr, dtype=np.float64)



def project_weight_vector(
    u: np.ndarray,
    G: np.ndarray,
    *,
    solver: Literal['quadprog'] = 'quadprog'
) -> np.ndarray:
    m = G.shape[0]
    w = solve_qp(G, np.zeros(m), -np.eye(m), -u, solver=solver)
    return w



def project_weights(
    U: Array,
    G: Array,
    *,
    solver: Literal['quadprog'] = 'quadprog'
    ) -> np.ndarray:

    G: np.ndarray = to_float64_array(G)
    U: np.ndarray = to_float64_array(U)

    W = np.apply_along_axis(
        lambda u: project_weight_vector(u, G, solver),
        axis=-1, arr=U
    )
    return W



def unconflicting_projection_gradients(
    jacobian: Array,
    preference_vector: Array | None = None,
    norm_eps: float = 1e-4,
    reg_eps: float = 1e-4,
    *,
    solver: Literal['quadprog'] = 'quadprog'
) -> Array:
    
    n_gradients: int = jacobian.shape[0]
    weight = preference_vector or jnp.full(n_gradients, 1/n_gradients, dtype=jnp.float32)
    U = jnp.diag(weight @ jacobian)
    G = gramutils.compute_normalized_regularized_gramian(jacobian, norm_eps, reg_eps)
    W = project_weights(U, G, solver)
    return jnp.sum(W, axis=0)