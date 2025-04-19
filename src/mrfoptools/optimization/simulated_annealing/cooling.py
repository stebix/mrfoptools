"""
Cooling schedules for simulated annealing.
"""
import jax.numpy as jnp


def linear_cooling(temp_init: float, r: float, n_iter: int) -> float:
    return temp_init - r * n_iter


def exponential_cooling(temp_init: float, r: float, n_iter: int) -> float:
    return temp_init * r ** n_iter


def logarithmic_cooling(temp_init: float, r: float, n_iter: int) -> float:
    return temp_init / jnp.log1p(n_iter)