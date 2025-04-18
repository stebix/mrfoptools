"""
Cooling schedules for simulated annealing.
"""
import jax.numpy as jnp


def linear_cooling(temp_start: float, r: float, iter: int) -> float:
    return temp_start - r * iter


def exponential_cooling(temp_start: float, r: float, iter: int) -> float:
    return temp_start * r ** iter


def logarithmic_cooling(temp_start: float, r: float, iter: int) -> float:
    return temp_start / jnp.log1p(iter)