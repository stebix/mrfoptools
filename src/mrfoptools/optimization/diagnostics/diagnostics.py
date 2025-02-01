"""
Tooling for gradient and cost diagnostics: Logging and visualization to tensorboard.

@Author: Jannik Stebani 2025
"""
from collections.abc import Mapping, Callable
import numpy as np
import jax
from torch.utils.tensorboard import SummaryWriter

from mrfoptools.optimization.costgrad import NumpyCostGradTuple


def log_costs(
    cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
    writer: SummaryWriter,
    iteration: int
) -> None:
    """
    Log scalar cost values for arbitrary (sub-) objectives to a summary writer.
    """
    prefix: str = 'costs'
    for name, cg_tuple in cost_grad_mapping.items():
        tag = '/'.join((prefix, name))
        writer.add_scalar(tag=tag, scalar_value=cg_tuple.cost, global_step=iteration)


def log_gradient_histograms(
    cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
    writer: SummaryWriter,
    iteration: int
) -> None:
    """
    Log gradient as histograms.
    """
    prefix: str = 'gradhists'
    for name, cg_tuple in cost_grad_mapping.items():
        tag = '/'.join((prefix, name))
        writer.add_histogram(tag=tag, values=cg_tuple.grad.ravel(), global_step=iteration)


def log_cosine_similarities(
    cosine_similarities: Mapping[str, float],
    writer: SummaryWriter,
    iteration: int
) -> None:
    """
    Log precomputed cosine similarites to a summary writer instance.
    """
    prefix: str = 'grad_cos_sim'
    for name, value in cosine_similarities.items():
        tag = '/'.join((prefix, name))
        writer.add_scalar(tag=tag, scalar_value=value, global_step=iteration)
    
    
def log_gradient_magnitude_similarities(
    magnitude_similarities: Mapping[str, float],
    writer: SummaryWriter,
    iteration: int
) -> None:
    """
    Log precomputed gradient magnitude similarities to a summary writer instance.
    """
    prefix: str = 'grad_mag_sim'
    for name, value in magnitude_similarities.items():
        tag = '/'.join((prefix, name))
        writer.add_scalar(tag=tag, scalar_value=value, global_step=iteration)


def build_relative_gain_logfunction(
    forward: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    initial_parameters: tuple,
    criterion: Callable[[jax.Array], jax.Array],
    writer: SummaryWriter,
    name: str,
    prefix: str = 'relative_gain'
) -> Callable:
    """
    Programmatically log improvements of a scalar cost value
    relative to a predefined initial value.
    """
    out = forward(*initial_parameters)
    init_cost = np.asarray(criterion(out))
    tag = '/'.join((prefix, name))
    
    def _log(cg_tuple: NumpyCostGradTuple, iteration: int):
        """Log relative gain over initial baseline cost value."""
        relative_improvement = cg_tuple.cost / init_cost
        writer.add_scalar(tag=tag, scalar_value=relative_improvement, global_step=iteration)
        
    return _log