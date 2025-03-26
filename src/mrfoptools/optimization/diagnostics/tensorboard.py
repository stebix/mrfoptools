from collections.abc import Mapping, Callable

import jax
import jax.numpy as jnp
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import mrfoptools.optimization.diagnostics.diagnostics as diag
import mrfoptools.optimization.gradtools.gradtools as gradtools

from mrfoptools.optimization.costgrad import CostContainer, GradientContainer

from mrfoptools.optimization.diagnostics.plothelpers import Plotter
from mrfoptools.optimization.diagnostics.references import RelativeGainEvaluator


ArrayLike = np.ndarray | jax.Array

class _DiagnosticLogger:

    def log_gradients(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        """Log the gradient(s) of the cost function(s)."""
        raise NotImplementedError
    
    def log_costs(
        self,
        cost_mapping: Mapping[str, CostContainer],
        iteration: int
    ) -> None:
        """Log the cost value(s) of the cost function(s)."""
        raise NotImplementedError

    def log_cosine_similarities(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        """Log the cosine similarities between the gradients of the cost functions."""
        raise NotImplementedError
    
    def log_gradient_magnitude_similarities(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        """Log the gradient magnitude similarities between the gradients of the cost functions."""
        raise NotImplementedError



def rad2deg(fa: ArrayLike) -> ArrayLike:
    return jnp.rad2deg(fa)

def imag(signal: ArrayLike) -> ArrayLike:
    return jnp.imag(signal)



class TensorboardLogger:
    """
    Log optimization diagnostics to a Tensorboard SummaryWriter instance.

    Parameters
    ----------
    writer : SummaryWriter
        Initialized tensorboard SummaryWriter instance.
    
    relative_gain_evaluator : RelativeGainEvaluator, optional
        Computes relative gains of cost functions with respect to base costs.
        Defaults to `None`, e.g. no comparison of relative gains.

    fa_postprocessor : Callable[[ArrayLike], ArrayLike], optional
        Postprocessor function for flip angles before logging.
        Defaults to `None`. If `None`, the default postprocessor is `jax.deg2rad`.
    """
    def __init__(
            self,
            writer: SummaryWriter,
            fa_plotter: Plotter,
            signal_plotter: Plotter,
            process_fa: Callable[[ArrayLike], ArrayLike] | None = None,
            process_signals: Callable[[ArrayLike], ArrayLike] | None = None,
            relative_gain_evaluator: RelativeGainEvaluator | None = None,
    ) -> None:
        self.writer = writer
        self.fa_plotter = fa_plotter
        self.signal_plotter = signal_plotter
        self.process_fa = process_fa or rad2deg
        self.process_signals = process_signals or imag
        self.relative_gain_evaluator = relative_gain_evaluator

    def log_gradients(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        diag.log_gradient_histograms(grad_mapping, self.writer, iteration)

    def log_costs(
        self,
        cost_mapping: Mapping[str, CostContainer],
        iteration: int
    ) -> None:
        diag.log_costs(cost_mapping, self.writer, iteration)

    def log_cosine_similarities(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        cosine_similarities = gradtools.compute_gradient_cosine_similarities(grad_mapping)
        diag.log_cosine_similarities(cosine_similarities, self.writer, iteration)

    def log_gradient_magnitude_similarities(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        magnitude_similarities = gradtools.compute_gradient_magnitude_similarities(grad_mapping)
        diag.log_gradient_magnitude_similarities(magnitude_similarities, self.writer, iteration)

    def log_relative_gains(
        self,
        cost_mapping: Mapping[str, CostContainer],
        iteration: int
    ) -> None:
        if not self.relative_gain_evaluator:
            return
        relative_gains = self.relative_gain_evaluator(cost_mapping)
        prefix: str = 'relative_gain'
        for name, value in relative_gains.items():
            tag = '/'.join((prefix, name))
            self.writer.add_scalar(tag=tag, scalar_value=float(value), global_step=iteration)

    def log_flipangles(
        self,
        flipangles: ArrayLike,
        iteration: int,
        *,
        close: bool = True
    ) -> None:
        """
        Log current state of flip angle train to tensorboard as matplotlib figure.
        """
        tag: str = 'plots/fa-trajectory'
        flipangles = self.process_fa(flipangles)
        fig, ax = self.fa_plotter.generate(flipangles)
        self.writer.add_figure(tag=tag, figure=fig, global_step=iteration, close=close)

    def log_signals(
        self,
        signals: ArrayLike,
        iteration: int,
        *,
        close: bool = True
    ) -> None:
        """
        Log current signals produced by the sequence to tensorboard as matplotlib figure.
        """
        tag: str = 'plots/signals'
        signals = self.process_signals(signals)
        fig, ax = self.signal_plotter.generate(signals)
        self.writer.add_figure(tag=tag, figure=fig, global_step=iteration, close=close)