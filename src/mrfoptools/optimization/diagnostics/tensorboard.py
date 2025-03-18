from collections.abc import Mapping, Callable
from numbers import Number

import attrs
import jax
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt


import mrfoptools.optimization.diagnostics.diagnostics as diag
import mrfoptools.optimization.gradtools.gradtools as gradtools

from mrfoptools.optimization.costgrad import NumpyCostGradTuple

ArrayLike = np.ndarray | jax.Array

class _DiagnosticLogger:

    def log_gradients(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        """Log the gradient(s) of the cost function(s)."""
        raise NotImplementedError
    
    def log_costs(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        """Log the cost value(s) of the cost function(s)."""
        raise NotImplementedError

    def log_cosine_similarities(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        """Log the cosine similarities between the gradients of the cost functions."""
        raise NotImplementedError
    
    def log_gradient_magnitude_similarities(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        """Log the gradient magnitude similarities between the gradients of the cost functions."""
        raise NotImplementedError



@attrs.define
class BaseCost:
    """
    Container for base cost values.
    """
    costname: str
    refname: str
    value: Number
    range: diag.CostValueRange


class RelativeGainEvaluator:
    """
    Compute relative gains of cost functions with respect to base costs.
    
    Relative gains are computed by matching base costs set in the initializer method
    via their `costname` to supplied transient costs in the
    `cost_grad_mapping` argument of the `__call__` method.
    """
    def __init__(
        self,
        *base_costs: BaseCost,
        prefix: str = 'relative_gain'
    ) -> None:
        self.base_costs: list[BaseCost] = base_costs
        self.prefix: str = prefix

    def __call__(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple]
    ) -> dict[str, float]:
        
        relative_gains: dict[str, float] = {}
        for base_cost in self.base_costs:
            logname = f'{self.prefix}/{base_cost.costname}-{base_cost.refname}'
            cg_tuple = cost_grad_mapping[base_cost.costname]
            if base_cost.range == diag.CostValueRange.POSITIVE:
                relative_gain = base_cost.value / cg_tuple.cost
            elif base_cost.range == diag.CostValueRange.NEGATIVE:
                relative_gain = cg_tuple.cost / base_cost.value
            else:
                raise ValueError(f'Invalid cost value range: {base_cost.range}')
            
            relative_gains[logname] = relative_gain
        return relative_gains
            
    def __str__(self):
        return f'{self.__class__.__name__}(base_costs=\'{self.base_costs}\')'
    
    def __repr__(self):
        return str(self)


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
    default_fa_postprocessor = jax.deg2rad

    def __init__(
            self,
            writer: SummaryWriter,
            relative_gain_evaluator: RelativeGainEvaluator | None = None,
            fa_postprocessor: Callable[[ArrayLike], ArrayLike] | None = None
    ) -> None:
        self.writer = writer
        self.relative_gain_evaluator = relative_gain_evaluator
        self.fa_postprocessor = fa_postprocessor or self.default_fa_postprocessor

    def log_gradients(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        diag.log_costs(cost_grad_mapping, self.writer, iteration)

    def log_costs(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        diag.log_costs(cost_grad_mapping, self.writer, iteration)

    def log_cosine_similarities(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        cosine_similarities = gradtools.compute_gradient_cosine_similarities(cost_grad_mapping)
        diag.log_cosine_similarities(cosine_similarities, self.writer, iteration)

    def log_gradient_magnitude_similarities(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        magnitude_similarities = gradtools.compute_gradient_magnitude_similarities(cost_grad_mapping)
        diag.log_gradient_magnitude_similarities(magnitude_similarities, self.writer, iteration)

    def log_relative_gains(
        self,
        cost_grad_mapping: Mapping[str, NumpyCostGradTuple],
        iteration: int
    ) -> None:
        if not self.relative_gain_evaluator:
            return
        relative_gains = self.relative_gain_evaluator(cost_grad_mapping)
        for tag, value in relative_gains.items():
            self.writer.add_scalar(tag=tag, scalar_value=float(value), global_step=iteration)

    def log_flipangles(
        self,
        flipangles: ArrayLike
    ) -> None:
        """
        Log current state of flip angle train to tensorboard as matplotlib figure.
        """
        fig, ax = plt.subplots()
        ax.plot(np.asarray(self.fa_postprocessor(flipangles)))