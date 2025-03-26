import os
from collections.abc import Mapping, Callable

import jax
import jax.numpy as jnp
import neptune.types
import numpy as np
import neptune
import matplotlib.pyplot as plt

from mrfoptools.optimization.costgrad import CostContainer, GradientContainer
from mrfoptools.namegen import generate_name
from mrfoptools.optimization.diagnostics.plothelpers import Plotter


ArrayLike = np.ndarray | jax.Array


def rad2deg(fa: ArrayLike) -> ArrayLike:
    return jnp.rad2deg(fa)

def imag(signal: ArrayLike) -> ArrayLike:
    return jnp.imag(signal)


def create_run(
    name: str | None = None,
    **kwargs
    ) -> neptune.Run:
    API_TOKEN = os.environ.get("NEPTUNE_API_TOKEN")
    name = name or generate_name()
    run = neptune.init_run(
        project='mr-fingerprinting/init-test',
        api_token=API_TOKEN,
        name=name,
        **kwargs
    )
    return run



class NeptuneLogger:

    def __init__(
        self,
        run: neptune.Run,
        fa_plotter: Plotter,
        signal_plotter: Plotter,
        process_fa: Callable[[ArrayLike], ArrayLike] = rad2deg,
        process_signals: Callable[[ArrayLike], ArrayLike] = imag
    ) -> None:
        self.run = run
        self.fa_plotter = fa_plotter
        self.signal_plotter = signal_plotter
        self.process_fa = process_fa
        self.process_signals = process_signals

    def log_costs(
        self,
        cost_mapping: Mapping[str, CostContainer],
        iteration: int
    ) -> None:
        for cost_name, cost_container in cost_mapping.items():
            self.run[f'costs/{cost_name}'].append(cost_container.cost, step=iteration)
    
    def log_gradients(
        self,
        grad_mapping: Mapping[str, GradientContainer],
        iteration: int
    ) -> None:
        for grad_name, grad_container in grad_mapping.items():
            self.run[f'grads/{grad_name}'].append(
                grad_container.grad.ravel().mean(), step=iteration
            )

    def log_gradient(
        self,
        gradient: ArrayLike,
        name: str,
        iteration: int
    ) -> None:
        """Log a single gradient vector."""
        self.run[f'grads/{name}'].append(gradient.ravel().mean(), step=iteration)

    
    def log_cosine_similarities(
        self,
        cosine_similarities: Mapping[str, float],
        iteration: int
    ) -> None:
        for name, value in cosine_similarities.items():
            self.run[f'grad_cos_sim/{name}'].append(value, step=iteration)

    
    def log_magnitude_similarities(
        self,
        magnitude_similarities: Mapping[str, float],
        iteration: int
    ) -> None:
        for name, value in magnitude_similarities.items():
            self.run[f'grad_mag_sim/{name}'].append(value, step=iteration)

    
    def log_relative_gains(
        self,
        relative_gains: Mapping[str, float],
        iteration: int
    ) -> None:
        prefix: str = 'relative_gain'
        for name, value in relative_gains.items():
            tag = '/'.join((prefix, name))
            self.run[tag].append(value, step=iteration)

    
    def log_flipangles(
        self,
        flipangles: ArrayLike,
        iteration: int,
        *,
        close: bool = True
    ) -> None:
        flipangles = self.process_fa(flipangles)
        fig, _ = self.fa_plotter.generate(flipangles)
        self.run['plots/flipangles'].append(fig, step=iteration)
        if close:
            plt.close(fig)

    def log_signals(
        self,
        signals: ArrayLike,
        iteration: int,
        *,
        close: bool = True
    ) -> None:
        signals = self.process_signals(signals)
        fig, _ = self.signal_plotter.generate(signals)
        self.run['plots/signals'].append(fig, step=iteration)
        if close:
            plt.close(fig)
