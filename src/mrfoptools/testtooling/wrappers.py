import time

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch
import numpy as np
import jax.numpy as jnp
import jax


ResultType = TypeVar('ResultType')



def casting(
    func: Callable,
    castables: tuple[type, ...],
    castfunc: Callable,
    postprocessor: Callable,
) -> Callable:
    """
    Create a casting wrapper that casts input arguments to a specific type
    and casts the output to a specific type.
    """
    def wrapper(*args, **kwargs):
        args = tuple(castfunc(arg) if isinstance(arg, castables) else arg for arg in args)
        kwargs = {
            k : castfunc(v)
            if isinstance(v, castables) else v
            for k, v in kwargs.items()
        }
        result = func(*args, **kwargs)
        return postprocessor(result)
    return wrapper



def jaxcasting(
    func: Callable,
) -> Callable:
    """
    Wrap callable to auto-cast input arguments to jax arrays
    an output values to numpy arrays.
    """
    castable: tuple[type, ...] = (np.ndarray,)
    castfunc = jnp.array
    def to_numpy(x):
        if isinstance(x, jax.Array):
            return np.array(x)
        return x

    return casting(func, castable, castfunc, to_numpy)


def torchcasting(
    func: Callable,
) -> Callable:
    """
    Wrap callable to auto-cast input arguments to torch tensors
    an output values to numpy arrays.
    """
    castable: tuple[type, ...] = (np.ndarray, float, np.float32, np.float64, np.complex64, np.complex128)
    castfunc = torch.tensor
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().resolve_conj().numpy()
        return x

    return casting(func, castable, castfunc, to_numpy)


def numpycasting(
    func: Callable,
) -> Callable:
    """
    Mock wrapper for compatibility with the other wrappers.
    """
    return func


class TimeUnit:
    SECOND =  'second'
    MILLISECOND = 'millisecond'
    MICROSECOND = 'microsecond'
    NANOSECOND = 'nanosecond'



@dataclass(frozen=True)
class BenchmarkingMeasurement:
    ID: str
    runtimes: tuple[float]
    repeats: int
    warmup: float | None = None
    unit: TimeUnit = TimeUnit.SECOND




def benchmark(
    func: Callable[..., ResultType],
    benchmark_cache: list,
    repeats: int = 3,
    do_warmup: bool = True,
    postprocessor: Callable[[ResultType], ResultType] = lambda x: x,
) -> ResultType:
    """
    Wrap function such that benchmarking results are recorded into the cache.
    """
    # timings_cache: list[tuple[str, float] | tuple[str, list[float]]] = []
    unit: TimeUnit = TimeUnit.SECOND

    def wrapper(*args, **kwargs):

        warmup_time = None

        if do_warmup:
            wup_start = time.perf_counter()
            _ = postprocessor(func(*args, **kwargs))
            wup_end = time.perf_counter()
            warmup_time = wup_end - wup_start

        runtimes: list[float] = []

        for _ in range(repeats):
            tstart = time.perf_counter()
            result = postprocessor(func(*args, **kwargs))
            tend = time.perf_counter()
            runtimes.append(tend - tstart)

        measurement = {'runtimes': runtimes, 'repeats': repeats, 'warmup': warmup_time, 'unit': unit}
        benchmark_cache.append(measurement)

        return result
    
    return wrapper


def jaxbenchmark(
    func: Callable,
    timings_cache: list,
    repeats: int = 3,
    do_warmup: bool = True,
) -> Callable:
    postprocessor = jax.block_until_ready
    return benchmark(func, timings_cache, repeats, do_warmup, postprocessor)


def torchbenchmark(
    func: Callable,
    timings_cache: list,
    repeats: int = 3,
    do_warmup: bool = True,
) -> Callable:
    return benchmark(func, timings_cache, repeats, do_warmup)


def numpybenchmark(
    func: Callable,
    timings_cache: list,
    repeats: int = 3,
    do_warmup: bool = True,
) -> Callable:
    return benchmark(func, timings_cache, repeats, do_warmup)