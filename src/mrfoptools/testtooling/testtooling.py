"""
Tooling to progrmmatically test equivalence and
performance of different implementations of the same function.
"""
import warnings
import time
import copy

from collections.abc import Callable, Sequence, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Any

import numpy as np
import jax
import jax.numpy as jnp
import torch

import mrfoptools.testtooling.wrappers as wrappers
from mrfoptools.testtooling.wrappers import BenchmarkingMeasurement

def jaxwrapper(
    func: Callable,
    *,
    repeats: int = 3
) -> tuple[Callable, list[float]]:
    """
    Wrap jax function to auto-cast arguments to jax arrays and record
    runtime performance.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    repeats : int, optional
        Number of times to repeat the function call for timing.
        Defaults to 3.

    Returns
    -------
    tuple[Callable, list[float]]
        Wrapped function and list of timings.
        Note that the timings are recorded upon
        calling the wrapped function.
    """
    timings: list[float] = []

    castable = (np.ndarray,)
    def wrapper(*args, **kwargs):
        args = tuple(jnp.array(arg) if isinstance(arg, castable) else arg for arg in args)
        kwargs = {
            k : jnp.array(v)
            if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        # warmup
        _ = func(*args, **kwargs).block_until_ready()
        
        for _ in range(repeats):
            tstart = time.perf_counter()
            result = func(*args, **kwargs).block_until_ready()
            tend = time.perf_counter()
            
            timings.append(tend - tstart)
            
        return np.array(result) if isinstance(result, jax.Array) else result
        
    return (wrapper, timings)



def torchwrapper(
    func: Callable,
    *,
    repeats: int = 3
) -> tuple[Callable, list[float]]:
    """
    Wrap torch function to auto-cast arguments to torch tensors and record
    runtime performance.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    repeats : int, optional
        Number of times to repeat the function call for timing.
        Defaults to 3.

    Returns
    -------
    tuple[Callable, list[float]]
        Wrapped function and list of timings.
        Note that the timings are recorded upon
        calling the wrapped function.
    """
    timings: list[float] = []
    
    castable = (np.ndarray, float, int, np.float32, np.complex64)

    def wrapper(*args, **kwargs):
        args = tuple(torch.tensor(arg) if isinstance(arg, castable) else arg for arg in args)
        kwargs = {
            k : torch.tensor(v)
            if isinstance(v, castable) else v
            for k, v in kwargs.items()
        }
        # warmup
        _ = func(*args, **kwargs)
        
        for _ in range(repeats):
            tstart = time.perf_counter()
            result = func(*args, **kwargs)
            tend = time.perf_counter()
            
            timings.append(tend - tstart)
            
        return result.detach().resolve_conj().numpy() if isinstance(result, torch.Tensor) else result
        
    return (wrapper, timings)



def numpywrapper(
    func: Callable,
    *,
    repeats: int = 3
) -> tuple[Callable, list[float]]:
    """
    Wrap numpy function to record runtime performance.

    Parameters
    ----------
    func : Callable
        Function to wrap.
    repeats : int, optional
        Number of times to repeat the function call for timing.
        Defaults to 3.

    Returns
    -------
    tuple[Callable, list[float]]
        Wrapped function and list of timings.
        Note that the timings are recorded upon
        calling the wrapped function.
    """
    timings: list[float] = []
    
    def wrapper(*args, **kwargs):
        # warmup
        _ = func(*args, **kwargs)
        
        for _ in range(repeats):
            tstart = time.perf_counter()
            result = func(*args, **kwargs)
            tend = time.perf_counter()
            
            timings.append(tend - tstart)
            
        return result
        
    return (wrapper, timings)



COMPILERS: dict[str, Callable[[Callable], Callable]] = {
    'jax': jax.jit,
    'torch': torch.compile,
    'numpy': lambda x: x
}

WRAPPERS: dict[str, Callable] = {
    'jax': jaxwrapper,
    'torch': torchwrapper,
    'numpy': numpywrapper
}



class ExecutionMode(Enum):
    COMPILED = 'compiled'
    EAGER = 'eager'


@dataclass
class Implementation:
    func: Callable
    ID: str | None = None

    def __post_init__(self):
        self.ID = self.ID or self.func.__name__

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def wrap_as_casting(self) -> 'Implementation':
        raise NotImplementedError
    
    def wrap_as_benchmarked(self) -> 'Implementation':
        raise NotImplementedError


@dataclass
class JaxImplementation(Implementation):
    def wrap_as_casting(self):
        return JaxImplementation(
            wrappers.jaxcasting(self.func),
            self.ID
        )
    
    def wrap_as_benchmarked(
            self,
            timings_cache: list,
            repeats: int = 3
            ) -> 'JaxImplementation':
        return JaxImplementation(
            wrappers.jaxcasting(wrappers.jaxbenchmark(self.func, timings_cache, repeats)),
            self.ID
        )


@dataclass
class TorchImplementation(Implementation):
    def wrap_as_casting(self):
        return TorchImplementation(
            wrappers.torchcasting(self.func),
            self.ID
        )
    
    def wrap_as_benchmarked(
            self,
            timings_cache: list,
            repeats: int = 3
            ) -> 'TorchImplementation':
        return TorchImplementation(
            wrappers.torchcasting(wrappers.torchbenchmark(self.func, timings_cache, repeats)),
            self.ID
        )


@dataclass
class NumpyImplementation(Implementation):
    def wrap_as_casting(self):
        return NumpyImplementation(
            wrappers.numpycasting(self.func),
            self.ID
        )
    
    def wrap_as_benchmarked(
            self,
            timings_cache: list,
            repeats: int = 3
            ) -> 'NumpyImplementation':
        return NumpyImplementation(
            wrappers.numpycasting(wrappers.numpybenchmark(self.func, timings_cache, repeats)),
            self.ID
        )


def get_numpy_compiler():
    try:
        import numba as nb
    except ImportError:
        msg = ('Numba not found. Compilation is unavailable.  '
               'Using default eager numpy implementation.')
        warnings.warn(msg)
        return lambda x: x
    
    return nb.njit


IMPL2COMPILER: dict[type, Callable] = {
    JaxImplementation: COMPILERS['jax'],
    TorchImplementation: COMPILERS['torch'],
    NumpyImplementation: get_numpy_compiler()
}


def autocompile(implementations: Sequence[Implementation]) -> list[Implementation]:
    """
    Create extended list of implementations by adding compiled
    versions of the given implementations.
    """
    extended_implementations: list[Implementation] = [
        *copy.copy(implementations)
    ]
    for impl in implementations:
        compiler = IMPL2COMPILER[type(impl)]
        func = compiler(impl.func)
        ID = f'{impl.ID}_compiled'
        extended_implementations.append(
            type(impl)(func=func, ID=ID)
        )
    return extended_implementations



def benchmark(
    implementations: Sequence[Implementation],
    args: tuple | None = None,
    kwargs: Mapping[str, Any] | None = None,
    *,
    repeats: int = 3
) -> tuple[list, list]:
    
    timings: list[BenchmarkingMeasurement] = []
    results: list = []

    args = args or ()
    kwargs = kwargs or {}

    for impl in implementations:
        # closure over the benchmark cache: Write measurements as dict into cache
        impl_benchmark_cache = []
        impl_wrapped = impl.wrap_as_benchmarked(impl_benchmark_cache, repeats)

        result = impl_wrapped(*args, **kwargs)

        results.append((impl.ID, result))
        params = impl_benchmark_cache[0]
        timings.append(
            BenchmarkingMeasurement(ID=impl.ID, **params)
        )

    return (timings, results)


class CompTest:
    abstol: float = 1e-6
    reltol: float = 1e-5
    compilers = COMPILERS
    wrappers = WRAPPERS

    def __init__(
        self,
        jax_impl: Callable,
        torch_impl: Callable,
        numpy_impl: Callable,
        *,
        abstol: float | None = None,
        reltol: float | None = None,
        equal_nan: bool = False,
    ) -> None:
        
        self.impls = self.construct_impls(
            ('jax', jax_impl),
            ('torch', torch_impl),
            ('numpy', numpy_impl)
        )

        self.abstol = abstol or self.abstol
        self.reltol = reltol or self.reltol
        self.equal_nan = equal_nan


    def construct_impls(self, *args: tuple[str, Callable]) -> dict[str, Callable]:
        impls: dict[str, Callable] = {}
        for name, func in args:
            impls[name] = func
            impls[f'{name}_compiled'] = self.compilers[name](func)
        return impls


    def equivalence(self, *args, **kwargs) -> None:
        """
        Test equivalence of the implementations.
        Inputs should be cast as numpy.ndarrays.
        """
        for impl in self.impls:
            wrapper = self.wrappers[impl]

        