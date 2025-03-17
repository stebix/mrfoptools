"""
Create stand in module for numba with focus on `njit` and `jit` decorators.
Intended uti;ization: Replace `import numba` for environments where numba is not available.
"""
from collections.abc import Callable


def jit(*args, **kwargs) -> Callable:
    """Mock jit wrapper - does nothing and calls the function as is."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def njit(*args, **kwargs) -> Callable:
    """Mock njit wrapper - does nothing and calls the function as is."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator