# project overview

In this scientific computing project, we implement physically accurate simulations of magnetic resonancen
imaging sequences based on the Bloch equation simulation or extended phase graph algorithm.

# tech stack

We use conda environments for pacakge and environment management.
For the numerical simulation, we utilize `numpy` and `jax`.
For a wide range of functions, we implement the same API in both frameworks.
Jax utilizes JIT-compilation and numpy code can be accelerated via the `numba` framework.
