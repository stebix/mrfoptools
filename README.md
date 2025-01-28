# mrfoptools

![example workflow](https://github.com/stebix/mrfoptools/actions/workflows/ci.yml/badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Tooling for programmatic optimization of MRF-specific flip angle and repetition time patterns
with gradient descent based algorithms.

We select `jax` as the numerical and automatic differentiation framework.

Concrete research purpose is the improvement of FISP-based MRF for inner ear (IE) imaging purposes.

Aims and Todos:

- [ ] Analytically optimal combination of gradients for sub-cost-functions (results from Multi-Task-Learning)

- [ ] Add parameterized Perlin-noise initialization functionality

Jannik Stebani 2024

### Acknowledgements

- Tom Griesler (PyTorch implementation @ Master Thesis)
- Max Gram     (Matalb EPG implementation)