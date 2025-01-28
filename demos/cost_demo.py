"""
Initial demonstration of cost functions.
"""
# ruff: noqa: F401
# ruff: noqa: F841
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mrfoptools.epg.signal as epgsig
import mrfoptools.optimization.costfuncs as costfuncs
import mrfoptools.optimization.optimization as optim
import mrfoptools.epg.sequences as seq


def main():
    T1 = jnp.array([500.0, 1000.0, 3000.0])
    T2 = jnp.array([50.0, 100.0, 300.0])
    M0 = 1.0

    tr_path = Path('/home/jannik/storage/code/thirdparty/tg_mrf_optimization/initialization/tr_cao.npy')
    fa_path = Path('/home/jannik/storage/code/thirdparty/tg_mrf_optimization/initialization/fa_cao.npy')

    fa_cao = np.load(fa_path)
    tr_cao = np.load(tr_path)
    phases = np.full_like(fa_cao, fill_value=np.pi/2)

    fa_cao_jax = jnp.array(fa_cao)
    tr_cao_jax = jnp.array(tr_cao)
    phases_jax = jnp.array(phases)

    n_blocks = 1
    n_shots = len(fa_cao)
    preparations = [epgsig.PreparationType.INVERSION]
    ti = [20]
    t2preptime = [0]

    TE = 1
    inv_eff = 1 # Assume perfect inversion
    delta_B1 = 1 # Assume perfect B1 transmission

    function = jax.jit(costfuncs.orthogonality_criterion,
                       static_argnames=('preparation', 'inversion_efficiency', 'delta_B1'))

    # warmup run
    _ = function(
        T1, T2, M0, fa_cao_jax, tr_cao_jax, phases_jax,
        preparations[0], ti[0], TE, inv_eff, delta_B1
    )
    # benchmark run
    tstart = time.perf_counter()
    oval = function(
        T1, T2, M0, fa_cao_jax, tr_cao_jax, phases_jax,
        preparations[0], ti[0], TE, inv_eff, delta_B1
    )
    jax.block_until_ready(oval)
    tend = time.perf_counter()
    print(f"Orthogonality criterion evaluation took {tend - tstart:.5f} seconds.")

    print(oval)

    grad_fa_func = jax.grad(function, argnums=3)

    gval = grad_fa_func(
        T1, T2, M0, fa_cao_jax, tr_cao_jax, phases_jax,
        preparations[0], ti[0], TE, inv_eff, delta_B1
    )

    print(gval.shape)

    print(gval)

    simres = seq.simulate_fisp(
        T1=T1, T2=T2, M0=M0, fa=fa_cao_jax, TR=tr_cao_jax, phases=phases_jax,
        TI=ti[0], TE=TE, max_states=1000
    )

    print(simres.shape)
    print(simres.dtype)

    raise Exception("Stop here")


    optres = optim.optimize(
        T1=T1, T2=T2, M0=M0, step_size=5, max_iterations=200,
        intial_fa=fa_cao_jax, TR=tr_cao_jax, phases=phases_jax,
        preparation=preparations[0],
        TI=ti[0],
        TE=TE,
        inversion_efficiency=inv_eff, delta_B1=delta_B1
    )

    fahist, losshist = optres
    print(losshist)


if __name__ == '__main__':
    main()