import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import torch
import rich

import mrfoptools.epg.epg as epg
import mrfoptools.epg.signal as epgsig
import mrfoptools.contrib.signalmodel_epg as contrib



def main():
    print('loading data')
    tr_path = Path('/home/jannik/storage/code/thirdparty/tg_mrf_optimization/initialization/tr_cao.npy')
    fa_path = Path('/home/jannik/storage/code/thirdparty/tg_mrf_optimization/initialization/fa_cao.npy')

    fa_cao = np.load(fa_path)
    tr_cao = np.load(tr_path)

    n_blocks = 1
    n_shots = len(fa_cao)
    preparations = [epgsig.PreparationType.INVERSION]
    ti = [20]
    t2preptime = [0]

    phases = np.full_like(fa_cao, fill_value=np.pi/2)

    t1 = 1000 # T1 in ms 
    t2 = 100 # T2 in ms
    m0 = 1
    te = 1
    inv_eff = 1 # Assume perfect inversion
    delta_B1 = 1 # Assume perfect B1 transmission

    torchimpl_start = time.perf_counter()
    signal_tom = contrib.calculate_signal_epg(
        t1, t2, m0, beats=n_blocks, shots=n_shots, prep=[1], fa=fa_cao,
        tr=tr_cao, ph=phases, ti=ti, t2te=t2preptime, te=te,
        inv_eff=inv_eff, delta_B1=delta_B1
    )
    torchimpl_end = time.perf_counter()
    print(f'torchimpl runtime: {torchimpl_end - torchimpl_start}')

    # manually prepare omega and precompute TE=specific terms and operators
    omega = jnp.hstack(
        (jnp.array([[0.0], [0.0], [m0]], dtype=jnp.complex64), jnp.zeros((3, 550), dtype=jnp.complex64))
    )
    inv_op = epg.inversion(inv_eff)
    bte = epg.b_epg(t1, te)
    rte = epg.r_epg(t1, t2, te)

    omega = epg.r_epg(t1, t2, ti[0]) @ inv_op @ omega
    omega = omega.at[2, 0].set(omega[2, 0] + m0 * epg.b_epg(t1, ti[0]))


    # warmup and jit
    _ = epgsig.compute_signal_inner_compiled(
        omega, T1=t1, T2=t2, M0=m0, fa=np.deg2rad(fa_cao), TR=tr_cao, phases=phases, TE=te, b_TE=bte, r_TE=rte
    ).block_until_ready()

    start_time_inner_compiled = time.perf_counter()
    sig_inner_compiled = epgsig.compute_signal_inner_compiled(
        omega, T1=t1, T2=t2, M0=m0, fa=np.deg2rad(fa_cao), TR=tr_cao, phases=phases, TE=te, b_TE=bte, r_TE=rte
    ).block_until_ready()
    end_time_inner_compiled = time.perf_counter()
    print(f'jax inner compiled runtime: {end_time_inner_compiled - start_time_inner_compiled}')

    
    ##################### COMPUTE LAX 
    fa_cao = jnp.deg2rad(jnp.array(fa_cao))
    tr_cao = jnp.array(tr_cao)
    phases = jnp.array(phases)

    # warmup and jit
    _ = epgsig.compute_signal_optimized(
        omega, T1=t1, T2=t2, M0=m0, fa=fa_cao, TR=tr_cao, phases=phases, TE=te, b_TE=bte, r_TE=rte
    ).block_until_ready()

    start_time_lax_optimized_uncompiled = time.perf_counter()
    sig_optimized = epgsig.compute_signal_optimized(
        omega, T1=t1, T2=t2, M0=m0, fa=fa_cao, TR=tr_cao, phases=phases, TE=te, b_TE=bte, r_TE=rte
    ).block_until_ready()
    end_time_lax_optimized_uncompiled = time.perf_counter()
    print(f'jax lax optimized uncompiled runtime: {end_time_lax_optimized_uncompiled - start_time_lax_optimized_uncompiled}')

    # warmup and jit
    lax_comp = jax.jit(epgsig.compute_signal_optimized)
    _ = lax_comp(
        omega, T1=t1, T2=t2, M0=m0, fa=fa_cao, TR=tr_cao, phases=phases, TE=te, b_TE=bte, r_TE=rte
    ).block_until_ready()

    start_time_optimized_compiled = time.perf_counter()
    sig_optimized_compiled = lax_comp(
        omega, T1=t1, T2=t2, M0=m0, fa=fa_cao, TR=tr_cao, phases=phases, TE=te, b_TE=bte, r_TE=rte
    ).block_until_ready()
    end_time_optimized_compiled = time.perf_counter()
    print(f'jax optimized compiled runtime: {end_time_optimized_compiled - start_time_optimized_compiled}')


    assert np.allclose(sig_optimized_compiled, sig_optimized, rtol=1e-5, atol=1e-5)
    assert np.allclose(sig_inner_compiled, sig_optimized, rtol=1e-5, atol=1e-5)
    assert np.allclose(signal_tom.detach().numpy(), sig_optimized, rtol=1e-5, atol=1e-5)




if __name__ == '__main__':
    main()

    print('finished')