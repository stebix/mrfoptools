import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import mrfoptools.epg.core as epg
import mrfoptools.epg.signal.signal as epgsig
import mrfoptools.contrib.signalmodel_epg as contrib

import mrfoptools.epg.sequences.fisp as seq

import mrfoptools.bloch.signal.signal as blochsig


def main():
    print('loading data')
    tr_path = Path('/home/jannik/storage/code/thirdparty/tg_mrf_optimization/initialization/tr_cao.npy')
    fa_path = Path('/home/jannik/storage/code/thirdparty/tg_mrf_optimization/initialization/fa_cao.npy')

    fa_cao = np.load(fa_path)
    tr_cao = np.load(tr_path)

    ti = [20]

    phases = np.full_like(fa_cao, fill_value=0.0)

    t1 = 1000 # T1 in ms 
    t2 = 100 # T2 in ms
    m0 = 1
    te = 1
    inv_eff = 1 # Assume perfect inversion

    n_iso = 333

    fa_cao_jax = jnp.deg2rad(jnp.array(fa_cao))
    tr_cao_jax = jnp.array(tr_cao)
    phases_jax = jnp.array(phases)

    signal = blochsig.compute_signal(
        t1, t2, fa_cao_jax, tr_cao_jax, phases_jax, m0, ti[0], te, n_iso, inv_eff, dephasing=2
    )

    print(signal.shape)


if __name__ == '__main__':
    main()