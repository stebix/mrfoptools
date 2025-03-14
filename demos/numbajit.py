"""
Test sequence numpy implementations.
"""
import os
import numpy as np
import time
import tqdm

import numba as nb

from mrfoptools.epg.sequences.numpy.fisp import simulate_fisp
from mrfoptools.epg.sequences.helpers import get_basic_configuration, extract_specialization_kwargs

nb.config.NUMBA_DEBUG = True

def main():
    """
    Basic smoke test for the numpy implementation of the FISP sequence.
    """
    repeats = 5

    pbar = tqdm.tqdm(total=8)

    for parallel in [True, False]:
        for n_species in [50, 100]:

            pbar.set_postfix_str(f'{n_species=} {parallel=}')

            conf = get_basic_configuration(backend='numpy', n_species=n_species, max_states=500)
            kwargs = extract_specialization_kwargs(conf)

            fa = conf.get('fa')
            TR = conf.get('tr')

            simulate_fisp_jit = nb.njit(simulate_fisp, fastmath=True, parallel=parallel, nogil=True)

            _ = simulate_fisp_jit(
                fa=fa,
                TR=TR,
                **kwargs
            )
            runtimes: list[float] = []

            for _ in range(repeats):
                tstart = time.time()
                sigs = simulate_fisp_jit(
                    fa=fa,
                    TR=TR,
                    **kwargs
                )
                tend = time.time()
                runtimes.append(tend - tstart)

            runtimes = np.array(runtimes)

            pbar.update()
            pbar.write(f'Params {n_species=} {parallel=} :: mean runtime -> {np.mean(runtimes)}')


    #print(simulate_fisp_jit.parallel_diagnostics(level=1))

if __name__ == '__main__':
   main()