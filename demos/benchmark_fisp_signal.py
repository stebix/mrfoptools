"""
Benchmark runtime scaling with increasing number of species.

@Author: Jannik Stebani 2025
"""
import time
import itertools
from collections import defaultdict

import jax
import jax.numpy as jnp
import rich
import rich.table
import tqdm

from rich.console import Console
from jax.typing import ArrayLike

from mrfoptools.epg.sequences.helpers import quickmake_simulate_fisp, get_basic_configuration



def main() -> None:


    T1_base = jnp.array([1500, 1750, 2000, 2250, 2500, 2750, 3000])
    T2_base = jnp.array([500, 500, 600, 700, 800, 900, 1000, 1500])

    T1_base = jnp.linspace(start=0.05, stop=3750, num=100)
    T2_base = jnp.linspace(start= 0.015, stop=3000, num=100)

    T1 = []
    T2 = []

    for t1val, t2val in itertools.product(T1_base, T2_base):
        T1.append(t1val)
        T2.append(t2val)

    T1 = jnp.array(T1)
    T2 = jnp.array(T2)

    n_species = [1, 3, 5, 10, 15, ]#25, 50, 75, 100, 125]#150, 200]

    basic_config = get_basic_configuration()
    fa = basic_config['fa']
    tr = basic_config['tr']

    repeats: int = 15

    benchmark_times: dict[int, list] = defaultdict(list)

    wrapped_n_species = tqdm.tqdm(n_species)

    for n in wrapped_n_species:

        wrapped_n_species.set_postfix_str(f'current n = {n}')
        T1_subset = T1[:n]
        T2_subset = T1[:n]

        simulate_fisp = jax.jit(quickmake_simulate_fisp(T1=T1_subset, T2=T2_subset, max_states=500))
        
        # warmup
        _ = simulate_fisp(fa, tr).block_until_ready()

        for _ in range(repeats):
            tstart = time.perf_counter()
            _ = simulate_fisp(fa, tr).block_until_ready()
            tend = time.perf_counter()

            duration = tend - tstart

            benchmark_times[n].append(duration)
   
    render(benchmark_times)


    print('Profiling for last element:')

    with jax.profiler.trace('/tmp/jax-trace', create_perfetto_link=True):
        _ = simulate_fisp(fa, tr).block_until_ready()


    print('Goodbye')


def render(results: dict[int, list[float]]) -> None:
    results = {k : jnp.array(v) for k, v in results.items()}

    table = rich.table.Table(title='Benchmark Results', style='green')

    table.add_column('Signal Atom Count', justify='right', style='cyan')
    table.add_column('Avg (ms)', style='magenta')
    table.add_column('StDev (ms)')
    table.add_column('Avg per Atom (ms) ')
    table.add_column('Median per Atom (ms) ')
    table.add_column('Min/Max per Atom (ms) ')


    f: float = 1e3

    for k, v in results.items():

        table.add_row(
            *[str(k), f'{jnp.mean(f*v):.3f}',
              f'{jnp.std(f*v, ddof=1):.4f}', f'{jnp.mean(f*v)/k:.4f}',
              f'{jnp.median(f*v)/k:.4f}', f'[{jnp.min(f*v)/k:.2f}, {jnp.max(f*v)/k:.2f}]']
        )

    console = Console()
    console.print(table)


if __name__ == '__main__':
    main()
        