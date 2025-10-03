"""
Sweep over constant flip angle initialization in the 45 - 60 degree range.
"""
import numpy as np

from pathlib import Path
from uuid import uuid4

import tqdm
import jax.numpy as jnp
from rich.console import Console

import mrfoptools.initialization.initialization as initools

from mrfoptools.namegen import generate_name
from mrfoptools.experiment.gradientbased import RunInfo, run_experiment, create_relaxometric_combinations
from mrfoptools.experiment.parameterclasses import Protocol, Initializations, Hyperparameter, BathtubLossParameters, InitializationType

def main():

    base_name: str = generate_name()
    sweep_ID: str = '-'.join((base_name, str(uuid4())))

    subrun_index = 2

    expdir = Path(
        f'$HOME/data/mrf-optruns-stable-archer/{sweep_ID}/subrun-{subrun_index}'
    )
    expdir.mkdir(exist_ok=True, parents=True)

    console = Console()
    console.log(
        f'Using experiment directory: \'{expdir}\''
    )

    runinfo = RunInfo(
        base_name='yun-traditional-species',
        run_dir=expdir,
        tags=['tradition-species', 'yun'],
        seed=1337,
        subrun_index=subrun_index,
        sweep_ID=sweep_ID
    )
    T1, T2 = create_relaxometric_combinations(
        T1=jnp.array([2250, 2500, 2750, 3000]),
        T2=np.linspace(500, 1250, num=8)
    )

    protocol = Protocol(
        M0=1.0,
        T1=T1,
        T2=T2,
        TE=2.2,
        TI=20,
        phase=0,
        inversion_efficiency=1.0,
        max_states=600,
        NR=1000
    )

    const_tr_init = 12
    const_phase_init = 0
    seed = 1337

    init_fa_pattern = jnp.deg2rad(
        initools.load_yun_pattern(style='tight', element='fa')
    )
    initializations = Initializations(
        fa=init_fa_pattern,
        tr=jnp.full(fill_value=const_tr_init, shape=protocol.NR),
        phases=jnp.full(fill_value=const_phase_init, shape=protocol.NR),
        seed=seed,
        type_=InitializationType.YUN_TIGHT,
        fa_value=None,
        tr_value=const_tr_init
    )
    hyperparameters = Hyperparameter(
        step_size=0.01,
        max_iterations=1000,
        min_fa=np.deg2rad(1),
        max_fa=np.deg2rad(90),
        bathtub_loss_parameters=BathtubLossParameters(
            radius=2.0,
            alpha=0.1,
            beta=0.5,
            gamma=10
        )
    )

    run_experiment(
        run_info=runinfo,
        protocol=protocol,
        initializations=initializations,
        hyperparameters=hyperparameters,
        leave_pbar=False
    )

if __name__ == '__main__':
    main()