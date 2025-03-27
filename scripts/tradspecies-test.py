"""
Sweep over constant flip angle initialization in the 45 - 60 degree range.
"""
import numpy as np

from pathlib import Path
from uuid import uuid4

import tqdm
import jax.numpy as jnp


import mrfoptools.initialization.initialization as initools

from mrfoptools.namegen import generate_name
from mrfoptools.experiment.gradientbased import RunInfo, construct_run_name, create_relaxometric_combinations, run_experiment
from mrfoptools.experiment.parameterclasses import Protocol, Initializations, Hyperparameter, BathtubLossParameters, InitializationType

def main():

    const_fa_values = np.linspace(45, 55, num=10)
    base_name: str = generate_name()
    sweep_ID: str = '-'.join((base_name, str(uuid4())))

    for i, const_fa_value in enumerate(tqdm.tqdm(const_fa_values, unit='subruns')):
        
        subrun_index = i + 1

        expdir = Path(
            f'$HOME/data/mrf-optruns-stable-archer/{sweep_ID}/subrun-{subrun_index}'
        )
        expdir.mkdir(exist_ok=True, parents=True)

        runinfo = RunInfo(
            base_name='constfa-traditional-species',
            run_dir=expdir,
            tags=['tradition-species', 'constfa'],
            seed=1337,
            subrun_index=subrun_index,
            sweep_ID=sweep_ID
        )
        # GM, WM, CSF, fat
        T1 = jnp.array([1300, 800, 3500, 375])
        T2 = jnp.array([100, 70, 1500, 60])

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

        const_fa_init = const_fa_value
        const_tr_init = 12
        const_phase_init = 0
        seed = 1337

        rng = np.random.default_rng(seed)
        init_fa_pattern = jnp.deg2rad(
            initools.create_constant_pattern(amplitude=const_fa_init, length=protocol.NR)
            + rng.normal(size=protocol.NR)
        )
        initializations = Initializations(
            fa=init_fa_pattern,
            tr=jnp.full(fill_value=const_tr_init, shape=protocol.NR),
            phases=jnp.full(fill_value=const_phase_init, shape=protocol.NR),
            seed=seed,
            type_=InitializationType.CONSTANT_PERTURBED,
            fa_value=const_fa_init,
            tr_value=const_tr_init
        )
        hyperparameters = Hyperparameter(
            step_size=0.01,
            max_iterations=500,
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