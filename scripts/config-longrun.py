"""
Perform experiments with different initialization strategies
and a long-running optimimzation loop with conFIG analytical gradient
combination.
"""
import numpy as np

from pathlib import Path
from uuid import uuid4
from collections.abc import Sequence

import tqdm
import jax
import jax.numpy as jnp
import optax
import datetime
import zoneinfo
import attrs
import logging

import mrfoptools.initialization.initialization as initools
import mrfoptools.optimization.blocks as optblocks
import mrfoptools.optimization.costfuncs as costfuncs
import mrfoptools.epg.sequences.jax.fisp as epgfisp
import mrfoptools.optimization.costgrad as costgrad
import mrfoptools.optimization.gradtools.gradtools as gradtools
import mrfoptools.optimization.gradtools.jacdesc as jacdesc 
import mrfoptools.optimization.diagnostics.references as refcs

from mrfoptools.namegen import generate_name
from mrfoptools.experiment.gradientbased import RunInfo, create_relaxometric_combinations, construct_run_name
from mrfoptools.experiment.parameterclasses import Protocol, Initializations, Hyperparameter, BathtubLossParameters, InitializationType
from mrfoptools.optimization.costgrad import cost_grad_builder
from mrfoptools.optimization.diagnostics.plothelpers import CachingPlotter
from mrfoptools.optimization.diagnostics.neptune import NeptuneLogger, create_run

from mrfoptools.io.io import SaveFormat
from mrfoptools.io.pickleutils import store_pickle
from mrfoptools.io.bag import OptimizationBag, store_optimization_bag



DEFAULT_LOGGER_NAME: str = '.'.join(('main', __name__))
logger = logging.getLogger(DEFAULT_LOGGER_NAME)


def main():

    base_name: str = generate_name()
    sweep_ID: str = '-'.join((base_name, str(uuid4())))


    NR = 1000
    seed = 1337
    const_fa_init = 49
    const_tr_init = 12
    const_phase_init = 0

    # create a constant flip angle initialization pattern
    rng = np.random.default_rng(seed)
    const_fa_init = jnp.deg2rad(
          initools.create_constant_pattern(amplitude=const_fa_init, length=NR)
        + rng.normal(size=NR)
    )

    const_initializations = Initializations(
        fa=const_fa_init,
        tr=jnp.full(fill_value=const_tr_init, shape=NR),
        phases=jnp.full(fill_value=const_phase_init, shape=NR),
        seed=seed,
        type_=InitializationType.CONSTANT_PERTURBED,
        fa_value=const_fa_init,
        tr_value=const_tr_init
    )
    yun_tight_fa_init = jnp.deg2rad(initools.load_yun_pattern(style='tight', element='fa'))
    yun_tight_initializations = Initializations(
        fa=yun_tight_fa_init,
        tr=jnp.full(fill_value=const_tr_init, shape=NR),
        phases=jnp.full(fill_value=const_phase_init, shape=NR),
        seed=seed,
        type_=InitializationType.YUN_TIGHT
    )

    yun_canonical_fa_init = jnp.deg2rad(initools.load_yun_pattern(style='canonical', element='fa'))
    yun_canonical_initializations = Initializations(
        fa=yun_canonical_fa_init,
        tr=jnp.full(fill_value=const_tr_init, shape=NR),
        phases=jnp.full(fill_value=const_phase_init, shape=NR),
        seed=seed,
        type_=InitializationType.YUN_CANONICAL
    )

    initializations_list = [yun_tight_initializations, yun_canonical_initializations, const_initializations]

    for i, initializations in enumerate(tqdm.tqdm(initializations_list, unit='subruns')):
        
        subrun_index = i + 1

        expdir = Path(
            f'/home/jannik/storage/mrf-optruns-stable-archer/{sweep_ID}/subrun-{subrun_index}'
        )
        expdir.mkdir(exist_ok=True, parents=True)

        runinfo = RunInfo(
            base_name='conFIG-longrun-initsweep',
            run_dir=expdir,
            tags=['init-sweep', 'conFIG', 'longrun'],
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

        hyperparameters = Hyperparameter(
            step_size=0.0005,
            max_iterations=40000,
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






def run_experiment(
        run_info: RunInfo,
        protocol: Protocol,
        initializations: Initializations,
        hyperparameters: Hyperparameter,
        save_format: SaveFormat | Sequence[SaveFormat] = SaveFormat.ZARR,
        leave_pbar: bool = True
    ):

    # base_name: str = generate_name()
    #sweep_ID: str = '-'.join((base_name, str(uuid4())))
    #sweep_run = create_run(name=f'{base_name}-sweep', tags=['sweep-level'])
    #sweep_run['sys/group_tags'].add(sweep_ID)

    run_name: str = construct_run_name(run_info.base_name, run_info.subrun_index)

    run = create_run(
        name=run_name,
        tags=run_info.tags)

    if run_info.sweep_ID is not None:    
        run['sys/group_tags'].add(run_info.sweep_ID)

    # construct the smoothness loss function
    radius = hyperparameters.bathtub_loss_parameters.radius
    alpha = hyperparameters.bathtub_loss_parameters.alpha
    beta = hyperparameters.bathtub_loss_parameters.beta
    gamma = hyperparameters.bathtub_loss_parameters.gamma
    bathtub_loss = costfuncs.construct_bathtub_loss(
        radius=radius, alpha=alpha, beta=beta, gamma=gamma
    )
    max_iterations = hyperparameters.max_iterations
    min_fa = hyperparameters.min_fa
    max_fa = hyperparameters.max_fa

    run['params/bathtub_radius'] = radius
    run['params/bathtub_alpha'] = alpha
    run['params/bathtub_beta'] = beta
    run['params/bathtub_gamma'] = gamma
    run['params/max_iterations'] = max_iterations
    run['params/min_fa'] = float(min_fa)
    run['params/max_fa'] = float(max_fa)

    def fa_diff_loss(fa: jax.Array) -> jax.Array:
        diff = fa[1:] - fa[:-1]
        smoothed = bathtub_loss(diff)
        return jnp.sum(smoothed)
    
    run['params/step_size'] = hyperparameters.step_size


    # extract protocol settings
    NR = protocol.NR
    M0 = protocol.M0
    TE = protocol.TE
    TI = protocol.TI
    max_states = protocol.max_states
    inversion_efficiency = protocol.inversion_efficiency
    T1 = protocol.T1
    T2 = protocol.T2
    logger.info(f'Utilizing {len(T1)} T1 values and {len(T2)} T2 values.')
    assert len(T1) == len(T2), 'T1 and T2 value count mismatch'
    n_species = T1.shape[0] # noqa F841


    phases = initializations.phases
    tr_pattern = initializations.tr
    initial_fa = initializations.fa

    reference_patterns = refcs.fetch_default_reference_fa_patterns(unit='rad')
    # also track relative gain against the initial flip angle pattern
    reference_patterns['initial'] = initial_fa

    forward_fatr = epgfisp.specialize_simulate_fisp(  # noqa F841
        T1=T1, T2=T2, M0=M0, phases=phases, TE=TE, TI=TI,
        inversion_efficiency=inversion_efficiency, max_states=max_states
    ) 
    forward = optblocks.build_specialized_forward(
        func=optblocks.forward,
        TR=tr_pattern, M0=M0, phases=phases, TE=TE, TI=TI, inversion_efficiency=inversion_efficiency, max_states=max_states
    )
    forward_jit = jax.jit(forward)


    def tv_step(T1, T2, fa):
        # signals = forward(T1, T2, fa)
        tv_cost = costfuncs.fa_total_variation_criterion(fa)
        return tv_cost

    def sn_step(T1, T2, fa):
        # signals = forward(T1, T2, fa)
        sn_cost = fa_diff_loss(fa)
        return sn_cost

    def sig_step(T1, T2, fa):
        signals = forward(T1, T2, fa)
        # sig_cost = costfuncs.mean_signal_criterion(signals)
        sig_cost = costfuncs.inverse_mean_signal_criterion(signals)
        return sig_cost

    def ortho_step(T1, T2, fa):
        signals = forward(T1, T2, fa)
        ortho_cost = costfuncs.orthogonality_criterion(signals)
        return ortho_cost

    tv_cost_value_grad = jax.jit(jax.value_and_grad(tv_step, argnums=2))
    sig_cost_value_grad = jax.jit(jax.value_and_grad(sig_step, argnums=2))
    ortho_cost_value_grad = jax.jit(jax.value_and_grad(ortho_step, argnums=2))
    smoothnes_cost_value_grad = jax.value_and_grad(sn_step, argnums=2)

    cg_functions = {'signal' : sig_cost_value_grad, 'orthogonality' : ortho_cost_value_grad, 'totvar' : tv_cost_value_grad, 'smoothness' : smoothnes_cost_value_grad}
    cost_grad_function = cost_grad_builder(cg_functions)


    metadata = {
        'timestamp' : (datetime.
                       datetime.
                       now(tz=zoneinfo.ZoneInfo('Europe/Berlin')).
                       isoformat()),
        'git_hash' : 'pseudo-git-hash :)',
    }


    fa_plotter = CachingPlotter.create_flipangle_plotter(
        baseline_data=initial_fa,
        is_radians=True,
        NR=NR
    )            
    signal_plotter = CachingPlotter.create_signal_plotter()

    fa = initial_fa.copy()

    # optimizer setup
    optimizer = optax.sgd(learning_rate=hyperparameters.step_size)
    # optimizer = optax.adam(learning_rate=hyperparameters.step_size)

    optimizer_state = optimizer.init(fa)
    logger.info(f'successfully initialized optimizer {optimizer} '
                f'with step size {hyperparameters.step_size}')



    # set up reference cost to evaluate relative fitness of optimization
    reference_costfuncs = {
        'signal': costfuncs.inverse_mean_signal_criterion,
        'orthogonality': costfuncs.orthogonality_criterion,
    }
    reference_specs = refcs.expand_to_reference_specs(reference_patterns, pre_args=(T1, T2), post_args=())
    reference_costs = refcs.ReferenceCost.from_mapping(
        refcs.compute_reference_costs(forward_jit, reference_costfuncs, reference_specs)
    )
    relative_gain_evaluator = refcs.RelativeGainEvaluator(*reference_costs, prefix='')


    neplogger = NeptuneLogger(
        run=run,
        fa_plotter=fa_plotter,
        signal_plotter=signal_plotter,
    )
    # log static run-wide applicable information to the Neptune framework
    neplogger.log_protocol(attrs.asdict(protocol))
    neplogger.log_initializations(attrs.asdict(initializations))

    fa_history = []
    cost_history = []
    sig_history = []

    for iteration in tqdm.trange(max_iterations, leave=leave_pbar):
        # compute cost and gradient for currrent parameters
        cost_grad_mapping = cost_grad_function(T1, T2, fa)
        cost_grad_mapping_numpy = costgrad.cast_to_numpy(cost_grad_mapping)
        # record first entries of history
        fa_history.append(fa)
        cost_history.append(cost_grad_mapping_numpy)

        neplogger.log_costs(cost_grad_mapping, iteration)
        neplogger.log_gradients(cost_grad_mapping, iteration)

        # Compute gradient diagnostic metrics and optimization fitness (relative gains)
        cossim = gradtools.compute_gradient_cosine_similarities(cost_grad_mapping_numpy)
        magsim = gradtools.compute_gradient_magnitude_similarities(cost_grad_mapping_numpy)
        relgains = relative_gain_evaluator(cost_grad_mapping_numpy)
        signals = np.asarray(forward_jit(T1, T2, fa))
        sig_history.append(signals)

        neplogger.log_relative_gains(relgains, iteration)
        neplogger.log_cosine_similarities(cossim, iteration)
        neplogger.log_magnitude_similarities(magsim, iteration)
        neplogger.log_flipangles(fa, iteration, close=True)
        neplogger.log_signals(signals, iteration, close=True)

        # TODO: Legacy manual gradient combination
        #gradient = (  f_smoothness * cost_grad_mapping['smoothness'].grad
        #            + f_signal * cost_grad_mapping['signal'].grad
        #            + f_orthogonality * cost_grad_mapping['orthogonality'].grad)
        
        # perform gradient deconfliction and analytical combination
        jacobian = jnp.stack(
            [cost_grad_mapping['smoothness'].grad,
             cost_grad_mapping['signal'].grad,
             cost_grad_mapping['orthogonality'].grad],
            axis=0
        )
        gradient = jacdesc.conFIG(jacobian)

        neplogger.log_gradient(gradient, 'mean-conFIG', iteration)
        
        update, optimizer_state = optimizer.update(gradient, optimizer_state, fa)
        fa = optax.apply_updates(fa, update)
        fa = optax.projections.projection_box(fa, lower=min_fa, upper=max_fa)


    save_formats = save_format if isinstance(save_format, Sequence) else [save_format]
    # save stuff
    history_data = {
        'fa_history': np.array(fa_history),
        'cg_history': costgrad.combine_cost_grad_mappings(cost_history),
        'sig_history': np.array(sig_history),
    }
    
    fname_pkl = f'{run_name}.pkl'
    fname_zarr = f'{run_info.base_name}.zarr'

    for save_format in save_formats:

        if save_format == SaveFormat.PICKLE:
            store_pickle(history_data, run_info.run_dir / fname_pkl)

        elif save_format == SaveFormat.ZARR:
            protocol = attrs.asdict(protocol)
            hyperparameters = attrs.asdict(hyperparameters)
            initializations = attrs.asdict(initializations)

            bag = OptimizationBag(
                protocol=protocol,
                hyperparameters=hyperparameters,
                histories=history_data,
                initializations=initializations,
                results={},
                metadata=metadata
            )
            store_optimization_bag(bag, run_info.run_dir / fname_zarr)

    run.stop()






if __name__ == '__main__':
    main()