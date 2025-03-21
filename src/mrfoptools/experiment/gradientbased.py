"""
Tooling to perform gradient-based optimization on MRF sequences.

@Author: Jannik Stebani 2025
"""
# ruff: noqa: F841
# ruff: noqa: F401

import argparse
import time
import numpy as np
import jax.numpy as jnp
import jax
import optax
import matplotlib.pyplot as plt
import tqdm
import itertools
import pickle
import datetime
import zoneinfo
from numbers import Number
from functools import partial

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

import mrfoptools.optimization.blocks as optblocks
import mrfoptools.initialization.initialization as initools
import mrfoptools.optimization.costfuncs as costfuncs
import mrfoptools.io.io as io
import mrfoptools.epg.sequences.jax.fisp as epgfisp
import mrfoptools.initialization.initialization as init
import mrfoptools.optimization.costgrad as costgrad

from mrfoptools.optimization.costgrad import cost_grad_builder
import mrfoptools.optimization.diagnostics.diagnostics as diags
import mrfoptools.optimization.gradtools.gradtools as gradtools

import mrfoptools.optimization.diagnostics.tensorboard as tbdiag

from mrfoptools.optimization.diagnostics.plothelpers import Plotter
from mrfoptools.optimization.diagnostics.tensorboard import RelativeGainEvaluator, BaseCost, TensorboardLogger
from mrfoptools.optimization.diagnostics.diagnostics import CostValueRange

from mrfoptools.io.bag import OptimizationBag, store_optimization_bag


bathtub_loss = costfuncs.construct_bathtub_loss(
    radius=2.0, alpha=0.1, beta=0.5, gamma=10
)

def fa_diff_loss(fa: jax.Array):
    diff = fa[1:] - fa[:-1]
    smoothed = bathtub_loss(diff)
    return jnp.linalg.norm(smoothed, ord=2)


def run_experiment(logdir_suffix: str):

    # optimization setting
    NR = 1000
    M0 = 1.0
    const_tr = 12
    const_fa_init = 49
    const_phase = 0
    TE = 1
    TI = 20
    step_size = 0.01
    inversion_efficiency = 1.0
    max_states = 1000
    max_iterations = 500

    min_fa = np.deg2rad(1)
    max_fa = np.deg2rad(90)

    T1_values = jnp.array([2250, 2500, 2750, 3000])
    # T2_values = jnp.array([500, 600, 700, 800, 900])
    T2_values = jnp.linspace(500, 1250, num=8)

    # Add random noise in units degrees (small perturbation)
    fa_pattern_np = initools.create_constant_pattern(amplitude=const_fa_init, length=NR)

    tr_pattern = jnp.array(initools.create_constant_pattern(amplitude=const_tr, length=NR))
    phases = jnp.array(initools.create_constant_pattern(amplitude=const_phase, length=NR))

    T1 = []
    T2 = []
    for t1v, t2v in itertools.product(T1_values, T2_values):
        T1.append(t1v)
        T2.append(t2v)
        
    T1 = jnp.array(T1)
    T2 = jnp.array(T2)

    print(f'T1 elements: {T1.shape}')
    print(f'T2 elements: {T2.shape}')


    forward_fatr = epgfisp.specialize_simulate_fisp(
        T1=T1, T2=T2, M0=M0, phases=phases, TE=TE, TI=TI,
        inversion_efficiency=inversion_efficiency, max_states=max_states
    )
    forward = optblocks.build_specialized_forward(
        func=optblocks.forward,
        TR=tr_pattern, M0=M0, phases=phases, TE=TE, TI=TI, inversion_efficiency=inversion_efficiency, max_states=600
    )

    seed = 1337
    key = jax.random.key(seed)
    initial_fa = jnp.deg2rad(
        jnp.array(fa_pattern_np) + jax.random.normal(key=key, shape=fa_pattern_np.shape)
    )
    initial_signals = forward(T1, T2, initial_fa)

    yun_amplitudes_literature = [35, 43, 70, 45, 27]
    amplitudes = jnp.deg2rad(jnp.array(yun_amplitudes_literature))
    yun_fa_manual = jnp.array(initools.create_sinusoidal_pattern(amplitudes, 200))
    yun_tr = tr_pattern #jnp.array(yun_pattern.repetition_times)
    yun_init_signals = forward_fatr(yun_fa_manual, yun_tr)


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
        sig_cost = costfuncs.mean_signal_criterion(signals)
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


    logdir = Path(f'/home/jannik/storage/mrf-optruns-march-exp/trial-{logdir_suffix}')
    logdir.mkdir()

    writer = SummaryWriter(log_dir=logdir)
    seed = int(time.time())
    key = jax.random.key(seed)


    initial_fa = yun_fa_manual

    protocol = {
        'T1': T1.tolist(),
        'T2': T2.tolist(),
        'TE': TE,
        'TI': TI,
        'phase': const_phase,
        'inversion_efficiency': inversion_efficiency,
        'max_states': max_states,
        'M0': M0,
        'NR': NR,
    }
    initializations = {
        'fa': np.asarray(initial_fa),
        'tr': np.asarray(tr_pattern),
        'seed': 1337
    }
    hyperparameters = {
        'step_size': step_size,
        'max_iterations': max_iterations,
        'min_fa': min_fa,
        'max_fa': max_fa
    }
    metadata = {
        'timestamp' : (datetime.
                       datetime.
                       now(tz=zoneinfo.ZoneInfo('Europe/Berlin')).
                       isoformat()),
        'git_hash' : 'pseudo-git-hash :)',
    }


    log_every_n: int = 3

    fa_plotter = Plotter(
        baseline_data=np.asarray(jnp.rad2deg(initial_fa)),
        title='FA Optimization',
        xlabel='NR index',
        ylabel='FA [deg]',
        legend=False,
        grid=True,
        ylim=(0, 94),
        xlim=(0, 1000),
        baseline_data_plot_kwargs={'label': 'initial',
                                   'ls' : 'dotted',
                                   'color' : 'black',
                                   'alpha' : 0.5},
    )
    signal_plotter = Plotter(
        title='Signal Optimization',
        xlabel='NR index',
        ylabel='Signal',
        legend=False,
        grid=True,
        ylim=(-0.5, 0.5),
        xlim=(0, 1000),
    )

    fa = initial_fa.copy()

    n_species = yun_init_signals.shape[0]
    print(f'n_species :: {n_species}')

    optimizer = optax.adam(learning_rate=step_size)
    optimizer_state = optimizer.init(fa)

    forward_jit = jax.jit(forward)
    # log_signals = signal_log_builder(forward=forward_jit, T1=T1, T2=T2, writer=writer, initial_signals=forward_jit(T1, T2, initial_fa))

    const_fa_init = jnp.array(initools.create_constant_pattern(amplitude=np.deg2rad(49), length=NR))

    # yun base costs
    yun_base_signals = forward_jit(T1, T2, yun_fa_manual)
    yun_base_signal_cost = costfuncs.mean_signal_criterion(yun_base_signals)
    yun_base_ortho_cost = costfuncs.orthogonality_criterion(yun_base_signals)
    yun_base_totvar_cost = costfuncs.fa_total_variation_criterion(yun_fa_manual)

    # constinit base costs
    constinit_base_signals = forward_jit(T1, T2, const_fa_init)
    constinit_base_signal_cost = costfuncs.mean_signal_criterion(constinit_base_signals)
    constinit_base_ortho_cost = costfuncs.orthogonality_criterion(constinit_base_signals)
    constinit_base_totvar_cost = costfuncs.fa_total_variation_criterion(const_fa_init)

    base_costs  = [
        BaseCost(costname='signal', refname='yun-base', value=yun_base_signal_cost, range=CostValueRange.NEGATIVE),
        BaseCost(costname='orthogonality', refname='yun-base', value=yun_base_ortho_cost, range=CostValueRange.POSITIVE),
        BaseCost(costname='totvar', refname='yun-base', value=yun_base_totvar_cost, range=CostValueRange.POSITIVE),
        BaseCost(costname='signal', refname='constfa-base', value=constinit_base_signal_cost, range=CostValueRange.NEGATIVE),
        BaseCost(costname='orthogonality', refname='constfa-base', value=constinit_base_ortho_cost, range=CostValueRange.POSITIVE),
        BaseCost(costname='totvar', refname='constfa-base', value=constinit_base_totvar_cost, range=CostValueRange.POSITIVE)
    ]
    relative_gain_evaluator = RelativeGainEvaluator(*base_costs)

    diaglogger = tbdiag.TensorboardLogger(
        writer,
        fa_plotter=fa_plotter,
        signal_plotter=signal_plotter,
        relative_gain_evaluator=relative_gain_evaluator)

    fa_history = []
    cost_history = [] # noqa: F841
    sig_history = []

    for iteration in tqdm.trange(max_iterations, leave=True):
        
        fa_history.append(fa)
        
        cost_grad_mapping = cost_grad_function(T1, T2, fa)
        cost_grad_mapping_numpy = costgrad.cast_to_numpy(cost_grad_mapping)

        cost_history.append(cost_grad_mapping_numpy)

        diaglogger.log_costs(cost_grad_mapping_numpy, iteration)
        diaglogger.log_gradients(cost_grad_mapping_numpy, iteration)
        diaglogger.log_cosine_similarities(cost_grad_mapping_numpy, iteration)
        diaglogger.log_gradient_magnitude_similarities(cost_grad_mapping_numpy, iteration)
        diaglogger.log_relative_gains(cost_grad_mapping_numpy, iteration)

        signals = np.asarray(forward_jit(T1, T2, fa))
        sig_history.append(signals)
        diaglogger.log_flipangles(fa, iteration)
        diaglogger.log_signals(signals, iteration)

        # ortho_grad_surgical = ortho_grad - jnp.dot(ortho_grad, sig_grad) / jnp.linalg.norm(sig_grad, ord=2) * sig_grad
        #gradients = jnp.stack([sig_grad, tv_grad, ortho_grad], axis=0)
        #gradient = compute_conFIG_gradient(gradients)
        
        gradient = 1.0 * cost_grad_mapping['smoothness'].grad + cost_grad_mapping['signal'].grad + 0.01 * 1/n_species * cost_grad_mapping['orthogonality'].grad
        #gradient = cost_grad_mapping['signal'].grad
        
        update, optimizer_state = optimizer.update(gradient, optimizer_state, fa)
        fa = optax.apply_updates(fa, update)
        fa = optax.projections.projection_box(fa, lower=min_fa, upper=max_fa)

    # save stuff
    test_data = {
        'fa_history': fa_history,
        'cg_history': cost_history,
        'sig_history': sig_history,
    }
    savepath = logdir / 'rundata.pkl'

    with open(savepath, 'wb') as f:
        pickle.dump(test_data, f)


    hist_data_np = {
        'fa_history': np.asarray(fa_history),
        'cg_history': np.asarray(cost_history),
        'sig_history': np.asarray(sig_history).astype(np.float32),
    }

    bag = OptimizationBag(
        protocol=protocol,
        hyperparameters=hyperparameters,
        histories=hist_data_np,
        initializations=initializations,
        results={},
        metadata=metadata
    )
    # store_optimization_bag(bag, logdir / 'optbag.zarr')


def make_parser():
    parser = argparse.ArgumentParser(
        description='Run an optimization experiment on the MRF sequence.'
    )
    parser.add_argument(
        'logdir_suffix',
        type=str,
        help='Suffix for logging directory to store the optimization run data.'
    )
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    run_experiment(logdir_suffix=args.logdir_suffix)

