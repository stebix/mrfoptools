import jax
import jax.numpy as jnp

import mrfoptools.optimization.simulated_annealing.calibration as calib

from mrfoptools.optimization.simulated_annealing.annealing_rebuild import *
from mrfoptools.optimization.simulated_annealing.variables import (
    Designation, ParameterType, ParameterAxis, Variable
)
from mrfoptools.parameterization.expansion import expand
from mrfoptools.optimization.costfuncs import minimum_average_criterion

from mrfoptools.epg.sequences.jax.fisp import specialize_simulate_fisp


def main():
    INIT_FA = 50
    INIT_TR = 12
    NR = 1000
    M0 = 1.0
    TE = 2.2
    TI = 20

    max_states = 750
    inversion_efficiency = 1.0


    fa = jnp.deg2rad(jnp.full(fill_value=INIT_FA, shape=NR, dtype=jnp.float32))
    tr = jnp.full(fill_value=INIT_TR, shape=NR, dtype=jnp.float32)
    phases = jnp.full_like(fa, fill_value=jnp.pi/2)

    T1 = jnp.array([1500, 2000, 2500, 3000])
    T2 = jnp.array([500, 500, 600, 700])

    simulate_fisp = specialize_simulate_fisp(
        T1=T1,
        T2=T2,
        M0=M0,
        phases=phases,
        TI=TI,
        TE=TE,
        max_states=max_states,
        inversion_efficiency=inversion_efficiency   
    )

    import line_profiler

    @line_profiler.profile
    def costfunc(
        fa_x, fa_y, tr_x, tr_y
    ):
        nreq = 1000
        w_time = 1.0
        w_signal = 1.0
        
        fa = expand(fa_x, fa_y, nreq=nreq, extrap=False, bounds=(jnp.deg2rad(1.0), jnp.deg2rad(90.0)))
        tr = expand(tr_x, tr_y, nreq=nreq, extrap=False, bounds=(6.0, 500.0))
        signals = simulate_fisp(fa, tr)
        return w_time * jnp.sqrt(jnp.sum(tr)) + w_signal / minimum_average_criterion(signals)


    n_controlpoints: int = 10

    fa_controlpoint_bounds = (jnp.deg2rad(1.0), jnp.deg2rad(90.0))
    tr_controlpoint_bounds = (5.0, 500.0)
    x_bounds = (0.0, 1000.0)

    fa_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.X),
        sort=True
    )
    tr_x = Variable.create_with_fixed_edges(
        parameters=jnp.linspace(0, 1000, num=n_controlpoints),
        bounds=x_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.TR, axis=ParameterAxis.X),
        sort=True
    )
    fa_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_FA),
        bounds=fa_controlpoint_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.FA, axis=ParameterAxis.Y),
        sort=False
    )
    tr_y = Variable.create_with_floating_edges(
        parameters=jnp.full(shape=n_controlpoints, fill_value=INIT_TR),
        bounds=tr_controlpoint_bounds,
        relscale=0.1,
        designation=Designation(type=ParameterType.TR, axis=ParameterAxis.Y),
        sort=False
    )
    variables = [fa_x, fa_y, tr_x, tr_y]

    key = jax.random.key(12390)

    r = calib.sweep_sample_perturbations_costs(
        key=key, variables=variables, j_max=2, sample_count=10, cost_func=costfunc
    )

    print(r)



if __name__ == '__main__':
    main()