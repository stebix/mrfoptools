"""
Script demo of the scale finding algorithm.
"""
import jax
import jax.numpy as jnp

from mrfoptools.optimization.simulated_annealing.calibration import find_scale_factors_legacy, VariableType

from mrfoptools.optimization.simulated_annealing.annealing import generate_cost_function

from mrfoptools.epg.sequences.helpers import quickmake_simulate_fisp

def main():
    simfisp = quickmake_simulate_fisp()
    cost_func = generate_cost_function(simfisp, weights=(1.0, 1.0))

    absbounds: dict[VariableType, tuple[float, float]] = {
        VariableType.FA_XCOORDS : (0.0, 1000.0),
        VariableType.FA_YCOORDS : (jnp.deg2rad(1.0), jnp.deg2rad(90.0)),
        VariableType.TR_XCOORDS : (0.0, 1000.0),
        VariableType.TR_YCOORDS : (5.0, 1250.0)
    }

    kwargs = {
        'j_max' : 4,
        'bounds_specification' : absbounds,
        'cost_func' : cost_func,
        'key' : jax.random.key(345783645),
        'sample_count' : 10,
    }

    sf = find_scale_factors_legacy(**kwargs)

    print(sf)


if __name__ == '__main__':
    main()