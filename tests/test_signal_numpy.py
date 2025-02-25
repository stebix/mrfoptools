import numpy as np

import mrfoptools.epg.core.numpy as npycore
import mrfoptools.epg.signal.numpy as npysig
import mrfoptools.epg.signal.numpy.preparations as npysigprep


from mrfoptools.initialization.initialization import load_yun_pattern


def init_helper() -> dict:
    """Setup for numpy-backend signal calculations."""
    pattern = load_yun_pattern(type_='both')
    fa = pattern.flip_angles
    tr = pattern.repetition_times
    phases = np.full_like(fa, 0.0)
    TI = 20
    TE = 1
    T1 = 1000
    T2 = 500
    M0 = 1.0
    inversion_efficiency = 1.0
    max_states = 500
    inversion_operator = npycore.inversion(inversion_efficiency)
    omega = npysigprep.prepare_inversion_omega(
        T1, T2, M0, max_states=max_states,
        inversion_operator=inversion_operator, TI=TI
    )
    parameters = {
        'omega' : omega,
        'T1' : T1,
        'T2' : T2,
        'M0' : M0,
        'fa' : fa,
        'TR' : tr,
        'phases' : phases,
        'TE' : TE,
        'b_TE' : npycore.b_epg(T1, TE).astype(omega.dtype),
        'r_TE' : npycore.r_epg(T1, T2, TE).astype(omega.dtype),
    }
    return parameters


class Test_compute_signal_numpynaive:

    def test_forward_simulation_smoke(self):
        parameters = init_helper()

        import numba
        compute_signal_jitted = numba.njit(npysig.compute_signal_numpynaive)

        func = npysig.compute_signal_numpynaive
        func = compute_signal_jitted

        print(f'{parameters["fa"].dtype}')
        print(f'{parameters["TR"].dtype}')
        print(f'{parameters["phases"].dtype}')

        signal = func(**parameters)
        assert signal.shape == parameters['fa'].shape