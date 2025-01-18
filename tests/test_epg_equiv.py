import jax.numpy as jnp
import numpy as np
import torch
import pytest

import mrfoptools.epg.epg as epg
import mrfoptools.contrib.signalmodel_epg as contrib


@pytest.mark.equivalencetest
def test_q_excitation_equivalence():
    alpha = np.deg2rad(45)
    phi = np.deg2rad(20)

    jaximp_result = epg.q_epg(alpha, phi)
    torchimp_result = contrib.q_epg(
        torch.as_tensor(alpha), torch.as_tensor(phi)
    ).detach().resolve_conj().numpy()

    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


@pytest.mark.equivalencetest
@pytest.mark.parametrize("T1", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("T2", [0.1, 0.2, 0.3])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.03])
def test_r_relaxation_equivalence(T1, T2, dt):
    jaximp_result = epg.r_epg(T1, T2, dt)
    torchimp_result = contrib.r_epg(
        torch.as_tensor(T1), torch.as_tensor(T2), torch.as_tensor(dt)
    ).detach().numpy()

    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


@pytest.mark.equivalencetest
@pytest.mark.parametrize("T1", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.03])
def test_dr_dT1_equivalence(T1, dt):
    jaximp_result = epg.dr_dT1_epg(T1, dt)
    torchimp_result = contrib.dr_dt1_epg(
        torch.as_tensor(T1), torch.as_tensor(dt)
    ).detach().numpy()

    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)


@pytest.mark.equivalencetest
def test_apply_gradient():
    mx, my, mz = 0.5, 0.3, 0.2
    omega = jnp.array(
        [
            [mx],
            [my],
            [mz],
        ]
    )
    omega_torch = torch.tensor([[mx], [my], [mz]])
    jaximp_result = epg.grad_shift(omega, dk=1)
    torchimp_result = contrib.epg_grad(omega_torch).detach().numpy()
    print(torchimp_result)
    print(jaximp_result)
    assert np.allclose(jaximp_result, torchimp_result, rtol=1e-5, atol=1e-5)



class Test_grad_shift:

    def test_smoke_basestate(self):
        """Test that grad shift applications modifies state matrix shape."""
        mx, my, mz = 0.5, 0.3, 0.2
        omega = jnp.array(
            [
                [mx],
                [my],
                [mz],
            ]
        )
        result = epg.grad_shift(omega, dk=1)
        assert isinstance(result, jnp.ndarray)
        assert result.shape == (3, 2)


    @pytest.mark.parametrize('n_grad_applications', [2, 3, 7])
    def test_smoke_iterative_application(self, n_grad_applications):
        """Test that mutiple grad shift applications modifies state matrix shape."""
        mx, my, mz = 0.5, 0.3, 0.2
        omega = jnp.array(
            [
                [mx],
                [my],
                [mz],
            ]
        )
        for _ in range(n_grad_applications):
            omega = epg.grad_shift(omega, dk=1)

        assert omega.shape == (3, n_grad_applications + 1)