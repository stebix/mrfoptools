import jax.numpy as jnp
import pytest

import mrfoptools.epg.epg as epg



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