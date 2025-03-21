import datetime
import zoneinfo
import jax.numpy as jnp 
import numpy as np
import pytest

@pytest.fixture
def metadata() -> dict:
    metadata = {
        'timestamp' : (datetime.
                       datetime.
                       now(tz=zoneinfo.ZoneInfo('Europe/Berlin')).
                       isoformat()
                       ),
        'author' : 'testuser',
        'git-hash' : '1234567890abcdef',
        'git-branch' : 'main',
    }
    return metadata


@pytest.fixture(params=(np, jnp))
def data(request) -> dict:
    xnp = request.param
    NR = 100
    NSTEPS = 10
    data = {
        'test_array_1' : xnp.zeros((NSTEPS, NR)),
        'test_array_2' : xnp.full((NSTEPS, NR), fill_value=25.1701),
        'test_complex_array' : xnp.full((10, 10), fill_value=1+1j),
        'test_irregular_array' : {
            'sampling_steps' : xnp.arange(33),
            'sampling_values' : xnp.full((33, 15), fill_value=1337)
        }
    }
    return data
