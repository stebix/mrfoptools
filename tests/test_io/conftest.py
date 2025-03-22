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


@pytest.fixture
def protocol() -> dict:
    protocol = {
        'sequence' : 'fisp',
        'NR' : 1000,
        'test' : 'nein-koenig-nein',
        'T1' : [1, 2, 3, 4, 5, 6],
        'T2' : [10, 20, 30, 40, 50, 60]
    }
    return protocol


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


@pytest.fixture
def initialization() -> dict:
    initializations = {
        'flipangles' : np.full(1000, fill_value=90.0),
        'tr' : np.full(1000, fill_value=12.0),
        'seed' : 1234
    }
    return initializations

@pytest.fixture
def results() -> dict:
    return {}


@pytest.fixture
def hyperparameters() -> dict:
    hyperparameter = {
        'optimization' : 'gradient_descent',
        'optimizer' : 'adam',
        'test_metadata_1' : 'test',
        'parameter' : 1000,
        'learning_rate' : 0.01,
        'in_depth_settings' : {
            'n_iterations' : 1000,
            'n_samples' : 100,
            'bathtub_parameters' : {
                'alpha' : 0.1,
                'beta' : 0.3,
                'gamma' : 0.7,
                'delta' : 2.5
            }
        }
    }
    return hyperparameter