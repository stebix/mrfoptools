import pytest
from functools import wraps

def skip_if_import_error(import_name):
    """Skip test if the specified import is unavailable."""
    try:
        __import__(import_name)
        skip = False
    except ImportError:
        skip = True
    
    return pytest.mark.skipif(skip, reason=f'\'{import_name}\' not installed')
