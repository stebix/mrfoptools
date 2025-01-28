"""
Test the initialization variants.

@Author: Jannik Stebani 2025
"""
import numpy as np

from mrfoptools.initialization.initialization import load_cao_pattern


def test_load_cao_fa_pattern():
    cao_fa_pattern = load_cao_pattern('fa')
    assert cao_fa_pattern.shape == (500,)
    assert np.isclose(np.min(cao_fa_pattern), 0)
    assert np.max(cao_fa_pattern) < 100


def test_load_cao_tr_pattern():
    cao_fa_pattern = load_cao_pattern('tr')
    assert cao_fa_pattern.shape == (500,)
    assert np.min(cao_fa_pattern) > 5
    assert np.max(cao_fa_pattern) < 20


def test_load_cao_pattern():
    pattern = load_cao_pattern('both')
    assert pattern.flip_angles.shape == (500,)
    assert pattern.repetition_times.shape == (500,)