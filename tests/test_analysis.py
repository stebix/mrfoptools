"""
Test suite for the analysis module in mrfoptools.epg.sequences.jax.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp

import pytest

import mrfoptools.epg.sequences.jax.pfisp2 as pfisp
import mrfoptools.epg.sequences.jax.analysis as analysis


class Test_expand_to_semantic_signal_segments:
    """
    Tests for the expand_to_semantic_signal_segments function.
    """
    @pytest.mark.integration
    def test_smoke(self):
        """
        Basic smoke test for expand_to_semantic_signal_segments.
        """
        n_species: int = 3
        n_tr: int = 100
        seed: int = 42
        key = jax.random.key(seed)
        signals = jax.random.normal(key=key, shape=(n_species, n_tr))

        parameters = SimpleNamespace(
            T1=jnp.linspace(1000.0, 2000.0, n_species),
            T2=jnp.linspace(100.0, 2000.0, n_species),
            M0=1.0
        )
        segments = analysis.expand_to_semantic_signal_segments(
            signals=signals,
            timepoints=jnp.arange(n_tr),
            parameters=parameters
        )

        assert len(segments) == n_species
        for segment in segments:
            assert isinstance(segment, analysis.SignalSection)
            assert segment.timepoints.shape == (n_tr,)
            assert segment.signal.shape == (n_tr,)