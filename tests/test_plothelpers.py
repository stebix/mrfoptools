import time
import numpy as np

import pytest

from mrfoptools.optimization.diagnostics.plothelpers import CachingPlotter


def test_caching_plotter_cache_size():
    rng = np.random.default_rng()
    plotter = CachingPlotter(cache_size=3)

    for i in range(3):
        data = rng.normal(size=15)
        _ = plotter.generate(data)
        assert len(plotter._cache) == i + 1

    for i in range(3):
        data = rng.normal(size=15)
        _ = plotter.generate(data)
        assert len(plotter._cache) == 3


def test_caching_plotter_cache_reuse():
    rng = np.random.default_rng()
    plotter = CachingPlotter(cache_size=3)

    datas = [rng.normal(size=15) for _ in range(3)]
    plots = [plotter.generate(data) for data in datas]

    for data in datas:
        plot_v2 = plotter.generate(data)
        assert plot_v2 in plots


@pytest.mark.benchmark
def test_performance_of_caching_plotter():
    rng = np.random.default_rng()
    plotter = CachingPlotter(cache_size=50)

    fresh_plot_times = []
    cached_plot_times = []

    for _ in range(50):
        data = rng.normal(size=1000)

        start = time.time()
        _ = plotter.generate(data)
        fresh_plot_times.append(time.time() - start)

        start = time.time()
        _ = plotter.generate(data)
        cached_plot_times.append(time.time() - start)

    
    fresh_plot_times = np.array(fresh_plot_times)
    cached_plot_times = np.array(cached_plot_times)

    msg = 'plotting new data should be slower than retrieving from cache'
    assert np.mean(fresh_plot_times) > np.mean(cached_plot_times), msg

    print(f'fresh plot time :: average {np.mean(fresh_plot_times):.4f} s +- {np.std(fresh_plot_times):.4f} s')
    print(f'cached plot time :: average {np.mean(cached_plot_times):.4f} s +- {np.std(cached_plot_times):.4f} s')

