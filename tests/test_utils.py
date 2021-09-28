from Radiomics.utils.utils import get_peak_from_histogram
import pytest 
import numpy as np

np.random.seed(9)


def test_get_peak_from_histogram():
    uniform = np.array([i for i in range(1000)])
    gaussian = np.random.normal(0, 100, 1000)
    steps = np.arange(-100, 100, step=10)
    uniform_bins, uniform_bin_edges = np.histogram(uniform, bins=steps)
    assert get_peak_from_histogram(uniform_bins, uniform_bin_edges) == 5
    gaussian_bins, gaussian_bin_edges = np.histogram(gaussian, bins=steps)
    assert abs(get_peak_from_histogram(gaussian_bins, gaussian_bin_edges)) < 10
    with pytest.raises(AssertionError):
        get_peak_from_histogram(bins=[], bin_edges=[])