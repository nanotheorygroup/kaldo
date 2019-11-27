"""
Unit and regression test for the finitedifference package.
"""

# Import package, test suite, and other packages as needed
import pytest
from ballistico.finitedifference import FiniteDifference
import numpy as np


def test_max_freq_imported_dlpoly():
    supercell = np.array([3, 3, 3])
    finite_difference = FiniteDifference.import_from_dlpoly('ballistico/tests/si-crystal', supercell)
    second_order = finite_difference.second_order / (28)
    eigv = np.linalg.eigh(second_order.reshape(162, 162))[0]
    freqs = np.sqrt(np.abs(eigv) * 9648.5) / (2 * np.pi)
    np.testing.assert_approx_equal(freqs.max(), 16.1, significant=3)
