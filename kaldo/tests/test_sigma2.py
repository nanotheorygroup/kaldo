"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
import os
import sys
import pytest


@pytest.fixture(scope="session")
def sigma2():
    root = "kaldo/tests/sigma2"
    result = ForceConstants.sigma2_tdep_MD(
        md_run=f"{root}/tdep_fit_configurations.xyz",
        primitive_file=f"{root}/infile.ucposcar",
        supercell_file=f"{root}/infile.ssposcar",
        fc_file=f"{root}/infile.forceconstant",
    )
    yield result


def test_sigma2(sigma2):
    expect_val = 0.1495
    test_input = sigma2
    assert pytest.approx(test_input, 0.01) == expect_val
