"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
import os
import sys
import pytest


@pytest.yield_fixture(scope="session")


def sigma2():
    home=os.getcwd()
    os.chdir(f"kaldo/tests/sigma2")
    result =  ForceConstants.sigma2_td_md(md_run='tdep_fit_configurations.xyz')
    yield result
    os.chdir(home)


def test_sigma2(sigma2):
    expect_val = .101
    test_input = sigma2
    assert pytest.approx(test_input,0.01) == expect_val


