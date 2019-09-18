"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
import ballistico
import pytest
import sys

def test_ballistico_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "ballistico" in sys.modules
