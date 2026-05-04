"""
Tests for the FourthOrder observable and the TDEP IFC4 reader.

Covers two layers:
  1. FourthOrder.load: diagonal-guard behaves identically to SecondOrder
     / ThirdOrder on non-diagonal fixtures.
  2. ForceConstants.from_folder with include_fourth=True:
       - attaches a .fourth attribute when format='tdep' + diagonal ssposcar.
       - default (include_fourth=False) leaves .fourth None.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Production-only fixture: large DFT-quality Si IFCs at 25^3 supercell.
# Set KALDO_TEST_SI_PROD to point at reference_si/T300_0 to enable.
# See kaldo/tests/_paths.py for details on env-var-gated test fixtures.
from kaldo.tests._paths import SI_PROD


# ---------------------------------------------------------------------------
# FourthOrder.load: diagonal guard
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SI_PROD.exists(), reason="Si production fixture unavailable")
def test_fourth_order_rejects_non_diagonal_supercell():
    """FourthOrder.load on a non-diagonal TDEP run must raise a clear error."""
    from kaldo.observables.fourthorder import FourthOrder
    with pytest.raises(ValueError, match=r"(?i)non.?diagonal|SNF|diagonal"):
        FourthOrder.load(folder=str(SI_PROD), supercell=(3, 3, 3), format="tdep")


def test_fourth_order_rejects_unsupported_format(tmp_path):
    """Only format='tdep' is wired today."""
    from kaldo.observables.fourthorder import FourthOrder
    with pytest.raises(ValueError, match=r"(?i)format|support"):
        FourthOrder.load(folder=str(tmp_path), supercell=(1, 1, 1), format="numpy")


# ---------------------------------------------------------------------------
# ForceConstants.from_folder include_fourth plumbing
# ---------------------------------------------------------------------------

SI_TDEP_DIR = Path(__file__).parent / "si-tdep"


@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_force_constants_fourth_default_is_none():
    """Default include_fourth=False: fc.fourth is None even on a TDEP folder."""
    from kaldo.forceconstants import ForceConstants
    fc = ForceConstants.from_folder(
        folder=str(SI_TDEP_DIR), supercell=(5, 5, 5), format="tdep",
    )
    assert fc.fourth is None


@pytest.mark.skipif(not SI_PROD.exists(), reason="Si production fixture unavailable")
def test_force_constants_include_fourth_trips_on_non_diagonal():
    """include_fourth=True must raise on non-diagonal (no silent quartet mis-index)."""
    from kaldo.forceconstants import ForceConstants
    with pytest.raises(ValueError, match=r"(?i)non.?diagonal|SNF|diagonal"):
        ForceConstants.from_folder(
            folder=str(SI_PROD), supercell=(3, 3, 3), format="tdep",
            include_fourth=True,
        )


def test_force_constants_include_fourth_rejects_non_tdep_format(tmp_path):
    """include_fourth=True is only defined for format='tdep'."""
    from kaldo.forceconstants import ForceConstants
    with pytest.raises(ValueError, match=r"(?i)include_fourth|tdep"):
        ForceConstants.from_folder(
            folder=str(tmp_path), supercell=(1, 1, 1), format="numpy",
            include_fourth=True,
        )
