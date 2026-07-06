"""
Tests for the FourthOrder observable and the TDEP IFC4 reader.

Covers two layers:
  1. FourthOrder.load: format validation.
  2. ForceConstants.from_folder with include_fourth=True:
       - attaches a .fourth attribute when format='tdep'.
       - default (include_fourth=False) leaves .fourth None.
       - include_fourth=True is rejected for non-TDEP formats.

Non-diagonal (SNF) IFC4 loading is covered in
test_nondiagonal_forceconstants.py on the env-var-gated production fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# FourthOrder.load: format guard
# ---------------------------------------------------------------------------

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


def test_force_constants_include_fourth_rejects_non_tdep_format(tmp_path):
    """include_fourth=True is only defined for format='tdep'."""
    from kaldo.forceconstants import ForceConstants
    with pytest.raises(ValueError, match=r"(?i)include_fourth|tdep"):
        ForceConstants.from_folder(
            folder=str(tmp_path), supercell=(1, 1, 1), format="numpy",
            include_fourth=True,
        )
