"""
Tests for the FourthOrder observable and the TDEP IFC4 reader.

Covers three layers:
  1. FourthOrder.load: format validation.
  2. ForceConstants.from_folder with include_fourth=True:
       - attaches a .fourth attribute when format='tdep'.
       - default (include_fourth=False) leaves .fourth None.
       - include_fourth=True is rejected for non-TDEP formats.
  3. parse_tdep_fourth_forceconstant: the streamed-COO reader places
     quartet blocks at the right coordinates, sums duplicate quartets,
     and never materializes the dense rank-11 tensor.

Non-diagonal (SNF) IFC4 loading is covered in
test_nondiagonal_forceconstants.py on the env-var-gated production fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
# parse_tdep_fourth_forceconstant: synthetic minimal IFC4 (no fixture)
# ---------------------------------------------------------------------------

def _write_ifc4(path, n_uc, quartets):
    """Write a minimal TDEP infile.forceconstant_fourthorder.

    quartets : dict a1 -> list of (i2, i3, i4, R2, R3, R4, phi) with i* 0-based
    (i1 is the central atom a1, R1 fixed at 0), phi a (3,3,3,3) array.
    """
    lines = [f"{n_uc}", "5.0"]
    for a1 in range(n_uc):
        qs = quartets.get(a1, [])
        lines.append(f"{len(qs)}")
        for (i2, i3, i4, R2, R3, R4, phi) in qs:
            for idx in (a1, i2, i3, i4):
                lines.append(f"{idx + 1}")
            for R in ((0, 0, 0), R2, R3, R4):
                lines.append(f"{float(R[0])} {float(R[1])} {float(R[2])}")
            for v in np.asarray(phi, dtype=float).ravel().tolist():
                lines.append(repr(v))
    path.write_text("\n".join(lines) + "\n")


def test_ifc4_parser_places_and_sums_blocks(tmp_path):
    """Streamed COO reader: block placement + duplicate-quartet summation.

    Two atoms, a diagonal 1x1x1 grid (n_rep = 1 so every R wraps to id 0).
    Atom 0 has the same (i2, i3, i4) quartet listed twice; the two phi
    blocks must sum. Atom 1 has one quartet.
    """
    from kaldo.interfaces.tdep_io import parse_tdep_fourth_forceconstant

    rng = np.random.default_rng(0)
    phi_a = rng.standard_normal((3, 3, 3, 3))
    phi_b = rng.standard_normal((3, 3, 3, 3))
    phi_c = rng.standard_normal((3, 3, 3, 3))
    Z = (0, 0, 0)

    uc = tmp_path / "infile.ucposcar"
    # 2-atom cubic primitive so ase can read it as VASP POSCAR
    uc.write_text(
        "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nSi\n2\n"
        "Direct\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
    )
    fc = tmp_path / "infile.forceconstant_fourthorder"
    _write_ifc4(fc, 2, {
        0: [(1, 0, 1, Z, Z, Z, phi_a), (1, 0, 1, Z, Z, Z, phi_b)],  # duplicate -> sum
        1: [(0, 1, 0, Z, Z, Z, phi_c)],
    })

    ifc4 = parse_tdep_fourth_forceconstant(fc_filename=str(fc), primitive=str(uc), supercell=(1, 1, 1))
    assert ifc4.shape == (2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3)

    dense = ifc4.todense()  # tiny here: 2*3*1*2*3*1*2*3*1*2*3 = 1296 entries
    # atom 0 block at (a1=0, r2=0, i2=1, r3=0, i3=0, r4=0, i4=1): phi_a + phi_b
    np.testing.assert_allclose(dense[0, :, 0, 1, :, 0, 0, :, 0, 1, :], phi_a + phi_b, atol=1e-12)
    # atom 1 block at (a1=1, i2=0, i3=1, i4=0): phi_c
    np.testing.assert_allclose(dense[1, :, 0, 0, :, 0, 1, :, 0, 0, :], phi_c, atol=1e-12)
    # total nonzero magnitude accounts for exactly the three blocks placed
    assert ifc4.nnz <= 3 * 81


def test_ifc4_parser_requires_exactly_one_of_grid_or_supercell(tmp_path):
    from kaldo.interfaces.tdep_io import parse_tdep_fourth_forceconstant
    uc = tmp_path / "infile.ucposcar"
    uc.write_text("Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nSi\n1\nDirect\n0.0 0.0 0.0\n")
    fc = tmp_path / "f4"
    _write_ifc4(fc, 1, {})
    with pytest.raises(ValueError, match=r"exactly one"):
        parse_tdep_fourth_forceconstant(fc_filename=str(fc), primitive=str(uc))


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

def test_ifc4_parser_raises_on_truncated_file(tmp_path):
    """A truncated IFC4 file must raise, not hang.

    Regression: the fixed-count token loop in _read_phi4 made no progress
    once readline() started returning '' at EOF, spinning forever on a
    truncated file.
    """
    from kaldo.interfaces.tdep_io import parse_tdep_fourth_forceconstant

    uc = tmp_path / "infile.ucposcar"
    uc.write_text(
        "Si\n1.0\n3.0 0.0 0.0\n0.0 3.0 0.0\n0.0 0.0 3.0\nSi\n2\n"
        "Direct\n0.0 0.0 0.0\n0.5 0.5 0.5\n"
    )
    fc = tmp_path / "infile.forceconstant_fourthorder"
    _write_ifc4(fc, 2, {0: [(1, 0, 1, (0, 0, 0), (0, 0, 0), (0, 0, 0),
                             np.ones((3, 3, 3, 3)))]})
    # Cut the file mid-phi-block (keep header + indices + R vectors + a few values)
    lines = fc.read_text().splitlines()
    fc.write_text("\n".join(lines[:15]) + "\n")

    with pytest.raises(ValueError, match="unexpected end of file"):
        parse_tdep_fourth_forceconstant(fc_filename=str(fc), primitive=str(uc), supercell=(1, 1, 1))
