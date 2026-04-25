"""
Cross-check tests linking ``ForceConstants.from_folder(supercell_matrix=M)``
to the ``kaldo.cumulant`` subpackage.

These live in their own module (instead of ``test_nondiagonal_forceconstants.py``)
because they import ``kaldo.cumulant`` and would force that subpackage to be
a dependency of the kaldo-core non-diagonal SNF feature. Keeping them
separate makes the dependency direction explicit: ``kaldo.cumulant``
consumes ``kaldo.ForceConstants``, not the other way around.

Covers (using the E.* labels from the original phased TDD):

  * E.3 — ``Phonons(fc).frequency`` on non-diagonal Si matches
    ``kaldo.cumulant.dynmat_and_eigs`` on the same TDEP IFC2 file.
  * E.4 — IFC3 cross-check: kaldo sparse-COO IFC3 vs cumulant list-of-tuples
    sum / L1 / Frobenius² agree on non-diagonal Si.
  * E.5 — ``F1_from_fc`` / ``F2_from_fc`` on non-diagonal Si production
    fixture match the pinned regression values at 2³ mesh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_PROD = Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0")
SI_MASS_AMU = 28.0855
SI_M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E5_F2_from_fc_on_nondiagonal_si_matches_pinned_2cubed():
    """F2_from_fc on non-diagonal Si production at 2^3 mesh matches pinned -6.43481e-5."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F2_from_fc

    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=SI_M, format="tdep",
    )
    res = F2_from_fc(
        fc, masses_amu=np.full(2, SI_MASS_AMU),
        kmesh=(2, 2, 2), T_K=300.0, sigma_THz=None,
        use_q_symmetry=True,
    )
    np.testing.assert_allclose(res["F2"], -6.43481e-5, rtol=5e-4)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E5_F1_from_fc_on_nondiagonal_si_matches_pinned_2cubed():
    """F1_from_fc on non-diagonal Si production at 2^3 mesh matches pinned -1.16592e-5."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import F1_from_fc

    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=SI_M, format="tdep",
        include_fourth=True,
    )
    res = F1_from_fc(
        fc, masses_amu=np.full(2, SI_MASS_AMU),
        kmesh=(2, 2, 2), T_K=300.0,
        use_q_symmetry=True,
    )
    np.testing.assert_allclose(res["F1"], -1.16592e-5, rtol=2e-4)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E4_third_order_cross_check_vs_cumulant():
    """Sum/|Phi|/Frobenius match between kaldo nondiag IFC3 reader and cumulant list."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.cumulant import read_tdep_ifc3

    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=SI_M, format="tdep",
    )
    dense = fc.third.value.todense()
    per_atom = read_tdep_ifc3(str(SI_PROD / "infile.forceconstant_thirdorder"), 2)
    sum_cum = 0.0; l1_cum = 0.0; l2sq_cum = 0.0
    for il in per_atom:
        for (_, _, _, _, phi) in il:
            sum_cum += phi.sum()
            l1_cum += np.abs(phi).sum()
            l2sq_cum += (phi ** 2).sum()
    np.testing.assert_allclose(float(dense.sum()), sum_cum, atol=1e-10)
    np.testing.assert_allclose(float(np.abs(dense).sum()), l1_cum, rtol=1e-12)
    np.testing.assert_allclose(float((dense ** 2).sum()), l2sq_cum, rtol=1e-12)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E3_phonons_frequency_matches_cumulant_dynmat_si_nondiag():
    """Phonons(fc).frequency on non-diagonal Si matches kaldo.cumulant.dynmat_and_eigs.

    Sorts spectra per q and globally to sidestep mode-ordering and MP mesh
    permutation. Tolerance: 2e-4 THz absolute, 5e-5 rel on non-acoustic modes.
    """
    import ase.io
    from kaldo.forceconstants import ForceConstants
    from kaldo.phonons import Phonons
    from kaldo.cumulant import read_tdep_pair_fcs, dynmat_and_eigs, AMU
    from kaldo.cumulant.harmonic import monkhorst_pack_qcart

    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=SI_M, format="tdep", only_second=True,
    )
    ph = Phonons(forceconstants=fc, kpts=(3, 3, 3), temperature=300,
                 is_classic=False, storage="memory")
    freq_kaldo = np.asarray(ph.frequency)

    uc = ase.io.read(str(SI_PROD / "infile.ucposcar"), format="vasp")
    uc_pos = np.asarray(uc.positions)
    uc_cell = np.asarray(uc.cell)
    masses_kg = np.full(2, SI_MASS_AMU * AMU)
    neighbors = read_tdep_pair_fcs(
        str(SI_PROD / "infile.forceconstant"), uc_pos, uc_cell,
    )
    qs = monkhorst_pack_qcart((3, 3, 3), uc_cell)
    freq_cum = np.zeros((qs.shape[0], 6))
    for i, q in enumerate(qs):
        w, _ = dynmat_and_eigs(neighbors, uc_pos, masses_kg, q)
        freq_cum[i] = w / (2 * np.pi * 1e12)

    a = np.sort(np.sort(freq_kaldo, axis=1), axis=0)
    b = np.sort(np.sort(freq_cum, axis=1), axis=0)
    assert a.shape == b.shape
    max_abs = np.max(np.abs(a - b))
    assert max_abs < 2e-4, f"spectra diverge by {max_abs:.2e} THz"
    mask = (np.abs(a) > 0.1) & (np.abs(b) > 0.1)
    rel = np.max(np.abs((a[mask] - b[mask]) / a[mask]))
    assert rel < 5e-5, f"non-acoustic modes diverge by rel {rel:.2e}"
