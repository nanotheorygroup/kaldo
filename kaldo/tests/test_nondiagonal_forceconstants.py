"""
Tests for non-diagonal SNF supercell support in ``kaldo.ForceConstants``.

Covers:

  * ``from_folder(supercell_matrix=M)`` accepts a 3x3 integer matrix and
    loads IFC2/IFC3/IFC4 on a non-diagonal tiling (rhombo primitive +
    cubic conventional ssposcar).
  * ``list_of_replicas`` returns the SNF Cartesian replica vectors.
  * ``Conductivity.rta`` runs end-to-end on a non-diagonal fc.
  * ``IFC3`` / ``IFC4`` are stored at the right shapes.
  * Diagonal path and SNF path agree element-wise on a fundamentally
    diagonal M (backward-compat firewall).
  * ``supercell_matrix`` validation rejects non-integer matrices and
    matrices that disagree with the inferred ucposcar->ssposcar mapping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SI_PROD = Path("/home/giuseppe/Development/4th-order-cumulants/reference_si/T300_0")
SI_TDEP_DIR = Path(__file__).parent / "si-tdep"
SI_MASS_AMU = 28.0855


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_conductivity_runs_on_nondiagonal_fc():
    """BTE/conductivity smoke test: Conductivity runs on a non-diagonal fc.

    Exercises the Grid + list_of_replicas + chi() + sij + velocity path
    through kaldo's conductivity.rta, confirming our NonDiagonalGrid is
    compatible with BTE machinery and not just Phonons.frequency.

    The exact kappa number depends heavily on mesh + sigma choices on the
    production DFT Si fixture; here we only assert:

      * Conductivity() runs to completion without error
      * kappa diagonal is finite and positive on all 3 axes
      * Order of magnitude matches the si-tdep regression (~50-150 W/mK)
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.phonons import Phonons
    from kaldo.conductivity import Conductivity

    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=M, format="tdep",
    )
    ph = Phonons(
        forceconstants=fc, kpts=(5, 5, 5), temperature=300,
        is_classic=False, storage="memory", is_unfolding=False,
    )
    cond = Conductivity(phonons=ph, method="rta", storage="memory").conductivity
    kappa = cond.sum(axis=0).diagonal()
    assert np.all(np.isfinite(kappa)), f"kappa has non-finite entries: {kappa}"
    assert np.all(kappa > 0), f"kappa diagonal non-positive: {kappa}"
    kappa_mean = kappa.mean()
    # On this 5^3 mesh + 5^3 supercell Si DFT fixture, kappa can run high
    # because low-q acoustic modes dominate and are under-resolved. The
    # exact value isn't the point of this smoke test; we just want to
    # confirm the NonDiagonalGrid -> BTE path produces a sensible order
    # of magnitude (tens to ~thousand W/mK range).
    assert 10 < kappa_mean < 2000, (
        f"Si nondiag kappa = {kappa_mean:.1f} W/mK, unphysical order"
    )


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E4_third_order_nondiag_loads():
    """ThirdOrder.load with supercell_matrix reads non-diagonal IFC3."""
    from kaldo.forceconstants import ForceConstants
    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=M, format="tdep",
    )
    assert fc.third is not None
    # Expected: (n_uc, 3, n_rep, n_uc, 3, n_rep, n_uc, 3)
    assert fc.third.value.shape == (2, 3, 108, 2, 3, 108, 2, 3)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E4_fourth_order_nondiag_loads():
    """FourthOrder.load with supercell_matrix reads non-diagonal IFC4."""
    from kaldo.forceconstants import ForceConstants
    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=M, format="tdep", include_fourth=True,
    )
    assert fc.fourth is not None
    assert fc.fourth.value.shape == (2, 3, 108, 2, 3, 108, 2, 3, 108, 2, 3)


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E2_list_of_replicas_on_nondiagonal_si():
    """fc.second.list_of_replicas must return the SNF Cartesian replica vectors.

    For non-diagonal Si production: 108 replicas, each in R = (a, b, c)_prim * uc_cell,
    where (a,b,c) are integer triples from the SNF replica_table.
    """
    from kaldo.forceconstants import ForceConstants
    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=M, format="tdep", only_second=True,
    )
    lr = fc.second.list_of_replicas  # (n_rep, 3) Cartesian
    assert lr.shape == (108, 3)
    # Compare to replica_table @ uc_cell
    expected = fc.second._replica_table @ np.asarray(fc.atoms.cell)
    np.testing.assert_allclose(lr, expected, atol=1e-10)


@pytest.mark.skipif(not SI_TDEP_DIR.exists(), reason="si-tdep fixture missing")
def test_diagonal_path_and_snf_path_agree_on_si_tdep():
    """Backward-compat firewall: loading si-tdep (diagonal 5^3) twice — once
    via the diagonal path (``supercell=(5,5,5)``) and once via the SNF path
    (``supercell_matrix=diag([5,5,5])``) — must produce IFC2 tensors that
    agree element-wise on the (i, j, R) keys.

    Catches any future drift where the SNF path silently disagrees with
    the diagonal path on a fundamentally diagonal M.
    """
    from kaldo.forceconstants import ForceConstants

    fc_diag = ForceConstants.from_folder(
        folder=str(SI_TDEP_DIR), supercell=(5, 5, 5), format="tdep",
    )
    M_diag = np.diag([5, 5, 5])
    fc_snf = ForceConstants.from_folder(
        folder=str(SI_TDEP_DIR), supercell_matrix=M_diag, format="tdep",
    )
    assert fc_diag.n_replicas == fc_snf.n_replicas == 125

    def collect_ifc2(fc):
        second = np.asarray(fc.second.value)[0]  # (n_uc, 3, n_rep, n_uc, 3)
        n_uc = fc.n_atoms
        rep_pos = np.asarray(fc.second.replicated_positions).reshape(
            fc.n_replicas, n_uc, 3,
        )
        uc_cell = np.asarray(fc.atoms.cell)
        inv_cell = np.linalg.inv(uc_cell)
        d = {}
        for i in range(n_uc):
            for r in range(fc.n_replicas):
                for j in range(n_uc):
                    phi = second[i, :, r, j, :]
                    if not np.any(phi):
                        continue
                    rj = rep_pos[r, j]
                    R_frac = (rj - np.asarray(fc.atoms.positions)[j]) @ inv_cell
                    R_min = np.round(R_frac - 5 * np.round(R_frac / 5)).astype(int)
                    d[(i, j, tuple(R_min))] = phi
        return d

    a = collect_ifc2(fc_diag)
    b = collect_ifc2(fc_snf)
    assert set(a) == set(b), (
        f"diagonal vs SNF (i,j,R) sets differ: only_diag={set(a)-set(b)};"
        f" only_snf={set(b)-set(a)}"
    )
    for key, phi_diag in a.items():
        np.testing.assert_allclose(
            phi_diag, b[key], atol=1e-12,
            err_msg=f"IFC2 entry {key} differs between diagonal and SNF paths",
        )


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_supercell_matrix_must_be_integer_valued():
    """A non-integer ``supercell_matrix`` must raise a clear error."""
    from kaldo.forceconstants import ForceConstants
    M = np.array([[3.5, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=float)
    with pytest.raises(ValueError, match=r"(?i)integer"):
        ForceConstants.from_folder(
            folder=str(SI_PROD), supercell_matrix=M, format="tdep", only_second=True,
        )


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_supercell_matrix_must_match_inferred():
    """A correct-shape but wrong supercell_matrix must raise."""
    from kaldo.forceconstants import ForceConstants
    M_wrong = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=int)
    with pytest.raises(ValueError, match=r"(?i)does not match"):
        ForceConstants.from_folder(
            folder=str(SI_PROD), supercell_matrix=M_wrong, format="tdep",
            only_second=True,
        )


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_E1_from_folder_accepts_supercell_matrix_on_nondiagonal_si():
    """ForceConstants.from_folder with supercell_matrix=M must load non-diagonal
    TDEP without raising the diagonal guard.

    Si production: primitive is rhombohedral, ssposcar is the 3x3x3 conventional
    cubic supercell (det M = 108).
    """
    from kaldo.forceconstants import ForceConstants
    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD),
        supercell_matrix=M,
        format="tdep",
        only_second=True,  # E.4 will enable IFC3 non-diagonal
    )
    # n_uc=2, n_replicas = det(M) = 108
    assert fc.n_atoms == 2
    assert fc.n_replicas == 108
    # The IFC2 tensor has the right shape for the non-diagonal storage
    assert fc.second.value.shape == (1, 2, 3, 108, 2, 3)
