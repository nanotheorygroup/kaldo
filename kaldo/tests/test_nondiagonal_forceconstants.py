"""
Tests for non-diagonal SNF supercell support in ``kaldo.ForceConstants``.

Covers:

  * ``from_folder(supercell_matrix=M)`` accepts a 3x3 integer matrix and
    loads IFC2/IFC3/IFC4 on a non-diagonal tiling (rhombo primitive +
    cubic conventional ssposcar).
  * ``list_of_replicas`` returns the per-pair Cartesian lattice vectors
    from the IFC2 file (the det(M) class table survives as metadata).
  * ``replicated_atoms`` carries the true ``M @ uc.cell`` supercell cell.
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

# Production-only fixture: large DFT-quality Si IFCs at 25^3 supercell.
# Set KALDO_TEST_SI_PROD to point at reference_si/T300_0 to enable.
# See kaldo/tests/_paths.py for details on env-var-gated test fixtures.
from kaldo.tests._paths import SI_PROD
SI_TDEP_DIR = Path(__file__).parent / "si-tdep"
SI_MASS_AMU = 28.0855


def test_replicated_atoms_cell_is_correct_on_nondiagonal_tiling():
    """``ForceConstant.replicated_atoms`` must carry the true supercell cell
    (``M @ uc.cell``) on a non-diagonal SNF tiling.

    Self-contained (no fixture): a 2-atom Si primitive tiled by a
    non-diagonal M into a cubic conventional supercell. Regression for the
    293 x 2.7 x 2.7 A "sliver" cell that ``atoms * (n_rep, 1, 1)`` used to
    produce, which silently corrupted anything consuming
    ``replicated_atoms`` on non-diagonal TDEP runs (e.g. the cumulant
    supercell sampler and any PBC-aware energy evaluation).
    """
    from ase.build import bulk, make_supercell
    from kaldo.observables.secondorder import SecondOrder
    from kaldo.interfaces.tdep_io import build_nondiag_observable_kwargs

    uc = bulk("Si", "diamond", a=5.43)
    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    sc = make_supercell(uc, M)  # cubic conventional, 216 atoms, det M = 108

    kw = build_nondiag_observable_kwargs(uc, sc)
    kw.pop("_mapping")
    n_rep = kw["supercell"][0]
    n_uc = len(uc)
    so = SecondOrder(value=np.zeros((1, n_uc, 3, n_rep, n_uc, 3)), folder="kALDo", **kw)

    ra = so.replicated_atoms
    assert len(ra) == len(sc) == 216
    np.testing.assert_allclose(np.asarray(ra.cell), np.asarray(sc.cell), atol=1e-10)

    # Every replicated atom sits on an ssposcar site modulo the (correct) cell.
    inv = np.linalg.inv(np.asarray(ra.cell))
    ra_frac = np.asarray(ra.positions) @ inv
    sc_frac = np.asarray(sc.positions) @ inv
    for p in ra_frac:
        d = sc_frac - p
        d -= np.round(d)
        assert np.min(np.linalg.norm(d, axis=1)) < 1e-9


@pytest.mark.skipif(not SI_PROD.exists(), reason="non-diagonal Si fixture unavailable")
def test_replicated_atoms_cell_matches_ssposcar_on_production_si():
    """On the production non-diagonal Si fixture, ``replicated_atoms.cell``
    equals the infile.ssposcar cell."""
    import ase.io
    from kaldo.forceconstants import ForceConstants

    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(folder=str(SI_PROD), supercell_matrix=M, format="tdep", only_second=True)
    sc = ase.io.read(str(SI_PROD / "infile.ssposcar"), format="vasp")
    np.testing.assert_allclose(
        np.asarray(fc.second.replicated_atoms.cell), np.asarray(sc.cell), atol=1e-8
    )


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
    """fc.second.list_of_replicas must return the per-pair Cartesian lattice vectors.

    Since issue #297 the SNF second-order grid stores the unique per-pair
    lattice vectors from the TDEP file (so Fourier interpolation matches
    TDEP/phonopy between commensurate q). The det(M) = 108 BvK class table
    survives as ``_replica_table`` metadata, and every per-pair vector must
    reduce to one of its classes.
    """
    from kaldo.forceconstants import ForceConstants
    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD), supercell_matrix=M, format="tdep", only_second=True,
    )
    lr = fc.second.list_of_replicas  # (n_R, 3) Cartesian, per-pair vectors
    grid_table = fc.second._direct_grid.grid(is_wrapping=True)
    np.testing.assert_allclose(lr, grid_table @ np.asarray(fc.atoms.cell), atol=1e-10)
    table = np.asarray(fc.second._replica_table)
    assert table.shape == (108, 3)
    Minv = np.linalg.inv(M.astype(float))
    for r in grid_table:
        diffs = (r - table) @ Minv
        assert np.any(np.all(np.abs(diffs - np.rint(diffs)) < 1e-6, axis=1))


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
    assert fc.n_atoms == 2
    # Since issue #297 the second-order replica axis holds the unique
    # per-pair lattice vectors from the file, not the det(M) = 108 classes.
    n_R = len(fc.second._direct_grid.grid(is_wrapping=True))
    assert fc.n_replicas == n_R
    assert fc.second.value.shape == (1, 2, 3, n_R, 2, 3)


# ---------------------------------------------------------------------------
# Issue #297: per-pair Fourier phases on the SNF path
# ---------------------------------------------------------------------------

_P_297 = np.array([[1, -1, 0], [0, 1, -1], [3, 3, 3]], dtype=int)  # det 9, anisotropic


def _write_297_model(folder, onsite_defect=0.0):
    """Write a self-contained TDEP model on the det-9 anisotropic tiling from
    issue #297: a stable two-species spring crystal whose IFC2 entries carry
    per-pair minimum-image lattice vectors, exactly as TDEP writes them.
    Returns (uc, entries) with entries = [(a1, a2, R, phi), ...].
    """
    import ase
    import ase.io
    from ase.build import make_supercell

    cell = np.array([[3.0, 0.0, 0.0], [0.3, 3.3, 0.0], [0.0, 0.2, 3.7]])
    uc = ase.Atoms("SiGe", cell=cell, scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]], pbc=True)
    sc = make_supercell(uc, _P_297)
    ase.io.write(str(folder / "infile.ucposcar"), uc, format="vasp")
    ase.io.write(str(folder / "infile.ssposcar"), sc, format="vasp")

    # Springs phi = -k(d) I between all pairs within rcut, keyed by the pair's
    # strict minimum-image lattice vector under the supercell lattice
    # (boundary ties are skipped so the file is unambiguous). On-site terms
    # enforce the acoustic sum rule, so the model is dynamically stable.
    rcut = 4.6
    shifts = np.array([[a, b, c]
                       for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                       if (a, b, c) != (0, 0, 0)])
    pos = uc.positions
    entries = []
    onsite = np.zeros((2, 3, 3))
    for i in range(2):
        for j in range(2):
            for Ra in range(-4, 5):
                for Rb in range(-4, 5):
                    for Rc in range(-4, 5):
                        R = np.array([Ra, Rb, Rc])
                        v = pos[j] + R @ cell - pos[i]
                        d = np.linalg.norm(v)
                        if d < 1e-9 or d > rcut:
                            continue
                        alt = np.linalg.norm(pos[j] + ((R - shifts @ _P_297) @ cell) - pos[i], axis=1)
                        if not np.all(d <= alt - 1e-9):
                            continue
                        k = 4.0 * np.exp(-d / 1.8)
                        entries.append((i, j, R.copy(), -k * np.eye(3)))
                        onsite[i] += k * np.eye(3)
    for i in range(2):
        entries.append((i, i, np.zeros(3, dtype=int), onsite[i] + onsite_defect * np.eye(3)))

    with open(folder / "infile.forceconstant", "w") as f:
        f.write("2\n100.0\n")
        for i in range(2):
            mine = [e for e in entries if e[0] == i]
            f.write(f"{len(mine)}\n")
            for _, a2, R, phi in mine:
                f.write(f"{a2 + 1}\n")
                f.write(" ".join(f"{x:.1f}" for x in R) + "\n")
                for row in phi:
                    f.write(" ".join(f"{x:.12f}" for x in row) + "\n")
    return uc, entries


def _per_pair_frequencies_thz(uc, entries, q):
    """Reference frequencies from the per-pair Fourier sum, converted with
    kaldo's own constants (value * mol/(10 J); nu = sqrt(lambda)/2pi)."""
    from ase import units

    masses = uc.get_masses()
    n = len(uc)
    D = np.zeros((n, 3, n, 3), dtype=complex)
    for a1, a2, R, phi in entries:
        D[a1, :, a2, :] += phi * np.exp(2j * np.pi * (q @ R)) / np.sqrt(masses[a1] * masses[a2])
    lam = np.linalg.eigvalsh(D.reshape(3 * n, 3 * n) * units.mol / (10 * units.J))
    return np.sign(lam) * np.sqrt(np.abs(lam)) / (2 * np.pi)


def test_tdep_snf_matches_per_pair_reference_at_incommensurate_q(tmp_path):
    """Issue #297: at q-points not commensurate with the ssposcar lattice the
    plain path must reproduce the per-pair Fourier sum (the convention of
    TDEP/phonopy/ALAMODE, whose vectors the file provides). The former
    class-collapsed table was exact on the commensurate set but off by up
    to ~85 cm^-1, with spurious imaginary acoustics, on band-path q.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    uc, entries = _write_297_model(tmp_path)
    fc = ForceConstants.from_folder(str(tmp_path), format="tdep", only_second=True)
    for q in ([0.6667, 0.3333, 0.1111], [0.1234, 0.4321, 0.2468]):
        q = np.array(q)
        hq = HarmonicWithQ(q_point=q, second=fc.second, storage="memory")
        got = np.sort(np.asarray(hq.frequency).ravel())
        ref = np.sort(_per_pair_frequencies_thz(uc, entries, q))
        np.testing.assert_allclose(got, ref, atol=1e-8)


def test_is_unfolding_rejected_on_snf_path(tmp_path):
    """is_unfolding assumes a diagonal (nx, ny, nz) supercell; on the SNF
    path it silently returns garbage, so it must be refused."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    _write_297_model(tmp_path)
    fc = ForceConstants.from_folder(str(tmp_path), format="tdep", only_second=True)
    with pytest.raises(NotImplementedError, match="non-diagonal"):
        HarmonicWithQ(q_point=np.array([0.1, 0.2, 0.3]), second=fc.second,
                      is_unfolding=True, storage="memory")


def test_tdep_snf_acoustic_sum_rule_targets_home_cell(tmp_path):
    """is_acoustic_sum=True subtracts the total row sum of each atom at the
    home cell (replica 0, R = [0, 0, 0]). On a per-pair table np.unique
    would put a negative vector at index 0, attaching the correction to a
    wrong lattice vector: invisible at Gamma (all phases are 1 there) but
    wrong at any other q. With a deliberate on-site defect, ASR-corrected
    loading of the defected file must reproduce the defect-free model.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    uc, _ = _write_297_model(tmp_path, onsite_defect=0.05)
    clean = tmp_path / "clean"
    clean.mkdir()
    _, entries_clean = _write_297_model(clean, onsite_defect=0.0)
    fc = ForceConstants.from_folder(str(tmp_path), format="tdep",
                                    only_second=True, is_acoustic_sum=True)
    assert np.array_equal(fc.second._direct_grid.grid(is_wrapping=True)[0], [0, 0, 0])
    q = np.array([0.1234, 0.4321, 0.2468])
    hq = HarmonicWithQ(q_point=q, second=fc.second, storage="memory")
    got = np.sort(np.asarray(hq.frequency).ravel())
    ref = np.sort(_per_pair_frequencies_thz(uc, entries_clean, q))
    np.testing.assert_allclose(got, ref, atol=1e-8)
