"""
Tests for non-diagonal SNF supercell support in ``kaldo.ForceConstants``.

Covers:

  * ``from_folder(supercell_matrix=M)`` accepts a 3x3 integer matrix and
    loads IFC2/IFC3 on a non-diagonal tiling (rhombohedral primitive +
    cubic conventional ssposcar).
  * physical quotient replicas remain distinct from literal IFC translations.
  * ``replicated_atoms`` carries the true ``M @ uc.cell`` supercell cell.
  * ``Conductivity.rta`` runs end-to-end on a non-diagonal fc.
  * ``IFC3`` / ``IFC4`` are stored at the right shapes.
  * Diagonal path and SNF path agree element-wise on a fundamentally
    diagonal M (backward-compat firewall).
  * ``supercell_matrix`` validation rejects non-integer matrices and
    matrices that disagree with the inferred ucposcar->ssposcar mapping.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# IFC4 remains an explicitly deferred, production-only integration test.
from kaldo.tests._paths import SI_PROD

SI_TDEP_DIR = Path(__file__).parent / "si-tdep"
SI_CONVENTIONAL_DIR = Path(__file__).parent / "data" / "input" / "tdep-si-conventional"
SI_CONVENTIONAL_MATRIX = np.array([[1, -1, 1], [1, 1, -1], [-1, 1, 1]], dtype=int)
SI_MASS_AMU = 28.0855


def test_lazy_forceconstants_preserve_nondiagonal_topology(tmp_path):
    """Lazy empty IFC2/IFC3 objects must retain the full integer matrix."""
    from ase.build import bulk
    from kaldo.forceconstants import ForceConstants

    atoms = bulk("Si", "diamond", a=5.43)
    matrix = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=int)
    forceconstants = ForceConstants(
        atoms=atoms,
        supercell=matrix,
        third_supercell=matrix,
        folder=str(tmp_path),
    )

    np.testing.assert_array_equal(forceconstants.second.supercell_grid.matrix, matrix)
    np.testing.assert_array_equal(forceconstants.third.supercell_grid.matrix, matrix)
    assert forceconstants.second.n_replicas == 4
    assert forceconstants.third.n_replicas == 4


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
    M = SI_CONVENTIONAL_MATRIX
    sc = make_supercell(uc, M)  # one cubic conventional cell, det(M) = 4

    kw = build_nondiag_observable_kwargs(uc, sc)
    kw.pop("_mapping")
    n_rep = kw["supercell_grid"].size
    n_uc = len(uc)
    so = SecondOrder(value=np.zeros((1, n_uc, 3, n_rep, n_uc, 3)), folder="kALDo", **kw)

    ra = so.replicated_atoms
    assert len(ra) == len(sc) == 8
    np.testing.assert_allclose(np.asarray(ra.cell), np.asarray(sc.cell), atol=1e-10)
    assert so.supercell_replicas.shape == (27 * n_rep, 3)
    np.testing.assert_allclose(
        so.supercell_positions,
        np.stack(
            np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        @ np.asarray(sc.cell),
        atol=1e-12,
    )

    # Every replicated atom sits on an ssposcar site modulo the (correct) cell.
    inv = np.linalg.inv(np.asarray(ra.cell))
    ra_frac = np.asarray(ra.positions) @ inv
    sc_frac = np.asarray(sc.positions) @ inv
    for p in ra_frac:
        d = sc_frac - p
        d -= np.round(d)
        assert np.min(np.linalg.norm(d, axis=1)) < 1e-9


def test_replicated_atoms_cell_matches_compact_ssposcar():
    """The loaded physical replica cell must equal ``infile.ssposcar``."""
    import ase.io
    from kaldo.forceconstants import ForceConstants

    fc = ForceConstants.from_folder(
        folder=str(SI_CONVENTIONAL_DIR),
        supercell_matrix=SI_CONVENTIONAL_MATRIX,
        format="tdep",
        only_second=True,
    )
    sc = ase.io.read(str(SI_CONVENTIONAL_DIR / "infile.ssposcar"), format="vasp")
    np.testing.assert_allclose(
        np.asarray(fc.second.replicated_atoms.cell), np.asarray(sc.cell), atol=1e-8
    )


def test_conductivity_runs_on_nondiagonal_fc():
    """BTE/conductivity smoke test: Conductivity runs on a non-diagonal fc.

    Exercises exact translation support, phases, flux, and velocity through
    ``conductivity.rta``, confirming non-diagonal topology reaches the BTE
    machinery and not only ``Phonons.frequency``.

    The IFCs are an analytic test model, not a material-property reference.
    This test therefore asserts only that:

      * Conductivity() runs to completion without error
      * kappa diagonal is finite and positive on all 3 axes
      * the non-diagonal IFC3 translation support reaches the BTE contraction
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.phonons import Phonons
    from kaldo.conductivity import Conductivity

    fc = ForceConstants.from_folder(
        folder=str(SI_CONVENTIONAL_DIR),
        supercell_matrix=SI_CONVENTIONAL_MATRIX,
        format="tdep",
    )
    ph = Phonons(
        forceconstants=fc,
        kpts=(3, 3, 3),
        temperature=300,
        is_classic=False,
        storage="memory",
        ifc_interpolation="auto",
    )
    cond = Conductivity(phonons=ph, method="rta", storage="memory").conductivity
    kappa = cond.sum(axis=0).diagonal()
    assert np.all(np.isfinite(kappa)), f"kappa has non-finite entries: {kappa}"
    assert np.all(kappa > 0), f"kappa diagonal non-positive: {kappa}"
    assert fc.third.n_translations == 7


def test_third_order_nondiagonal_loads_with_literal_support():
    """ThirdOrder.load with supercell_matrix reads non-diagonal IFC3."""
    from kaldo.forceconstants import ForceConstants

    fc = ForceConstants.from_folder(
        folder=str(SI_CONVENTIONAL_DIR),
        supercell_matrix=SI_CONVENTIONAL_MATRIX,
        format="tdep",
    )
    assert fc.third is not None
    # Literal TDEP axes may exceed the four physical quotient classes.
    assert fc.third.n_replicas == 4
    assert fc.third.n_translations == 7
    assert fc.third.value.shape == (2, 3, 7, 2, 3, 7, 2, 3)


def test_compact_fixture_manifest_and_acoustic_sum_rules():
    """Generated topology and exact IFC2/IFC3 sum rules must remain intact."""
    from kaldo.forceconstants import ForceConstants

    manifest = json.loads((SI_CONVENTIONAL_DIR / "expected.json").read_text())
    fc = ForceConstants.from_folder(
        folder=str(SI_CONVENTIONAL_DIR),
        supercell_matrix=SI_CONVENTIONAL_MATRIX,
        format="tdep",
    )

    assert manifest["determinant"] == fc.n_replicas == 4
    assert manifest["supercell_atoms"] == len(fc.second.replicated_atoms) == 8
    expected_support = np.asarray(manifest["translation_support"])
    np.testing.assert_array_equal(
        fc.second.translation_support.translations, expected_support
    )
    np.testing.assert_array_equal(
        fc.third.translation_support.translations, expected_support
    )

    ifc2 = np.asarray(fc.second.value)[0]
    # Sum over every translated partner atom for each central atom and pair
    # of Cartesian components.
    np.testing.assert_allclose(ifc2.sum(axis=(2, 3)), 0.0, atol=1e-14)

    ifc3 = np.asarray(fc.third.value.todense())
    # Translating either partner index rigidly cannot change the bond energy.
    np.testing.assert_allclose(ifc3.sum(axis=(2, 3)), 0.0, atol=1e-14)
    np.testing.assert_allclose(ifc3.sum(axis=(5, 6)), 0.0, atol=1e-14)


@pytest.mark.skipif(
    not SI_PROD.exists(), reason="non-diagonal IFC4 fixture unavailable"
)
def test_E4_fourth_order_nondiag_loads():
    """FourthOrder.load with supercell_matrix reads non-diagonal IFC4."""
    from kaldo.forceconstants import ForceConstants

    M = np.array([[3, -3, 3], [3, 3, -3], [-3, 3, 3]], dtype=int)
    fc = ForceConstants.from_folder(
        folder=str(SI_PROD),
        supercell_matrix=M,
        format="tdep",
        include_fourth=True,
    )
    assert fc.fourth is not None
    assert fc.fourth.value.shape == (2, 3, 108, 2, 3, 108, 2, 3, 108, 2, 3)


def test_physical_replicas_and_ifc_support_remain_distinct():
    """Physical quotient representatives and IFC Fourier support must differ."""
    from kaldo.forceconstants import ForceConstants

    fc = ForceConstants.from_folder(
        folder=str(SI_CONVENTIONAL_DIR),
        supercell_matrix=SI_CONVENTIONAL_MATRIX,
        format="tdep",
        only_second=True,
    )
    assert fc.second.n_replicas == 4
    assert fc.second.replica_translations.shape == (4, 3)
    assert fc.second.n_translations == 7
    expected = fc.second.translation_support.translations @ np.asarray(fc.atoms.cell)
    np.testing.assert_allclose(fc.second.list_of_replicas, expected, atol=1e-10)


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
        folder=str(SI_TDEP_DIR),
        supercell=(5, 5, 5),
        format="tdep",
    )
    M_diag = np.diag([5, 5, 5])
    fc_snf = ForceConstants.from_folder(
        folder=str(SI_TDEP_DIR),
        supercell_matrix=M_diag,
        format="tdep",
    )
    assert fc_diag.n_replicas == fc_snf.n_replicas == 125

    def collect_ifc2(fc):
        second = np.asarray(fc.second.value)[0]  # (n_uc, 3, n_rep, n_uc, 3)
        n_uc = fc.n_atoms
        d = {}
        for i in range(n_uc):
            for r, translation in enumerate(fc.second.translation_support.translations):
                for j in range(n_uc):
                    phi = second[i, :, r, j, :]
                    if not np.any(phi):
                        continue
                    R_min = np.round(
                        translation - 5 * np.round(translation / 5)
                    ).astype(int)
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
            phi_diag,
            b[key],
            atol=1e-12,
            err_msg=f"IFC2 entry {key} differs between diagonal and SNF paths",
        )


def test_supercell_matrix_must_be_integer_valued():
    """A non-integer ``supercell_matrix`` must raise a clear error."""
    from kaldo.forceconstants import ForceConstants

    M = SI_CONVENTIONAL_MATRIX.astype(float)
    M[0, 0] += 0.5
    with pytest.raises(ValueError, match=r"(?i)integer"):
        ForceConstants.from_folder(
            folder=str(SI_CONVENTIONAL_DIR),
            supercell_matrix=M,
            format="tdep",
            only_second=True,
        )


def test_supercell_matrix_must_match_inferred():
    """A correct-shape but wrong supercell_matrix must raise."""
    from kaldo.forceconstants import ForceConstants

    M_wrong = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=int)
    with pytest.raises(ValueError, match=r"(?i)does not match"):
        ForceConstants.from_folder(
            folder=str(SI_CONVENTIONAL_DIR),
            supercell_matrix=M_wrong,
            format="tdep",
            only_second=True,
        )


def test_from_folder_accepts_compact_nondiagonal_si():
    """ForceConstants.from_folder with supercell_matrix=M must load non-diagonal
    TDEP without raising the diagonal guard.

    The two-atom rhombohedral primitive maps to one eight-atom conventional
    cubic cell with determinant four.
    """
    from kaldo.forceconstants import ForceConstants

    fc = ForceConstants.from_folder(
        folder=str(SI_CONVENTIONAL_DIR),
        supercell_matrix=SI_CONVENTIONAL_MATRIX,
        format="tdep",
        only_second=True,
    )
    assert fc.n_atoms == 2
    assert fc.n_replicas == 4
    # Seven literal bond translations are retained on the IFC2 axis.
    assert fc.second.value.shape == (1, 2, 3, 7, 2, 3)


# ---------------------------------------------------------------------------
# Per-pair Fourier phases on a non-diagonal TDEP supercell
# ---------------------------------------------------------------------------

_PER_PAIR_SUPERCELL = np.array([[1, -1, 0], [0, 1, -1], [3, 3, 3]], dtype=int)


def _write_per_pair_tdep_model(folder, onsite_defect=0.0):
    """Write a stable TDEP IFC2 model carrying literal pair translations.

    The determinant-nine, anisotropic supercell has several shortest pair
    vectors in the same periodic class.  TDEP writes those vectors literally;
    replacing them with one representative per class therefore changes the
    Fourier phases away from the commensurate supercell grid.
    """
    import ase.io
    from ase import Atoms
    from ase.build import make_supercell

    cell = np.array([[3.0, 0.0, 0.0], [0.3, 3.3, 0.0], [0.0, 0.2, 3.7]])
    primitive = Atoms(
        "SiGe",
        cell=cell,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        pbc=True,
    )
    supercell = make_supercell(primitive, _PER_PAIR_SUPERCELL)
    ase.io.write(folder / "infile.ucposcar", primitive, format="vasp")
    ase.io.write(folder / "infile.ssposcar", supercell, format="vasp")

    # Build central springs between strict minimum-image pairs. Boundary ties
    # are skipped so every record has one unambiguous literal translation.
    cutoff = 4.6
    neighboring_shifts = np.array(
        [
            [a, b, c]
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for c in (-1, 0, 1)
            if (a, b, c) != (0, 0, 0)
        ]
    )
    positions = primitive.positions
    entries = []
    onsite = np.zeros((2, 3, 3))
    for atom_i in range(2):
        for atom_j in range(2):
            for r_a in range(-4, 5):
                for r_b in range(-4, 5):
                    for r_c in range(-4, 5):
                        translation = np.array([r_a, r_b, r_c])
                        displacement = (
                            positions[atom_j] + translation @ cell - positions[atom_i]
                        )
                        distance = np.linalg.norm(displacement)
                        if distance < 1.0e-9 or distance > cutoff:
                            continue
                        alternative_distances = np.linalg.norm(
                            positions[atom_j]
                            + (translation - neighboring_shifts @ _PER_PAIR_SUPERCELL)
                            @ cell
                            - positions[atom_i],
                            axis=1,
                        )
                        if not np.all(distance <= alternative_distances - 1.0e-9):
                            continue
                        spring = 4.0 * np.exp(-distance / 1.8)
                        entries.append(
                            (atom_i, atom_j, translation.copy(), -spring * np.eye(3))
                        )
                        onsite[atom_i] += spring * np.eye(3)
    for atom_i in range(2):
        entries.append(
            (
                atom_i,
                atom_i,
                np.zeros(3, dtype=int),
                onsite[atom_i] + onsite_defect * np.eye(3),
            )
        )

    with (folder / "infile.forceconstant").open("w") as stream:
        stream.write("2\n100.0\n")
        for atom_i in range(2):
            atom_entries = [entry for entry in entries if entry[0] == atom_i]
            stream.write(f"{len(atom_entries)}\n")
            for _, atom_j, translation, tensor in atom_entries:
                stream.write(f"{atom_j + 1}\n")
                stream.write(" ".join(str(value) for value in translation) + "\n")
                for row in tensor:
                    stream.write(" ".join(f"{value:.12f}" for value in row) + "\n")
    return primitive, entries


def _direct_per_pair_frequencies(primitive, entries, q_point):
    """Evaluate the literal ``sum_R Phi(R) exp(2 pi i q.R)`` oracle."""
    from ase import units

    masses = primitive.get_masses()
    n_atoms = len(primitive)
    dynamical = np.zeros((n_atoms, 3, n_atoms, 3), dtype=complex)
    for atom_i, atom_j, translation, tensor in entries:
        phase = np.exp(2j * np.pi * np.dot(q_point, translation))
        dynamical[atom_i, :, atom_j, :] += (
            phase * tensor / np.sqrt(masses[atom_i] * masses[atom_j])
        )
    eigenvalues = np.linalg.eigvalsh(
        dynamical.reshape(3 * n_atoms, 3 * n_atoms) * units.mol / (10 * units.J)
    )
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)


def test_tdep_nondiagonal_matches_literal_per_pair_phases(tmp_path):
    """Literal TDEP vectors must control dispersion off the BvK q mesh.

    Folding the vectors to one representative per periodic class is exactly
    equivalent on the commensurate grid, which allowed the historical bug to
    evade frequency tests.  At incommensurate q it gives a different spectrum;
    ``auto`` must instead reproduce an independent direct Fourier sum.
    """
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    primitive, entries = _write_per_pair_tdep_model(tmp_path)
    forceconstants = ForceConstants.from_folder(
        str(tmp_path), format="tdep", only_second=True
    )
    assert forceconstants.second.translation_support.provenance == "file"
    # The file happens to contain nine distinct vectors for a determinant-nine
    # supercell, but they are not one-per-class: several vectors share a class
    # while other classes are absent. Equal axis lengths must not be mistaken
    # for a compact periodic representation.
    class_ids = forceconstants.second.translation_support.class_ids
    assert len(np.unique(class_ids)) < len(class_ids)

    collapse_errors = []
    for q_point in (
        np.array([0.6667, 0.3333, 0.1111]),
        np.array([0.1234, 0.4321, 0.2468]),
    ):
        literal = HarmonicWithQ(
            q_point=q_point,
            second=forceconstants.second,
            ifc_interpolation="auto",
            storage="memory",
        )
        collapsed = HarmonicWithQ(
            q_point=q_point,
            second=forceconstants.second,
            ifc_interpolation="periodic",
            storage="memory",
        )
        expected = np.sort(_direct_per_pair_frequencies(primitive, entries, q_point))
        actual = np.sort(np.asarray(literal.frequency).ravel())
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-8)
        collapse_errors.append(
            np.max(np.abs(actual - np.sort(np.asarray(collapsed.frequency).ravel())))
        )
    # One point is accidentally isospectral for this central-force model; the
    # other exposes the historical class-representative error by >1 THz.
    assert max(collapse_errors) > 1.0


def test_tdep_per_pair_and_periodic_agree_on_commensurate_grid(tmp_path):
    """Class folding must remain exactly equivalent at BvK q-points."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    _write_per_pair_tdep_model(tmp_path)
    forceconstants = ForceConstants.from_folder(
        str(tmp_path), format="tdep", only_second=True
    )
    # M @ q is integer, so exp(2*pi*i*q.(R+nM)) is independent of the
    # representative chosen for each periodic translation class.
    q_point = np.linalg.solve(_PER_PAIR_SUPERCELL, np.array([1.0, 0.0, 0.0]))
    literal = HarmonicWithQ(
        q_point=q_point,
        second=forceconstants.second,
        ifc_interpolation="auto",
        storage="memory",
    )
    collapsed = HarmonicWithQ(
        q_point=q_point,
        second=forceconstants.second,
        ifc_interpolation="periodic",
        storage="memory",
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(literal.frequency).ravel()),
        np.sort(np.asarray(collapsed.frequency).ravel()),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_tdep_acoustic_sum_rule_corrects_literal_home_cell(tmp_path):
    """The acoustic correction must be attached to the literal ``R=0`` slot."""
    from kaldo.forceconstants import ForceConstants
    from kaldo.observables.harmonic_with_q import HarmonicWithQ

    primitive, _ = _write_per_pair_tdep_model(tmp_path, onsite_defect=0.05)
    clean_folder = tmp_path / "clean"
    clean_folder.mkdir()
    _, clean_entries = _write_per_pair_tdep_model(clean_folder)

    forceconstants = ForceConstants.from_folder(
        str(tmp_path),
        format="tdep",
        only_second=True,
        is_acoustic_sum=True,
    )
    np.testing.assert_array_equal(
        forceconstants.second.translation_support.translations[0],
        np.zeros(3, dtype=int),
    )
    q_point = np.array([0.1234, 0.4321, 0.2468])
    harmonic = HarmonicWithQ(
        q_point=q_point,
        second=forceconstants.second,
        ifc_interpolation="auto",
        storage="memory",
    )
    expected = np.sort(_direct_per_pair_frequencies(primitive, clean_entries, q_point))
    np.testing.assert_allclose(
        np.sort(np.asarray(harmonic.frequency).ravel()),
        expected,
        rtol=0.0,
        atol=1.0e-8,
    )
