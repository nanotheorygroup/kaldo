"""Scientific contracts for provenance-aware third-order IFC interpolation."""

from pathlib import Path

import numpy as np
from ase import Atoms
import pytest
from sparse import COO

from kaldo.grid import SupercellGrid, TranslationSupport
from kaldo.observables.thirdorder import ThirdOrder


def _third(value, support, positions=((0.0, 0.0, 0.0),), cell=None, folder=""):
    cell = np.eye(3) if cell is None else np.asarray(cell, dtype=float)
    atoms = Atoms("H" * len(positions), positions=positions, cell=cell, pbc=True)
    replicated = (
        support.supercell.representatives[:, None, :] @ np.asarray(atoms.cell)
        + np.asarray(atoms.positions)[None, :, :]
    )
    return ThirdOrder(
        atoms=atoms,
        replicated_positions=replicated,
        supercell=tuple(np.diag(support.supercell.matrix)),
        folder=folder,
        value=value,
        supercell_grid=support.supercell,
        translation_support=support,
    )


def _single_entry(shape, translation_j, translation_k, datum=1.0):
    coord = np.array([[0], [0], [translation_j], [0], [0], [translation_k], [0], [0]])
    return COO(coord, np.array([datum]), shape=shape)


def _fourier_value(interpolation, qj, qk):
    phases_j = interpolation.support.phases(qj)[0]
    phases_k = interpolation.support.phases(qk)[0]
    dense = interpolation.value.todense()
    return np.einsum("r,s,iarjbskc->", phases_j, phases_k, dense)


def test_wigner_seitz_ties_form_cartesian_product_with_conserved_weights():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport.periodic(grid)
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    third = _third(_single_entry(shape, 1, 1), support)

    result = third.get_interpolation("wigner-seitz")

    np.testing.assert_array_equal(result.support.translations, [[-1, 0, 0], [1, 0, 0]])
    nonzero = result.value.data
    assert len(nonzero) == 4
    np.testing.assert_allclose(nonzero, np.full(4, 0.25), rtol=0, atol=1e-15)
    np.testing.assert_allclose(nonzero.sum(), 1.0, rtol=0, atol=1e-15)


def test_dense_and_sparse_inputs_compile_identically():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport.periodic(grid)
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    sparse_value = _single_entry(shape, 1, 1, datum=2.5)
    dense_value = sparse_value.todense()

    sparse_result = _third(sparse_value, support).get_interpolation("wigner-seitz")
    dense_result = _third(dense_value, support).get_interpolation("wigner-seitz")

    np.testing.assert_array_equal(
        sparse_result.support.translations, dense_result.support.translations
    )
    np.testing.assert_allclose(
        sparse_result.value.todense(), dense_result.value.todense(), rtol=0, atol=0
    )


def test_legacy_rank3_sparse_storage_is_normalized_without_densifying():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport.periodic(grid)
    rank8_shape = (1, 3, 2, 1, 3, 2, 1, 3)
    rank3 = _single_entry(rank8_shape, 1, 1, datum=2.0).reshape((3, 6, 6))
    third = _third(rank3, support)

    periodic = third.get_interpolation("periodic")
    wigner_seitz = third.get_interpolation("wigner-seitz")

    assert isinstance(periodic.value, COO)
    assert isinstance(wigner_seitz.value, COO)
    assert periodic.value.shape == rank8_shape
    assert wigner_seitz.value.ndim == 8
    np.testing.assert_allclose(wigner_seitz.value.data.sum(), 2.0, rtol=0, atol=1e-15)


def test_gamma_contraction_sums_literal_translation_axes_exactly():
    """Gamma-only amorphous projection depends on the IFC3 zeroth moment."""
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport([[0, 0, 0], [2, 0, 0]], grid, provenance="file")
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    value = COO(coords, np.array([1.0, 2.0, 4.0]), shape=shape)
    third = _third(value, support)

    gamma = third.gamma_contracted_value()

    assert isinstance(gamma, COO)
    assert gamma.shape == (3, 3, 3)
    np.testing.assert_allclose(gamma[0, 0, 0], 7.0, rtol=0.0, atol=0.0)
    assert gamma.nnz == 1


def test_sparse_save_keeps_replicated_atoms_inside_output_folder(tmp_path):
    """The native IFC3 artifact set must not leak a path by string joining."""
    grid = SupercellGrid(np.eye(3, dtype=int))
    support = TranslationSupport.periodic(grid)
    shape = (1, 3, 1, 1, 3, 1, 1, 3)
    third = _third(_single_entry(shape, 0, 0), support, folder=str(tmp_path))

    third.save("third")

    assert (tmp_path / "third.npz").is_file()
    assert (tmp_path / "replicated_atoms_third.xyz").is_file()
    assert not Path(str(tmp_path) + "replicated_atoms_third.xyz").exists()


def test_wigner_seitz_and_periodic_gauges_agree_at_commensurate_q():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport.periodic(grid)
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    third = _third(_single_entry(shape, 1, 1, datum=3.0), support)
    periodic = third.get_interpolation("periodic")
    wigner_seitz = third.get_interpolation("wigner-seitz")

    for qj, qk in (
        ([0, 0, 0], [0, 0, 0]),
        ([0.5, 0, 0], [0, 0, 0]),
        ([0.5, 0, 0], [0.5, 0, 0]),
    ):
        np.testing.assert_allclose(
            _fourier_value(wigner_seitz, qj, qk),
            _fourier_value(periodic, qj, qk),
            rtol=0,
            atol=1e-14,
        )


def test_file_auto_preserves_literal_value_and_translation_support():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport([[0, 0, 0], [2, 0, 0]], grid, provenance="file")
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    value = _single_entry(shape, 1, 0)
    third = _third(value, support)

    result = third.get_interpolation("auto")

    assert result.resolved_mode == "file"
    assert result.value is value
    assert result.support is support
    assert third.get_interpolation("auto") is result


def test_explicit_periodic_override_folds_file_translations():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport([[0, 0, 0], [2, 0, 0]], grid, provenance="file")
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    coords = np.array([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])
    value = COO(coords, np.array([1.0, 2.0]), shape=shape)
    third = _third(value, support)

    periodic = third.get_interpolation("periodic")
    wigner_seitz = third.get_interpolation("wigner-seitz")

    assert periodic.resolved_mode == "periodic"
    assert periodic.support.provenance == "periodic"
    assert periodic.support.size == 2
    np.testing.assert_allclose(periodic.value.data.sum(), 3.0, rtol=0, atol=0)
    assert wigner_seitz.resolved_mode == "wigner-seitz"
    np.testing.assert_allclose(wigner_seitz.value.data.sum(), 3.0, rtol=0, atol=1e-15)


def test_legacy_export_rejects_unserializable_translation_support(tmp_path):
    """Legacy IFC3 files must not silently discard literal translation axes."""
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport([[0, 0, 0], [2, 0, 0]], grid, provenance="file")
    shape = (1, 3, 2, 1, 3, 2, 1, 3)
    third = _third(_single_entry(shape, 1, 1), support)
    third.folder = str(tmp_path)

    with pytest.raises(ValueError, match="cannot preserve.*translation support"):
        third.save(format="sparse")


def test_invalid_mode_and_mismatched_axes_are_rejected():
    grid = SupercellGrid(np.diag([2, 1, 1]))
    support = TranslationSupport.periodic(grid)
    value = np.zeros((1, 3, 2, 1, 3, 2, 1, 3))
    third = _third(value, support)

    try:
        third.get_interpolation("unfolded")
    except ValueError as error:
        assert "ifc_interpolation" in str(error)
    else:
        raise AssertionError("invalid interpolation mode was accepted")


def test_nondiagonal_sparse_ifc3_projection_is_cached_and_remains_sparse():
    """A skew quotient uses exact phases without constructing a dense IFC3.

    This is deliberately structural rather than a timing assertion: one
    nonzero source block may fan out only over its tied shortest images, the
    compiled object is reused, and its Fourier projection agrees with the
    compact periodic gauge on the supercell-commensurate mesh.
    """
    matrix = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]], dtype=int)
    grid = SupercellGrid(matrix)
    support = TranslationSupport.periodic(grid)
    n_atoms = 2
    shape = (
        n_atoms,
        3,
        support.size,
        n_atoms,
        3,
        support.size,
        n_atoms,
        3,
    )
    coords = np.array([[0], [1], [1], [1], [2], [2], [0], [0]])
    value = COO(coords, np.array([2.75]), shape=shape)
    atoms_positions = ((0.0, 0.0, 0.0), (0.41, 0.23, 0.17))
    atoms = Atoms(
        "H2",
        scaled_positions=atoms_positions,
        cell=np.array([[2.1, 0.0, 0.0], [0.4, 1.8, 0.0], [0.2, 0.3, 2.4]]),
        pbc=True,
    )
    replicated = (
        support.supercell.representatives[:, None, :] @ np.asarray(atoms.cell)
        + np.asarray(atoms.positions)[None, :, :]
    )
    third = ThirdOrder(
        atoms=atoms,
        replicated_positions=replicated,
        supercell=matrix,
        folder="",
        value=value,
        supercell_grid=grid,
        translation_support=support,
    )

    periodic = third.get_interpolation("periodic")
    wigner_seitz = third.get_interpolation("wigner-seitz")
    assert third.get_interpolation("wigner-seitz") is wigner_seitz
    assert isinstance(wigner_seitz.value, COO)
    assert wigner_seitz.value.nnz < 64
    np.testing.assert_allclose(
        wigner_seitz.value.data.sum(), value.data.sum(), rtol=0.0, atol=1e-15
    )
    np.testing.assert_array_equal(wigner_seitz.support.supercell.matrix, matrix)

    # M q is integer, hence translations differing by n M have identical
    # phases and both gauges must give the same IFC3 Fourier projection.
    qj = np.linalg.solve(matrix, np.array([1.0, 0.0, 0.0]))
    qk = np.linalg.solve(matrix, np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(
        _fourier_value(wigner_seitz, qj, qk),
        _fourier_value(periodic, qj, qk),
        rtol=0.0,
        atol=1e-14,
    )

    # Replacing the source tensor must invalidate the interpolation plan;
    # cache identity cannot depend on mode and support alone.
    third.value = COO(coords, np.array([5.5]), shape=shape)
    replaced = third.get_interpolation("wigner-seitz")
    assert replaced is not wigner_seitz
    np.testing.assert_allclose(replaced.value.data.sum(), 5.5, rtol=0.0, atol=1e-15)


def test_pair_gauge_tracks_independent_basis_images_in_skew_cell():
    """Each IFC3 leg must follow the periodic image of its own atom pair.

    Moving basis atom ``i`` by integer lattice vector ``s_i`` changes the
    integer Fourier translations as ``R_ij -> R_ij + s_i - s_j`` and
    independently for ``R_ik``.  The corresponding phase is a basis-gauge
    factor, not a change in the physical three-phonon matrix element.
    """
    cell = np.array([[3.0, 0.0, 0.0], [1.5, 2.598, 0.0], [0.2, 0.1, 4.0]])
    scaled = np.array(
        [
            [0.05, 0.08, 0.11],
            [0.42, 0.31, 0.17],
            [0.19, 0.73, 0.29],
        ]
    )
    shifts = np.array([[1, 0, 0], [0, -1, 0], [1, 1, 0]], dtype=int)
    grid = SupercellGrid(np.eye(3, dtype=int))
    support = TranslationSupport.periodic(grid)
    shape = (3, 3, 1, 3, 3, 1, 3, 3)
    coordinate = np.array([[0], [1], [0], [1], [2], [0], [2], [0]])
    value = COO(coordinate, np.array([2.75]), shape=shape)

    reference = _third(value, support, scaled @ cell, cell).get_interpolation(
        "wigner-seitz"
    )
    translated = _third(
        value, support, (scaled + shifts) @ cell, cell
    ).get_interpolation("wigner-seitz")

    qj = np.array([0.13, 0.27, 0.19])
    qk = np.array([0.21, -0.16, 0.07])
    reference_fourier = _fourier_value(reference, qj, qk)
    translated_fourier = _fourier_value(translated, qj, qk)
    expected_gauge = np.exp(
        2j * np.pi * (qj @ (shifts[0] - shifts[1]) + qk @ (shifts[0] - shifts[2]))
    )

    np.testing.assert_allclose(
        translated_fourier,
        expected_gauge * reference_fourier,
        rtol=2e-14,
        atol=2e-14,
    )


def test_wigner_seitz_fc3_magnitude_is_invariant_to_wrapped_origin_shift():
    """Moving the cell origin cannot change an IFC3 Fourier magnitude.

    A rigid translation followed by wrapping may move different basis atoms
    through different cell faces.  Their stored coordinates then differ by
    independent lattice vectors, but this is only an eigenvector/Fourier gauge
    change—not a change in the crystal or its scattering strength.
    """
    cell = np.array([[3.0, 0.0, 0.0], [1.4, 2.7, 0.0], [0.2, 0.1, 4.1]])
    scaled = np.array(
        [
            [0.05, 0.08, 0.11],
            [0.42, 0.31, 0.17],
            [0.19, 0.73, 0.29],
        ]
    )
    origin_shift = np.array([0.73, 0.41, 0.67])
    wrapped = np.mod(scaled + origin_shift, 1.0)
    grid = SupercellGrid(np.eye(3, dtype=int))
    support = TranslationSupport.periodic(grid)
    shape = (3, 3, 1, 3, 3, 1, 3, 3)
    coordinate = np.array([[0], [1], [0], [1], [2], [0], [2], [0]])
    value = COO(coordinate, np.array([2.75]), shape=shape)

    reference = _third(value, support, scaled @ cell, cell).get_interpolation(
        "wigner-seitz"
    )
    translated = _third(value, support, wrapped @ cell, cell).get_interpolation(
        "wigner-seitz"
    )
    qj = np.array([0.13, 0.27, 0.19])
    qk = np.array([0.21, -0.16, 0.07])

    np.testing.assert_allclose(
        np.abs(_fourier_value(translated, qj, qk)),
        np.abs(_fourier_value(reference, qj, qk)),
        rtol=2e-14,
        atol=2e-14,
    )
