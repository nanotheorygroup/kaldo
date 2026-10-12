"""SparseSecondOrder derives displacements, dynamical blocks, and storage from its structure."""

from ase import Atoms
import numpy as np
import pytest
from scipy.sparse import bsr_array

from kaldo.helpers.units import EV_TO_10J_PER_MOL
from kaldo.observables.sparse_secondorder import SparseSecondOrder


def _pair(positions, cell, pbc=True):
    """Two atoms coupled by a symmetric, sum-rule-satisfying spring."""
    spring = np.diag([1.0, 2.0, 3.0])
    blocks = np.array([spring, -spring, -spring, spring])
    matrix = bsr_array((blocks, np.array([0, 1, 0, 1]), np.array([0, 2, 4])), shape=(6, 6))
    atoms = Atoms("SiGe", positions=positions, cell=cell, pbc=pbc)
    return SparseSecondOrder(matrix, atoms=atoms)


def test_minimum_image_displacement_crosses_the_cell_boundary():
    ifc = _pair([[0.5, 0, 0], [9.5, 0, 0]], np.eye(3) * 10)
    displacement = ifc.block_displacement()
    # Atom 0 sits 1 Angstrom to the +x side of atom 1's periodic image.
    np.testing.assert_allclose(displacement, [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 0]], atol=1e-12)
    harmonic = ifc.to_sparse_harmonic()
    masses = ifc.atoms.get_masses()
    np.testing.assert_allclose(harmonic.blocks[1], -np.diag([1., 2., 3.]) * EV_TO_10J_PER_MOL /
                               np.sqrt(masses[0] * masses[1]))
    assert harmonic.validation_report()["skew_derivative_residual"] == 0


def test_ambiguous_periodic_image_is_rejected():
    with pytest.raises(ValueError, match="image is ambiguous"):
        _pair([[0, 0, 0], [5, 0, 0]], np.eye(3) * 10).block_displacement()


def test_nonperiodic_displacement_is_the_position_difference():
    ifc = _pair([[0, 0, 0], [7, 0, 0]], np.eye(3) * 10, pbc=False)
    np.testing.assert_allclose(ifc.block_displacement()[1], [-7, 0, 0])


def test_save_and_load_keep_structure_and_explicit_displacement(tmp_path):
    ifc = _pair([[0.5, 0, 0], [9.5, 0, 0]], np.eye(3) * 10)
    explicit = np.array([[0., 0, 0], [-9., 0, 0], [9., 0, 0], [0., 0, 0]])
    ifc = SparseSecondOrder(ifc.matrix, atoms=ifc.atoms, displacement=explicit)
    ifc.save(tmp_path)
    loaded = SparseSecondOrder.load(tmp_path)
    np.testing.assert_array_equal(loaded.atoms.get_masses(), ifc.atoms.get_masses())
    np.testing.assert_array_equal(loaded.block_displacement(), explicit)
    np.testing.assert_allclose(loaded.dynmat.toarray(), ifc.mass_weighted_bsr().toarray(), rtol=0, atol=0)


def test_accessors_without_structure_explain_what_is_missing():
    ifc = SparseSecondOrder(_pair([[0, 0, 0], [1, 0, 0]], np.eye(3) * 10).matrix)
    with pytest.raises(ValueError, match="needs the structure"):
        ifc.dynmat
