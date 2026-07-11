"""
Unit and regression test for the kaldo package.
"""

# Import package, test suite, and other packages as needed
from kaldo.forceconstants import ForceConstants
import numpy as np
import pytest


@pytest.fixture(scope="session")
def forceconstants():
    print("Preparing phonons object.")
    forceconstants = ForceConstants.from_folder(folder="kaldo/tests/si-crystal/", supercell=[3, 3, 3], format="eskm")
    return forceconstants


def test_c11(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 0, 0], 142.97, significant=3)


def test_c12(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[0, 0, 1, 1], 75.85, significant=3)


def test_c44(forceconstants):
    cijkl = forceconstants.elastic_prop()
    np.testing.assert_approx_equal(cijkl[1, 2, 1, 2], 69.06, significant=3)


def test_elastic_prop_invariant_under_grid_relabeling(tmp_path):
    """elastic_prop must not depend on the replica enumeration convention.

    A C-ordered SecondOrder and a self-consistent F-relabeled twin (same
    physics: value replica axis permuted together with the declared grid)
    must give identical elastic tensors. A random seeded value tensor on an
    asymmetric supercell makes the check maximally discriminating: any code
    path that overrides the declared enumeration instead of trusting the
    grid/value pair mispairs replica vectors against the dynamical matrix
    and fails this test.
    """
    from ase.build import bulk
    from kaldo.grid import Grid
    from kaldo.observables.secondorder import SecondOrder

    atoms = bulk("Si", "diamond", a=5.43)
    supercell = (2, 2, 3)
    n_uc, n_rep = len(atoms), int(np.prod(supercell))

    rng = np.random.default_rng(0)
    value_c = rng.standard_normal((1, n_uc, 3, n_rep, n_uc, 3)).astype(np.float64)

    # Permutation p with grid_F[k] == grid_C[p[k]] (pure enumeration order)
    grid_c = Grid(supercell, order="C").grid(is_wrapping=False)
    grid_f = Grid(supercell, order="F").grid(is_wrapping=False)
    p = np.array([np.flatnonzero((grid_c == gf).all(axis=1))[0] for gf in grid_f])
    value_f = value_c[:, :, :, p, :, :]

    second_c = SecondOrder.from_supercell(atoms, grid_type="C", supercell=supercell,
                                          value=value_c, folder=str(tmp_path / "c"))
    second_f = SecondOrder.from_supercell(atoms, grid_type="F", supercell=supercell,
                                          value=value_f, folder=str(tmp_path / "f"))

    fc_c = ForceConstants(atoms=atoms, supercell=supercell, second_order=second_c,
                          folder=str(tmp_path / "c"))
    fc_f = ForceConstants(atoms=atoms, supercell=supercell, second_order=second_f,
                          folder=str(tmp_path / "f"))

    np.testing.assert_allclose(fc_f.elastic_prop(), fc_c.elastic_prop(),
                               rtol=1e-10, atol=1e-10)
