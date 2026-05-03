"""
Tests for the realspace symmetry reduction in calculate_second/calculate_third.

For correct implementation, the use_symmetry=True path must produce force
constants numerically equal to the use_symmetry=False baseline (modulo
floating-point noise) for any supercell shape the symmetry path supports.

System: Cu FCC conventional cell (4 atoms) with ASE's built-in EMT calculator.
Cu is chosen because EMT supports it, the conventional cell is high-symmetry
(Oh point group), and the small atom count keeps the test fast.
"""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from kaldo.controllers.displacement import calculate_second, calculate_third


def _cu_atoms(supercell):
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    replicated_atoms = atoms.repeat(supercell)
    replicated_atoms.calc = EMT()
    return atoms, replicated_atoms


# Cubic and slab supercells; the slab case (2,1,1) is the regression case for
# the supercell-incompatible-op bug — most spacegroup ops swap axes and must
# be filtered out, leaving only the in-plane subgroup.
SUPERCELLS = [(2, 2, 2), (2, 1, 1), (3, 1, 1)]


@pytest.mark.parametrize("supercell", SUPERCELLS)
def test_calculate_second_use_symmetry_matches_baseline(supercell):
    atoms, rep = _cu_atoms(supercell)

    baseline = calculate_second(
        atoms, rep, second_order_delta=1e-5, n_workers=1, calculator=EMT(),
    )
    sym = calculate_second(
        atoms, rep, second_order_delta=1e-5, n_workers=1, calculator=EMT(),
        use_symmetry=True,
    )

    np.testing.assert_allclose(
        np.asarray(sym), np.asarray(baseline),
        rtol=1e-7, atol=1e-9,
        err_msg=f"Second-order use_symmetry result mismatch on supercell={supercell}",
    )


@pytest.mark.parametrize("supercell", SUPERCELLS)
def test_calculate_third_use_symmetry_matches_baseline(supercell):
    atoms, rep = _cu_atoms(supercell)

    baseline = calculate_third(
        atoms, rep, third_order_delta=1e-5, n_workers=1, calculator=EMT(),
    )
    sym = calculate_third(
        atoms, rep, third_order_delta=1e-5, n_workers=1, calculator=EMT(),
        use_symmetry=True,
    )

    base_dense = baseline.todense() if hasattr(baseline, 'todense') else np.asarray(baseline)
    sym_dense  = sym.todense()      if hasattr(sym, 'todense')      else np.asarray(sym)

    # FD-noise floor for third-order with delta=1e-5 is ~1e-4 absolute on
    # IFC values of order 1-10. The symmetry path computes the canonical
    # block via FD on different atom displacements than the baseline does
    # for the rotated-equivalent block, so the two computations don't
    # agree to machine precision even when the rotation algebra is exact.
    # atol=1e-3 covers this; rtol=1e-6 keeps physics-relevant entries tight.
    np.testing.assert_allclose(
        sym_dense, base_dense,
        rtol=1e-6, atol=1e-3,
        err_msg=f"Third-order use_symmetry result mismatch on supercell={supercell}",
    )


def test_use_symmetry_rejects_non_diagonal_supercell():
    """A non-diagonal (sheared) supercell must raise NotImplementedError
    rather than silently producing wrong results."""
    atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)
    # Build a non-diagonal supercell by manually constructing the cell.
    rep = atoms.repeat((2, 2, 2))
    sheared_cell = rep.cell.array.copy()
    sheared_cell[0, 1] = 0.5  # introduce a tiny shear
    rep.set_cell(sheared_cell, scale_atoms=False)
    rep.calc = EMT()

    with pytest.raises(NotImplementedError, match="diagonal|integer"):
        calculate_second(
            atoms, rep, second_order_delta=1e-5, n_workers=1, calculator=EMT(),
            use_symmetry=True,
        )


def test_use_symmetry_rejects_scratch_dir(tmp_path):
    """use_symmetry=True must not silently combine with scratch_dir."""
    atoms, rep = _cu_atoms((2, 2, 2))
    with pytest.raises(ValueError, match="scratch_dir"):
        calculate_second(
            atoms, rep, second_order_delta=1e-5, n_workers=1, calculator=EMT(),
            use_symmetry=True, scratch_dir=str(tmp_path),
        )


def test_use_symmetry_with_parallel_via_observable(tmp_path):
    """SecondOrder.calculate / ThirdOrder.calculate with use_symmetry=True
    AND n_workers>1 must NOT auto-fill scratch_dir from self.folder.
    Otherwise calculate_second/third would reject the combination and
    the user's parallel symmetry call would always fail.
    Regression test for CodeRabbit finding on PR #253."""
    import shutil
    from kaldo.forceconstants import ForceConstants

    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)

    folder = str(tmp_path / "fc_sym_parallel")
    shutil.rmtree(folder, ignore_errors=True)

    fc = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=folder)
    # SecondOrder.calculate with use_symmetry=True + n_workers=2 must succeed
    fc.second.calculate(calculator=EMT, n_workers=2, use_symmetry=True)
    # ThirdOrder.calculate with use_symmetry=True + n_workers=2 must succeed
    fc.third.calculate(calculator=EMT, n_workers=2, use_symmetry=True)
