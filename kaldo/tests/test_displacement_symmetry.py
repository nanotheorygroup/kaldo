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


def _invariance_violation_second(phi6, atoms, supercell):
    """Max |Phi[g(b)] - R Phi[b] R^T| over all supercell-compatible ops and blocks."""
    from kaldo.controllers.displacement import _get_symmetry_maps
    m = _get_symmetry_maps(atoms, supercell)
    nx, ny, nz = supercell
    n_unit = len(atoms)
    n_rep = nx * ny * nz
    phi = np.asarray(phi6).reshape(n_unit, 3, n_rep, n_unit, 3)
    worst = 0.0
    for k in range(len(m['rotations'])):
        R = m['rotations_cart'][k]
        for i in range(n_unit):
            ip = m['atom_map'][k, i]
            for j in range(n_unit):
                jp = m['atom_map'][k, j]
                lv = (m['grid'] @ m['rotations'][k].T
                      + (m['cell_shifts'][k, j] - m['cell_shifts'][k, i])) % m['supercell_shape']
                lnew = lv[:, 0] * ny * nz + lv[:, 1] * nz + lv[:, 2]
                for li in range(n_rep):
                    dev = np.abs(phi[ip, :, lnew[li], jp, :] - R @ phi[i, :, li, j, :] @ R.T).max()
                    worst = max(worst, dev)
    return worst


def test_symmetrize_second_projects_noise_to_invariant_subspace():
    from kaldo.controllers.displacement import symmetrize_ifc_second
    atoms, rep = _cu_atoms((2, 2, 2))
    baseline = calculate_second(atoms, rep, second_order_delta=1e-5, n_workers=1, calculator=EMT())
    rng = np.random.default_rng(0)
    noisy = np.asarray(baseline) + rng.normal(scale=1e-2, size=np.shape(baseline))
    assert _invariance_violation_second(noisy, atoms, (2, 2, 2)) > 1e-3
    projected = symmetrize_ifc_second(noisy, atoms, (2, 2, 2))
    assert _invariance_violation_second(projected, atoms, (2, 2, 2)) < 1e-12


def test_symmetrize_second_is_a_fixed_point_on_symmetric_input():
    from kaldo.controllers.displacement import symmetrize_ifc_second
    atoms, rep = _cu_atoms((2, 2, 2))
    baseline = np.asarray(calculate_second(atoms, rep, second_order_delta=1e-5, n_workers=1, calculator=EMT()))
    projected = symmetrize_ifc_second(baseline, atoms, (2, 2, 2))
    # EMT is an analytic, symmetry-respecting potential: projection only removes FD noise.
    np.testing.assert_allclose(projected, baseline, rtol=1e-6, atol=1e-8)
    # Idempotence: projecting twice changes nothing at machine precision.
    np.testing.assert_allclose(symmetrize_ifc_second(projected, atoms, (2, 2, 2)), projected,
                               rtol=0, atol=1e-13)


def test_symmetrize_third_projects_noise_and_preserves_symmetric_input():
    from sparse import COO
    from kaldo.controllers.displacement import symmetrize_ifc_third
    atoms, rep = _cu_atoms((2, 2, 2))
    baseline = calculate_third(atoms, rep, third_order_delta=1e-5, n_workers=1, calculator=EMT())
    dense = baseline.todense()
    # Symmetric input is a (near-)fixed point: only FD noise is removed.
    # atol matches the FD-noise floor documented for the use_symmetry
    # baseline comparison above (~1e-4 absolute at delta=1e-5).
    projected = symmetrize_ifc_third(baseline, atoms, (2, 2, 2)).todense()
    np.testing.assert_allclose(projected, dense, rtol=1e-5, atol=1e-3)
    # Noisy input: projection is idempotent (P P x == P x) at machine precision.
    rng = np.random.default_rng(1)
    mask = dense != 0
    noisy_dense = dense + rng.normal(scale=1e-3, size=dense.shape) * mask
    noisy = COO.from_numpy(noisy_dense)
    p1 = symmetrize_ifc_third(noisy, atoms, (2, 2, 2))
    p2 = symmetrize_ifc_third(p1, atoms, (2, 2, 2))
    np.testing.assert_allclose(p2.todense(), p1.todense(), rtol=0, atol=1e-12)
    # And it actually moved the noisy input.
    assert np.abs(p1.todense() - noisy_dense).max() > 1e-5


def test_calculate_second_symmetrize_flag_and_method(tmp_path):
    from kaldo.forceconstants import ForceConstants
    from kaldo.controllers.displacement import symmetrize_ifc_second
    atoms, _ = _cu_atoms((2, 2, 2))
    fc = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=str(tmp_path / 'a'))
    fc.second.calculate(EMT(), delta_shift=1e-3, is_storing=False)  # symmetrize defaults to True
    assert _invariance_violation_second(np.asarray(fc.second.value), atoms, (2, 2, 2)) < 1e-12
    # Opt-out reproduces raw finite differences; the public method projects them.
    fc2 = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=str(tmp_path / 'b'))
    fc2.second.calculate(EMT(), delta_shift=1e-3, is_storing=False, symmetrize=False)
    raw = np.asarray(fc2.second.value).copy()
    fc2.second.symmetrize()
    np.testing.assert_allclose(np.asarray(fc2.second.value),
                               symmetrize_ifc_second(raw, atoms, (2, 2, 2)),
                               rtol=0, atol=1e-14)


def test_calculate_third_symmetrize_flag_and_method(tmp_path):
    from kaldo.forceconstants import ForceConstants
    from kaldo.controllers.displacement import symmetrize_ifc_third
    atoms, _ = _cu_atoms((2, 2, 2))
    fc = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=str(tmp_path / 'a'))
    fc.third.calculate(EMT(), delta_shift=1e-3, is_storing=False, symmetrize=False)
    raw = fc.third.value
    sym_ref = symmetrize_ifc_third(raw, atoms, (2, 2, 2)).todense()
    fc.third.symmetrize()
    np.testing.assert_allclose(fc.third.value.todense(), sym_ref, rtol=0, atol=1e-14)
    # Default path applies the same projection during calculate.
    fc2 = ForceConstants(atoms=atoms, supercell=(2, 2, 2), folder=str(tmp_path / 'b'))
    fc2.third.calculate(EMT(), delta_shift=1e-3, is_storing=False)
    np.testing.assert_allclose(fc2.third.value.todense(), sym_ref, rtol=0, atol=1e-12)
