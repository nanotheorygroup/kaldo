"""Periodic 3D exactness of colored IFC2 against independent finite-difference references."""

from functools import partial

import numpy as np
from ase.build import bulk
from ase.calculators.lj import LennardJones
from scipy.optimize._numdiff import approx_derivative  # Private, test-only independent reference.
from scipy.sparse import csr_array, kron

from kaldo.controllers.colored_fd import (merge_colored_second_order, plan_colored_second_order,
                                          run_colored_second_order)
from kaldo.controllers.displacement import calculate_second
from kaldo.tests.colored_fd_helpers import PeriodicToyEAM, rattled_simple_cubic as _rattled_simple_cubic


def _colored_ifc(atoms, calculator, radius, root):
    plan = plan_colored_second_order(atoms, radius, radius, plan_dir=root / "plan",
                                     calculator_description="periodic-reference")
    run_colored_second_order(plan.directory, calculator, workdir=root / "shard",
                             calculator_description="periodic-reference")
    return plan, merge_colored_second_order(plan.directory, [root / "shard"], output_dir=root / "merged",
                                            symmetrize=False, apply_acoustic_sum=False)


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def test_toy_eam_forces_are_energy_gradient():
    atoms = _rattled_simple_cubic(4, seed=0)
    atoms.calc = PeriodicToyEAM()
    forces = atoms.get_forces()
    step = 1e-5
    for atom, axis in [(0, 0), (17, 1), (63, 2)]:
        energies = []
        for sign in (1, -1):
            moved = atoms.copy()
            moved.calc = PeriodicToyEAM()
            moved.positions[atom, axis] += sign * step
            energies.append(moved.get_potential_energy())
        np.testing.assert_allclose(-(energies[0] - energies[1]) / (2 * step), forces[atom, axis],
                                   rtol=1e-6, atol=1e-8)


def test_periodic_many_body_matches_single_atom_columns(tmp_path):
    atoms = _rattled_simple_cubic(6, seed=1)
    plan, merged = _colored_ifc(atoms, PeriodicToyEAM, 2.15, tmp_path)
    assert plan.n_groups < len(atoms) / 3
    dense = merged.to_dense()
    reference = atoms.copy()
    reference.calc = PeriodicToyEAM()
    for atom in (0, 7, 43, 130, 215):
        for axis in range(3):
            forces = []
            for sign in (1, -1):
                reference.positions[:] = atoms.positions
                reference.positions[atom, axis] += sign * plan.delta_shift
                forces.append(reference.get_forces())
            column = -(forces[0] - forces[1]) / (2 * plan.delta_shift)
            assert _relative_error(dense[0, atom, axis, 0], column) < 1e-10


def test_periodic_pair_potential_matches_dense_calculate_second(tmp_path):
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((4, 3, 3))
    atoms.rattle(0.03, seed=3)
    calculator = partial(LennardJones, sigma=2.3, epsilon=0.1, rc=2.7, smooth=True)
    plan, merged = _colored_ifc(atoms, calculator, 2.75, tmp_path)
    assert plan.n_groups < len(atoms) / 4
    dense = calculate_second(atoms, atoms.copy(), plan.delta_shift, calculator=calculator)
    assert _relative_error(merged.to_dense(), dense) < 1e-10


def test_periodic_many_body_matches_scipy_approx_derivative(tmp_path):
    atoms = _rattled_simple_cubic(6, seed=2)
    plan, merged = _colored_ifc(atoms, PeriodicToyEAM, 2.15, tmp_path)
    assert plan.n_groups < len(atoms) / 3
    work = atoms.copy()
    work.calc = PeriodicToyEAM()

    def minus_forces(flat_positions):
        work.positions = flat_positions.reshape(-1, 3)
        return -work.get_forces().ravel()

    sparsity = csr_array(kron(plan.color.astype(np.float64), np.ones((3, 3))))
    jacobian = approx_derivative(minus_forces, atoms.positions.ravel(), method="3-point",
                                 abs_step=plan.delta_shift, sparsity=sparsity)
    assert _relative_error(merged.matrix.toarray(), jacobian.toarray()) < 1e-9
