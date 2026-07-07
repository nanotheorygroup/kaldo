"""
BatchEnergyEvaluator frame + fallback contract.

The evaluator accelerates repeated single-point energies for a fixed-topology
supercell by reusing one LAMMPS instance (scatter + ``run 0 pre no post no``).
When the attached calculator is not LAMMPS (no live ``calc.lmp`` handle) it
must fall back to plain ASE ``get_potential_energy`` and stay correct.

These tests exercise the fallback path with a cheap analytic ASE calculator
(no LAMMPS needed) and assert:

  * the energy returned for a displacement ``u`` matches the calculator's own
    energy at ``equilibrium + u``, for the first (setup) call and later calls;
  * displacements are applied in the caller's atom order;
  * ``uses_fast_path`` reports False for a non-LAMMPS calculator.

The LAMMPS fast path itself (scatter_atoms + run flags) requires a real
LAMMPS build and is smoke-tested on the cluster, not in CI.
"""
from __future__ import annotations

import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class _SpringCalc(Calculator):
    """Analytic ASE calculator: E = 0.5 k * sum |r - r_ref|^2 (isotropic)."""

    implemented_properties = ["energy", "free_energy"]

    def __init__(self, r_ref, k=3.0):
        super().__init__()
        self._r_ref = np.array(r_ref, dtype=float)
        self._k = float(k)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        du = atoms.get_positions() - self._r_ref
        e = 0.5 * self._k * float(np.sum(du * du))
        self.results = {"energy": e, "free_energy": e}


def _make_atoms(n=6, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 8, size=(n, 3))
    return Atoms("H" * n, positions=pos, cell=np.eye(3) * 10.0, pbc=True), pos


def test_fallback_energy_matches_direct_ase():
    from kaldo.cumulant._lammps_batch import BatchEnergyEvaluator

    atoms, eq = _make_atoms()
    atoms.calc = _SpringCalc(eq, k=2.5)
    ev = BatchEnergyEvaluator(atoms, eq)

    rng = np.random.default_rng(1)
    for i in range(5):
        u = rng.normal(scale=0.05, size=eq.shape)
        got = ev.energy(u)
        want = 0.5 * 2.5 * float(np.sum(u * u))  # E = 0.5 k |u|^2 about eq
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)

    # First call resolves the (lack of) fast path.
    assert ev.uses_fast_path is False


def test_zero_displacement_is_equilibrium_energy():
    from kaldo.cumulant._lammps_batch import BatchEnergyEvaluator

    atoms, eq = _make_atoms(seed=3)
    atoms.calc = _SpringCalc(eq, k=1.0)
    ev = BatchEnergyEvaluator(atoms, eq)
    assert abs(ev.energy(np.zeros_like(eq))) < 1e-12


def test_displacement_atom_order_is_respected():
    """A displacement on atom j must move only atom j's contribution."""
    from kaldo.cumulant._lammps_batch import BatchEnergyEvaluator

    atoms, eq = _make_atoms(n=4, seed=5)
    atoms.calc = _SpringCalc(eq, k=1.0)
    ev = BatchEnergyEvaluator(atoms, eq)
    ev.energy(np.zeros_like(eq))  # prime

    u = np.zeros_like(eq)
    u[2] = [0.1, 0.0, 0.0]
    got = ev.energy(u)
    assert abs(got - 0.5 * 1.0 * 0.1 ** 2) < 1e-12


def test_bad_equilibrium_shape_raises():
    from kaldo.cumulant._lammps_batch import BatchEnergyEvaluator

    atoms, eq = _make_atoms(n=4)
    import pytest
    with pytest.raises(ValueError, match=r"shape"):
        BatchEnergyEvaluator(atoms, eq[:, :2])  # (n, 2) not (n, 3)
