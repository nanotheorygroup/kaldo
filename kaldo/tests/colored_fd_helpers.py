"""Finite-range reference calculators and structures for colored IFC tests."""

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class ToyFiniteRange(Calculator):
    """Chain spring or compact-support EAM-like force model, in eV/Angstrom."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, cutoff=1.3, many_body=False):
        super().__init__()
        self.cutoff = cutoff
        self.many_body = many_body
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        positions = self.atoms.positions
        n_atoms = len(positions)
        pairs = []
        density = np.zeros(n_atoms)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                vector = positions[j] - positions[i]
                distance = np.linalg.norm(vector)
                if distance < self.cutoff:
                    value = (1 - distance / self.cutoff)**4
                    derivative = -4 * (1 - distance / self.cutoff)**3 / self.cutoff
                    density[i] += value
                    density[j] += value
                    pairs.append((i, j, vector / distance, distance, derivative))
        forces = np.zeros((n_atoms, 3))
        if self.many_body:
            energy = float(np.dot(density, density) / 2)
            for i, j, unit, _, derivative in pairs:
                pair_force = (density[i] + density[j]) * derivative * unit
                forces[i] += pair_force
                forces[j] -= pair_force
        else:
            energy = 0.0
            for i, j, unit, distance, _ in pairs:
                energy += 0.5 * (distance - 1.0)**2
                pair_force = (distance - 1.0) * unit
                forces[i] += pair_force
                forces[j] -= pair_force
        self.results = {"energy": energy, "forces": forces}


class PeriodicToyEAM(Calculator):
    """Compact-support embedding energy with forces dependent to twice the pair cutoff."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, cutoff=1.05):
        super().__init__()
        self.cutoff = cutoff

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        n_atoms = len(self.atoms)
        period = np.diag(np.asarray(self.atoms.cell))
        assert np.all(self.cutoff < period / 2)
        pair_vectors = self.atoms.positions[np.newaxis] - self.atoms.positions[:, np.newaxis]
        pair_vectors -= period * np.round(pair_vectors / period)
        pair_distances = np.linalg.norm(pair_vectors, axis=2)
        first, second = np.nonzero((pair_distances < self.cutoff) & ~np.eye(n_atoms, dtype=np.bool_))
        distance = pair_distances[first, second]
        vector = pair_vectors[first, second]
        reduced = 1 - distance / self.cutoff
        density = np.bincount(first, weights=reduced**4, minlength=n_atoms)
        slope = -4 * reduced**3 / self.cutoff
        pull = (density[first] * slope / distance)[:, np.newaxis] * vector
        gradient = np.zeros((n_atoms, 3))
        np.add.at(gradient, second, pull)
        np.add.at(gradient, first, -pull)
        self.results = {"energy": 0.5 * float(density @ density), "forces": -gradient}


def chain(n=9):
    return Atoms("Si" * n, positions=np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))),
                 cell=[n + 4, 6, 6], pbc=False)


def rattled_simple_cubic(n_side, seed):
    grid = np.indices((n_side,) * 3).reshape(3, -1).T.astype(np.float64)
    atoms = Atoms("Si" * len(grid), positions=grid, cell=np.eye(3) * n_side, pbc=True)
    atoms.rattle(0.03, seed=seed)
    return atoms
