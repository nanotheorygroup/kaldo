"""
Fast repeated single-point energies from one persistent LAMMPS instance.

The cumulant 0th-order correction evaluates the potential energy of ~10^5
displaced supercell configurations that share topology (same atoms, same
box, only positions move). Driving that through ``ase`` ``LAMMPSlib``'s
``get_potential_energy`` per configuration is slow: every call re-enters the
ASE calculator machinery.

``BatchEnergyEvaluator`` mirrors LatticeDynamicsToolkit's
``LAMMPSCalculator`` / ``single_point_potential_energy``:

  * After one ASE setup call (builds the box and an initial neighbor list),
    configure ``fix nve`` + zero velocities and enlarge the neighbor skin
    to the nearest-neighbor distance so thermal displacements never invalidate
    the list.
  * Hot loop: ``scatter_atoms('x')``, zero forces/velocities, then
    ``run 1 pre no post yes``.  ``pre no`` skips the neighbor rebuild;
    ``run 1`` (not ``run 0``) is what actually recomputes forces/energy.
    Zeroing ``f`` and ``v`` keeps the Verlet step from moving atoms.

``run 0 pre no`` is wrong here: it skips force evaluation and leaves thermo
``pe`` stale.  ``run 0 pre yes`` is correct but rebuilds neighbors every
config and is slower.

NOTE: exercising the fast path requires a working LAMMPS Python module. The
frame contract (positions in ssposcar order, energy in eV) is unit-tested
with a surrogate; the ``scatter_atoms`` path is smoke-tested when LAMMPS is
importable.
"""
from __future__ import annotations

import numpy as np


def _nearest_neighbor_distance(positions, cell, pbc=True):
    """Minimum image nearest-neighbor distance (Angstrom)."""
    from ase.geometry import find_mic

    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    dmin = np.inf
    # O(N^2) is fine for typical SC sizes (~10^2-10^3 atoms) and runs once.
    for i in range(n):
        delta = positions[i + 1:] - positions[i]
        if len(delta) == 0:
            continue
        mic, _ = find_mic(delta, cell, pbc=pbc)
        norms = np.linalg.norm(mic, axis=1)
        if norms.size:
            dmin = min(dmin, float(norms.min()))
    if not np.isfinite(dmin) or dmin <= 0.0:
        # Fallback: half the shortest cell edge.
        lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
        dmin = 0.5 * float(lengths.min())
    return dmin


class BatchEnergyEvaluator:
    """Repeated single-point potential energies for a fixed-topology cell.

    Parameters
    ----------
    atoms : ase.Atoms
        Supercell at equilibrium, already carrying a configured ``LAMMPSlib``
        calculator (``atoms.calc``). The calculator's live LAMMPS handle is
        reused for the fast path.
    equilibrium_positions : (n_at, 3) float ndarray
        Reference Cartesian positions (Angstrom) in the atom order the caller
        will supply displacements in. Displacements ``u`` are added to these.
    rebuild_every : int, optional
        Unused (kept for API compatibility). Neighbor lists are built once
        with a large skin; the hot loop uses ``run 1 pre no`` like LDT.
    """

    def __init__(self, atoms, equilibrium_positions, *, rebuild_every: int = 0):
        self.atoms = atoms
        self.eq = np.ascontiguousarray(equilibrium_positions, dtype=float)
        self.n_at = len(atoms)
        self.rebuild_every = int(rebuild_every)  # retained; unused (see docstring)
        self._i = 0
        self._started = False
        self._lmp = None      # live LAMMPS handle, if the fast path is available
        self._prism = None
        self._f_zeros = None
        self.uses_fast_path = None  # resolved on the first energy() call
        self.neighbor_skin = None

        if self.eq.shape != (self.n_at, 3):
            raise ValueError(
                f"equilibrium_positions shape {self.eq.shape} != ({self.n_at}, 3)"
            )

    def _ensure_started(self, u0):
        """Run one full ASE evaluation, then configure the Julia-style fast path.

        If the calculator exposes a live LAMMPS handle (``calc.lmp``), cache it
        and set up ``nve`` + large neighbor skin for ``run 1 pre no`` single-
        points. Otherwise fall back to plain ASE ``get_potential_energy``.
        """
        self.atoms.set_positions(self.eq + u0)
        e0 = float(self.atoms.get_potential_energy())
        self._lmp = getattr(self.atoms.calc, "lmp", None)
        self._prism = getattr(self.atoms.calc, "prism", None)
        self.uses_fast_path = self._lmp is not None
        self._started = True
        if self._lmp is not None:
            self._configure_single_point_mode()
        return e0

    def _configure_single_point_mode(self):
        """Match LDT ``LAMMPSCalculator(..., single_point=True)`` setup."""
        lmp = self._lmp
        # ASE may have left integrator/fix state; replace with a hold nve.
        try:
            lmp.command("unfix all")
        except Exception:
            pass
        lmp.command("fix hold all nve")
        lmp.command("velocity all set 0.0 0.0 0.0")

        # Skin = nn distance so harmonic thermal moves never invalidate the list.
        cell = np.asarray(self.atoms.get_cell(), dtype=float)
        pbc = self.atoms.pbc if hasattr(self.atoms, "pbc") else True
        skin = _nearest_neighbor_distance(self.eq, cell, pbc=pbc)
        self.neighbor_skin = skin
        lmp.command(f"neighbor {skin:.10g} bin")
        # Do not auto-rebuild; the large skin keeps the initial list valid.
        lmp.command("neigh_modify delay 0 every 1 check no")
        lmp.command("run 0 post no")  # rebuild once with the new skin

        self._f_zeros = np.zeros((self.n_at, 3), dtype=float)

    def _ase_energy(self, positions_ase):
        self.atoms.set_positions(positions_ase)
        return float(self.atoms.get_potential_energy())

    def _scatter(self, name, arr):
        """Scatter a (n_at, 3) array into LAMMPS (``x`` goes through the prism)."""
        pos = arr
        if name == "x" and self._prism is not None:
            pos = self._prism.vector_to_lammps(arr)
        flat = np.ascontiguousarray(pos, dtype=np.float64).ravel()
        import ctypes
        # from_buffer avoids the ~100x slower flat.tolist() copy into a ctor.
        c_buf = (ctypes.c_double * flat.size).from_buffer(flat)
        self._lmp.scatter_atoms(name, 1, 3, c_buf)

    def _fast_energy(self, positions_ase):
        """Julia ``single_point_potential_energy``: scatter + ``run 1 pre no``.

        ``run 0 pre no`` skips force evaluation (stale ``pe``).
        ``run 0 pre yes`` recomputes but rebuilds neighbor lists every config.
        ``run 1 pre no`` recomputes forces/energy without rebuilding neighbors;
        zeroing ``f``/``v`` keeps the Verlet step from moving atoms.
        """
        lmp = self._lmp
        self._scatter("x", positions_ase)
        self._scatter("f", self._f_zeros)
        lmp.command("velocity all set 0.0 0.0 0.0")
        lmp.command("run 1 pre no post yes")
        return float(lmp.get_thermo("pe"))

    def energy(self, u):
        """Potential energy (eV) of the equilibrium cell displaced by ``u``.

        ``u`` has shape (n_at, 3) in the same atom order as
        ``equilibrium_positions``.
        """
        u = np.asarray(u, dtype=float)
        if not self._started:
            e = self._ensure_started(u)
            self._i = 1
            return e
        pos = self.eq + u
        e = self._fast_energy(pos) if self._lmp is not None else self._ase_energy(pos)
        self._i += 1
        return e
