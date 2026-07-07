"""
Fast repeated single-point energies from one persistent LAMMPS instance.

The cumulant 0th-order correction evaluates the potential energy of ~10^5
displaced supercell configurations that share topology (same atoms, same
box, only positions move). Driving that through ``ase`` ``LAMMPSlib``'s
``get_potential_energy`` per configuration is slow: every call re-enters the
ASE calculator machinery and issues a bare ``run 0``, which triggers a full
LAMMPS setup (neighbor-list rebuild) each time.

``BatchEnergyEvaluator`` keeps a single LAMMPS instance alive and, for each
configuration, only

  * ``scatter_atoms('x', ...)`` the new coordinates, and
  * ``run 0 pre no post no`` -- reuse the existing neighbor lists and skip
    pre/post setup,

then reads the potential energy back. Topology never changes, so the neighbor
lists stay valid across configurations (LAMMPS' neighbor skin absorbs the
sub-Angstrom thermal displacements sampled here; a periodic full rebuild is
available via ``rebuild_every`` as a safety valve for large excursions).

Setup (box, atom types, pair style, first neighbor build) is delegated to
``LAMMPSlib`` so this module inherits ASE's cell-prism handling and unit
conversions; only the hot loop is specialized.

NOTE: exercising the fast path requires a working LAMMPS Python module. The
frame contract (positions in ssposcar order, energy in eV) is unit-tested
with a surrogate; the ``scatter_atoms``/``run 0 pre no post no`` path itself
must be smoke-tested against a real LAMMPS build.
"""
from __future__ import annotations

import numpy as np


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
        Force a full neighbor rebuild (``run 0 pre yes``) every this many
        configurations. ``0`` (default) never forces a rebuild after the
        initial setup. Use a positive value if sampled displacements are
        large relative to the neighbor skin.
    """

    def __init__(self, atoms, equilibrium_positions, *, rebuild_every: int = 0):
        self.atoms = atoms
        self.eq = np.ascontiguousarray(equilibrium_positions, dtype=float)
        self.n_at = len(atoms)
        self.rebuild_every = int(rebuild_every)
        self._i = 0
        self._started = False
        self._lmp = None      # live LAMMPS handle, if the fast path is available
        self._prism = None
        self.uses_fast_path = None  # resolved on the first energy() call

        if self.eq.shape != (self.n_at, 3):
            raise ValueError(
                f"equilibrium_positions shape {self.eq.shape} != ({self.n_at}, 3)"
            )

    def _ensure_started(self, u0):
        """Run one full ASE evaluation to build the box/types/neighbor lists.

        If the calculator exposes a live LAMMPS handle (``calc.lmp``), cache it
        and use the scatter + ``run 0 pre no post no`` fast path for subsequent
        configurations. Otherwise fall back to plain ASE ``get_potential_energy``
        (correct, just not accelerated) so any ASE calculator still works.
        """
        self.atoms.set_positions(self.eq + u0)
        e0 = float(self.atoms.get_potential_energy())
        self._lmp = getattr(self.atoms.calc, "lmp", None)
        self._prism = getattr(self.atoms.calc, "prism", None)
        self.uses_fast_path = self._lmp is not None
        self._started = True
        return e0

    def _ase_energy(self, positions_ase):
        self.atoms.set_positions(positions_ase)
        return float(self.atoms.get_potential_energy())

    def _fast_energy(self, positions_ase):
        """scatter positions + run 0 (no setup) + read potential energy."""
        lmp = self._lmp
        pos = positions_ase
        if self._prism is not None:
            pos = self._prism.vector_to_lammps(pos)
        flat = np.ascontiguousarray(pos, dtype=float).ravel()

        import ctypes
        c_pos = (ctypes.c_double * flat.size)(*flat.tolist())
        lmp.scatter_atoms("x", 1, 3, c_pos)

        force_rebuild = (
            self.rebuild_every > 0 and (self._i % self.rebuild_every == 0)
        )
        # pre no  -> skip setup (neighbor rebuild); pre yes on the safety tick.
        lmp.command("run 0 pre %s post no" % ("yes" if force_rebuild else "no"))
        # 'pe' is the potential-energy thermo keyword LAMMPS always defines.
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
