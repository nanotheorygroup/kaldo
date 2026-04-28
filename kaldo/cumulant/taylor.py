"""
Supercell Taylor-series V_n contractors (n = 2, 3, 4).

Consumes pre-remapped supercell IFCs from Julia (``out/ifcs_sc_remapped.h5``),
which sidesteps subtle axis-permutation conventions between TDEP and our
own supercell enumeration. The HDF5 axis order reversal below compensates
for Julia's column-major storage.

Formulas:

    V_2 = (1/2)  sum  phi^{(2)}_{ab}(a1, a2) u_{a1, a} u_{a2, b}
    V_3 = (1/6)  sum  phi^{(3)}_{abc}(a1, a2, a3) u_{a1, a} u_{a2, b} u_{a3, c}
    V_4 = (1/24) sum  phi^{(4)}_{abcd}(a1, a2, a3, a4) u_1 u_2 u_3 u_4

with u in Angstroms and phi^{(n)} in eV/A^n; returns V in eV.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class SCContractors:
    """Fast (numpy-einsum) supercell V_2/V_3/V_4 evaluators."""

    def __init__(self, h5_path: Path):
        with h5py.File(h5_path, "r") as f:
            self.n_atoms_sc = int(f["n_atoms_sc"][()])

            # IFC2
            self.a1_2 = f["ifc2/a1"][:].astype(np.int64)
            self.a2_2 = f["ifc2/a2"][:].astype(np.int64)
            phi2 = f["ifc2/phi_eV_per_A2"][:]
            if phi2.shape[-2:] != (3, 3):
                phi2 = np.moveaxis(phi2, -1, 0)
            self.phi2 = phi2.astype(np.float64)

            # IFC3 - Julia col-major -> reverse last 3 axes
            self.a1_3 = f["ifc3/a1"][:].astype(np.int64)
            self.a2_3 = f["ifc3/a2"][:].astype(np.int64)
            self.a3_3 = f["ifc3/a3"][:].astype(np.int64)
            phi3 = f["ifc3/phi_eV_per_A3"][:]
            if phi3.shape[-3:] != (3, 3, 3):
                phi3 = np.moveaxis(phi3, -1, 0)
            self.phi3 = np.transpose(phi3, (0, 3, 2, 1)).astype(np.float64)

            # IFC4 - Julia col-major -> reverse last 4 axes
            self.a1_4 = f["ifc4/a1"][:].astype(np.int64)
            self.a2_4 = f["ifc4/a2"][:].astype(np.int64)
            self.a3_4 = f["ifc4/a3"][:].astype(np.int64)
            self.a4_4 = f["ifc4/a4"][:].astype(np.int64)
            phi4 = f["ifc4/phi_eV_per_A4"][:]
            if phi4.shape[-4:] != (3, 3, 3, 3):
                phi4 = np.moveaxis(phi4, -1, 0)
            self.phi4 = np.transpose(phi4, (0, 4, 3, 2, 1)).astype(np.float64)

    def V2(self, u_flat):
        """V_2 in eV for ``u_flat`` of shape (n_sc, 3) in Angstrom."""
        u1 = u_flat[self.a1_2]
        u2 = u_flat[self.a2_2]
        return 0.5 * np.einsum("pa,pab,pb->", u1, self.phi2, u2)

    def V3(self, u_flat):
        """V_3 in eV."""
        u1 = u_flat[self.a1_3]
        u2 = u_flat[self.a2_3]
        u3 = u_flat[self.a3_3]
        return np.einsum("pa,pb,pc,pabc->", u1, u2, u3, self.phi3) / 6.0

    def V4(self, u_flat):
        """V_4 in eV."""
        u1 = u_flat[self.a1_4]
        u2 = u_flat[self.a2_4]
        u3 = u_flat[self.a3_4]
        u4 = u_flat[self.a4_4]
        return np.einsum("pa,pb,pc,pd,pabcd->", u1, u2, u3, u4, self.phi4) / 24.0
