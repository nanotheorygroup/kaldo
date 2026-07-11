"""
Supercell Taylor-series V_n contractors (n = 2, 3, 4).

Two entry points:

* ``SCContractors(h5_path)`` -- loads pre-remapped supercell IFCs from a
  Julia-produced HDF5 file (legacy path, kept for cross-validation).
* ``SCContractors.from_tdep_folder(folder)`` -- builds the same flat
  quartet/triplet/pair tables directly from a TDEP run, no Julia step.

Both produce arrays in kaldo's row-major convention:

    a{n}_{2,3,4} : (n_quartets,) supercell-atom indices
    phi{2,3,4}   : (n_quartets, 3, 3, ...)  cartesian rank-n block

The contractions are then independent of how the tables were built:

    V_2 = (1/2)  sum_p  phi2[p]_{ab}  u[a1_2[p], a]  u[a2_2[p], b]
    V_3 = (1/6)  sum_p  phi3[p]_{abc} u[a1_3[p], a]  u[a2_3[p], b]  u[a3_3[p], c]
    V_4 = (1/24) sum_p  phi4[p]_{abcd} ...

with u in Angstroms and phi^{(n)} in eV/A^n; returns V in eV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


class SCContractors:
    """Fast (numpy-einsum) supercell V_2/V_3/V_4 evaluators."""

    def __init__(self, h5_path: Path | None = None, *, _from_arrays=None):
        if _from_arrays is not None:
            self.__dict__.update(_from_arrays)
            return
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

    @classmethod
    def from_tdep_folder(cls, folder, *, include_fourth: bool = True) -> "SCContractors":
        """Build flat per-quartet/triplet/pair tables from a TDEP folder.

        Unlike the HDF5 path, this stays inside Python: TDEP IFC files are
        read into the per-primitive-atom representation by ``read_tdep_*``,
        then expanded to the supercell flat form via the SNF replica
        mapping. No tensor is ever materialized; only the nonzero entries.
        """
        from .tdep_reader import (
            read_tdep_pair_fcs, read_tdep_ifc3, read_tdep_ifc4,
        )
        from kaldo.interfaces.tdep_io import build_supercell_replica_mapping
        import ase.io

        folder = Path(folder)
        uc = ase.io.read(str(folder / "infile.ucposcar"), format="vasp")
        sc = ase.io.read(str(folder / "infile.ssposcar"), format="vasp")
        n_uc = len(uc)
        n_sc = len(sc)
        uc_pos = np.asarray(uc.positions)
        uc_cell = np.asarray(uc.cell)

        mapping = build_supercell_replica_mapping(uc, sc)
        sc_atom_index = _build_inverse_sc_index(mapping)

        arrs = {"n_atoms_sc": int(n_sc)}

        neighbors = read_tdep_pair_fcs(
            str(folder / "infile.forceconstant"), uc_pos, uc_cell,
        )
        a1_2, a2_2, phi2 = _expand_pairs(
            neighbors, uc_pos, uc_cell, mapping, sc_atom_index,
        )
        arrs.update(a1_2=a1_2, a2_2=a2_2, phi2=phi2)

        triplets = read_tdep_ifc3(
            str(folder / "infile.forceconstant_thirdorder"), n_uc,
        )
        a1_3, a2_3, a3_3, phi3 = _expand_triplets(
            triplets, mapping, sc_atom_index,
        )
        arrs.update(a1_3=a1_3, a2_3=a2_3, a3_3=a3_3, phi3=phi3)

        if include_fourth:
            quartets = read_tdep_ifc4(
                str(folder / "infile.forceconstant_fourthorder"), n_uc,
            )
            a1_4, a2_4, a3_4, a4_4, phi4 = _expand_quartets(
                quartets, mapping, sc_atom_index,
            )
            arrs.update(a1_4=a1_4, a2_4=a2_4, a3_4=a3_4, a4_4=a4_4, phi4=phi4)

        return cls(_from_arrays=arrs)

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


# ---------------------------------------------------------------------------
# TDEP -> flat supercell quartet/triplet/pair expansion
# ---------------------------------------------------------------------------
#
# read_tdep_ifc{2,3,4} return one list per primitive atom:
#
#     pairs[i_uc]    = [(j_uc, lv2_frac, phi(3,3))]
#     triplets[i_uc] = [(j_uc, k_uc, lv2_frac, lv3_frac, phi(3,3,3))]
#     quartets[i_uc] = [(j_uc, k_uc, l_uc, lv2_frac, lv3_frac, lv4_frac, phi(3,3,3,3))]
#
# Here lvX_frac is the integer lattice vector (in primitive basis) between
# the cell of atom 1 and the cell of atom X. Under translational symmetry,
# each per-primitive-atom entry implies one supercell quartet per replica
# R_i of atom i_uc: atom 1 sits at (i_uc, R_i), atom j at (j_uc, R_i + lv2),
# atom k at (k_uc, R_i + lv3), atom l at (l_uc, R_i + lv4), with all
# lattice vectors wrapped through the SNF supercell PBC.


def _build_inverse_sc_index(mapping) -> np.ndarray:
    """``(atom_uc, replica_id) -> i_sc`` lookup as a (n_uc, n_rep) int table."""
    n_uc = int(np.max(mapping["atom_of_sc"])) + 1
    n_rep = mapping["replica_table"].shape[0]
    inv = np.full((n_uc, n_rep), -1, dtype=np.int64)
    inv[mapping["atom_of_sc"], mapping["replica_id_of_sc"]] = np.arange(
        len(mapping["atom_of_sc"]), dtype=np.int64,
    )
    return inv


def _replica_lookup_table(mapping) -> dict:
    """Dict from ``(rx, ry, rz)`` tuple to replica index for the wrapped form."""
    table = mapping["replica_table"]
    return {(int(r[0]), int(r[1]), int(r[2])): i for i, r in enumerate(table)}


def _wrap_sums_to_replica_ids(R_sums: np.ndarray, mapping, lut, *, _cache=None) -> np.ndarray:
    """Vectorized wrap of (..., 3) integer lattice vectors to replica ids.

    Uses ``kaldo.grid.wrap_lattice_vector_to_replica`` per row, with a
    Python-level memoization on the unique input rows. The kaldo helper
    handles both the sc-fractional ``[0,1)`` form and norm-minimal form
    of the replica table; we just spare the cost of calling it for
    duplicate inputs (every R_i + R_partner sum repeats heavily).
    """
    from kaldo.grid import wrap_lattice_vector_to_replica
    table = mapping["replica_table"]; M = mapping["M"]
    flat = R_sums.reshape(-1, 3).astype(int)
    # Unique rows then map back.
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    cache = _cache if _cache is not None else {}
    out_unique = np.empty(uniq.shape[0], dtype=np.int64)
    for i, r in enumerate(uniq):
        key = (int(r[0]), int(r[1]), int(r[2]))
        # First try direct table hit (fast path).
        idx = lut.get(key)
        if idx is None:
            idx = cache.get(key)
            if idx is None:
                idx = int(wrap_lattice_vector_to_replica(r, table, M))
                cache[key] = idx
            if idx < 0:
                raise RuntimeError(f"Lattice vector {r} did not map to a replica")
        out_unique[i] = idx
    return out_unique[inv].reshape(R_sums.shape[:-1])


def _sc_replicas_of(i_uc, mapping):
    """Indices of all sc atoms whose primitive id is ``i_uc``, plus their R_i table."""
    mask = mapping["atom_of_sc"] == i_uc
    sc_idx = np.where(mask)[0]
    R_i = mapping["replica_vector_of_sc"][sc_idx]      # (n_rep, 3)
    return sc_idx, R_i


def _expand_pairs(neighbors, uc_pos, uc_cell, mapping, sc_atom_index):
    """Expand TDEP pair list to flat (a1, a2, phi) supercell tables."""
    lut = _replica_lookup_table(mapping)
    inv_uc_cell = np.linalg.inv(uc_cell)
    a1_chunks = []
    a2_chunks = []
    phi_chunks = []
    for i_uc, il in enumerate(neighbors):
        sc_idx_i, R_i_tab = _sc_replicas_of(i_uc, mapping)
        n_rep = sc_idx_i.shape[0]
        if not il:
            continue
        # Stack the per-i_uc partner descriptors.
        partner_uc = []
        R_j_list = []
        phi_list = []
        for (j_uc, r_j_cart, _lv, phi) in il:
            tau_j = uc_pos[j_uc]
            R_j = np.rint((r_j_cart - tau_j) @ inv_uc_cell).astype(int)
            partner_uc.append(int(j_uc))
            R_j_list.append(R_j)
            phi_list.append(np.asarray(phi, dtype=np.float64))
        partner_uc = np.asarray(partner_uc, dtype=np.int64)        # (n_pair,)
        R_j_arr = np.asarray(R_j_list, dtype=int)                   # (n_pair, 3)
        phi_arr = np.asarray(phi_list, dtype=np.float64)            # (n_pair, 3, 3)
        n_pair = partner_uc.shape[0]

        # Broadcast: (n_rep, n_pair, 3)
        R_sum = R_i_tab[:, None, :] + R_j_arr[None, :, :]
        rid_j = _wrap_sums_to_replica_ids(R_sum, mapping, lut)      # (n_rep, n_pair)
        j_sc = sc_atom_index[partner_uc[None, :].repeat(n_rep, axis=0), rid_j]

        a1 = sc_idx_i[:, None].repeat(n_pair, axis=1)               # (n_rep, n_pair)
        a1_chunks.append(a1.ravel())
        a2_chunks.append(j_sc.ravel())
        phi_chunks.append(np.tile(phi_arr, (n_rep, 1, 1, 1)).reshape(-1, 3, 3))

    a1_arr = np.concatenate(a1_chunks).astype(np.int64) if a1_chunks else np.zeros(0, np.int64)
    a2_arr = np.concatenate(a2_chunks).astype(np.int64) if a2_chunks else np.zeros(0, np.int64)
    phi = np.concatenate(phi_chunks, axis=0) if phi_chunks else np.zeros((0, 3, 3))
    return a1_arr, a2_arr, phi


def _expand_triplets(triplets, mapping, sc_atom_index):
    lut = _replica_lookup_table(mapping)
    a1_c = []
    a2_c = []
    a3_c = []
    phi_c = []
    for i_uc, ts in enumerate(triplets):
        sc_idx_i, R_i_tab = _sc_replicas_of(i_uc, mapping)
        n_rep = sc_idx_i.shape[0]
        if not ts:
            continue
        j_uc = np.asarray([t[0] for t in ts], dtype=np.int64)
        k_uc = np.asarray([t[1] for t in ts], dtype=np.int64)
        R_j = np.rint(np.asarray([t[2] for t in ts])).astype(int)   # (n_t, 3)
        R_k = np.rint(np.asarray([t[3] for t in ts])).astype(int)
        phi_arr = np.asarray([t[4] for t in ts], dtype=np.float64)  # (n_t, 3, 3, 3)
        n_t = j_uc.shape[0]

        R_sum_j = R_i_tab[:, None, :] + R_j[None, :, :]
        R_sum_k = R_i_tab[:, None, :] + R_k[None, :, :]
        rid_j = _wrap_sums_to_replica_ids(R_sum_j, mapping, lut)
        rid_k = _wrap_sums_to_replica_ids(R_sum_k, mapping, lut)
        j_sc = sc_atom_index[j_uc[None, :].repeat(n_rep, 0), rid_j]
        k_sc = sc_atom_index[k_uc[None, :].repeat(n_rep, 0), rid_k]

        a1 = sc_idx_i[:, None].repeat(n_t, axis=1)
        a1_c.append(a1.ravel())
        a2_c.append(j_sc.ravel())
        a3_c.append(k_sc.ravel())
        phi_c.append(np.tile(phi_arr, (n_rep, 1, 1, 1, 1)).reshape(-1, 3, 3, 3))

    if not a1_c:
        return (np.zeros(0, np.int64),) * 3 + (np.zeros((0, 3, 3, 3)),)
    return (
        np.concatenate(a1_c).astype(np.int64),
        np.concatenate(a2_c).astype(np.int64),
        np.concatenate(a3_c).astype(np.int64),
        np.concatenate(phi_c, axis=0),
    )


def _expand_quartets(quartets, mapping, sc_atom_index):
    lut = _replica_lookup_table(mapping)
    a1_c = []
    a2_c = []
    a3_c = []
    a4_c = []
    phi_c = []
    for i_uc, qs in enumerate(quartets):
        sc_idx_i, R_i_tab = _sc_replicas_of(i_uc, mapping)
        n_rep = sc_idx_i.shape[0]
        if not qs:
            continue
        j_uc = np.asarray([q[0] for q in qs], dtype=np.int64)
        k_uc = np.asarray([q[1] for q in qs], dtype=np.int64)
        l_uc = np.asarray([q[2] for q in qs], dtype=np.int64)
        R_j = np.rint(np.asarray([q[3] for q in qs])).astype(int)
        R_k = np.rint(np.asarray([q[4] for q in qs])).astype(int)
        R_l = np.rint(np.asarray([q[5] for q in qs])).astype(int)
        phi_arr = np.asarray([q[6] for q in qs], dtype=np.float64)  # (n_q, 3, 3, 3, 3)
        n_q = j_uc.shape[0]

        R_sum_j = R_i_tab[:, None, :] + R_j[None, :, :]
        R_sum_k = R_i_tab[:, None, :] + R_k[None, :, :]
        R_sum_l = R_i_tab[:, None, :] + R_l[None, :, :]
        rid_j = _wrap_sums_to_replica_ids(R_sum_j, mapping, lut)
        rid_k = _wrap_sums_to_replica_ids(R_sum_k, mapping, lut)
        rid_l = _wrap_sums_to_replica_ids(R_sum_l, mapping, lut)
        j_sc = sc_atom_index[j_uc[None, :].repeat(n_rep, 0), rid_j]
        k_sc = sc_atom_index[k_uc[None, :].repeat(n_rep, 0), rid_k]
        l_sc = sc_atom_index[l_uc[None, :].repeat(n_rep, 0), rid_l]

        a1 = sc_idx_i[:, None].repeat(n_q, axis=1)
        a1_c.append(a1.ravel())
        a2_c.append(j_sc.ravel())
        a3_c.append(k_sc.ravel())
        a4_c.append(l_sc.ravel())
        phi_c.append(np.tile(phi_arr, (n_rep, 1, 1, 1, 1, 1)).reshape(-1, 3, 3, 3, 3))

    if not a1_c:
        return (np.zeros(0, np.int64),) * 4 + (np.zeros((0, 3, 3, 3, 3)),)
    return (
        np.concatenate(a1_c).astype(np.int64),
        np.concatenate(a2_c).astype(np.int64),
        np.concatenate(a3_c).astype(np.int64),
        np.concatenate(a4_c).astype(np.int64),
        np.concatenate(phi_c, axis=0),
    )
