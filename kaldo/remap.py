"""
Generic supercell IFC remapping utilities.

These helpers remap primitive IFC containers on ``ForceConstants`` into
supercell-flat pair/triplet/quartet tables. Returned data is ephemeral and
not cached on the ``ForceConstants`` instance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ReplicaTable:
    """Replica mapping information for a primitive->supercell tiling."""
    def __init__(self, forceconstants):
        from kaldo.interfaces.tdep_io import build_supercell_replica_mapping

        uc = forceconstants.atoms
        sc = forceconstants.second.replicated_atoms
        mapping = build_supercell_replica_mapping(uc, sc)
        table = np.rint(np.asarray(mapping["replica_table"], dtype=float)).astype(int)
        M = np.rint(np.asarray(mapping["M"], dtype=float)).astype(int)

        self.table = table
        self.supercell_matrix = M


class IFC2Remap:
    """Supercell-flat second-order IFC entries.

    Attributes
    ----------
    a1, a2 : np.ndarray[int64]
        Supercell atom indices for each pair term.
    phi : np.ndarray[float64]
        3x3 IFC blocks in eV/Angstrom^2 for each (a1, a2) term.
    """

    def __init__(self, a1, a2, phi):
        self.a1 = np.asarray(a1, dtype=np.int64)
        self.a2 = np.asarray(a2, dtype=np.int64)
        self.phi = np.asarray(phi, dtype=np.float64)


class IFC3Remap:
    """Supercell-flat third-order IFC entries."""

    def __init__(self, a1, a2, a3, phi):
        self.a1 = np.asarray(a1, dtype=np.int64)
        self.a2 = np.asarray(a2, dtype=np.int64)
        self.a3 = np.asarray(a3, dtype=np.int64)
        self.phi = np.asarray(phi, dtype=np.float64)


class IFC4Remap:
    """Supercell-flat fourth-order IFC entries."""

    def __init__(self, a1, a2, a3, a4, phi):
        self.a1 = np.asarray(a1, dtype=np.int64)
        self.a2 = np.asarray(a2, dtype=np.int64)
        self.a3 = np.asarray(a3, dtype=np.int64)
        self.a4 = np.asarray(a4, dtype=np.int64)
        self.phi = np.asarray(phi, dtype=np.float64)


@dataclass
class SupercellIFCRemap:
    """Supercell-flat IFC tables plus references to primitive IFC objects."""

    n_atoms_sc: int
    replica_table: ReplicaTable
    second_ifc: object
    third_ifc: object | None
    fourth_ifc: object | None
    ifc2: IFC2Remap
    ifc3: IFC3Remap | None
    ifc4: IFC4Remap | None


def remap_to_supercell_ifcs(forceconstants, *, require_third=False, require_fourth=False):
    """Remap IFC2/3/4 into supercell-flat tables.

    Returns
    -------
    SupercellIFCRemap
    """
    from kaldo.grid import wrap_lattice_vector_to_replica

    second = forceconstants.second
    third = forceconstants.third
    fourth = forceconstants.fourth
    if require_third and third is None:
        raise ValueError("third-order IFCs are required for requested remap")
    if require_fourth and fourth is None:
        raise ValueError("fourth-order IFCs are required for requested remap")

    n_uc = forceconstants.n_atoms
    n_rep = forceconstants.n_replicas
    n_sc = n_uc * n_rep
    rt = ReplicaTable(forceconstants)
    table = rt.table
    M = rt.supercell_matrix

    if len(table) != n_rep:
        raise ValueError("replica table size mismatch while building supercell remap")

    def sc_idx(replica_id, atom_id):
        return int(replica_id) * n_uc + int(atom_id)

    # IFC2
    second_dense = np.asarray(second.value)[0]  # (n_uc,3,n_rep,n_uc,3)
    a1_2, a2_2, phi2 = [], [], []
    for r0_id, R0 in enumerate(table):
        for a1 in range(n_uc):
            i_sc = sc_idx(r0_id, a1)
            for r2 in range(n_rep):
                R2 = np.rint(table[r2]).astype(int)
                r2_id = wrap_lattice_vector_to_replica(R0 + R2, table, M)
                if r2_id < 0:
                    raise ValueError(f"failed to wrap IFC2 replica vector {R0 + R2}")
                for a2 in range(n_uc):
                    block = second_dense[a1, :, r2, a2, :]
                    if not np.any(block):
                        continue
                    a1_2.append(i_sc)
                    a2_2.append(sc_idx(r2_id, a2))
                    phi2.append(block)
    out2 = IFC2Remap(a1_2, a2_2, phi2)

    out3 = None
    if third is not None:
        third_dense = np.asarray(third.value.todense())
        a1_3, a2_3, a3_3, phi3 = [], [], [], []
        for r0_id, R0 in enumerate(table):
            for a1 in range(n_uc):
                i_sc = sc_idx(r0_id, a1)
                for r2 in range(n_rep):
                    R2 = np.rint(table[r2]).astype(int)
                    r2_id = wrap_lattice_vector_to_replica(R0 + R2, table, M)
                    if r2_id < 0:
                        raise ValueError("failed to wrap IFC3 replica vector R2")
                    for a2 in range(n_uc):
                        for r3 in range(n_rep):
                            R3 = np.rint(table[r3]).astype(int)
                            r3_id = wrap_lattice_vector_to_replica(R0 + R3, table, M)
                            if r3_id < 0:
                                raise ValueError("failed to wrap IFC3 replica vector R3")
                            for a3 in range(n_uc):
                                block = third_dense[a1, :, r2, a2, :, r3, a3, :]
                                if not np.any(block):
                                    continue
                                a1_3.append(i_sc)
                                a2_3.append(sc_idx(r2_id, a2))
                                a3_3.append(sc_idx(r3_id, a3))
                                phi3.append(block)
        out3 = IFC3Remap(a1_3, a2_3, a3_3, phi3)

    out4 = None
    if fourth is not None:
        fourth_dense = np.asarray(fourth.value.todense())
        a1_4, a2_4, a3_4, a4_4, phi4 = [], [], [], [], []
        for r0_id, R0 in enumerate(table):
            for a1 in range(n_uc):
                i_sc = sc_idx(r0_id, a1)
                for r2 in range(n_rep):
                    R2 = np.rint(table[r2]).astype(int)
                    r2_id = wrap_lattice_vector_to_replica(R0 + R2, table, M)
                    if r2_id < 0:
                        raise ValueError("failed to wrap IFC4 replica vector R2")
                    for a2 in range(n_uc):
                        for r3 in range(n_rep):
                            R3 = np.rint(table[r3]).astype(int)
                            r3_id = wrap_lattice_vector_to_replica(R0 + R3, table, M)
                            if r3_id < 0:
                                raise ValueError("failed to wrap IFC4 replica vector R3")
                            for a3 in range(n_uc):
                                for r4 in range(n_rep):
                                    R4 = np.rint(table[r4]).astype(int)
                                    r4_id = wrap_lattice_vector_to_replica(R0 + R4, table, M)
                                    if r4_id < 0:
                                        raise ValueError("failed to wrap IFC4 replica vector R4")
                                    for a4 in range(n_uc):
                                        block = fourth_dense[a1, :, r2, a2, :, r3, a3, :, r4, a4, :]
                                        if not np.any(block):
                                            continue
                                        a1_4.append(i_sc)
                                        a2_4.append(sc_idx(r2_id, a2))
                                        a3_4.append(sc_idx(r3_id, a3))
                                        a4_4.append(sc_idx(r4_id, a4))
                                        phi4.append(block)
        out4 = IFC4Remap(a1_4, a2_4, a3_4, a4_4, phi4)

    return SupercellIFCRemap(
        n_atoms_sc=int(n_sc),
        replica_table=rt,
        second_ifc=second,
        third_ifc=third,
        fourth_ifc=fourth,
        ifc2=out2,
        ifc3=out3,
        ifc4=out4,
    )
