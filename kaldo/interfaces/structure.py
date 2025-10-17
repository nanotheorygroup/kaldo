from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import ase.io
from ase import Atoms

from kaldo.helpers.logger import get_logger
import kaldo.interfaces.shengbte_io as shengbte_io

log = get_logger()


@dataclass
class ResolvedStructure:
    unit_atoms: Atoms
    supercell: tuple[int, int, int]
    replicated_atoms: Atoms | None = None
    grid_order: Literal["C", "F"] = "C"
    charges: Any | None = None
    found_files: dict[str, Path] = field(default_factory=dict)

    def with_supercell(self, new_supercell: tuple[int, int, int]) -> "ResolvedStructure":
        return ResolvedStructure(
            unit_atoms=self.unit_atoms,
            supercell=new_supercell,
            replicated_atoms=None,
            grid_order=self.grid_order,
            charges=self.charges,
            found_files=self.found_files.copy(),
        )


def _unit_from_replicated(replicated: Atoms, supercell: tuple[int, int, int]) -> Atoms:
    n_replicas = int(np.prod(supercell))
    n_total = len(replicated)
    if n_replicas == 0 or n_total % n_replicas != 0:
        raise ValueError("Unable to infer unit cell from replicated atoms.")
    n_unit = n_total // n_replicas
    unit = Atoms(
        symbols=replicated.get_chemical_symbols()[:n_unit],
        positions=replicated.positions[:n_unit],
        cell=replicated.cell / np.array(supercell),
        pbc=[1, 1, 1],
    )
    return unit


def resolve_structure(folder: str | Path,
                      *,
                      format_hint: str | None = None,
                      filenames: dict[str, str] | None = None,
                      supercell_hint: tuple[int, int, int] | None = None) -> ResolvedStructure:
    folder = Path(folder)
    filenames = filenames or {}

    # CONTROL (ShengBTE)
    control_name = filenames.get("control", "CONTROL")
    control_path = folder / control_name
    if control_path.exists():
        atoms, supercell, charges = shengbte_io.import_control_file(str(control_path))
        return ResolvedStructure(
            unit_atoms=atoms,
            supercell=tuple(supercell),
            grid_order="F",
            charges=charges,
            found_files={"control": control_path},
        )

    # POSCAR / CONTCAR
    structure_names = [filenames.get("structure"), "POSCAR", "CONTCAR"]
    for name in structure_names:
        if not name:
            continue
        path = folder / name
        if path.exists():
            atoms = ase.io.read(path, format="vasp")
            return ResolvedStructure(
                unit_atoms=atoms,
                supercell=supercell_hint or (1, 1, 1),
                found_files={"structure": path},
            )

    # CONFIG / replicated_atoms.xyz
    replica_names = [filenames.get("replicas"), "CONFIG", "replicated_atoms.xyz"]
    for name in replica_names:
        if not name:
            continue
        path = folder / name
        if not path.exists():
            continue
        fmt = "dlp4" if name.upper() == "CONFIG" else None
        replicated = ase.io.read(path, format=fmt)
        supercell = supercell_hint or (1, 1, 1)
        unit = _unit_from_replicated(replicated, supercell)
        return ResolvedStructure(
            unit_atoms=unit,
            supercell=supercell,
            replicated_atoms=replicated,
            found_files={"replicas": path},
        )

    # TDEP (infile.ucposcar / infile.ssposcar)
    uc_path = folder / filenames.get("tdep_uc", "infile.ucposcar")
    sc_path = folder / filenames.get("tdep_sc", "infile.ssposcar")
    if uc_path.exists() and sc_path.exists():
        uc = ase.io.read(uc_path, format="vasp")
        sc = ase.io.read(sc_path, format="vasp")
        supercell = supercell_hint
        if supercell is None:
            transform = sc.cell @ np.linalg.inv(uc.cell)
            supercell = tuple(np.round(np.diag(transform)).astype(int))
        return ResolvedStructure(
            unit_atoms=uc,
            supercell=supercell,
            replicated_atoms=sc,
            found_files={"structure": uc_path, "replicas": sc_path},
        )

    # HiPhive (atom_prim.xyz)
    prim_path = folder / filenames.get("hiphive_prim", "atom_prim.xyz")
    if prim_path.exists():
        atoms = ase.io.read(prim_path)
        return ResolvedStructure(
            unit_atoms=atoms,
            supercell=supercell_hint or (1, 1, 1),
            found_files={"structure": prim_path},
        )

    raise FileNotFoundError(
        "Unable to determine structure. Provide control/structure filenames or a supercell hint."
    )
