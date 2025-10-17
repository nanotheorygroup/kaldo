from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import ase.io
from ase import Atoms

from kaldo.helpers.logger import get_logger

log = get_logger()


@dataclass
class ForceConstantData:
    """Container returned by format-specific loaders."""

    order: Literal[2, 3]
    value: Any
    unit_atoms: Atoms
    supercell: tuple[int, int, int]
    replicated_atoms: Atoms | None = None
    grid_order: Literal["C", "F"] = "C"
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_replicas(resolved, folder: Path, candidates: Sequence[str]) -> Atoms:
    """
    Return replicated atoms for loaders, reusing the resolved structure or
    reading one of the candidate files. When nothing is available we tile the
    unit cell according to the supercell.
    """
    if resolved.replicated_atoms is not None:
        return resolved.replicated_atoms

    for name in candidates:
        path = folder / name
        if not path.exists():
            continue
        fmt = None
        lower = name.lower()
        if lower == "config":
            fmt = "dlp4"
        elif lower in {"poscar", "contcar"} or lower.endswith("ssposcar") or lower.endswith("ucposcar"):
            fmt = "vasp"
        elif lower.endswith(".xyz"):
            fmt = "extxyz"
        return ase.io.read(path, format=fmt)

    log.warning("Replicated atoms missing; tiling unit cell as fallback.")
    unit = resolved.unit_atoms
    sx, sy, sz = resolved.supercell
    return unit * (sx, sy, sz)


def attach_charges(atoms: Atoms, charges: np.ndarray | None) -> None:
    if charges is None:
        return
    atoms.info["dielectric"] = charges[0]
    atoms.set_array("charges", charges[1:])
