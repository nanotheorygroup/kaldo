"""
HiPhive force-constant loaders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from hiphive import ForceConstants

from kaldo.interfaces.common import ForceConstantData, ensure_replicas
from kaldo.helpers.logger import get_logger

log = get_logger()


def _read_forceconstants(folder: Path, filename: str) -> ForceConstants:
    path = folder / filename
    if not path.exists():
        raise FileNotFoundError(f"HiPhive file '{path}' not found.")
    return ForceConstants.read(str(path))


def _reshape_second(fcs: ForceConstants, n_replicas: int, n_prim: int) -> np.ndarray:
    array = fcs.get_fc_array(2).transpose(0, 2, 1, 3)
    array = array.reshape((n_replicas, n_prim, 3, n_replicas, n_prim, 3))
    return array[0, np.newaxis, ...]


def _reshape_third(fcs: ForceConstants, n_prim: int, supercell: Tuple[int, int, int]) -> np.ndarray:
    array = fcs.get_fc_array(3).transpose(0, 3, 1, 4, 2, 5)
    n_rep = int(np.prod(supercell))
    array = array.reshape((n_rep, n_prim, 3, n_rep, n_prim, 3, n_rep, n_prim, 3))
    return array[0, ...].reshape((3 * n_prim, 3 * n_rep * n_prim, 3 * n_rep * n_prim))


def load_second_hiphive(*, folder: Path, resolved, filename: str = "model2.fcs", **_) -> ForceConstantData:
    fcs = _read_forceconstants(folder, filename)
    n_prim = resolved.unit_atoms.positions.shape[0]
    n_rep = int(np.prod(resolved.supercell))
    value = _reshape_second(fcs, n_rep, n_prim)
    replicas = ensure_replicas(resolved, folder, ("replicated_atoms.xyz",))
    return ForceConstantData(
        order=2,
        value=value,
        unit_atoms=resolved.unit_atoms,
        supercell=resolved.supercell,
        replicated_atoms=replicas,
    )


def load_third_hiphive(*, folder: Path, resolved, filename: str = "model3.fcs", **_) -> ForceConstantData:
    fcs = _read_forceconstants(folder, filename)
    n_prim = resolved.unit_atoms.positions.shape[0]
    value = _reshape_third(fcs, n_prim, resolved.supercell)
    replicas = ensure_replicas(resolved, folder, ("replicated_atoms.xyz",))
    return ForceConstantData(
        order=3,
        value=value,
        unit_atoms=resolved.unit_atoms,
        supercell=resolved.supercell,
        replicated_atoms=replicas,
    )
