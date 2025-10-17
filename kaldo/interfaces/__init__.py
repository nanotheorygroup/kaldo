from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import numpy as np
from scipy.sparse import load_npz
from sparse import COO

from kaldo.interfaces.common import ForceConstantData, ensure_replicas
from kaldo.interfaces.structure import ResolvedStructure, resolve_structure
from kaldo.interfaces.eskm_io import (
    load_second_eskm,
    load_second_lammps,
    load_third_eskm,
    load_third_lammps,
)
from kaldo.interfaces.hiphive_io import load_second_hiphive, load_third_hiphive
from kaldo.interfaces.shengbte_io import (
    load_second_vasp,
    load_second_qe,
    load_third_vasp,
    load_third_d3q,
)
from kaldo.interfaces.tdep_io import load_second_tdep, load_third_tdep

__all__ = [
    "ForceConstantData",
    "ResolvedStructure",
    "resolve_structure",
    "load_forceconstants",
]


def load_second_numpy(*, folder: Path, resolved: ResolvedStructure, filename: str = "second.npy") -> ForceConstantData:
    array = np.load(folder / filename, allow_pickle=True)
    replicas = ensure_replicas(resolved, folder, ("replicated_atoms.xyz",))
    return ForceConstantData(
        order=2,
        value=array,
        unit_atoms=resolved.unit_atoms,
        supercell=resolved.supercell,
        replicated_atoms=replicas,
    )


def load_third_numpy(*, folder: Path, resolved: ResolvedStructure, filename: str = "third.npz") -> ForceConstantData:
    path = folder / filename
    n_unit = resolved.unit_atoms.positions.shape[0]
    n_rep = int(np.prod(resolved.supercell))
    if path.suffix == ".npz":
        coo = COO.from_scipy_sparse(load_npz(path))
        value = coo.reshape((3 * n_unit, 3 * n_rep * n_unit, 3 * n_rep * n_unit))
    else:
        arr = np.load(path, allow_pickle=True)
        value = arr.reshape((3 * n_unit, 3 * n_rep * n_unit, 3 * n_rep * n_unit))
    replicas = ensure_replicas(resolved, folder, ("replicated_atoms_third.xyz", "replicated_atoms.xyz"))
    return ForceConstantData(
        order=3,
        value=value,
        unit_atoms=resolved.unit_atoms,
        supercell=resolved.supercell,
        replicated_atoms=replicas,
    )


SECOND_ALIAS = {
    "numpy": "numpy",
    "eskm": "eskm",
    "lammps": "lammps",
    "shengbte": "vasp",
    "vasp": "vasp",
    "shengbte-qe": "qe",
    "shengbte-d3q": "vasp",
    "qe": "qe",
    "hiphive": "hiphive",
    "tdep": "tdep",
}

THIRD_ALIAS = {
    "numpy": "numpy",
    "sparse": "numpy",
    "eskm": "eskm",
    "lammps": "lammps",
    "shengbte": "vasp",
    "vasp": "vasp",
    "shengbte-d3q": "d3q",
    "d3q": "d3q",
    "shengbte-qe": "vasp",
    "hiphive": "hiphive",
    "tdep": "tdep",
}


SECOND_LOADERS: dict[str, Callable[..., ForceConstantData]] = {
    "numpy": load_second_numpy,
    "eskm": load_second_eskm,
    "lammps": load_second_lammps,
    "vasp": load_second_vasp,
    "qe": load_second_qe,
    "hiphive": load_second_hiphive,
    "tdep": load_second_tdep,
}

THIRD_LOADERS: dict[str, Callable[..., ForceConstantData]] = {
    "numpy": load_third_numpy,
    "eskm": load_third_eskm,
    "lammps": load_third_lammps,
    "vasp": load_third_vasp,
    "d3q": load_third_d3q,
    "hiphive": load_third_hiphive,
    "tdep": load_third_tdep,
}


def _normalize_token(order: int, token: str) -> str:
    aliases = SECOND_ALIAS if order == 2 else THIRD_ALIAS
    canonical = aliases.get(token, token)
    if canonical != token:
        warnings.warn(f"'{token}' is deprecated; use '{canonical}'", DeprecationWarning, stacklevel=3)
    return canonical


def load_forceconstants(order: int,
                        token: str,
                        resolved: ResolvedStructure,
                        *,
                        folder: str | Path,
                        **options) -> ForceConstantData:
    folder = Path(folder)
    canonical = _normalize_token(order, token)
    loaders = SECOND_LOADERS if order == 2 else THIRD_LOADERS
    try:
        loader = loaders[canonical]
    except KeyError:
        raise ValueError(f"{canonical} is not a supported format for order {order}.")
    return loader(folder=folder, resolved=resolved, **options)
