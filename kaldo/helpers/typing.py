from typing import Literal, Annotated
from numpy.typing import NDArray, ArrayLike
import numpy as np
from os import PathLike
from ase import Atoms

NDArray = NDArray
ArrayLike = ArrayLike
Supercell = tuple[int, int, int]  # Supercell: The configuration of the supercell as (n_x, n_y, n_z)
PathLike = PathLike | str
Atoms = Atoms

ThirdorderData = np.ndarray[tuple[int, int, int, int, int, int, int, int], np.dtype[np.float64]]
# ThirdorderData = np.ndarray[tuple[int[Literal["atom_i"]], int[Literal["alpha"]], int[Literal["second_cell_id"]], int[Literal["atom_j"]], int[Literal["beta"]], int[Literal["third_cell_id"]], int[Literal["atom_k"]], int[Literal["gamma"]]], np.dtype[np.float64]]
# ThirdorderData = Annotated[NDArray[np.float64], Literal["atom_i", "alpha", "second_cell_id", "atom_j", "beta", "third_cell_id", "atom_k", "gamma"]]
