from numpy.typing import NDArray, ArrayLike
from os import PathLike
from ase import Atoms

NDArray = NDArray
ArrayLike = ArrayLike
Supercell = tuple[int, int, int]  # Supercell: The configuration of the supercell as (n_x, n_y, n_z)
PathLike = PathLike | str
Atoms = Atoms
