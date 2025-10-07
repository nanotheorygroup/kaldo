"""
Helper functions for calculating sigma2 values from MD trajectories and force constants.
"""
import numpy as np
from ase.geometry import find_mic
from ase.io import read
from sklearn.metrics import mean_squared_error
from kaldo.observables.secondorder import parse_tdep_forceconstant


def calculate_displacement(atoms, initial_structure):
    """
    Calculate the displacement between two atomic structures using minimum image convention.

    Parameters
    ----------
    atoms : ase.Atoms
        The current atomic structure.
    initial_structure : ase.Atoms
        The reference atomic structure.

    Returns
    -------
    np.ndarray
        The displacement array with minimum image convention applied.
    """
    disp = atoms.positions - initial_structure.positions
    return find_mic(disp.reshape(-1, 3), atoms.cell)[0].reshape(initial_structure.positions.shape)


def calculate_harmonic_force(disp, second_order_fc):
    """
    Calculate harmonic forces from displacements and second order force constants.

    Parameters
    ----------
    disp : np.ndarray
        Displacement array.
    second_order_fc : np.ndarray
        Second order force constant matrix.

    Returns
    -------
    np.ndarray
        Harmonic forces reshaped to match displacement shape.
    """
    force_harmonic_vec = -np.dot(second_order_fc, disp.flatten())
    return force_harmonic_vec.reshape(disp.shape)


def calculate_sigma(md_forces, harmonic_forces):
    """
    Calculate sigma value comparing MD forces to harmonic forces.

    Parameters
    ----------
    md_forces : np.ndarray
        Forces from molecular dynamics.
    harmonic_forces : np.ndarray
        Harmonic forces calculated from force constants.

    Returns
    -------
    float
        The sigma value (normalized RMSE).
    """
    return np.sqrt(mean_squared_error(md_forces, harmonic_forces)) / np.std(md_forces)


def sigma2_tdep_md(fc_file: str = 'infile.forceconstant',
                   primitive_file: str = 'infile.ucposcar',
                   supercell_file: str = 'infile.ssposcar',
                   md_run: str = 'dump.xyz') -> float:
    """
    Calculate the sigma2 value using TDEP and MD data.

    Parameters
    ----------
    fc_file : str, optional
        Path to the force constant file. Default is ``'infile.forceconstant'``.
    primitive_file : str, optional
        Path to the primitive cell file. Default is ``'infile.ucposcar'``.
    supercell_file : str, optional
        Path to the supercell file. Default is ``'infile.ssposcar'``.
    md_run : str, optional
        Path to the MD trajectory file. Default is ``'dump.xyz'``.

    Returns
    -------
    float
        The average sigma2 value.

    """
    initial_structure = read(supercell_file, format="vasp")
    second_order_fc = parse_tdep_forceconstant(
        fc_file=fc_file,
        primitive=primitive_file,
        supercell=supercell_file,
        symmetrize=True,
        two_dim=True
    )
    full_md_traj = read(md_run, index=":")
    displacements = [
        calculate_displacement(atoms, initial_structure)
        for atoms in full_md_traj
    ]
    force_harmonic = [
        calculate_harmonic_force(disp, second_order_fc)
        for disp in displacements
    ]
    sigma_values = [
        calculate_sigma(atoms.get_forces(), harm_force)
        for atoms, harm_force in zip(full_md_traj, force_harmonic)
    ]

    return np.mean(sigma_values)
