import numpy as np
import ase.io
import ballistico.atoms_helper as atoms_helper
from ballistico.finite_difference import FiniteDifference
from ase.calculators.lammpslib import LAMMPSlib
from ballistico.ballistico_phonons import BallisticoPhonons as Phonons
from ballistico.conductivity_controller import ConductivityController
from ballistico.plotter import Plotter
import ballistico.io_helper as io_helper
np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # We start from a atoms
    atoms = ase.io.read ('cubic-si.xyz')
    
    # and replicate it
    supercell = np.array ([3, 3, 3])
    n_replicas = np.prod(supercell)
    replicated_geometry, _ = atoms_helper.replicate_atoms(atoms, supercell)
    
    # then we store it
    ase.io.write ('dlpoly_files/CONFIG', replicated_geometry, format='dlp4')

    # we create our system
    temperature = 300

    # our Phonons object built on the system
    kpts = np.array ([5, 5, 5])
    is_classic = False

    # import the calculated second order
    second_order = io_helper.import_second_dlpoly (atoms, supercell)

    # import the calculated third order
    third_order = io_helper.import_third_order_dlpoly(atoms, supercell)


    # Create a finite difference object
    finite_difference = FiniteDifference(atoms=atoms,
                                         supercell=supercell,
                                         second_order=second_order,
                                         third_order=third_order,
                                         is_persistency_enabled=True)
    
    # Create a phonon object
    phonons = Phonons (finite_difference=finite_difference,
                       kpts=kpts,
                       is_classic=is_classic,
                       temperature=temperature,
                       sigma_in=None,
                       is_persistency_enabled=True)
    
    # Create a plot helper object
    plotter = Plotter (phonons=phonons,
                       is_showing=False,
                       folder='plot/ballistico/',
                       is_persistency_enabled=True)

    # call the method plot everything
    plotter.plot_everything()
    
    # calculate the conductivity creating a conductivity object and calling the
    # calculate_conductivity method
    ConductivityController (phonons).calculate_conductivity (is_classical=is_classic)
