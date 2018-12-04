import numpy as np
import ase
import ase.io
import ballistico.geometry_helper as geometry_helper
import ballistico.atoms_helper as atoms_helper
from ballistico.finite_difference import FiniteDifference
from ase.calculators.lammpslib import LAMMPSlib
from ballistico.ballistico_phonons import BallisticoPhonons as Phonons
from ballistico.conductivity_controller import ConductivityController
from ballistico.plotter import Plotter
import ballistico.io_helper as io_helper
from ase.build import bulk
from ase.calculators.espresso import Espresso

if __name__ == "__main__":
    # We start from a atoms
    # atoms = ase.io.read ('si-bulk.xyz')
    atoms = bulk ('Si', 'diamond', a=5.43)

    # and replicate it
    supercell = np.array ([3, 3, 3])
    n_replicas = np.prod(supercell)
    
    # we create our system
    temperature = 300

    # our Phonons object built on the system
    kpts = np.array ([5, 5, 5])
    is_classic = False

    calculator = LAMMPSlib
    calculator_inputs = ["pair_style tersoff",
                              "pair_coeff * * forcefields/Si.tersoff Si"]
    pseudopotentials = None
    
    # calculator = Espresso
    # calculator_inputs = {'system': {'ecutwfc': 16.0},
    #                      'electrons': {'conv_thr': 1e-8},
    #                      'disk_io': 'low',
    #                      'pseudo_dir': '/home/giuseppe/espresso/pseudo/'}
    # pseudopotentials = {'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'}

    # Create a finite difference object
    finite_difference = FiniteDifference(atoms=atoms,
                                         supercell=supercell,
                                         calculator=calculator,
                                         calculator_inputs=calculator_inputs,
                                         pseudopotentials=pseudopotentials,
                                         is_persistency_enabled=True,
                                         is_third_order_symmetry_enabled=True)
    
    # Create a phonon object
    phonons = Phonons (finite_difference=finite_difference,
                       kpts=kpts,
                       is_classic=is_classic,
                       temperature=temperature,
                       sigma_in=None,
                       is_persistency_enabled=False)
    
    # Create a plot helper object
    plotter = Plotter (phonons=phonons,
                       is_showing=True,
                       folder='plot/ballistico/',
                       is_persistency_enabled=True)


    # call the method plot everything
    plotter.plot_everything()
    
    # calculate the conductivity creating a conductivity object and calling the
    # calculate_conductivity method
    ConductivityController (phonons).calculate_conductivity (is_classical=is_classic)
