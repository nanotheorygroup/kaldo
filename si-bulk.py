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
from ballistico.shengbte_phonons_controller import ShengbtePhononsController as Sheng
from ase.build import bulk
from ase.calculators.espresso import Espresso
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # We start from a atoms
    # atoms = ase.io.read ('si-bulk.xyz')
    atoms = bulk('Si', 'diamond', a=5.432)
    is_classic = False

    # and replicate it
    supercell = np.array([3, 3, 3])
    n_replicas = np.prod(supercell)

    temperature = 300
    # we create our system
    # temperature = 300

    # our Phonons object built on the system
    kpts = np.array([7, 7, 7])

    calculator = LAMMPSlib
    calculator_inputs = {'lmpcmds': ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"],
                         'log_file': 'log_lammps.out'}

    # calculator = Espresso
    # calculator_inputs = {'pseudopotentials':{'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'},
    #                 'tstress':True,
    #                 'tprnfor':True,
    #                 'input_data':
    #                      {'system': {'ecutwfc': 16.0},
    #                                            'electrons': {'conv_thr': 1e-8},
    #                                            'disk_io': 'low',
    #                                            'pseudo_dir': '/home/giuseppe/espresso/pseudo/'},
    #                 'koffset':(2, 2, 2),
    #                 'kpoints':(1, 1, 1)}

    third_order_symmerty_inputs = {'NNEIGH': 4, 'SYMPREC': 1e-5}

    # Create a finite difference object
    finite_difference = FiniteDifference(atoms=atoms,
                                         supercell=supercell,
                                         calculator=calculator,
                                         calculator_inputs=calculator_inputs,
                                         is_persistency_enabled=False,
                                         third_order_symmerty_inputs=third_order_symmerty_inputs)

    # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference, kpts=kpts, is_classic=is_classic,
                      temperature=temperature, is_persistency_enabled=False)
    # Create a plot helper object
    plotter = Plotter (phonons=phonons,
                       is_showing=True,
                       folder='plot/ballistico/',
                       is_persistency_enabled=True).plot_everything()

    # Create a phonon object
    phonons = Sheng(finite_difference=finite_difference, kpts=kpts, is_classic=is_classic,
                      temperature=temperature, is_persistency_enabled=False)
    # Create a plot helper object
    plotter = Plotter (phonons=phonons,
                       is_showing=True,
                       folder='plot/ballistico/',
                       is_persistency_enabled=True).plot_everything()




    # calculate the conductivity creating a conductivity object and calling the
    # calculate_conductivity method
    # heat_capacity = phonons.c_v.mean()
    conductivity = ConductivityController(phonons).calculate_conductivity(is_classical=is_classic)[0, 0]
    print(conductivity)
