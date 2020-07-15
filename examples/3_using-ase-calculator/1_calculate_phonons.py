import numpy as np
from kaldo.forceconstants import ForceConstants
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import kaldo.controllers.plotter as plotter

td='ald'

atoms = bulk('Si', 'diamond', a=5.432)
is_classic = False

# and replicate it
supercell = np.array([3, 3, 3])

lammps_inputs = {'lmpcmds': ["pair_style tersoff",
                             "pair_coeff * * forcefields/Si.tersoff Si"],
                 'log_file': 'log_lammps.out',
                 'keep_alive':True}


forceconstants = ForceConstants(atoms=atoms,
                                     supercell=supercell,
                                     folder='forceconstant')
forceconstants.second.calculate(calculator=LAMMPSlib(**lammps_inputs))
forceconstants.third.calculate(calculator=LAMMPSlib(**lammps_inputs))

n_replicas = np.prod(supercell)
kpts = [5, 5, 5]
temperature = 300

# # Create a phonon object
phonons = Phonons(forceconstants=forceconstants,
                    kpts=kpts,
                    is_classic=is_classic,
                    temperature=temperature,
                    folder='ald_out')
plotter.plot_dispersion(phonons)

print('Inverse conductivity in W/m/K')
print(Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0))

print('RTA conductivity in W/m/K')
print(Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0))

plotter.plot_dos(phonons)
plotter.plot_vs_frequency(phonons, phonons.heat_capacity, 'cv')
plotter.plot_vs_frequency(phonons, phonons.bandwidth, 'gamma_THz')
plotter.plot_vs_frequency(phonons, phonons.phase_space, 'phase_space')
