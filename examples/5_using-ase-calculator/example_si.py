from ase.phonons import Phonons
import numpy as np
from kaldo.finitedifference import FiniteDifference
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
import kaldo.controllers.plotter as plotter
from kaldo.phonons import Phonons as kaldoPhonons
import matplotlib.pyplot as plt

# We start from a atoms
# atoms = ase.io.read ('si-bulk.xyz')
atoms = bulk('Si', 'diamond', a=5.432)
lammps_inputs = {'lmpcmds': ["pair_style tersoff",
                             "pair_coeff * * forcefields/Si.tersoff Si(D)"],
                 'log_file': 'log_lammps.out',
                 'keep_alive':True}

# Phonon calculator
supercell = (3, 3, 3)

ph = Phonons(atoms, LAMMPSlib(**lammps_inputs), supercell=supercell, delta=0.05)
ph.run()

# Read forces and assemble the dynamical matrix
ph.read(acoustic=True)
ph.clean()

path = atoms.cell.bandpath('GXUGLWX', npoints=100)
bs = ph.get_band_structure(path)


# Plot the band structure and DOS:
fig, ax = plt.subplots()
ev_to_tHz = 241.79893
emax = 0.07
bs.plot(ax=ax, emin=0.0, emax=emax)

fig.show()


# and replicate it

lammps_inputs = {'lmpcmds': ["pair_style tersoff",
                             "pair_coeff * * forcefields/Si.tersoff Si(D)"],
                'log_file': 'log_lammps.out'}

# Create a finite difference object
finite_difference = FiniteDifference(atoms=atoms,
                                     supercell=supercell,
                                     calculator=LAMMPSlib,
                                     calculator_inputs=lammps_inputs,
                                     is_reduced_second=False,
                                     folder='si-fd')

n_replicas = np.prod(supercell)
kpts = [5, 5, 5]
temperature = 300
is_classic = False

# # Create a phonon object
phonons = kaldoPhonons(finite_difference=finite_difference,
                    kpts=kpts,
                    is_classic=is_classic,
                    temperature=temperature,
                    folder='si-out')

plotter.plot_dispersion(phonons)
print('done')
