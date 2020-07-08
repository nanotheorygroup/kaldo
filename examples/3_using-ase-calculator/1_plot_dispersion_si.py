from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.phonons import Phonons
from kaldo.forceconstants import ForceConstants
import kaldo.controllers.plotter as plotter
from kaldo.phonons import Phonons as KaldoPhonons
import matplotlib.pyplot as plt
import numpy as np

# We start from a atoms

atoms = bulk('Si', 'diamond', a=5.432)

# Config lammps input

lammps_inputs = {'lmpcmds': ["pair_style tersoff",
                             "pair_coeff * * forcefields/Si.tersoff Si"],
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


# Create a force constant object

forceconstants = ForceConstants(atoms=atoms,supercell=supercell,folder='si-fd')

# Compute 2nd and 3rd IFCs with the defined calculators

forceconstants.second.calculate(LAMMPSlib(**lammps_inputs))
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs))

kpts = [5, 5, 5]
temperature = 300
is_classic = False

# Create a phonon object

phonons = KaldoPhonons(forceconstants=forceconstants,
                    kpts=kpts,
                    is_classic=is_classic,
                    temperature=temperature,
                    folder='si-out')


# Plotting dispersion relation with built-in plotter

plotter.plot_dispersion(phonons,with_velocity=False)
print('done')