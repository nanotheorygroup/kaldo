from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
from ase.phonons import Phonons
from kaldo.forceconstants import ForceConstants
import kaldo.controllers.plotter as plotter
from kaldo.phonons import Phonons as KaldoPhonons
import matplotlib.pyplot as plt
import numpy as np

# We start from a atoms

atoms = bulk('SiC', 'zincblende', a=4.325)

# and replicate it
supercell = (3, 3, 3)
lammps_inputs = {'lmpcmds': ["pair_style tersoff",
                             "pair_coeff * * forcefields/SiCGe.tersoff Si(D) C"],
                'log_file': 'log_lammps_sic.out',
                'keep_alive':True}

# Config phonon calculator

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
emax = 0.135
bs.plot(ax=ax, emin=0.0, emax=emax)
fig.show()

is_classic = False
# Create a force constant object

forceconstants = ForceConstants(atoms=atoms,supercell=supercell,folder='si-fd')

# Compute 2nd and 3rd IFCs with the defined calculators

forceconstants.second.calculate(LAMMPSlib(**lammps_inputs))
forceconstants.third.calculate(LAMMPSlib(**lammps_inputs))


kpts = [5, 5, 5]
temperature = 300

# Create a phonon object

phonons = KaldoPhonons(forceconstants=forceconstants,
                    kpts=kpts,
                    is_classic=is_classic,
                    temperature=temperature,
                    folder='sic-out')

# Plotting dispersion relation with built-in plotter

plotter.plot_dispersion(phonons,with_velocity=False)
print('done')