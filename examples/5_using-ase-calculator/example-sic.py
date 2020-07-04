from ase.phonons import Phonons
import numpy as np
from kaldo.forceconstants import ForceConstants
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib
import kaldo.controllers.plotter as plotter
from kaldo.phonons import Phonons as kaldoPhonons
import matplotlib.pyplot as plt


# and replicate it
supercell = (3, 3, 3)

lammps_input = {'lmpcmds': ["pair_style tersoff",
                             "pair_coeff * * forcefields/SiCGe.tersoff Si(D) C"],
                'log_file': 'log_lammps.out',
                'keep_alive':True}

atoms = bulk('SiC', 'zincblende', a=4.325)

# Phonon calculator
ph = Phonons(atoms, LAMMPSlib(**lammps_input), supercell=supercell, delta=0.05)
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

# Create a finite difference object
forceconstants = ForceConstants(atoms=atoms,
						  supercell=supercell,
						  calculator=LAMMPSlib,
						  calculator_inputs=lammps_input,
						  is_reduced_second=True,
						  folder='sic-fd')


kpts = [5, 5, 5]
temperature = 300

# # Create a phonon object
phonons = kaldoPhonons(forceconstants=forceconstants,
                    kpts=kpts,
                    is_classic=is_classic,
                    temperature=temperature,
                    folder='sic-out')

plotter.plot_dispersion(phonons)
print('done')
