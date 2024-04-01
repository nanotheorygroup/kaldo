# Runs kALDo on monolayer b-InSe using force constants by DFPT
#
# Usage: python 1_run_kaldo.py
# u/n controls if kALDo unfolds the force constants
# disp will exit the calculations after making a dispersion
# overwrite allows the script to replace data written in previous runs

# Harmonic ----------------------
# Dispersion args
npoints = 200 # points along path
pathstring = 'GMKG' # actual path
unfold_bool = True

# Anharmonic --------------------
# Threading per process
nthread = 2
# K-pt grid
k = 7 # cubed
# Conductivity method
cond_method = 'inverse'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
import os
import sys
import numpy as np
from ase.io import read
# Replicas
nrep = int(7)
nrep_third = int(5)
supercell = np.array([nrep, nrep, 1])
kpts, kptfolder = [k, k, 1], '{}_{}_{}'.format(k,k,k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])

# Detect harmonic
harmonic = False
if 'harmonic' in sys.argv:
    harmonic = True
# Control data IO + overwriting controls
qe_data = 'espresso_fcs'
overwrite = False
prefix=os.environ['kaldo_ald']+'/'
if 'overwrite' in sys.argv:
    overwrite = True
if os.path.isdir(prefix):
    print('!! - '+prefix+' directory already exists')
    print('!! - continuing may overwrite, or load previous data\n')
    if not overwrite:
        print('!! - overwrites disallowed, exiting safely..')
        exit()
else:
    os.mkdir(prefix)
# Control threading behavior
os.environ['CUDA_VISIBLE_DEVICES']=" "
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(nthread)
tf.config.threading.set_intra_op_parallelism_threads(nthread)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Print out detected settings
print('\n\n\tCalculating for supercell {} -- '.format(supercell))
print('\t\t In folder:       QE:{}'.format(qe_data))
print('\t\t Out folder:      {}'.format(prefix))
print('\t\t Dispersion only: {}'.format(harmonic))
print('\t\t Overwrite permission: {}\n\n'.format(overwrite))
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Begin simulation
# Import kALDo
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.controllers.plotter import plot_dispersion

# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
                       folder=qe_data,
                       supercell=supercell,
                       only_second=harmonic,
                       third_supercell=third_supercell,
                       is_acoustic_sum=True,
                       format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder=prefix,
              is_unfolding=unfold_bool,
              storage='memory',
              is_nac=True,)

# Harmonic data along path
# Although you need to specify the k-pt grid for the Phonons object, we don't
# actually use it for dispersion relations and velocities the sampling is taken
# care of by the path specified and the npoints variable set above.
# Note: Choice of k-pt grid WILL effect DoS calculations for amorphous models.
atoms = read('espresso_fcs/POSCAR', format='vasp')
cell = atoms.cell
lat = cell.get_bravais_lattice()
path = cell.bandpath(pathstring, npoints=npoints)
print('Unit cell detected: {} '.format(atoms))
print('Special points on cell: ')
print(lat.get_special_points())
print('Path: {}'.format(path))
np.savetxt('path.txt', np.hstack([path.kpts, np.ones((path.kpts.shape[0], 1))]))
plot_dispersion(phonons, is_showing=True,
            manually_defined_path=path,) #folder=prefix+'/dispersion')
if harmonic:
    print('\n\n\n\tHarmonic quantities generated, exiting safely ..')
    quit(0)

# Conductivity & Anharmonics
# Different methods of calculating the conductivity can be compared
# but full inversion of the three-phonon scattering matrix typically agrees with
# experiment more than say the relaxation time approximation (not applicable for
# systems without symmetry).
# Calculating this K will produce as a byproduct things like the phonon
# bandwidths, phase space etc, but some need to be explicitly called to output
# numpy files (e.g. Phonons.participation_ratio, find more in Phonons docs).
#
# All of the anharmonic quantities should be converged according to a sensitivity
# analysis of their value against increasing k-points.
cond = Conductivity(phonons=phonons, method=cond_method, storage='numpy')
cond_matrix = cond.conductivity.sum(axis=0)
diag = np.diag(cond_matrix)
offdiag = np.abs(cond_matrix).sum() - np.abs(diag).sum()
print('Conductivity from full inversion (W/m-K):\n%.3f' % (np.mean(diag)))
print('Sum of off-diagonal terms: %.3f' % offdiag)
print('Full matrix:')
print(cond_matrix)
