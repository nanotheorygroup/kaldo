# Runs kALDo on monolayer MgO using force constants by DFPT
nthr=8 # adjust to your computer's limits
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(nthr)
tf.config.threading.set_inter_op_parallelism_threads(nthr)
# GPU-usage
# Optionally you can configure for GPU with
# devices = tf.config.list_physical_devices('GPU')
# index = 0 # Index of which GPU to select
# tf.config.set_visible_devices(devices[index])

import numpy as np
from scipy import constants
import h5py

# Replicas
k = 9 # cubed
nrep = int(9)
nrep_third = int(5)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k, k, k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])
radps_to_THz = 1 / (2 * np.pi)
thz_to_invcm = constants.value('hertz-inverse meter relationship') * 1e12 / 100
# sheng_data = './BTE.omega'
# gamma_freqs = np.loadtxt(sheng_data)[0, :] * radps_to_THz
#phonopy_data = h5py.File('phonopy/with_NAC/anharmonic.hdf5', 'r')
# print(f'Frequencies at Gamma (shengbte): {gamma_freqs}')

### Begin simulation
# Import kALDo
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity

# Create harmonic ForceConstants object from QE-data
forceconstant_nac = ForceConstants.from_folder(
    folder='./forces',
    supercell=supercell,
    third_supercell=third_supercell,
    only_second=False,
    is_acoustic_sum=True,
    format='shengbte-d3q')
# If you load in forceconstants from another format, but still want to use the NAC
# Adjust the dielectric tensor, and Born effective charges BEFORE creating the Phonons object
# This is sample code for how to do that:
#
# atoms = forceconstant.atoms
# born_charges = np.tile(np.eye(3), len(atoms)).reshape((3,len(atoms),3))
# born_charges = np.transpose(born_charges, axes=(1,0,2))
# forconstant.atoms.info['dielectric'] = np.eye(3) * 4
# forceconstant.atoms.set_array('charges', born_charges)

# Create Phonon Object
phonons_nac = Phonons(forceconstants=forceconstant_nac,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='./',
                  is_unfolding=True,
                  storage='numpy', )

nac_cond_matrix = (Conductivity(phonons=phonons_nac, method='rta').conductivity.sum(axis=0))
nac_inv_matrix = (Conductivity(phonons=phonons_nac, method='inverse').conductivity.sum(axis=0))

forceconstant = ForceConstants.from_folder(
    folder='./forces/uncharged',
    supercell=supercell,
    third_supercell=third_supercell,
    only_second=False,
    is_acoustic_sum=True,
    format='shengbte-d3q')
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='./nocharges',
                  is_unfolding=True,
                  storage='numpy', )
cond_matrix = (Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0))
inv_matrix = (Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0))


print("uncharged")
print(cond_matrix)
print(inv_matrix)
print("charged")
print(nac_cond_matrix)
print(nac_inv_matrix)
