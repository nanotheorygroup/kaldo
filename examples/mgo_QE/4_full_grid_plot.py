# Runs kALDo on monolayer MgO using force constants by DFPT
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
from scipy import constants
import h5py

# Replicas
k = 12  # cubed
nrep = int(4)
nrep_third = int(2)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k, k, k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])
radps_to_THz = 1 / (2 * np.pi)
thz_to_invcm = constants.value('hertz-inverse meter relationship') * 1e12 / 100
sheng_data = './BTE.omega'
gamma_freqs = np.loadtxt(sheng_data)[0, :] * radps_to_THz
phonopy_data = h5py.File('phonopy/with_NAC/anharmonic.hdf5', 'r')
print(f'Frequencies at Gamma (shengbte): {gamma_freqs}')

### Begin simulation
# Import kALDo
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
    folder='./',
    supercell=supercell,
    only_second=True,
    is_acoustic_sum=True,
    format='shengbte-qe')
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
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='./',
                  is_unfolding=False,
                  storage='numpy', )

phonons.frequency
phonons.velocity

nrep = int(9)
supercell = np.array([nrep, nrep, nrep])
forceconstant = ForceConstants.from_folder(
    folder='./nocharges',
    supercell=supercell,
    only_second=True,
    is_acoustic_sum=True,
    format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='./nocharges',
                  is_unfolding=False,
                  storage='numpy', )
phonons.frequency
phonons.velocity

import numpy as np
from matplotlib import pyplot as plt
from palettable.colorbrewer.qualitative import Set1_6
from palettable.cartocolors.qualitative import Prism_6 # note you can reverse order with {set}_r
from sklearn.neighbors import KernelDensity
set1 = Set1_6.hex_colors
plt.style.use('/home/nwlundgren/.config/matplotlib/matplotlibrc')
outfold = 'plots/'

#### FIGURE 1
fig = plt.figure(figsize=(16,12))
grid = plt.GridSpec(1, 2, wspace=0.35, hspace=0.08)
ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[0, 1])

sheng_x = np.load(f'{kptfolder}/frequency.npy')
none_x = np.load(f'nocharges/{kptfolder}/frequency.npy')
phonopy_x = phonopy_data['frequency'][:,:]
sheng_y = np.linalg.norm(np.load(f'{kptfolder}/velocity.npy'), axis=-1)
none_y = np.linalg.norm(np.load(f'nocharges/{kptfolder}/velocity.npy'), axis=-1)
phonopy_y = np.linalg.norm(phonopy_data['group_velocity'], axis=-1)
ax0.scatter(sheng_x, sheng_y/10, color=set1[1], s=15, marker='o', zorder=0, alpha=0.35)
ax0.scatter(none_x, none_y/10, color=set1[0], s=15, marker='s', zorder=3, alpha=0.35)
ax0.scatter(phonopy_x, phonopy_y/10, color=set1[2], s=100, marker='d', zorder=2, alpha=0.35)
ax0.set_ylabel(r'$V_{g}$ (km/s)', rotation=70, ha='right', color='k')
# note0 = ax0.annotate('Acoustic Modes', xy=(0.7, 0.92),
#                      xycoords='axes fraction', fontsize=18, )
# note0.set_bbox(dict(facecolor='white', alpha=1, linewidth=0.7, edgecolor='k'))
ax0.set_title(r'$V_{g}$ vs Frequency', color='k')

sheng_x = np.load(f'{kptfolder}/frequency.npy')
none_x = np.load(f'nocharges/{kptfolder}/frequency.npy')
sheng_kernel = KernelDensity(kernel='gaussian', bandwidth=1/4.135).fit(sheng_x.reshape(-1, 1))
none_kernel = KernelDensity(kernel='gaussian', bandwidth=1/4.135).fit(none_x.reshape(-1, 1))
phonopy_kernel = KernelDensity(kernel='gaussian', bandwidth=1/4.135).fit(phonopy_x.reshape(-1, 1))
sheng_y = np.exp(sheng_kernel.score_samples(sheng_x.reshape(-1, 1)))
none_y = np.exp(none_kernel.score_samples(none_x.reshape(-1, 1)))
phonopy_y = np.exp(phonopy_kernel.score_samples(phonopy_x.reshape(-1, 1)))
ax1.scatter(sheng_x, sheng_y, color=set1[1], s=15, marker='o', zorder=0, alpha=0.35)
ax1.scatter(none_x, none_y, color=set1[0], s=15, marker='s', zorder=2, alpha=0.35)
ax1.scatter(phonopy_x, phonopy_y, color=set1[2], s=100, marker='d', zorder=1, alpha=0.35)
ax1.set_ylabel(r'$DoS$', rotation=70, ha='right', color='k')
# note0 = ax0.annotate('Acoustic Modes', xy=(0.7, 0.92),
#                      xycoords='axes fraction', fontsize=18, )
# note0.set_bbox(dict(facecolor='white', alpha=1, linewidth=0.7, edgecolor='k'))
ax1.set_title(r'$V_{g}$ vs Frequency', color='k')

from matplotlib.lines import Line2D
Lines = [Line2D([], [], lw=0, marker='o', markersize=5, color=set1[1]),
         Line2D([], [], lw=0, marker='s', markersize=5, color=set1[0]),
         Line2D([], [], lw=0, marker='d', markersize=5, color=set1[2])]
Labels = ['ShengBTE',
          'No Charge',
          'Phonopy']
ax0.legend(Lines, Labels, loc='upper right', fontsize=20)

plt.suptitle('MgO Velocity Corrections', fontweight='bold', fontsize=25)
plt.savefig(f'{outfold}/grid_compare.png')
