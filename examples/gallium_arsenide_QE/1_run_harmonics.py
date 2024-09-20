# Runs kALDo on monolayer GaAs using force constants by DFPT
#
# Usage: python 1_run_kaldo.py
#
# Harmonic ----------------------
# Dispersion args
npoints = 200
pathstring = 'GXWGK'
outfold = 'plots/'
# GXWGK - special points with 200 points
# G   X   W   G   K
# 0  53  67 140 199
# -------------------------------

# Anharmonic --------------------
# Threading per process
nthread = 2
# Conductivity method
cond_method = 'inverse'
# K-pt grid (technically harmonic, but the dispersion is the only property we really deal
# with here, and it isn't relevant to that)
k = 2  # cubed

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
import numpy as np
from scipy import constants

# Replicas
nrep = int(4)
nrep_third = int(2)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k, k, k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])
radps_to_THz = 1 / (2 * np.pi)
thz_to_invcm = constants.value('hertz-inverse meter relationship') * 1e12 / 100
sheng_data = 'refs/BTE.omega'
gamma_freqs = np.loadtxt(sheng_data)[0, :] * radps_to_THz
print(f'Frequencies at Gamma (shengbte): {gamma_freqs}')

### Begin simulation
# Import kALDo
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
    folder='./corrected/',
    supercell=supercell,
    only_second=True,
    is_acoustic_sum=True,
    format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='corrected',
                  is_unfolding=True,
                  storage='numpy',
                  is_nac=True, )

atoms = forceconstant.atoms
cell = atoms.cell
lat = cell.get_bravais_lattice()
path = cell.bandpath(pathstring, npoints=npoints)
print('Unit cell detected: {} '.format(atoms))
print('Special points on cell: ')
print(lat.get_special_points())
print('Path: {}'.format(path))
np.savetxt('path.txt', np.hstack([path.kpts, np.ones((path.kpts.shape[0], 1))]))
freqs = []
vels = []
vnorms = []
for i, kpoint in enumerate(path.kpts):
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=True,
                           is_nac=True, )
    freqs.append(phonon.frequency.squeeze())
    vels.append(phonon.velocity.reshape((-1, 1)))
    vnorms.append(np.linalg.norm(phonon.velocity.squeeze(), axis=-1).squeeze())

np.savetxt(f'{outfold}/{pathstring}.pts-nac', path.kpts)
np.savetxt(f'{outfold}/{pathstring}.freqs-nac', np.array(freqs))
np.savetxt(f'{outfold}/{pathstring}.vels-nac', np.array(vels).squeeze())
np.savetxt(f'{outfold}/{pathstring}.vnorms-nac', np.array(vnorms).squeeze())

freqs = []
vels = []
vnorms = []
for i, kpoint in enumerate(path.kpts):
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=True,
                           is_nac=False, )
    freqs.append(phonon.frequency.squeeze())
    vels.append(phonon.velocity.reshape((-1, 1)))
    vnorms.append(np.linalg.norm(phonon.velocity.squeeze(), axis=-1).squeeze())

np.savetxt(f'{outfold}/{pathstring}.pts-none', path.kpts)
np.savetxt(f'{outfold}/{pathstring}.freqs-none', np.array(freqs))
np.savetxt(f'{outfold}/{pathstring}.vels-none', np.array(vels).squeeze())
np.savetxt(f'{outfold}/{pathstring}.vnorms-none', np.array(vnorms).squeeze())


from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from palettable.colorbrewer.qualitative import Set1_6

set1 = Set1_6.hex_colors


special_points = ['G', 'X', 'W', 'G', 'K']
special_indices = np.array([0, 53, 79, 140, 199], dtype=int) * 1 / 199

#### FIGURE 1
fig = plt.figure(figsize=(16, 12))
grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.08)
ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[1, 0])
ax2 = fig.add_subplot(grid[:, 1])

nac_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-nac')
none_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
nac_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-nac')
none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
for column in range(3):
    ax0.scatter(nac_x[:, column], nac_y[:, column] / 10, color=set1[0], s=15, zorder=1)
    ax0.scatter(none_x[:, column], none_y[:, column] / 10, color=set1[2], s=12, marker='x', zorder=0)
ax0.set_xticks([0, 2, 4, 6,], [0, 2, 4, 6,], color='k')
ax0.set_xlim([0, 7])
ax0.set_ylim([0, 5.1])
ax0.grid(color='k', alpha=0.8)
ax0.set_ylabel(r'$V_{g}$ (km/s)', rotation=70, ha='right', color='k')
note0 = ax0.annotate('Acoustic Modes', xy=(0.7, 0.92),
                     xycoords='axes fraction', fontsize=18, )
note0.set_bbox(dict(facecolor='white', alpha=1, linewidth=0.7, edgecolor='k'))
ax0.set_title(r'$V_{g}$ vs Frequency', color='k')

nac_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-nac')
none_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
nac_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-nac')
none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
for column in range(3, 6):
    ax1.scatter(nac_x[:, column], nac_y[:, column] / 10, color=set1[0], s=15, zorder=1)
    ax1.scatter(none_x[:, column], none_y[:, column] / 10, color=set1[2], s=12, marker='x', zorder=0)
ax1.set_xticks([6, 6.5, 7, 7.5, 8], [6, '', 7, '', 8], color='k')
ax1.set_xlim([6, 8.5])
ax1.set_yticks([0, 0.5, 1, 1.5, 2], [0, "", 1, "", 2], color='k')
ax1.grid(color='k', alpha=0.8)
ax1.set_xlabel('Frequency (THz)', color='k')
ax1.set_ylabel(r'$V_{g}$ (km/s)', rotation=70, ha='right', color='k')
note1 = ax1.annotate('Optical Modes', xy=(0.7, 0.92),
                     xycoords='axes fraction', fontsize=18, )
note1.set_bbox(dict(facecolor='white', alpha=1, linewidth=0.7, edgecolor='k'))

nac_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-nac')
none_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
kaldo_x = np.linspace(0, 1, nac_y.shape[0])
for column in range(nac_y.shape[1]):
    ax2.scatter(kaldo_x, nac_y[:, column], color=set1[0], s=15, zorder=1)
    ax2.scatter(kaldo_x, none_y[:, column], color=set1[2], s=12, marker='x', zorder=0)
ax2.set_xticks(special_indices, labels=special_points, color='k')
ax2.grid(color='k', alpha=0.8)
ax2.set_yticks([0, 5, 10], [0, 5, 10], color='k')
ax2.set_ylabel(r'$\nu$ (THz)', rotation=70, ha='right', color='k')
ax2.set_title('Phonon Dispersion')

lines = [Line2D([], [], lw=5, color=set1[0]),
         Line2D([], [], lw=0, marker='x', color=set1[2]),
         ]
labels = ['QE', 'None']
ax2.legend(lines, labels, title='Correction', bbox_to_anchor=(0.95, 0.95))
plt.suptitle('GaAs Velocity Corrections', fontweight='bold', fontsize=25)
plt.savefig(f'{outfold}/phonon_velocities.png', bbox_inches='tight')

