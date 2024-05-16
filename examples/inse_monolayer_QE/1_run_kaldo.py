# Runs kALDo on monolayer b-InSe using force constants by DFPT
#
# Usage: python 1_run_kaldo.py
# u/n controls if kALDo unfolds the force constants
# disp will exit the calculations after making a dispersion
# overwrite allows the script to replace data written in previous runs

# Harmonic ----------------------
# Dispersion args
npoints = 399
pathstring = 'GMKG'
unfold_bool = True

# Anharmonic --------------------
# Threading per process
nthread = 2
# Conductivity method
cond_method = 'inverse'
# K-pt grid (technically harmonic, but the dispersion is the only property we really deal
# with here, and it isn't relevant to that)
k = 7 # cubed

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
import os
import sys
import numpy as np
from ase.io import read
from scipy import constants
# Replicas
nrep = int(7)
nrep_third = int(5)
supercell = np.array([nrep, nrep, 1])
kpts, kptfolder = [k, k, 1], '{}_{}_{}'.format(k, k, 1)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])

# Import target frequencies
matdyn_frequencies = np.loadtxt('matdyn_reference/with_correction.gp')
thz_to_invcm = constants.value('hertz-inverse meter relationship')*1e12/100
matdyn_frequencies = matdyn_frequencies/thz_to_invcm
print(f'{matdyn_frequencies[0]=}')
qe_data = 'espresso_fcs'


### Begin simulation
# Import kALDo
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
                       folder=qe_data,
                       supercell=supercell,
                       only_second=True,
                       third_supercell=third_supercell,
                       is_acoustic_sum=True,
                       format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder='./',
              is_unfolding=unfold_bool,
              storage='memory',
              is_nac=True,)
from kaldo.observables.harmonic_with_q import HarmonicWithQ
# phonon = HarmonicWithQ(np.array([0.25, 0.25, 0.000]),
#                        phonons.forceconstants.second,
#                        distance_threshold=phonons.forceconstants.distance_threshold,
#                        storage='memory',
#                        is_nw=phonons.is_nw,
#                        is_unfolding=phonons.is_unfolding,
#                        is_nac=True, )
# print(phonon.frequency)
# exit()

atoms = read('espresso_fcs/POSCAR', format='vasp')
cell = atoms.cell
lat = cell.get_bravais_lattice()
path = cell.bandpath(pathstring, npoints=npoints)
print('Unit cell detected: {} '.format(atoms))
print('Special points on cell: ')
print(lat.get_special_points())
print('Path: {}'.format(path))
np.savetxt('path.txt', np.hstack([path.kpts, np.ones((path.kpts.shape[0], 1))]))
freqs = []
for i,kpoint in enumerate(path.kpts):
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=phonons.is_unfolding,
                           is_nac=True,)
    print(kpoint, phonon.frequency.squeeze()[:3])
    freqs.append(phonon.frequency.squeeze())

print(np.array(freqs).squeeze().shape)
np.savetxt('plots/7_7_1/kpts', path.kpts)
np.savetxt('plots/7_7_1/dispersion', np.array(freqs))


from matplotlib import pyplot as plt
plt.style.use('/home/nwlundgren/spanners/configurations/nicholas.mplstyle')
# Unit conversions
# THz to cm-1
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot()

colors = ['r', 'b',]
kaldo_y = np.loadtxt('plots/7_7_1/dispersion')
kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
for column in range(kaldo_y.shape[1]):
    ax.plot(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='b', lw=2, zorder=10)
plt.show()
#plot_dispersion(phonons, is_showing=True,
#           manually_defined_path=path,) #folder=prefix+'/dispersion')
#plot_dispersion(phonons, is_showing=True, n_k_points=150) #folder=prefix+'/dispersion')
