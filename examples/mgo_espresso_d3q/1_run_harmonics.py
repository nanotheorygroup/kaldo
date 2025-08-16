# Runs kALDo on MgO using force constants by DFPT
#
# Usage: python 1_run_kaldo.py
# Pier 1 Imports --
import sys
import numpy as np
# Import kALDo™
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.controllers.plotter import plot_dos

# Harmonic ----------------------
# Dispersion args
npoints = 200
pathstring = 'GXUGLW'
unfold_bool = True
outfold = 'plots/'
# special points with 200 points
# G   X  U  G   L   W
# 0  49  66 119 163 199
# -------------------------------


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
# Replicas
k = 5 # cubed
nrep = int(5) # cubed
nrep_third = int(5) # cubed
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k, k, k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])

### Begin simulation
# Create harmonic ForceConstants object from QE-data
forceconstant_nac = ForceConstants.from_folder(
                       folder='./forces',
                       supercell=supercell,
                       only_second=True,
                       is_acoustic_sum=True,
                       format='shengbte-qe')
forceconstant = ForceConstants.from_folder(
                       folder='./forces/uncharged',
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
phonons_nac = Phonons(forceconstants=forceconstant_nac,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder='./forces',
              is_unfolding=unfold_bool,
              storage='numpy',)
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder='./forces/uncharged',
              is_unfolding=unfold_bool,
              storage='numpy',)

# Density of States
plot_dos(phonons, is_showing=False)
plot_dos(phonons_nac, is_showing=False)
print('dos figures saved!')

# Dispersion
atoms = forceconstant.atoms
cell = atoms.cell
lat = cell.get_bravais_lattice()
path = cell.bandpath(pathstring, npoints=npoints)
print('Unit cell detected: {} '.format(atoms))
print('Special points on cell: ')
print(lat.get_special_points())
print('Path: {}'.format(path))
# Save path for Matdyn
np.savetxt('tools/kpath.txt', np.hstack([path.kpts, np.ones((path.kpts.shape[0], 1))]))

freqs_nac, freqs = [], []
vels_nac, vels = [], []
vnorms_nac, vnorms = [], []
for i,kpoint in enumerate(path.kpts):
    phonon_nac = HarmonicWithQ(kpoint,
                           phonons_nac.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=phonons.is_unfolding)
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=phonons.is_unfolding)
    freqs.append(phonon.frequency.squeeze())
    vels.append(phonon.velocity.flatten())
    vnorms.append(np.linalg.norm(phonon.velocity.squeeze(), axis=-1).squeeze())
    freqs_nac.append(phonon_nac.frequency.squeeze())
    vels_nac.append(phonon_nac.velocity.flatten())
    vnorms_nac.append(np.linalg.norm(phonon_nac.velocity.squeeze(), axis=-1).squeeze())
np.savetxt(f'{outfold}/{pathstring}.pts', path.kpts)
np.savetxt(f'{outfold}/{pathstring}.freqs-nac', np.array(freqs_nac))
np.savetxt(f'{outfold}/{pathstring}.freqs', np.array(freqs))
np.savetxt(f'{outfold}/{pathstring}.vels-nac', np.array(vels_nac).squeeze())
np.savetxt(f'{outfold}/{pathstring}.vels', np.array(vels).squeeze())
np.savetxt(f'{outfold}/{pathstring}.vnorms-nac', np.array(vnorms_nac).squeeze())
np.savetxt(f'{outfold}/{pathstring}.vnorms', np.array(vnorms).squeeze())
