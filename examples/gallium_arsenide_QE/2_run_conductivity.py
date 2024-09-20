# Runs kALDo on monolayer GaAs using force constants by DFPT
#
# Usage: python 1_run_kaldo.py
# u/n controls if kALDo unfolds the force constants
# disp will exit the calculations after making a dispersion
# overwrite allows the script to replace data written in previous runs
import os

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
k = 5  # cubed

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
import numpy as np
from scipy import constants

# Replicas
nrep = int(5)
nrep_third = int(5)
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
from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons

# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
    folder='./uncorrected/',
    supercell=supercell,
    third_supercell=third_supercell,
    only_second=False,
    is_acoustic_sum=True,
    format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='uncorrected',
                  is_unfolding=True,
                  storage='numpy',
                  is_nac=False, )
conductivity = Conductivity(phonons=phonons,
                            method='inverse',
                            storage='numpy').conductivity.sum(axis=0)
# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
    folder='./corrected/',
    supercell=supercell,
    third_supercell=third_supercell,
    only_second=False,
    is_acoustic_sum=True,
    format='shengbte-qe')
phonons_nac = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder='corrected',
                  is_unfolding=True,
                  storage='numpy',
                  is_nac=True, )
conductivity_nac = Conductivity(phonons=phonons_nac,
                            method='inverse',
                            storage='numpy').conductivity.sum(axis=0)
conductivity_diff = conductivity - conductivity_nac
print(conductivity)
print(conductivity_diff)
