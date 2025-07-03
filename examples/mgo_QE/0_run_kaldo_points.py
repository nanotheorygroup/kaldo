# Harmonic ----------------------
# Dispersion args
#npoints = 100 # points along path
#pathstring = 'GMKG' # actual path
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
import numpy as np
from scipy import constants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import kaldo

print(kaldo.__file__)
# Replicas
nrep = int(9)
nrep_third = int(2)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k,k,k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])

# Import target frequencies
thz_to_invcm = constants.value('hertz-inverse meter relationship')*1e12/100
# Sheng
radps_to_THz = 1/(2*np.pi)
sheng_freqs=np.loadtxt('sheng/BTE.omega')*radps_to_THz
sheng_vels=np.loadtxt('sheng/BTE.v').reshape((-1, 6, 3))
# Matdyn
cm_to_THz = constants.value('hertz-inverse meter relationship')*1e12/100
matdyn_freqs = np.loadtxt('matdyn/dynamics/FREQ.gp')[:, 1:]/cm_to_THz


### Begin simulation
# Import kALDo
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
                       folder='./',
                       supercell=supercell,
                       only_second=True,
                       third_supercell=third_supercell,
                       is_acoustic_sum=False,
                       format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder='./',
              is_unfolding=unfold_bool,
              storage='memory',)

print("fc and phonons made")

points = np.array([
    [0.000, 0.000, 0.000],
    [0.500, 0.000, 0.000],
    [0.500, 0.500, 0.000],
    [0.000, 0.000, 0.3333333333],
    [0.500, 0.000, 0.3333333333],
    [0.500, 0.500, 0.3333333333],
])
freqs = []
vels = np.zeros((points.shape[0], 3*2, 3), dtype=float)
for i,kpoint in enumerate(points):
    print("\n\nkpoint: ", kpoint)
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=True,
                           is_nac=True,)
    freqs.append(phonon.frequency.squeeze())
    vels[i, ...] = phonon.velocity
    print('\tFrequencies:')
    print("\tkaldo", np.round(freqs[i],5))
    print("\tmatdyn", matdyn_freqs[i,:])
    print("\tsheng", sheng_freqs[i,:])
    print('\tVelocities:')
    for mode in range(6):
        print(f"\tkaldo m{mode} {np.round(vels[i, mode, :], 6)}")
        print(f"\tsheng m{mode} {sheng_vels[i, mode, :]}")

with open('point-compare.txt', 'w') as f:
    for i, kpoint in enumerate(points):
        print("\n\nkpoint: ", kpoint, file=f)
        print('\tFrequencies:', file=f)
        print(f"kaldo {np.round(freqs[i], 4)}", file=f)
        print(f"matdyn {matdyn_freqs[i, :]}", file=f)
        print(f"sheng {sheng_freqs[i,:]}", file=f)
        print('\tVelocities:', file=f)
        for mode in range(6):
            print(f"{mode} {np.round(vels[i, mode, :],5)}", file=f)
            print(f"{mode} {sheng_vels[i, mode, :]}", file=f)
