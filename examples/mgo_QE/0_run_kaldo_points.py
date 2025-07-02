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
# Replicas
nrep = int(9)
nrep_third = int(2)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k,k,k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])

# Import target frequencies
thz_to_invcm = constants.value('hertz-inverse meter relationship')*1e12/100
qe_data = 'espresso_fcs'
# Sheng
radps_to_THz = 1/(2*np.pi)
sheng_data=np.loadtxt('sheng/BTE.omega')*radps_to_THz
# Matdyn
cm_to_THz = constants.value('hertz-inverse meter relationship')*1e12/100
matdyn_data = np.loadtxt('matdyn/FREQ.gp')[:, 1:]/cm_to_THz

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
    print("kpoint: ", kpoint)
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=phonons.is_unfolding,
                           is_nac=True,)
    freqs.append(phonon.frequency.squeeze())
    vels[i, ...] = phonon.velocity
    print("kaldo", freqs[-1])
    #print("matdyn", matdyn_data[i,:])
    print("sheng", sheng_data[i,:])
print(freqs)
