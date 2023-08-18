import os
import numpy as np
from scipy import constants
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.observables.harmonic_with_q import HarmonicWithQ
np.set_printoptions(suppress=True, linewidth=200)
os.environ['CUDA_VISIBLE_DEVICES']=""

#### Si SPECIFIC ##########################
k = 7
rep = 3
kpts = [k, k, k]
supercell = np.array([rep, rep, rep])
###########################################
#### InSe SPECIFIC ##########################
#k = 13
#rep = 9
#kpts = [k, k, 1]
#supercell = np.array([rep, rep, 1])
###########################################

## Hardcoded Parameters - eg pathnames
thz_to_icm = 1e12*constants.value('hertz-inverse meter relationship')/100
temperature = 300
unfold_bool = True
ald_folder='./'
fcs_folder='forces'
qpt_folder = 'kaldo.out.per_qpt/'
dyn_folder = 'kaldo.out.dyn/'
txtpath = 'path.out.txt'
freqs_fn = 'kaldo.out.freq.gp'
vels_fn = 'kaldo.out.vels.gp'
repacked_fn = 'kaldo.out'
to_calculate = True

# Set up qptdata
if not os.path.isdir(qpt_folder):
    os.mkdir(qpt_folder)
#else:
#    print('Data already exists in qpt folder')
#    to_calculate=False

# Set up dynamical data
if not os.path.isdir(dyn_folder):
    os.mkdir(dyn_folder)

forceconstant = ForceConstants.from_folder(folder=fcs_folder,
                           supercell=supercell,
                           only_second=True,
                           is_acoustic_sum=True,
                           format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  is_unfolding=unfold_bool,
                  temperature=temperature,
                  folder=ald_folder,
                  storage='numpy')


if to_calculate:
    print('\n\n\n!! kALDo FC object created, executing calculations along path')
    freqs_file = open(freqs_fn, 'w')
    vels_file = open(vels_fn, 'w')
    qs_to_run = np.loadtxt(txtpath)
    nqs = qs_to_run.shape[0]
    for i,q in enumerate(qs_to_run):
        if (i%20)==0:
            print('\t{} of {}'.format(i, nqs))
        hrm = HarmonicWithQ(q_point=q, supercell=supercell,
                second=forceconstant.second, is_unfolding=phonons.is_unfolding)
        vel = hrm.velocity.reshape((-1,))
        vel_str = np.array2string(vel, precision=5).strip('[]')
        vels_file.write(str(i)+' '+vel_str+'\n')
        freq = hrm.frequency*thz_to_icm
        freq_str = np.array2string(freq, precision=5).strip('[]')
        freqs_file.write(str(i)+' '+freq_str+'\n')
    freqs_file.close()
    vels_file.close()
    print('\t Calculations complete')

print('!! Attempting repackaging ..')
gamma = qpt_folder+'0.0_0.0_0.0.npy'
filenames = [fn for fn in os.listdir(qpt_folder) if fn != gamma]
print('\t Found {} qpt files '.format(len(filenames)))
repack = np.load(gamma)
for f in filenames:
    repack = np.append(repack, np.load(qpt_folder+f))
np.save(repacked_fn, repack)
print('\t Completed packing')
print('\t Final shape: {}'.format(repack.shape))
print('!! Exiting ..')
print('\n\n')
