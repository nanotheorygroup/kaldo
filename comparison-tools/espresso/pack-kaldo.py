import numpy as np
from ase.io import read
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import os
os.environ['CUDA_VISIBLE_DEVICES']=""

k = 7
rep = 3
temperature = 300
unfold_bool = True
kpts = [k, k, k]
supercell = np.array([rep, rep, rep])
fcs_folder='forces'
ald_folder='./'
qpt_folder = 'kaldo.out.per_qpt/'
txtpath = 'path.out.txt'
repacked_md = 'md.out'
repacked_fn = 'kaldo.out'
atoms = read(fcs_folder+'/POSCAR', format='vasp')
to_calculate = True

if not os.path.isdir(qpt_folder):
    os.mkdir(qpt_folder)
else:
    print('Data already exists in qpt folder')
    to_calculate=False

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
    print('\n\n\n!! kALDo objects created, executing calculations along path')
    print('Working ...')
    qs_to_run = np.loadtxt(txtpath)
    nqs = qs_to_run.shape[0]
    print('\t0 of {}'.format(nqs))
    for i,q in enumerate(qs_to_run):
        if (i%20)==0:
            print('\t{} of {}'.format(i, nqs))
        hrm = HarmonicWithQ(q_point=q, supercell=supercell,
                second=forceconstant.second, is_unfolding=phonons.is_unfolding)
        hrm.frequency # cause kALDo to calculate the frequency at q
        #hrm.velocity
    print('!! Calculations complete, repackaging ..')

print('!! Attempting packing ..')
gamma = qpt_folder+'0.0_0.0_0.0.npy'
filenames = [fn for fn in os.listdir(qpt_folder) if fn != gamma]
print('\tFound {} per_qpt files ')
repack = np.load(gamma)
for f in filenames:
    repack = np.append(repack, np.load(qpt_folder+f))

np.save(repacked_fn, repack)
print('!! Completed packing')
print('\t Final shape: {}'.format(repack.shape))
print('Exiting ..')
print('\n\n')

# qe_data = np.load(fcs_folder+repacked_md+'.npy')['qvec']
# __, args = np.unique(qe_data, axis=0, return_index=True)
# qs_to_run = qe_data[np.sort(args)] @ cellt
