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
folder = 'workup/'
kaldo_qptdat = 'kaldo_per_qpt/'
repacked_md = 'compiled-matdyn'
repacked_fn = 'compiled-matdyn'
atoms = read(folder+'POSCAR', format='vasp')
cellt = atoms.cell.array.T/(atoms.cell.array[0,2]*2)

forceconstant = ForceConstants.from_folder(folder=folder,
                           supercell=supercell,
                           only_second=True,
                           is_acoustic_sum=True,
                           format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  is_unfolding=unfold_bool,
                  temperature=temperature,
                  folder=folder,
                  storage='numpy')

print('\n\n\n!! kALDo objects created, executing calculations along path')
print('Working ...')
qe_data = np.load(folder+repacked_md+'.npy')['qvec']
__, args = np.unique(qe_data, axis=0, return_index=True)
qs_to_run = qe_data[np.sort(args)] @ cellt
nqs = qs_to_run.shape[0]; print('\t0 of {}'.format(nqs))
for i,q in enumerate(qs_to_run):
    if (i%20)==0:
        print('\t{} of {}'.format(i, nqs))
    hrm = HarmonicWithQ(q_point=q, supercell=supercell,
                second=forceconstant.second, is_unfolding=phonons.is_unfolding)
    hrm.frequency # cause kALDo to calculate the frequency at q

print('!! Calculations complete, repackaging ..')
gamma = kaldo_qptdat+'0.0_0.0_0.0.npy'
filenames = [fn for fn in os.listdir(kaldo_qptdat) if fn != gamma]
print('Compiling {} q-points'.format(len(filenames)+1))
repack = np.load(gamma)
for f in filenames:
    repack = np.append(repack, np.load(kaldo_qptdat+f))

print('!! Completed')
print('!! final shape {}'.format(repack.shape))
np.save(repacked_fn, repack)