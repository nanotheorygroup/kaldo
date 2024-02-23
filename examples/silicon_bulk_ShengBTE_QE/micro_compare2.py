import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from opt_einsum import contract
import spglib
np.set_printoptions(linewidth=150, suppress=True)
q_of_interest = np.array([0.51, 0, 0.51])

systems = ['3u', '3n', '8n']
fc3 = ForceConstants.from_folder(
    folder='kaldo_inputs/3x3x3/',
    supercell=[3,3,3],
    only_second=True,
    is_acoustic_sum=True,
    format='shengbte-qe')



vel_3u =HarmonicWithQ(q_point=q_of_interest, second=fc3.second, is_unfolding=True).velocity.squeeze()
print(vel_3u)
print(np.linalg.norm(vel_3u, axis=-1))
# np.save('velocity', vel_3u)
#
# symdic = spglib.get_symmetry(fc3.atoms)
# rotations = symdic['rotations']
# print(rotations.shape, vel_3u.T.shape)

# print(np.matmul(rotations[3], vel_3u.T).T)
# print(np.linalg.norm(vel_3u, axis=-1))
