import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.observables.harmonic_with_q import HarmonicWithQ
import matplotlib.pyplot as plt
# fig=plt.figure()
# ax=fig.add_subplot(projection='3d')
#np.set_printoptions(linewidth=250, suppress=True, precision=2)

q_of_interest = np.array([0.5, 0.25, 0.75])
qdot_interest = q_of_interest / np.linalg.norm(q_of_interest)

shengdata = np.loadtxt('qsvs.txt')
sheng_qs = shengdata[:, :3]
sheng_vs = shengdata[:, 3:]

systems = ['3u', '3n', '8n']
fc3 = ForceConstants.from_folder(
    folder='kaldo_inputs/3x3x3/',
    supercell=[3,3,3],
    only_second=True,
    is_acoustic_sum=True,
    format='shengbte-qe')
fc8 = ForceConstants.from_folder(
    folder='kaldo_inputs/8x8x8/',
    supercell=[8,8,8],
    only_second=True,
    is_acoustic_sum=True,
    format='shengbte-qe')

xmax = 0
ymax = 0
with open('velocity_by_q.txt', 'w') as fout:
    for q, vel_s in zip(sheng_qs, sheng_vs):
        q_hat = np.abs(qdot_interest.dot(q)) / np.linalg.norm(q)

        vel_3u = np.linalg.norm(HarmonicWithQ(q_point=q, second=fc3.second, is_unfolding=True).velocity, axis=-1).flatten()
        vel_3n = np.linalg.norm(HarmonicWithQ(q_point=q, second=fc3.second, is_unfolding=False).velocity, axis=-1).flatten()
        vel_8n = np.linalg.norm(HarmonicWithQ(q_point=q, second=fc8.second, is_unfolding=False).velocity, axis=-1).flatten()
        #vel_8u = np.linalg.norm(HarmonicWithQ(q_point=q, second=fc8.second, is_unfolding=True).velocity, axis=-1).flatten()

        diffs = [np.linalg.norm((vel_s - vxx)) for vxx in [vel_3u, vel_3n, vel_8n]]

        # ax.scatter([q_hat, ]*6, vel_3u, color='r')
        # ax.scatter([q_hat, ]*6, vel_8n, color='g')
        # ax.scatter([q_hat, ]*6, vel_s*10, color='b')

        stack = np.hstack([q, diffs])
        print(np.array2string(stack, max_line_width=None, precision=5, separator=' ')\
              .replace('[', '')\
              .replace(']', ''), file=fout)

        if np.max(stack)>ymax:
            ymax=np.max(stack)
        if q_hat > xmax:
            xmax = q_hat

# plt.xlim([0, xmax])
# plt.ylim([0, ymax])
# plt.savefig('compare.png')