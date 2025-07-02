import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from palettable.colorbrewer.qualitative import Set1_6
from palettable.cartocolors.qualitative import Prism_6 # note you can reverse order with {set}_r
set1 = Set1_6.hex_colors
plt.style.use('/home/nwlundgren/.config/matplotlib/matplotlibrc')
pathstring = 'GXWGK'
unfold_bool = True
outfold = 'plots/'

# GXWGK - special points with 200 points
# G   X   W   G   K
# 0  53  67 140 199
# -------------------------------
special_points = ['G', 'X', 'W', 'G', 'K']
special_indices = np.array([0, 53, 67, 140, 199], dtype=int) * 1/199

#### FIGURE 1
fig = plt.figure(figsize=(16,12))
grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.08)
ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[1, 0])
ax2 = fig.add_subplot(grid[:, 1])

sheng_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-sheng')
none_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-sheng')
none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
for column in range(3):
    ax0.scatter(sheng_x[:, column], sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
    ax0.scatter(none_x[:, column], none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
ax0.set_xticks([0,2.5,5,7.5,10,12.5], [0, '', 5, '', 10, ''], color='k')
ax0.set_xlim([0,12.5,])
ax0.set_yticks([0,3,6,9,12,15], [0,3,6,9,12,15], color='k')
ax0.set_ylim([0,15])
ax0.set_ylabel(r'$V_{g}$ (km/s)', rotation=70, ha='right', color='k')
note0 = ax0.annotate('Acoustic Modes', xy=(0.7, 0.92),
                     xycoords='axes fraction', fontsize=18, )
note0.set_bbox(dict(facecolor='white', alpha=1, linewidth=0.7, edgecolor='k'))
ax0.set_title(r'$V_{g}$ vs Frequency', color='k')


sheng_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-sheng')
none_x = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-sheng')
none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
for column in range(3,6):
    ax1.scatter(sheng_x[:, column], sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
    ax1.scatter(none_x[:, column], none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
ax1.set_xticks([10,12.5,15,17.5,20,22.5], [10, '', 15, '', 20, ''], color='k')
ax1.set_xlim([10,22.5])
ax1.set_yticks([0,2,4,6], [0,2,4,6], color='k')
ax1.set_ylim([0,6])
ax1.set_xlabel('Frequency (THz)', color='k')
ax1.set_ylabel(r'$V_{g}$ (km/s)', rotation=70, ha='right', color='k')
note1 = ax1.annotate('Optical Modes', xy=(0.7, 0.92),
                     xycoords='axes fraction', fontsize=18, )
note1.set_bbox(dict(facecolor='white', alpha=1, linewidth=0.7, edgecolor='k'))

sheng_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-sheng')
none_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
for column in range(sheng_y.shape[1]):
    ax2.scatter(kaldo_x, sheng_y[:, column], color=set1[1], s=15, zorder=1)
    ax2.scatter(kaldo_x, none_y[:, column], color=set1[2], s=12, marker='x', zorder=0)
ax2.set_xticks(special_indices, labels=special_points, color='k')
ax2.set_yticks([0,5,10,15,20], [0,5,10,15,20], color='k')
ax2.set_ylabel(r'$\nu$ (THz)', rotation=70, ha='right', color='k')
ax2.set_title('Phonon Dispersion')

lines = [Line2D([],[], lw=5, color=set1[1]),
         Line2D([],[], lw=0, marker='x', color=set1[2]),
        ]
labels = ['Sheng', 'None']
ax2.legend(lines, labels, title='Correction', bbox_to_anchor=(0.95, 0.95))
plt.suptitle('MgO Velocity Corrections', fontweight='bold', fontsize=25)
plt.savefig(f'{outfold}/{pathstring}.by_mode.png')
#
# qe_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-qe')
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
# kaldo_x = np.linspace(0, 1, qe_y.shape[0])
# for column in range(qe_y.shape[1]):
#     ax1.scatter(kaldo_x/kaldo_x.max(), qe_y[:, column]/10, color=set1[0], s=15, zorder=1)
#     ax1.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
#     ax1.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
# ax1.set_xticks(special_indices, labels=special_points)
# ax1.set_ylabel('Group Velocity (km/s)')
# ax1.set_title('GV Norms', fontsize=20)
#
# qe_y = np.loadtxt(f'{outfold}/{pathstring}.vels-qe')
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vels-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.vels-none')
# kaldo_x = np.linspace(0, 1, qe_y.shape[0])
# for column in range(qe_y.shape[1]):
#     ax2.scatter(kaldo_x/kaldo_x.max(), qe_y[:, column]/10, color=set1[0], s=15, zorder=1)
#     ax2.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
#     ax2.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
# ax2.set_xticks(special_indices, labels=special_points)
# ax2.set_ylabel('Group Velocity (km/s)')
# ax2.set_title('GV by Component', fontsize=20)
#
# lines = [Line2D([],[], lw=5, color=set1[0]),
#          Line2D([],[], lw=5, color=set1[1]),
#          Line2D([],[], lw=0, marker='x', color=set1[2]),
#         ]
# labels = ['QE', 'Sheng', 'None']
# ax1.legend(lines, labels, title='Correction', bbox_to_anchor=(1.2, 1.25))
# plt.suptitle('MgO NAC Corrections with Linear Response',)
# plt.savefig(f'{outfold}/{pathstring}.dispersion.png')
#
# #### FIGURE 2
# fig = plt.figure(figsize=(16,12))
# grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.2)
# ax0 = fig.add_subplot(grid[:, 0])
# ax1 = fig.add_subplot(grid[0, 1])
# ax2 = fig.add_subplot(grid[1, 1])
#
# qe_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-qe')
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
# kaldo_x = np.linspace(0, 1, qe_y.shape[0])
# for column in range(qe_y.shape[1]):
#     ax0.scatter(kaldo_x, qe_y[:, column], color=set1[0], s=15, zorder=1)
#     ax0.scatter(kaldo_x, sheng_y[:, column], color=set1[1], s=15, zorder=1)
#     ax0.scatter(kaldo_x, none_y[:, column], color=set1[2], s=12, marker='x', zorder=0)
# ax0.set_xticks(special_indices, labels=special_points)
# ax0.set_ylabel('Frequency (THz)')
# ax0.set_title('Phonon Dispersion')
#
# qe_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-qe')
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
# kaldo_x = np.linspace(0, 1, qe_y.shape[0])
# for column in range(qe_y.shape[1]):
#     ax1.scatter(kaldo_x/kaldo_x.max(), qe_y[:, column]/10, color=set1[0], s=15, zorder=1)
#     ax1.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
#     ax1.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
# ax1.set_xticks(special_indices, labels=special_points)
# ax1.set_ylabel('Group Velocity (km/s)')
# ax1.set_title('GV Norms', fontsize=20)
#
# qe_y = np.loadtxt(f'{outfold}/{pathstring}.vels-qe')
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vels-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.vels-none')
# kaldo_x = np.linspace(0, 1, qe_y.shape[0])
# for column in range(qe_y.shape[1]):
#     ax2.scatter(kaldo_x/kaldo_x.max(), qe_y[:, column]/10, color=set1[0], s=15, zorder=1)
#     ax2.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
#     ax2.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
# ax2.set_xticks(special_indices, labels=special_points)
# ax2.set_ylabel('Group Velocity (km/s)')
# ax2.set_title('GV by Component', fontsize=20)
#
# lines = [Line2D([],[], lw=5, color=set1[0]),
#          Line2D([],[], lw=5, color=set1[1]),
#          Line2D([],[], lw=0, marker='x', color=set1[2]),
#         ]
# labels = ['QE', 'Sheng', 'None']
# ax1.legend(lines, labels, title='Correction', bbox_to_anchor=(1.2, 1.25))
# plt.suptitle('MgO NAC Corrections with Linear Response',)
# plt.savefig(f'{outfold}/{pathstring}.dispersion.png')
#
# #### FIGURE 1
# fig = plt.figure(figsize=(16,12))
# grid = plt.GridSpec(2, 2, wspace=0.08, hspace=0.08)
# ax0 = fig.add_subplot(grid[:, 0])
# ax1 = fig.add_subplot(grid[0, 1])
# ax2 = fig.add_subplot(grid[1, 1])
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.freqs')
# matdyn_y = np.loadtxt('dynamics/FREQ.gp')[:,1:] / thz_to_invcm
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax0.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)
#     ax0.scatter(kaldo_x/kaldo_x.max(), matdyn_y[:, column], color='b', lw=2, zorder=10)
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vels')
# kaldonc_y = np.loadtxt(f'{outfold}/{pathstring}.vels.nonac')
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax1.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)
#     ax1.scatter(kaldo_x / kaldo_x.max(), kaldonc_y[:, column], color='g', lw=0.5, zorder=0)
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
# kaldonc_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms.nonac')
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax2.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)
#     ax2.scatter(kaldo_x / kaldo_x.max(), kaldonc_y[:, column], color='g', lw=0.5, zorder=0)
# plt.savefig(f'{outfold}/{pathstring}.disp.png')
# plt.clf()
#
#
# #### FIGURE 2
# fig = plt.figure(figsize=(16,12))
# grid = plt.GridSpec(2, 2, wspace=0.08, hspace=0.08)
# ax0 = fig.add_subplot(grid[:, 0])
# ax1 = fig.add_subplot(grid[0, 1])
# ax2 = fig.add_subplot(grid[1, 1])
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
# kaldonc_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms.nonac')
# kaldo_x = np.loadtxt(f'{outfold}/{pathstring}.freqs')
# for column in range(kaldo_y.shape[1]):
#     ax0.scatter(kaldo_x[:, column], kaldo_y[:, column], color='r', lw=2, zorder=10)
#     ax0.scatter(kaldo_x[:, column], kaldonc_y[:, column], color='g', lw=0.5, zorder=0)
# plt.savefig(f'{outfold}/{pathstring}.disp.png')
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
# kaldonc_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms.nonac')
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax1.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)
#     ax1.scatter(kaldo_x / kaldo_x.max(), kaldonc_y[:, column], color='g', lw=0.5, zorder=0)
# plt.savefig(f'{outfold}/{pathstring}.disp.png')
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.freqs')
# matdyn_y = np.loadtxt('dynamics/FREQ.gp')[:,1:] / thz_to_invcm
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax2.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)
#     ax2.scatter(kaldo_x/kaldo_x.max(), matdyn_y[:, column], color='b', lw=2, zorder=10)
# plt.savefig(f'{outfold}/{pathstring}.vels.png')
# plt.clf()
#
#
# #### FIGURE 3
# fig = plt.figure(figsize=(16,12))
# grid = plt.GridSpec(2, 2, wspace=0.08, hspace=0.08)
# ax0 = fig.add_subplot(grid[:, 0])
# ax1 = fig.add_subplot(grid[0, 1])
# ax2 = fig.add_subplot(grid[1, 1])
# #
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
# kaldo_x = np.loadtxt(f'{outfold}/{pathstring}.freqs')
# for column in range(kaldo_y.shape[1]):
#     ax0.scatter(kaldo_x[:, column], kaldo_y[:, column], lw=2, zorder=10)
# plt.savefig(f'{outfold}/{pathstring}.disp.png')
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax1.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], lw=2, zorder=10)
#
#
# kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.freqs')
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax2.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], lw=2, zorder=10)
# plt.savefig(f'{outfold}/{pathstring}.v_branches.png')
# plt.show()

