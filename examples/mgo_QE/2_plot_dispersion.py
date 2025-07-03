import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from palettable.colorbrewer.qualitative import Set1_6
from palettable.cartocolors.qualitative import Prism_6 # note you can reverse order with {set}_r
set1 = Set1_6.hex_colors
plt.style.use('/home/nwlundgren/.config/matplotlib/matplotlibrc')
pathstring = 'GXUGLW'
unfold_bool = True
outfold = 'plots/'

# special points with 200 points
# G   X  U  G   L   W
# 0  49  66 119 163 199
# G   X  K  G   L   W
# 0  49  79 127 166 199
# -------------------------------
special_points = [point for point in pathstring.upper()]
# point_coordinates = []
# point_coordinates.append(bp.special_points[point])
# for coord,point in zip(coord_list,special_points):
#     condition = (kpts==coord).all(axis=1)
#     results = np.where(condition)[0]
#     if results.size > 1:
#         print('error on ', point)
#     print(f"{point=} - {coord} - index: {np.where(condition)[0]}")
# special_indices = np.array([0, 49, 66, 119, 163, 199], dtype=int) * 1/199
special_indices = np.array([0,  49,  79, 127, 166, 199], dtype=int) * 1/199

#### FIGURE 1
fig = plt.figure(figsize=(16,12))
grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.2)
ax0 = fig.add_subplot(grid[:, 0])
ax1 = fig.add_subplot(grid[0, 1])
ax2 = fig.add_subplot(grid[1, 1])

sheng_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-sheng')
none_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
matdyn_x = np.loadtxt('matdyn/dispersion/FREQ.gp')[:, 0]
matdyn_x /= matdyn_x.max()
matdyn_y = np.loadtxt('matdyn/dispersion/FREQ.gp')[:, 1:] / 33.356
for column in range(sheng_y.shape[1]):
    ax0.scatter(kaldo_x, sheng_y[:, column], color=set1[1], s=12, zorder=1, alpha=0.5)
    ax0.scatter(kaldo_x, none_y[:, column], color=set1[2], s=15, marker='X', zorder=0, alpha=0.5)
    ax0.scatter(matdyn_x, matdyn_y[:, column], color=set1[0], s=12, marker='D', zorder=1, alpha=0.35)
phonopy_data = np.loadtxt('phonopy/with_NAC/MgO_band_with_NAC.dat')
phonopy_x = phonopy_data[:, 0] / phonopy_data[:, 0].max()
phonopy_y = phonopy_data[:, 1]
ax0.scatter(phonopy_x, phonopy_y, color=set1[3], marker='P', s=12, zorder=2, alpha=0.5)
ax0.set_xticks(special_indices, labels=special_points)
ax0.set_ylabel('Frequency (THz)')
ax0.set_title('Phonon Dispersion')

sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-sheng')
none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
for column in range(sheng_y.shape[1]):
    ax1.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
    ax1.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
ax1.set_xticks(special_indices, labels=special_points)
ax1.set_ylabel('Group Velocity (km/s)')
ax1.set_title('GV Norms', fontsize=20)

sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vels-sheng')
none_y = np.loadtxt(f'{outfold}/{pathstring}.vels-none')
kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
for column in range(sheng_y.shape[1]):
    ax2.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
    ax2.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
ax2.set_xticks(special_indices, labels=special_points)
ax2.set_ylabel('Group Velocity (km/s)')
ax2.set_title('GV by Component', fontsize=20)

lines = [Line2D([],[], lw=0, marker='o', markersize=8, color=set1[1]),
         Line2D([],[], lw=0, marker='X', markersize=8, color=set1[2]),
         Line2D([],[], lw=0, marker='D', markersize=8, color=set1[0]),
         Line2D([],[], lw=0, marker='P', markersize=8, color=set1[3]),
        ]
labels = ['kALDo', 'None', 'Matdyn', 'Phonopy']
ax1.legend(lines, labels, title='Correction', bbox_to_anchor=(1.2, 1.25))
plt.suptitle('MgO NAC Corrections with Linear Response', fontweight='bold', fontsize=25)
plt.savefig(f'{outfold}/{pathstring}.dispersion.png')

# #### FIGURE 2
# fig = plt.figure(figsize=(16,12))
# grid = plt.GridSpec(2, 2, wspace=0.15, hspace=0.2)
# ax0 = fig.add_subplot(grid[:, 0])
# ax1 = fig.add_subplot(grid[0, 1])
# ax2 = fig.add_subplot(grid[1, 1])
#
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-none')
# kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
# for column in range(sheng_y.shape[1]):
#     ax0.scatter(kaldo_x, sheng_y[:, column], color=set1[1], s=15, zorder=1)
#     ax0.scatter(kaldo_x, none_y[:, column], color=set1[2], s=12, marker='x', zorder=0)
# ax0.set_xticks(special_indices, labels=special_points)
# ax0.set_ylabel('Frequency (THz)')
# ax0.set_title('Phonon Dispersion')
# phonopy_data = np.loadtxt('phonopy/with_NAC/MgO_band_with_NAC.dat')
# phonopy_x = phonopy_data[:, 0]
# phonopy_y = phonopy_data[:, 1]
# ax0.scatter(phonopy_x, phonopy_y, color=set1[3], s=100, marker='P', zorder=2, alpha=0.35)
#
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-none')
# kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
# for column in range(sheng_y.shape[1]):
#     ax1.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[0], s=15, zorder=1)
#     ax1.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
#     ax1.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
# ax1.set_xticks(special_indices, labels=special_points)
# ax1.set_ylabel('Group Velocity (km/s)')
# ax1.set_title('GV Norms', fontsize=20)
#
# sheng_y = np.loadtxt(f'{outfold}/{pathstring}.vels-sheng')
# none_y = np.loadtxt(f'{outfold}/{pathstring}.vels-none')
# kaldo_x = np.linspace(0, 1, sheng_y.shape[0])
# for column in range(sheng_y.shape[1]):
#     ax2.scatter(kaldo_x/kaldo_x.max(), sheng_y[:, column]/10, color=set1[1], s=15, zorder=1)
#     ax2.scatter(kaldo_x / kaldo_x.max(), none_y[:, column]/10, color=set1[2], s=12, marker='x', zorder=0)
# ax2.set_xticks(special_indices, labels=special_points)
# ax2.set_ylabel('Group Velocity (km/s)')
# ax2.set_title('GV by Component', fontsize=20)
#
# lines = [Line2D([],[], lw=5, color=set1[1]),
#          Line2D([],[], lw=0, marker='x', color=set1[2]),
#         ]
# labels = ['Sheng', 'None']
# ax1.legend(lines, labels, title='Correction', bbox_to_anchor=(1.2, 1.25))
# plt.suptitle('MgO NAC Corrections with Linear Response',)
# plt.savefig(f'{outfold}/{pathstring}.dispersion.png')