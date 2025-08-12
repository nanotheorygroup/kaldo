import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
plt.style.use('../.kaldo_style_guide.mpl')

pathstring = 'GXUGLW'
unfold_bool = True
outfold = 'plots/'
# special points with 200 points
# G   X  U  G   L   W
# 0  49  66 119 163 199
# -------------------------------
special_points = [point for point in pathstring.upper()]
special_indices = np.array([0, 49, 66, 119, 163, 199], dtype=int) * 1/199

#### FIGURE 1
fig = plt.figure(figsize=(18,12))
grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.4)
ax0, ax1 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
red="#E41A1C"
blue="#377EB8"
green='#4DAF4A'

# Dispersion
uncorrected_y = np.loadtxt(f'{outfold}/{pathstring}.freqs')
corrected_y = np.loadtxt(f'{outfold}/{pathstring}.freqs-nac')
uncorrected_x = corrected_x = np.linspace(0, 1, corrected_y.shape[0])
matdyn_y = np.loadtxt('tools/refs.freq')[:, 1:] / 33.356 # cm-1
matdyn_x = np.arange(matdyn_y.shape[0])/(matdyn_y.shape[0])
for column in range(corrected_y.shape[1]):
    ax0.scatter(corrected_x, corrected_y[:, column], color=red, marker='o', s=12, zorder=2, alpha=0.5)
    ax0.scatter(uncorrected_x, uncorrected_y[:, column], color=blue, marker='X', s=12, zorder=1, alpha=0.5)
    ax0.scatter(matdyn_x, matdyn_y[:, column], color=green, marker='D', s=12, zorder=0, alpha=0.5)
ax0.set_xticks(special_indices, labels=special_points,)
ax0.set_ylabel('Frequency (THz)', fontsize=28)
ax0.set_title('Phonon Dispersion', fontsize=32)
ax0.set_xlim([0,1])

# Velocity Plot
corrected_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms-nac')
uncorrected_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
for column in range(corrected_y.shape[1]):
    ax1.scatter(corrected_x, corrected_y[:, column], color=red, s=12, zorder=2, alpha=0.5)
    ax1.scatter(uncorrected_x, uncorrected_y[:, column], color=blue, s=12, zorder=1, alpha=0.5)
ax1.set_xticks(special_indices, labels=special_points)
ax1.set_ylabel('Group Velocity (km/s)', fontsize=28)
ax1.set_title('GV Norms', fontsize=32)

labels = ['NAC', 'Uncorrected', 'Matdyn',]
markers = ['o', 'X', 'D']
colors = [red, blue, green]
lines = [Line2D([],[], lw=0, marker=marker, markersize=8, color=color) \
            for marker, color in zip(markers, colors)]
ax1.legend(lines, labels, bbox_to_anchor=(0.75, 0.8))
fig.savefig(f'{outfold}/{pathstring}.dispersion.png')

