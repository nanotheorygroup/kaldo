import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotting_dictionaries import linestyledic

fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot()

prefix= 'data/'

bands_max = 0
lin = []; lab = []
for system in ['1u', '1n', '3u', '3n', '8u', '8n']:
    infold = prefix+system+'/dispersion/'
    color = linestyledic[system[0]]
    ls = linestyledic[system[1]]

    # Load data
    q=np.loadtxt(infold+'q')
    bands=np.loadtxt(infold+'dispersion')

    # Plot
    ax.plot(q, bands, color=color, linestyle=ls, label=system)
    lin.append(Line2D([], [], color=color, linestyle=ls, linewidth=4))
    lab.append(system)
    if bands.max()>bands_max:
        bands_max=bands.max()

ax.legend(lin, lab, loc='center right')
xlabels = np.loadtxt(infold+'point_names', dtype='str')
xticks = np.loadtxt(infold+'Q_val') - 0.02; xticks[-1] +=0.02
ax.set_yticks([0, 4, 8, 12, 16], [0, 4, 8, 12, 16], color='k', fontsize=18)
ax.set_xticks(xticks, xlabels, fontsize=20, color='k')
ax.grid('x', color='k', alpha=0.5, zorder=-1)
ax.grid('y', 'major', color='k', alpha=0.5, zorder=-1)
ax.set_xlabel('Q-vector', fontsize=20, color='k')
ax.set_ylabel(r'$\omega (THz)$', fontsize=20, color='k')
ax.set_xlim([0, 1])
ax.set_ylim([0, bands_max*1.05])
ax.set_title('Crystal Silicon - Dispersion', fontsize=24, color='k')
plt.savefig('dispersions.png')