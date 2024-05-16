# This plots phononic properties as sampled by the k-pt grid we recieved in
# 1_run_kaldo.py. The plots are made by specifying a property like "bandwidth"
# to the script, and compares systems in the environment variable "kaldo_systems"
#
# Plots are sent to the "kaldo-outputs" directory titled "<property>.png"
#
# Usage: `python 4_plot_phonon_data.py <property>`
#
# Currently these need to match those selected in 1_run_kaldo.py
# todo: Do we make it environment variable?
k = 7
temperature = 300
is_classic = False

# You shouldn't need to edit this at all, unless you'd like to change the style
# but is commented thoroughly for reproducibility.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools.plotting_dictionaries import scatter_dic, stats_bool_dic,\
     scale_dic, label_dic
#from kaldo.helpers.storage import get_folder_from_label as gfml
#from kaldo.helpers.storage import DEFAULT_STORE_FORMATS
#plt.style.use('../.kaldo_style_guide.mpl')
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot()

# Retrieve systems to plot and quantities on axes.
systems_to_run = os.environ['kaldo_systems'].split(',')
to_plot_x = 'frequency'
to_plot_y = sys.argv[1]

## Build path for data using parameters + script arguments #######
# Environment variables
prefix = os.environ['kaldo_ald']+'/'
kaldo_output_folder = os.environ['kaldo_outputs']+'/'
# Set up k-grid specific file path
kpts, kptfolder = [k, k, k], '{}_{}_{}/'.format(k,k,k)
# Set up statistics specific file path
if is_classic:
    tempstats = '{}/classic/'.format(temperature)
else:
    tempstats = '{}/quantum/'.format(temperature)
tempstats_x = '' if (not stats_bool_dic[to_plot_x]) else tempstats
tempstats_y = '' if (not stats_bool_dic[to_plot_y]) else tempstats
##################################################################

# Plot everything, add artists to legend, keep running total of maximums
lin = []; lab = [];
xmin = 1e-2; xmax = 0
ymin = 1e-2; ymax = 0
scale = 'linear' if not (to_plot_y in scale_dic.keys()) else scale_dic[to_plot_y] # important for log scale
for system in systems_to_run:
    # todo: control how props behave, e.g. vector quantities vs scalars
    # x-axis:
    abscicca = np.load(prefix+system+'/'+kptfolder+tempstats_x+to_plot_x+'.npy')

    # y-axis:
    if to_plot_y=='phase_space':
        ordinate = np.load(prefix+system+'/'+kptfolder+tempstats_y+'_ps_gamma.npy')[:,0]
    elif to_plot_y=='velocity':
        ordinate = np.linalg.norm(\
            np.load(prefix+system+'/'+kptfolder+tempstats_y+to_plot_y+'.npy'), axis=-1)
    else:
        ordinate = np.load(prefix+system+'/'+kptfolder+tempstats_y+to_plot_y+'.npy')

    # Pull style info
    color = np.array(scatter_dic[system[0]])
    marker = scatter_dic[system[1]]

    # Plot
    ax.scatter(abscicca, ordinate, color=color, marker=marker, s=20, alpha=0.6, zorder=1)

    # Add artist info for legend
    lin.append(Line2D([], [], color=color, marker=marker, linewidth=0, markersize=15, markeredgewidth=5)); lab.append(system)

    # Track min/max
    xmin = xmin if (xmin < abscicca.min()) else abscicca.min()
    if (scale == 'linear'):
        ymin = ymin if (ymin < ordinate.min()) else ordinate.min()
    else:
        nonzeros = ordinate.flatten().nonzero()
        logmin = ordinate.flatten()[nonzeros].min()
        ymin = ymin if (ymin < logmin) else logmin
    xmax = xmax if (xmax > abscicca.max()) else abscicca.max()
    ymax = ymax if (ymax > ordinate.max()) else ordinate.max()

# General use
ax.tick_params(axis='both', labelsize=20, color='k')
ax.grid('x', color='k', alpha=0.5, zorder=-1)
ax.grid('y', 'major', color='k', alpha=0.5, zorder=-1)

# Set scale based on dictionary
scale = 'linear' if not (to_plot_y in scale_dic.keys()) else scale_dic[to_plot_y]
ax.set_yscale(scale)

# Set plot boundaries using data tracked in for loop
ax.set_xlim([xmin * 0.9, xmax * 1.1])
ax.set_ylim([ymin * 0.9, ymax * 1.1])

# Create custom legend
leg = ax.legend(lin, lab, loc='lower right', fontsize=18, framealpha=1)
leg.set_title('Supercell + Unfold', prop={'size': 20})

# Set title based on property name
title = ' '.join(to_plot_y.split('_')).title()
ax.set_title('Crystal Silicon - '+title, fontsize=28, color='k')

# Set label based on dictionary
xlabel = label_dic[to_plot_x]
ylabel = label_dic[to_plot_y]
ax.set_xlabel(xlabel, fontsize=28, color='k')
ax.set_ylabel(ylabel, fontsize=28, color='k')

# Save figure
figurename = kaldo_output_folder+'/'+to_plot_y+'.png'
plt.savefig(figurename)