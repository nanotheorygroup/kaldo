# This plots phononic properties along a path through q-space controlled by
# the special points passed in 1_run_kaldo.py. And systems are plotted based
# on the environment variable "kaldo_systems"
#
# Plots are sent to the "kaldo-outputs" directory
#
# Usage: `python 2_plot_dispersions.py vd`

# You shouldn't need to edit this at all, unless you'd like to change the style
# but is commented thoroughly for reproducibility.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools.plotting_dictionaries import line_dic, label_dic
#plt.style.use('../.kaldo_style_guide')
fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot()

# Directory, system, and plot selection
prefix = os.environ['kaldo_ald']+'/'
output_prefix = os.environ['kaldo_outputs']+'/'
systems_to_run=os.environ['kaldo_systems'].split(',')
plot_dispersion = False; plot_velocity = False
if 'd' in sys.argv[1]:
    plot_dispersion = True
if 'v' in sys.argv[1]:
    plot_velocity = True

# First, dispersion plot.
if plot_dispersion:
    print('Plotting dispersion ..')

    freq_max = 0
    lin = []; lab = []
    for system in systems_to_run:
        # Construct file path, and get line style from dics
        # imported from tools/plotting_dictionaries.py
        print('\t Adding {}'.format(system))
        infold = prefix+system+'/dispersion/'
        color = line_dic[system[0]]
        ls = line_dic[system[1]]
        lw = 1.3 if (not ls == ':') else 3

        # Load data
        q=np.loadtxt(infold+'q')
        bands=np.loadtxt(infold+'dispersion')

        # Plot
        ax.plot(q, bands, color=color, linestyle=ls, label=system, lw=lw)

        # Add artists for custom legend
        lin.append(Line2D([], [], color=color, linestyle=ls, linewidth=5))
        lab.append(system)

        # Track maximums
        if bands.max() > freq_max:
            freq_max = bands.max()

    # General use -- todo: move to our style guide
    ax.tick_params(axis='both', labelsize=20, color='k')
    ax.grid('x', color='k', alpha=0.5, zorder=-1)
    ax.grid('y', 'major', color='k', alpha=0.5, zorder=-1)

    # Legend from artists made in for loop
    leg = ax.legend(lin, lab, loc='center right', fontsize=18, framealpha=1)
    leg.set_title('Supercell + Unfold', prop={'size': 20})

    # Custom x-ticks/labels from kaldo_ald text files
    xlabels = np.loadtxt(infold+'point_names', dtype='str')
    xticks = np.loadtxt(infold+'Q_val')
    ax.set_xticks(xticks, xlabels)
    ax.set_xlabel('Q-vector', fontsize=20, color='k')
    ax.set_xlim([0, 1])

    # Arbitrary y-ticks good for silicon
    ax.set_yticks([0, 4, 8, 12, 16], [0, 4, 8, 12, 16])
    ax.set_ylabel(label_dic['frequency'], fontsize=20, color='k')
    ax.set_ylim([0, freq_max*1.05])

    ax.set_title('Crystal Silicon - Phonon Dispersion', fontsize=24, color='k')
    plt.savefig(output_prefix+'dispersions.png')

## Plot velocity
if plot_velocity:
    print('Plotting velocities ..')
    if plot_dispersion: #wipe figure
        fig.clf(True); ax = fig.add_subplot()

    vel_max = 0
    lin = []; lab = []
    for system in systems_to_run:
        # Construct file path, and get line style from dics
        # imported from tools/plotting_dictionaries.py
        print('\t Adding {}'.format(system))
        infold = prefix+system+'/dispersion/'
        color = line_dic[system[0]]
        ls = line_dic[system[1]]
        lw = 1.3 if (not ls==':') else 3

        # Load data
        q=np.loadtxt(infold+'q')
        vels=np.loadtxt(infold+'velocity_norm')

        # Plot
        ax.plot(q, vels, color=color, linestyle=ls, label=system, lw=lw)

        # Add artists for custom legend
        lin.append(Line2D([], [], color=color, linestyle=ls, linewidth=5))
        lab.append(system)

        # Track maximum
        if vels.max()>vel_max:
            vel_max=vels.max()

    # General use -- todo: move tick+label font+color to our style guide
    ax.tick_params(axis='both', labelsize=20, color='k')
    ax.grid('x', color='k', alpha=0.5, zorder=-1)
    ax.grid('y', 'major', color='k', alpha=0.5, zorder=-1)

    # Legend from artists made in for loop
    leg = ax.legend(lin, lab, loc='upper center', fontsize=18)
    leg.set_title('Supercell + Unfold', prop={'size': 20})

    # Custom x-ticks/labels from kaldo_ald text files
    xlabels = np.loadtxt(infold + 'point_names', dtype='str')
    xticks = np.loadtxt(infold + 'Q_val')
    ax.set_xticks(xticks, xlabels)
    ax.set_xlabel('Q-vector', fontsize=20, color='k')
    ax.set_xlim([0, 1])

    # Arbitrary y-ticks good for silicon
    #ax.set_yticks([0, 4, 8, 12, 16], [0, 4, 8, 12, 16])
    ax.set_ylabel(label_dic['velocity'], fontsize=20, color='k')
    ax.set_ylim([0, vel_max*1.05])

    ax.set_title('Crystal Silicon - Phonon Velocity (norm)', fontsize=24, color='k')
    plt.savefig(output_prefix+'velocities.png')
