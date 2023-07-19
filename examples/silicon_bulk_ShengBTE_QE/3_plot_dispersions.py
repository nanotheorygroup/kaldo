# This plots phononic properties along a path through q-space
# controlled by the special points passed in 1_run_kaldo.py
# Plots are sent to the kaldo-outputs directory

# Usage: python 2_plot_dispersions.py vd
# The argument should be v/d/vd for velocity
# dispersion or both


## Plotting script parameters ############
import os
import sys
prefix = os.environ['kaldo_ald']
output_prefix = os.environ['kaldo_outputs']
systems_to_run=os.environ['kaldo_systems'].split(',')
disp = False; vel = False
if 'd' in sys.argv[1]:
    disp = True
if 'v' in sys.argv[1]:
    vel = True
##########################################
##########################################

# Begin main
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools.plotting_dictionaries import linestyle_dic, label_dic
fig = plt.figure(figsize=(18,12))
ax = fig.add_subplot()

## plot dispersion
print('Plotting dispersion ..')
bands_max = 0
lin = []; lab = []
for system in systems_to_run:
    if not disp:
        continue
    print('\t Adding {}'.format(system))
    infold = prefix+'/'+system+'/dispersion/'
    color = linestyle_dic[system[0]]
    ls = linestyle_dic[system[1]]

    # Load data
    q=np.loadtxt(infold+'q')
    bands=np.loadtxt(infold+'dispersion')

    # Plot
    ax.plot(q, bands, color=color, linestyle=ls, label=system)
    lin.append(Line2D([], [], color=color, linestyle=ls, linewidth=4))
    lab.append(system)
    if bands.max()>bands_max:
        bands_max=bands.max()
# Labels, units, legend etc.
ax.tick_params(axis='both', fontsize=18)
ax.legend(lin, lab, loc='center right')
xlabels = np.loadtxt(infold+'point_names', dtype='str')
xticks = np.loadtxt(infold+'Q_val') - 0.02; xticks[-1] +=0.02
ax.set_xticks(xticks, xlabels)
ax.set_xticklabels(fontsize=20, color='k')

ax.set_yticks([0, 4, 8, 12, 16], [0, 4, 8, 12, 16])
ax.set_yticklabels(fontsize=18, color='k')

ax.grid('x', color='k', alpha=0.5, zorder=-1)
ax.grid('y', 'major', color='k', alpha=0.5, zorder=-1)

ax.set_xlabel('Q-vector', fontsize=20, color='k')
ax.set_ylabel(r'$\omega_{\mu} (THz)$', fontsize=20, color='k')

ax.set_xlim([0, 1])
ax.set_ylim([0, bands_max*1.05])

ax.set_title('Crystal Silicon - Dispersion', fontsize=24, color='k')
if disp:
    plt.savefig(output_prefix+'/dispersions.png')

## Plot velocity
print('Plotting velocities ..')
fig.clf(True); ax = fig.add_subplot()
vel_max = 0
lin = []; lab = []
for system in systems_to_run:
    if not vel:
        continue
    print('\t Adding {}'.format(system))
    infold = prefix+system+'/dispersion/'
    color = linestyle_dic[system[0]]
    ls = linestyle_dic[system[1]]

    # Load data
    q=np.loadtxt(infold+'q')
    vels=np.loadtxt(infold+'velocity_norm')

    # Plot
    ax.plot(q, vels, color=color, linestyle=ls, label=system)
    lin.append(Line2D([], [], color=color, linestyle=ls, linewidth=4))
    lab.append(system)
    if vels.max()>vel_max:
        vel_max=vels.max()

ax.tick_params(axis='both', labelsize=20, color='k')
ax.legend(lin, lab, loc='center right')
xlabels = np.loadtxt(infold+'point_names', dtype='str')
xticks = np.loadtxt(infold+'Q_val') - 0.02; xticks[-1] +=0.02
ax.set_xticks(xticks, xlabels, fontsize=20, color='k')
ax.grid('x', color='k', alpha=0.5, zorder=-1)
ax.grid('y', 'major', color='k', alpha=0.5, zorder=-1)
ax.set_xlabel('Q-vector', fontsize=20, color='k')
ax.set_ylabel(r'$v_{\mu} (\AA / ps)$', fontsize=20, color='k')
ax.set_xlim([0, 1])
ax.set_ylim([0, vel_max*1.05])
ax.set_title('Crystal Silicon - Velocities', fontsize=24, color='k')
if vel:
    plt.savefig(output_prefix+'/velocities.png')
