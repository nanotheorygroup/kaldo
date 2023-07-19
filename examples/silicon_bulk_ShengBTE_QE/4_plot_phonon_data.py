import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tools.plotting_dictionaries import scatterstyle_dic, stats_bool_dic,\
     scale_dic, label_dic, filename_dic
#from kaldo.helpers.storage import get_folder_from_label as gfml
#from kaldo.helpers.storage import DEFAULT_STORE_FORMATS

# Currently these need to match those selected in 1_run_kaldo.py
k = 7
temperature = 300
is_classic = False

# Control which systems to plot and what quantities you're plotting
prefix = os.environ['kaldo_ald']+'/'
systems_to_run = os.environ['kaldo_systems'].split(',')
kaldo_output_folder = os.environ['kaldo_outputs']
to_plot_x = 'frequency'
to_plot_y = sys.argv[1]

kpts, kptfolder = [k, k, k], '{}_{}_{}/'.format(k,k,k)
if is_classic:
    tempstats = '{}/classic/'.format(temperature)
else:
    tempstats = '{}/quantum/'.format(temperature)


# Instantiate figure + set up styles, pathnames, etc.
plt.style.use('../.kaldo_style_guide.mpl')
fig = plt.figure(figsize=(18, 12)); ax = fig.add_subplot()
tempstats_x = '' if (not stats_bool_dic[to_plot_x]) else tempstats
tempstats_y = '' if (not stats_bool_dic[to_plot_y]) else tempstats
figurename = kaldo_output_folder+'/'+to_plot_y+'.png'
scale = 'linear' if not (to_plot_y in scale_dic.keys()) else scale_dic[to_plot_y]
xlabel = label_dic[to_plot_x]
ylabel = label_dic[to_plot_y]

# Plot everything, add artists to legend, keep running total of maximums
lin = []; lab = []; ymax = 0; xmax = 0
for system in systems_to_run:
    ''' # to do - control how props behave
    '''
    abscicca = np.load(prefix+system+'/'+kptfolder+tempstats_x+to_plot_x+'.npy')
    if to_plot_y=='phase_space':
        ordinate = np.load(prefix+system+'/'+kptfolder+tempstats_y+'_ps_gamma.npy')[:,0]
    elif to_plot_y=='velocity':
        ordinate = np.linalg.norm(\
            np.load(prefix+system+'/'+kptfolder+tempstats_y+to_plot_y+'.npy'), axis=-1)
    else:
        ordinate = np.load(prefix+system+'/'+kptfolder+tempstats_y+to_plot_y+'.npy')

    color = np.array(scatterstyledic[system[0]])
    marker = scatterstyledic[system[1]]
    ax.scatter(abscicca, ordinate, color=color, marker=marker, s=5, alpha=0.5, zorder=1)
    lin.append(Line2D([], [], color=color, marker=marker)); lab.append(system)
    xmax = xmax if (xmax > abscicca.max()) else abscicca.max()
    ymax = ymax if (ymax > ordinate.max()) else ordinate.max()

# Add axis labels, units, etc.
ax.grid('x', color='k', alpha=0.5, zorder=0)
ax.grid('y', 'major', color='k', alpha=0.5, zorder=0)

ax.set_yscale(scale)

ax.set_xlim([0, xmax * 1.05])
ax.set_ylim([0, ymax * 1.05])

ax.legend(lin, lab)

title = ' '.join(to_plot_y.split('_')).title()
ax.set_title('Crystal Silicon '+title, fontsize=24, color='k')
ax.set_xlabel(xlabel, fontsize=20, color='k')
ax.set_ylabel(ylabel, fontsize=20, color='k')

plt.savefig(figurename)
