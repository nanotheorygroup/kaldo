import matplotlib.pyplot as plt
import numpy as np

# Here is a quick plotting tool to compare some features of a-Si.
# There are tools to make quick plots comparing different observables
# against each other or just to compare different phononic observables
# by frequency.


# Adjust pyplot defaults
params = {'figure.figsize': (12, 8),
         'legend.fontsize': 20,
         'axes.labelsize':  20,
         'axes.titlesize':  24,
         'xtick.labelsize': 18,
         'ytick.labelsize': 18}
plt.rcParams.update(params)

# Enter desired plots as a dictionary key with None as its value
# Options:
# dos, bandwidth, participation, diffusivity
# conductivity, cumulative_conductivity
desired_plots = {'dos': None,
                 'bandwidth': None}

# Enter the systems you want to compare
# {size (string) : [list of concentrations (floats)]}
desired_comparisons = {'1728': [0.1] }
                    # '13824': [0.1]}

def plot_pDOS(size, conc, ax):
    folder = 'structures/'+str(size)+'_atom/aSiGe_C'+str(int(conc*100))
    frequencies = np.load(folder+'/frequency.npy')
    histogram = np.histogram(frequencies, bins=200)
    ax.plot(frequencies, histogram, label=size+'[Ge]: '+str(int(100*conc)))

def plot_observable_against_frequency(size, conc, observable, ax):
    folder = 'structures/'+str(size)+'_atom/aSiGe_C'+str(int(conc*100))
    frequencies = np.load(folder+'/frequency.npy')
    observable = np.load(folder+'/'+observable+'.npy')
    ax.plot(frequencies, observable, label=size+'[Ge]: '+str(int(100*conc)))

def plot_obs_against_obs(size, conc, observable1, observable2, ax):
    folder = 'structures/' + str(size) + '_atom/aSiGe_C' + str(int(conc * 100))
    observable1 = np.load(folder+'/'+observable1+'.npy')
    observable2 = np.load(folder+'/'+observable2+'.npy')
    ax.plot(observable1, observable2, label=size+'[Ge]: '+str(int(100*conc)))

# Plot figures
plots = desired_plots.keys()
for size in desired_comparisons.keys():
    concentrations = desired_comparisons[size]
    for conc in concentrations:
        for obs in plots:
            if obs=='dos':
                if desired_plots[obs] == None:
                    plt.figure()
                    desired_plots[obs] = [plt.gca(), plt.gcf()]
                    plt.title('Density of States')
                    plt.xlabel('Frequency (THz)')
                    plt.ylabel('Intensity')
                ax = desired_plots['dos'][0]
                plot_pDOS(size, conc, ax)
            elif obs=='cumulative_conductivity':
                if desired_plots[obs] == None:
                    plt.figure()
                    desired_plots[obs] = [plt.gca(), plt.gcf()]
                    plt.title('Cumulative Conductivity')
                    plt.xlabel('Frequency (THz)')
                    plt.ylabel(r'\kappa _{nm} (W/m/K)')
                ax = desired_plots['dos'][0]
                plot_pDOS(size, conc, ax)
            else:
                if desired_plots[obs] == None:
                    plt.figure()
                    desired_plots[obs] = [plt.gca(), plt.gcf()]
                    plt.title(obs)
                    plt.xlabel('Frequency (THz)')
                    if obs=='diffusivity':
                        plt.ylabel(r'D_{nm} (mm^2/s)')
                    if obs=='conductivity':
                        plt.ylabel(r'\kappa _{nm} (W/m/K)')
                    if obs=='participation':
                        plt.ylabel('Participation Ratio (%)')
                    if obs=='bandwidth':
                        plt.ylabel('Bandwidth (1/ps)')

                ax = desired_plots[obs][0]
                plot_observable_against_frequency(size, conc, obs, ax)

# Save figures
for key in plots:
    figure = desired_plots[key][1]
    figure.savefig(key+'.png')