from ase.io import read
from pylab import *
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore")
def cumulative_cond_cal(observables, kappa_tensor, prefactor=1/3):

    """Compute cumulative conductivity based on either frequency or mean-free path

       input:
       observables: (ndarray) either phonon frequency or mean-free path
       cond_tensor: (ndarray) conductivity tensor
       prefactor: (float) prefactor to average kappa tensor, 1/3 for bulk material

       output:
       observables: (ndarray) sorted phonon frequency or mean-free path
       kappa_cond (ndarray) cumulative conductivity

    """

    # Sum over kappa by directions
    kappa = np.einsum('maa->m', prefactor * kappa_tensor)

    # Sort observables
    observables_argsort_indices = np.argsort(observables)
    cumulative_kappa = np.cumsum(kappa[observables_argsort_indices])
    return observables[observables_argsort_indices], cumulative_kappa



def set_fig_properties(ax_list, panel_color_str='black', line_width=2):
    tl = 4
    tw = 2
    tlm = 2

    for ax in ax_list:
        ax.tick_params(which='major', length=tl, width=tw)
        ax.tick_params(which='minor', length=tlm, width=tw)
        ax.tick_params(which='both', axis='both', direction='in',
                       right=True, top=True)
        ax.spines['bottom'].set_color(panel_color_str)
        ax.spines['top'].set_color(panel_color_str)
        ax.spines['left'].set_color(panel_color_str)
        ax.spines['right'].set_color(panel_color_str)

        ax.spines['bottom'].set_linewidth(line_width)
        ax.spines['top'].set_linewidth(line_width)
        ax.spines['left'].set_linewidth(line_width)
        ax.spines['right'].set_linewidth(line_width)

        for t in ax.xaxis.get_ticklines(): t.set_color(panel_color_str)
        for t in ax.yaxis.get_ticklines(): t.set_color(panel_color_str)
        for t in ax.xaxis.get_ticklines(): t.set_linewidth(line_width)
        for t in ax.yaxis.get_ticklines(): t.set_linewidth(line_width)
# Denote plot default format
aw = 2
fs = 12
font = {'size': fs}
matplotlib.rc('font', **font)
matplotlib.rc('axes', linewidth=aw)

# Configure Matplotlib to use a LaTeX-like style without LaTeX
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

data_folder = './'

# Derive symbols for high symmetry directions in Brillouin zone
dispersion = np.loadtxt(data_folder  + 'plots/5_5_5/dispersion')
point_names = np.loadtxt(data_folder  + 'plots/5_5_5/point_names', dtype=str)

point_names_list = []
for point_name in point_names:
    if point_name == 'G':
        point_name = r'$\Gamma$'
    elif point_name == 'U':
        point_name = 'U=K'
    point_names_list.append(point_name)

# Load in the "tick-mark" values for these symbols
q = np.loadtxt(data_folder  + 'plots/5_5_5/q')
Q = np.loadtxt(data_folder  + 'plots/5_5_5/Q_val')

# Load in grid and dos individually
dos_Ge = np.load(data_folder + 'plots/5_5_5/dos.npy')

# Plot dispersion
fig = figure(figsize=(8,6))
set_fig_properties([gca()])
plot(q[0], dispersion[0, 0], 'r-', ms=1)
plot(q, dispersion, 'r-', ms=1)
for i in range(1, 4):
    axvline(x=Q[i], ymin=0, ymax=2, ls='--',  lw=1, c="k")
axhline(y =0, ls = '--', c='k', lw=1)
ylabel('Frequency (THz)', fontsize=14)
xlabel(r'Wave vector ($\frac{2\pi}{a}$)', fontsize=14)
gca().set_yticks(np.arange(0, 21, 3))
xticks(Q, point_names_list)
ylim([-0.5, 18])
xlim([Q[0], Q[4]])
dosax = fig.add_axes([0.91, .11, .08, .77])
set_fig_properties([gca()])

# Plot per projection
for d in np.expand_dims(dos_Ge[1],0):
    dosax.plot(d, dos_Ge[0],c='r')
dosax.set_yticks([])
dosax.set_xticks([])
dosax.set_xlabel("DOS")
ylim([-0.5, 18])
show()

# Load in group velocity, heat_capacity (cv) and frequency data
frequency =  np.load(
    data_folder + 'ALD_Si46/5_5_5/frequency.npy',
    allow_pickle=True)

cv =  np.load(
    data_folder + 'ALD_Si46/5_5_5/300/quantum/heat_capacity.npy',
    allow_pickle=True)


group_velocity = np.load(
    data_folder + 'ALD_Si46/5_5_5/velocity.npy')

phase_space = np.load(data_folder +
 'ALD_Si46/5_5_5/300/quantum/_ps_and_gamma.npy',
                      allow_pickle=True)[:,0]


# Compute norm of group velocity
# Convert the unit from angstrom / picosecond to kilometer/ second
group_velocity_norm = np.linalg.norm(
    group_velocity.reshape(-1, 3), axis=1) / 10.0

# Plot observables in subplot
figure(figsize=(12, 3.5))
subplot(1,3, 1)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C')[3:], 1e23*cv.flatten(order='C')[3:],
        facecolor='w', edgecolor='r', s=10, marker='8')
ylabel (r"$C_{v}$ ($10^{23}$ J/K)")
xlabel('Frequency (THz)', fontsize=14)
ylim(0.8*1e23*cv.flatten(order='C')[3:].min(), 1.05*1e23*cv.flatten(order='C')[3:].max())
gca().set_xticks(np.arange(0, 20, 5))

subplot(1 ,3, 2)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C'),
        group_velocity_norm, facecolor='w', edgecolor='r', s=10, marker='^')
xlabel('Frequency (THz)', fontsize=14)
ylabel(r'$|v| \ (\frac{km}{s})$', fontsize=14)

subplot(1 ,3, 3)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C'),
        phase_space, facecolor='w', edgecolor='r', s=10, marker='o')
xlabel('Frequency (THz)', fontsize=14)
ylabel('Phase space', fontsize=14)
subplots_adjust(wspace=0.33)
tight_layout()
show()

# Load in scattering rate
scattering_rate = np.load(
    data_folder +
    'ALD_Si46/5_5_5/300/quantum/bandwidth.npy', allow_pickle=True)

# Derive lifetime, which is inverse of scattering rate
life_time = scattering_rate ** (-1)

# Denote lists to intake mean free path in each direction
mean_free_path = []
for i in range(3):
    mean_free_path.append(np.loadtxt(
    data_folder +
    'ALD_Si46/5_5_5/rta/300/quantum/mean_free_path_' +
    str(i) + '.dat'))

# Convert list to numpy array and compute norm for mean free path
# Convert the unit from angstrom  to nanometer
mean_free_path = np.array(mean_free_path).T
mean_free_path_norm = np.linalg.norm(
    mean_free_path.reshape(-1, 3), axis=1) / 10.0

# Plot observables in subplot
figure(figsize=(12, 3.5))
subplot(1,3, 1)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C'),
        life_time, facecolor='w', edgecolor='r', s=10, marker='s')
gca().set_xticks(np.arange(0, 20, 5))
yscale('log')
ylabel(r'$\tau \ (ps)$', fontsize=14)
xlabel('Frequency (THz)', fontsize=14)

subplot(1,3, 2)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C'),
        scattering_rate, facecolor='w', edgecolor='r', s=10, marker='d')
ylabel(r'$\Gamma \ (THz)$', fontsize=14)
xlabel('Frequency (THz)', fontsize=14)

subplot(1,3, 3)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C'),
        mean_free_path_norm, facecolor='w', edgecolor='r', s=10, marker='8')
ylabel(r'$\lambda \ (nm)$', fontsize=14)
xlabel('Frequency (THz)', fontsize=14)
yscale('log')
ylim([1e-3, 1e5])
subplots_adjust(wspace=0.33)
tight_layout()
show()

# Denote zeros to intake kappa tensor
kappa_tensor = np.zeros([mean_free_path.shape[0], 3, 3])

for i in range(3):
    for j in range(3):
        kappa_tensor[:, i, j] = np.loadtxt(data_folder +
    'ALD_Si46/5_5_5/rta/300/quantum/conductivity_' +
    str(i) + '_' + str(j) + '.dat')

# Sum over the 0th dimension to recover 3-by-3 kappa matrix
kappa_matrix = kappa_tensor.sum(axis=0)
print("Bulk thermal conductivity: %.1f W m^-1 K^-1\n"
      %np.mean(np.diag(kappa_matrix)))
print("kappa matrix: ")
print(kappa_matrix)
print('\n')

# Compute kappa in per mode and cumulative representations
kappa_per_mode = kappa_tensor.sum(axis=-1).sum(axis=1)
freq_sorted, kappa_cum_wrt_freq = cumulative_cond_cal(
    frequency.flatten(order='C'), kappa_tensor)
lambda_sorted, kappa_cum_wrt_lambda = cumulative_cond_cal(
    mean_free_path_norm, kappa_tensor)

# Plot observables in subplot
figure(figsize=(12, 3.5))
subplot(1,3, 1)
set_fig_properties([gca()])
scatter(frequency.flatten(order='C'),
        kappa_per_mode, facecolor='w', edgecolor='r', s=10, marker='>', label='$\kappa_{per \ mode}$')
axhline(y =0, ls = '--', c='k', lw=1)
ylabel(r'$\kappa_{per \ mode}\;\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$',fontsize=14)
xlabel('Frequency (THz)', fontsize=14)
legend(loc=1, fontsize=12)
ylim([-0.1, 1.5])

subplot(1,3, 2)
set_fig_properties([gca()])
plot(freq_sorted, kappa_cum_wrt_freq, 'r',
     label=r'$\kappa_{pure} \approx 43.2 \;\frac{\rm{W}}{\rm{m}\cdot\rm{K}} $')
gca().set_yticks(np.arange(0, 60, 10))
ylabel(r'$\kappa_{cumulative, \omega}\;\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$',fontsize=14)
xlabel('Frequency (THz)', fontsize=14)
legend(loc=4, fontsize=12)

subplot(1,3, 3)
set_fig_properties([gca()])
plot(lambda_sorted, kappa_cum_wrt_lambda, 'r')
gca().set_yticks(np.arange(0, 60, 10))
xlabel(r'$\lambda \ (nm)$', fontsize=14)
ylabel(r'$\kappa_{cumulative, \lambda}\;\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$',fontsize=14)
xscale('log')
xlim([1e-1, 3e3])
subplots_adjust(wspace=0.33)
tight_layout()
show()

# Denote temperature array
temperatures = np.arange(60, 1060, 50, dtype=int)

# Denote zeros to intake kappa tensor
kappa_vary_temperature_rta = []
kappa_vary_temperature_qhgk = []
kappa_tensor_rta = np.copy(kappa_tensor)
kappa_tensor_qhgk = np.copy(kappa_tensor)
for temperature in temperatures:
    temperature_str = str(temperature)
    kappa_data_path_rta = data_folder + \
    'ALD_Si46/5_5_5/rta/' + temperature_str + '/quantum/conductivity_' + \
    str(i) + '_' + str(j) + '.dat'
    
    kappa_data_path_qhgk = data_folder + \
    'ALD_Si46/5_5_5/qhgk/' + temperature_str + '/quantum/conductivity_' + \
    str(i) + '_' + str(j) + '.dat'
    for i in range(3):
        for j in range(3):
            kappa_tensor_rta[:, i, j] = np.loadtxt(kappa_data_path_rta)
            kappa_tensor_qhgk[:, i, j] = np.loadtxt(kappa_data_path_qhgk)

    # Sum over the 0th dimension to recover 3-by-3 kappa matrix
    # and take the average of body-diagonal
    kappa_rta = np.diag(kappa_tensor_rta.sum(axis=0)).mean()
    kappa_qhgk = np.diag(kappa_tensor_qhgk.sum(axis=0)).mean()
    kappa_vary_temperature_rta.append(kappa_rta.copy())
    kappa_vary_temperature_qhgk.append(kappa_qhgk.copy())


# Cast list to array
kappa_vary_temperature_rta = np.array(kappa_vary_temperature_rta)
kappa_vary_tempeature_qhgk = np.array(kappa_vary_temperature_qhgk)

figure(figsize=(8,6))
set_fig_properties([gca()])
plt.semilogy(temperatures, kappa_vary_temperature_rta, c='r', lw=1.5, label=r'$\kappa$ALDo_rta')
plt.semilogy(temperatures, kappa_vary_temperature_qhgk, c='b', lw=1.5, label=r'$\kappa$ALDo_qhgk')
scatter(300, 45.4, edgecolor='c', facecolor='w', marker='d',s=60, label = r'DFT (PBE functional) + BTE $\approx  45.4 \;\frac{\rm{W}}{\rm{m}\cdot\rm{K}}$')
scatter(300, 53.0, edgecolor='m', facecolor='w', marker='s',s=60, label = r'DFT (LDA functional) + BTE  $\approx  53.0 \;\frac{\rm{W}}{\rm{m}\cdot\rm{K}}$')
gca().errorbar(300, 43.1, yerr=4.1, fmt="s", capsize=3,
                   markerfacecolor='w', markeredgecolor='g', ecolor='g', marker='>',markersize=6,label=r'$\kappa_{emd}^{\text{ADP potential}} \approx  43.1 \pm 4.1 \;\frac{\rm{W}}{\rm{m}\cdot\rm{K}}$')
gca().set_xticks(np.arange(100, 1000, 200))
ylabel(r'$\kappa\;\left(\frac{\rm{W}}{\rm{m}\cdot\rm{K}}\right)$',fontsize=14)
xlabel('Temperature (K)',fontsize=14)
legend(loc=1, fontsize=12)
tight_layout()
show()
