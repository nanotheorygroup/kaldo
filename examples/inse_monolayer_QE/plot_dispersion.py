import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants as con
plt.style.use('/home/nwlundgren/spanners/configurations/nicholas.mplstyle')
# Unit conversions
# THz to cm-1
thz_to_invcm = con.value('hertz-inverse meter relationship')*1e12/100
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot()

colors = ['r', 'b',]

for color, file in zip(colors, ['matdyn_reference/with_correction.gp',]):
    monolayer_data = np.loadtxt(file)
    xmax = monolayer_data[:, 0].max()
    ax.plot(monolayer_data[:, 0]/xmax, monolayer_data[:, 1:]/thz_to_invcm, color=color, lw=2, zorder=1)



kaldo_y = np.loadtxt('plots/7_7_1/dispersion')
kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
for column in range(kaldo_y.shape[1]):
    ax.plot(kaldo_x, kaldo_y[:, column], color='b', lw=2, zorder=10)

#kaldo_x = np.loadtxt('kaldo_dispersion/dispersion_nc/q')
#kaldo_y = np.loadtxt('kaldo_dispersion/dispersion_nc/dispersion')
#for column in range(kaldo_y.shape[1]):
#    ax.plot(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='k', lw=2, zorder=9)

#lines = [plt.Line2D([],[], color='k', lw=5),
lines=   [plt.Line2D([],[], color='r', lw=5),
         plt.Line2D([],[], color='b', lw=5),]
labels = ['matdyn', 'kaldo']
ax.legend(lines,labels, loc='upper center',).set_zorder(100)

# points = ['G', 'M', 'K', 'G']
# pointpos = np.loadtxt('plots/no_correction/Q_val')
# ax.set_xticks(pointpos, points)

ax.set_xlabel('k-point')
ax.set_ylabel('Frequency (THz)')
ax.set_title('Dispersion - InSe - With NAC Correction')

plt.savefig('plots/corrected.png')
