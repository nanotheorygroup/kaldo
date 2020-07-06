from ase.io import read
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np


# Ballistico can config the finite difference object
# straightly from the docsource folder. Currently support
# format includes "eskm", "hiphive", "shengbte" and "shengbte-qe" 

forceconstants = ForceConstants.from_folder(folder='structure_a_si_512', format='eskm')

# Here we will demonstrate the effect of band widths on the lifetimes. The effect will be shown for each
# broadening shape (a for loop is used in the name of efficiency). The graphs will be constructed here as well.
# Please see the preceeding section on shapes for a more in depth explanation.

shapes = ['triangle','gauss','lorentz']
labels = ['Triangle','Gaussian','Lorentzian']

# Initialize the string for phonon with one of the
# three bandwidths to perform the simulation

simulation_bandwidth_str = 'Gaussian'
index = np.where(simulation_bandwidth_str == np.array(labels))[0]

# Compute phonon bandwith with different widths, in the unit of THz

widths = [0.5/4.135, 1/4.135]
widths_l = [.121, .242]
plt.figure(figsize=(16,12))
plt.suptitle("Amorphous Si (Tersoff '89)", y=0.98, fontsize=18, fontweight='bold')
for i in index:
    for j in range(0,len(widths)):
        phonon_object = Phonons(forceconstants=forceconstants,
                                is_classic=False,
                                temperature=300,
                                third_bandwidth=widths[j],
                                broadening_shape=shapes[i],
                                is_tf_backend=False,
                                folder='ald_'+shapes[i])
        freq = phonon_object.frequency
        gamma = phonon_object.bandwidth
        lbl = 'Width: ' + str(widths_l[j]) + 'THz'
        plt.scatter(freq, gamma, label=lbl, s=5)
    plt.title('Broadening Shape: '+str(labels[i]), fontsize=18,fontweight='bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(r'$\nu$ (THz)', fontsize=18, fontweight='bold')
    plt.ylabel(r'$\Gamma$ (THz)', fontsize=18, fontweight='bold')
    lgnd = plt.legend(scatterpoints=1, fontsize=14)
    for handle in lgnd.legendHandles:
      handle.set_sizes([50.0])
#plt.savefig('Ballistico_third_bandwidth.png')
plt.show()
