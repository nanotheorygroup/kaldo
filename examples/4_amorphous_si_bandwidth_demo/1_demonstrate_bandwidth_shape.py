from ase.io import read
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np

##########################################
# 1. Loading Data
##########################################

# Ballistico can config the finite difference object
# straightly from the docsource folder. Currently support
# format includes "eskm", "hiphive", "shengbte" and "shengbte-qe" 

forceconstants = ForceConstants.from_folder(folder='structure_a_si_512', format='eskm')


##########################################
# 2. Calculating Bandwidths
##########################################
# With the newly created finite difference object, you can create a Phonon object to calculate
# the dispersion relation, phase space and mode bandwidths for the system.

# Here we will demonstrate the effects that the different band broadening shapes have on the numerical data of amorphous
# silicon at 300K. First, we create three Phonon objects with three different shapes. Gauss, Lorentzian, and triangular.


phonons_triangle = Phonons(forceconstants=forceconstants,
                           is_classic=False,
                           temperature=300,
                           third_bandwidth=0.5/4.135,
                           broadening_shape='triangle',
                           is_tf_backend=False,
                           folder='ald_triangle')

phonons_gaussian = Phonons(forceconstants=forceconstants,
                           is_classic=False,
                           temperature=300,
                           third_bandwidth=0.5/4.135,
                           broadening_shape='gauss',
                           is_tf_backend=False,
                           folder='ald_gauss')

phonons_lorentzian = Phonons(forceconstants=forceconstants,
                             is_classic=False,
                             temperature=300,
                             third_bandwidth=0.5/4.135,
                             broadening_shape='lorentz',
                             is_tf_backend=False,
                             folder='ald_lorentz')


##########################################
# 3, Visualize the Broadening!
##########################################

phonons_list = [phonons_gaussian, phonons_triangle, phonons_lorentzian]
phonons_labels = ['Gaussian', 'Triangular', 'Lorentzian']
plt.figure(figsize=(16,12))

# Initialize the string for phonon with one of the
# three bandwidths to perform the simulation

simulation_bandwidth_str = 'Gaussian'
index = np.where(simulation_bandwidth_str == np.array(phonons_labels))[0]
for i in index:
    ph = phonons_list[i]
    freq = ph.frequency[0][3:]
    gamma = ph.bandwidth[0][3:]
    lbl = 'Broadening: '+phonons_labels[i]
    plt.scatter(freq, gamma, label=lbl, s=10)
plt.suptitle("Amorphous Si (made with Tersoff '89)(512 atoms)", y=0.98, fontsize=21, fontweight='bold')
plt.title('Band-broadening Shape Contrast, width = 0.121', fontsize=18, fontweight='bold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel(r'$\nu$ (THz)', fontsize=18, fontweight='bold')
plt.ylabel(r'$\Gamma$ (THz)', fontsize=18, fontweight='bold')
lgnd = plt.legend(scatterpoints=1, fontsize=14)
for handle in lgnd.legendHandles:
    handle.set_sizes([50.0])
#plt.savefig('Ballistico_bandwidth_shapes.png')
plt.show()