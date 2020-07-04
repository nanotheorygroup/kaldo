from ase.io import read
import matplotlib.pyplot as plt
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons



##########################################
# 1. Loading Data
##########################################
# kaldo offers a number of ways of uploading your system data into the program
# below, a few of the options will be displayed. But first, a quick explanation of what we need to get started.
#
# In order to calculate the phonon lifetimes, we need three files
# 1. A configuration of atoms.
#       These can be saved in any format that can be read by ase
#       (see here: https://wiki.fysik.dtu.dk/ase/ase/io/io.html)
# 2. A dynamical matrix
# 3. A third order IFC matrix
#   The third order and dynamical matrix that will be used here were generated with a customized version of DL_POLY_4
#   (You can register to get DL_POLY at https://www.scd.stfc.ac.uk/Pages/DL_POLY-Registration.aspx)

# To upload your data originally, try making a ForceConstants object with .from_files or .from_folder
# First identify file names
folder = 'structure'
config_file = str(folder) + '/CONFIG'
dynmat_file = str(folder) + '/Dyn.form'
third_file = str(folder) + '/THIRD'
#
# # Then read in files and create a finite difference object
atoms = read(config_file, format='dlp4')
finite_diff = ForceConstants.from_files(folder=folder,replicated_atoms=atoms,dynmat_file=dynmat_file, third_file=third_file)

# kaldo also saves your data in more quickly loaded numpy arrays. This can save valuable time when plotting or
# in instances where you need to handle the same data repeatedly!
# If you'd like to see it in action, just comment out the previous finite difference object and uncomment line 41!

#finite_diff = ForceConstants.from_folder(folder='structure',format='numpy')


##########################################
# 2. Calculating Bandwidths
##########################################
# With the newly created finite difference object, you can create a Phonon object to calculate
# the dispersion relation, phase space and mode bandwidths for the system.

# Here we will demonstrate the effects that the different band broadening shapes have on the numerical data of amorphous
# silicon at 300K. First we create three Phonon objects with three different shapes. Gauss, Lorentzian, and triangular.

phonons_triangle = Phonons(forceconstants=finite_diff,
                           is_classic=False,
                           temperature=300,
                           third_bandwidth=0.5/4.135,
                           broadening_shape='triangle',
                           is_tf_backend=False,
                           folder='structure/ald_triangle')

phonons_gaussian = Phonons(forceconstants=finite_diff,
                           is_classic=False,
                           temperature=300,
                           third_bandwidth=0.5/4.135,
                           broadening_shape='gauss',
                           is_tf_backend=False,
                           folder='structure/ald_gauss')

phonons_lorentzian = Phonons(forceconstants=finite_diff,
                             is_classic=False,
                             temperature=300,
                             third_bandwidth=0.5/4.135,
                             broadening_shape='lorentz',
                             is_tf_backend=False,
                             folder='structure/ald_lorentz')


##########################################
# 3, Visualize the Broadening!
##########################################
phonons_list = [phonons_gaussian, phonons_triangle, phonons_lorentzian]
phonons_labels = ['Gaussian', 'Triangular', 'Lorentzian']
plt.figure(figsize=(16,12))
for i in range(0,len(phonons_list)):
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
plt.savefig('kaldo_bandwidth_shapes.png')
plt.show()
