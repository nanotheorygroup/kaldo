from ase.io import read
import matplotlib.pyplot as plt
from ballistico.finitedifference import FiniteDifference
from ballistico.phonons import Phonons

# First identify file names
folder = 'structure'
config_file = str(folder) + '/CONFIG'
dynmat_file = str(folder) + '/Dyn.form'
third_file = str(folder) + '/THIRD'

# Then read in files and create a finite difference object
atoms = read(config_file, format='dlp4')
finite_diff = FiniteDifference.from_files(folder=folder, replicated_atoms=atoms, dynmat_file=dynmat_file,third_file=third_file)

# If all of the relevant numpy files exist (see last example), feel free to uncomment this line and comment
# out the preceeding FiniteDifference creation
# finite_diff = FiniteDifference.from_folder(folder='structure',format='numpy')


# Here we will demonstrate the effect of band widths on the lifetimes. The effect will be shown for each
# broadening shape (a for loop is used in the name of efficiency). The graphs will be constructed here as well.
# Please see the preceeding section on shapes for a more in depth explanation.

shapes = ['triangle','gauss','lorentz']
labels = ['Triangle','Gaussian','Lorentzian']
widths = [0.5/4.135, 1/4.135]
widths_l = [.121, .242]
fig, axs = plt.subplots(ncols=1,nrows=3, figsize=(6,14))
plt.suptitle("Amorphous Si (Tersoff '89)", y=0.98, fontsize=18, fontweight='bold')
for i in range(0,len(shapes)):
    ax = axs[i]
    for j in range(0,len(widths)):
        phonon_object = Phonons(finite_difference=finite_diff,
                                is_classic=False,
                                temperature=300,
                                third_bandwidth=widths[j],
                                broadening_shape=shapes[i],
                                is_tf_backend=False,
                                folder='structure/ald_'+shapes[i])
        freq = phonon_object.frequency
        gamma = phonon_object.bandwidth
        lbl = 'Width: ' + str(widths_l[j])
        ax.scatter(freq, gamma, label=lbl, s=5)
    ax.set_title('Broadening Shape: '+str(labels[i]), fontsize=14,fontweight='bold')
    ax.tick_params(labelsize=10)
    ax.set_xlabel(r'$\nu$ (THz)', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'$\Gamma$ (THz)', fontsize=10, fontweight='bold')
    lgnd = ax.legend(scatterpoints=1, fontsize=12)
    for handle in lgnd.legendHandles:
        handle.set_sizes([50.0])
plt.show()
#plt.savefig('Ballistico_third_bandwidth.png')