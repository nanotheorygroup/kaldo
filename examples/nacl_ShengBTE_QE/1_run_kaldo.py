# Runs kALDo on NaCl using force constants with or without Born Effective Charges
#
# Usage: python 1_run_kaldo.py <harmonic> <overwrite>
# harmonic will exit the calculations after making a dispersion
# overwrite allows the script to replace data written in previous runs
import sys

# Charges -----------------------
nac = False
if "nac" in sys.argv:
    nac = True
nac_string = 'with' if nac else 'no'
# Harmonic ----------------------
# Dispersion args
npoints = 200 # points along path
pathstring = 'GXULGK' # actual path

# Anharmonic --------------------
# Threading per process
nthread = 4
# K-pt grid
k = 15 # cubed
# Temperature (in K)
temperature = 300
# Conductivity method
cond_method = 'inverse'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
import os
import sys
import numpy as np
from ase.io import read
# Input/Output folders
fcs_folder = os.environ["kaldo_inputs"]+f"/{nac_string}_charges/"
prefix = os.environ["kaldo_ald"] # prefix to ald folder (where data is kept)
ald_folder = prefix + f"/{nac_string}_charges" # actual data folder
plot_folder = os.environ["kaldo_outputs"]+f"/{nac_string}_charges/" # where plots are saved

# Supecell Information
nrep = int(8)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k,k,k)
third_supercell = np.array([3,3,3])

# Detect harmonic
# This lets you calculate only the harmonic quantities first before you
# spend the time calculating third order quantities
harmonic = False
if "harmonic" in sys.argv:
    harmonic = True

# Unfolding - currently required to be True for this example
unfold_bool = True
unfold = "u"

# Setup folders
# inputs
if not os.path.isfile(fcs_folder+'/espresso.ifc2'):
    print("!! - No force constants detected in "+fcs_folder)
    print("!! - Please unzip the tarball found in the \"tools\" directory in this directory")
    print("!! - Exiting ..")
    quit(1)
# outputs
if os.path.isdir(prefix):
    print("\n!! - "+prefix+" directory already exists")
    if os.path.isdir(ald_folder):
        print("!! - "+ald_folder+" directory already exists")
        print("!! - continuing may overwrite, or load previous data\n")
    else:
        os.mkdir(ald_folder)
else:
    os.mkdir(prefix)
    os.mkdir(ald_folder)
# plots
if not os.path.isdir(os.environ["kaldo_outputs"]):
    os.mkdir(os.environ["kaldo_outputs"])
    os.mkdir(plot_folder)
else:
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)
    else:
        print('!! - '+plot_folder+' already exists')
        print('!! - continuing may overwrite previous data\n')

# Control threading behavior using parameter from above
os.environ["CUDA_VISIBLE_DEVICES"]="" # No GPUs
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(nthread)
tf.config.threading.set_intra_op_parallelism_threads(nthread)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Print out detected settings
print("## Simulation Settings:")
print("\n\n\tCalculating for NaCl -- ")
print("\t\t NAC correction (bool): {}".format(nac))
print("\t\t In folder:       {}".format(fcs_folder))
print("\t\t Out folder:      {}".format(ald_folder))
print("\t\t Dispersion only: {}".format(harmonic))
print("\n\n## Beginning Simulation:")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

### Begin simulation
# Import kALDo
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
from kaldo.controllers.plotter import plot_dispersion

# Create kALDo objects
forceconstant = ForceConstants.from_folder(
                       folder=fcs_folder,
                       supercell=supercell,
                       only_second=harmonic,
                       third_supercell=third_supercell,
                       is_acoustic_sum=True,
                       format="shengbte-qe")
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=temperature,
              folder=ald_folder,
              is_unfolding=unfold_bool,
              storage="numpy")

# Harmonic data along path
# Although you need to specify the k-pt grid for the Phonons object, we don't
# actually use it for dispersion relations and velocities. Instead, the sampling is taken
# care of by the path specified and the npoints variable set above.
print("\n\n## Calculating Dispersion! ######################")
atoms = read(fcs_folder+"/POSCAR", format="vasp")
cell = atoms.cell
lat = cell.get_bravais_lattice() # get Bravais Lattice of unit cell
path = cell.bandpath(pathstring, npoints=npoints)
print("\t\tUnit cell detected: {}".format(atoms))
print("\t\tSpecial points on cell:")
print("\t\t{}".format(lat.get_special_points()))
print("\t\tPath: {}".format(path))
plot_dispersion(phonons, is_showing=False,
            manually_defined_path=path, folder=ald_folder+"/dispersion")
if harmonic:
    print("\n\n\tHarmonic quantities generated!\n\texiting safely ..")
    quit(0)
else:
    print(f"Please find the plotted results in {ald_folder+'/dispersion'}")
    print("## End Harmonic Calculations ######################\n\n")

# Conductivity & Anharmonics
# Different methods of calculating the conductivity can be compared
# but full inversion of the three-phonon scattering matrix typically agrees with
# experiment more than say the relaxation time approximation (not applicable for
# systems without symmetry).
# Calculating this K_latt will produce as a byproduct things like the phonon
# bandwidths, phase space etc, but some need to be explicitly called to output
# numpy files (e.g. Phonons.participation_ratio, find more in Phonons docs).
#
# All of the anharmonic quantities should be converged according to a sensitivity
# analysis of their value against increasing k-points.
# This ensures smooth interpolation between points being sampled
print('## Calculating Conductivity! ######################')
cond = Conductivity(phonons=phonons, method=cond_method, storage='numpy')
cond_matrix = cond.conductivity.sum(axis=0)
diag = np.diag(cond_matrix)
offdiag = np.abs(cond_matrix).sum() - np.abs(diag).sum()
print("Conductivity from full inversion (W/m-K):\n%.3f" % (np.mean(diag)))
print("Sum of off-diagonal terms: %.3f" % offdiag)
print("Full matrix:")
print(cond_matrix)
print('## End Conductivity Calculations ######################')
print('## End of Simulation ######################\n\n')

