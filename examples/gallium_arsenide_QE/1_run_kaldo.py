# Runs kALDo on monolayer b-InSe using force constants by DFPT
#
# Usage: python 1_run_kaldo.py
# u/n controls if kALDo unfolds the force constants
# disp will exit the calculations after making a dispersion
# overwrite allows the script to replace data written in previous runs

# Harmonic ----------------------
# Dispersion args
npoints = 150
pathstring = 'GXWGL'
unfold_bool = True
outfold = 'plots_nc/'

# Anharmonic --------------------
# Threading per process
nthread = 2
# Conductivity method
cond_method = 'inverse'
# K-pt grid (technically harmonic, but the dispersion is the only property we really deal
# with here, and it isn't relevant to that)
k = 2 # cubed

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# You shouldn't need to edit below this line, but it should be well commented
# so that you can reference it for how to set up your own workflow
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Settings detected by environment variables, POSCARs and arguments
import os
import sys
import numpy as np
from ase.io import read
from scipy import constants
# Replicas
nrep = int(5)
nrep_third = int(2)
supercell = np.array([nrep, nrep, nrep])
kpts, kptfolder = [k, k, k], '{}_{}_{}'.format(k, k, k)
third_supercell = np.array([nrep_third, nrep_third, nrep_third])
thz_to_invcm = constants.value('hertz-inverse meter relationship')*1e12/100
qe_data = './'


### Begin simulation
# Import kALDo
from kaldo.observables.harmonic_with_q import HarmonicWithQ
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons


# Create harmonic ForceConstants object from QE-data
forceconstant = ForceConstants.from_folder(
                       folder=qe_data,
                       supercell=supercell,
                       only_second=True,
                       is_acoustic_sum=True,
                       format='shengbte-qe')
phonons = Phonons(forceconstants=forceconstant,
              kpts=kpts,
              is_classic=False,
              temperature=300,
              folder='./',
              is_unfolding=unfold_bool,
              storage='memory',
              is_nac=True,)
# print('full:')
# #print(phonons.frequency)
# #print(phonons.velocity)
# print('done!')

# velocities = np.zeros((3, 6, 3))
# kpoints = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]
# for i,kpt in enumerate(kpoints):
#     phonon = HarmonicWithQ(np.array(kpt),
#                            phonons.forceconstants.second,
#                            distance_threshold=phonons.forceconstants.distance_threshold,
#                            storage='memory',
#                            is_nw=phonons.is_nw,
#                            is_unfolding=phonons.is_unfolding,
#                            is_nac=True, )
#     print(f'kpt: {kpt}')
#     print('frequency')
#     print(phonon.frequency * 2 * np.pi)
#     print('velocity')
#     print(phonon.velocity)
#     print('kpt done!')
#     velocities[i, :, : ] = phonon.velocity
#
# velocities = np.transpose(velocities, axes=(1, 0, 2))
# velocities = np.reshape(velocities, (-1, 3))
# print(velocities)

# phonon = HarmonicWithQ(np.array([0.25, 0.25, 0.000]),
#                        phonons.forceconstants.second,
#                        distance_threshold=phonons.forceconstants.distance_threshold,
#                        storage='memory',
#                        is_nw=phonons.is_nw,
#                        is_unfolding=phonons.is_unfolding,
#                        is_nac=True, )
# print(phonon.frequency)
# exit()

atoms = forceconstant.atoms
cell = atoms.cell
lat = cell.get_bravais_lattice()
path = cell.bandpath(pathstring, npoints=npoints)
print('Unit cell detected: {} '.format(atoms))
print('Special points on cell: ')
print(lat.get_special_points())
print('Path: {}'.format(path))
np.savetxt('path.txt', np.hstack([path.kpts, np.ones((path.kpts.shape[0], 1))]))
freqs = []
vels = []
vnorms = []
for i,kpoint in enumerate(path.kpts):
    phonon = HarmonicWithQ(kpoint,
                           phonons.forceconstants.second,
                           distance_threshold=phonons.forceconstants.distance_threshold,
                           storage='memory',
                           is_nw=phonons.is_nw,
                           is_unfolding=phonons.is_unfolding,
                           is_nac=False,)
    #print(kpoint, phonon.frequency.squeeze()[:3])
    freqs.append(phonon.frequency.squeeze())
    vels.append(phonon.velocity.reshape((-1, 1)))
    vnorms.append(np.linalg.norm(phonon.velocity.squeeze(), axis=-1).squeeze())

np.savetxt(f'{outfold}/{pathstring}.pts', path.kpts)
np.savetxt(f'{outfold}/{pathstring}.freqs', np.array(freqs))
np.savetxt(f'{outfold}/{pathstring}.vels', np.array(vels).squeeze())
np.savetxt(f'{outfold}/{pathstring}.vnorms', np.array(vnorms).squeeze())

#print(np.array(freqs).squeeze().shape)
np.savetxt(f'plots/{kptfolder}/kpts', path.kpts)
np.savetxt(f'plots/{kptfolder}/dispersion', np.array(freqs))


from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('/home/nwlundgren/spanners/configurations/nicholas.mplstyle')
fig = plt.figure(figsize=(16,12))
grid = plt.GridSpec(2, 2, wspace=0.08, hspace=0.08)
ax0 = fig.add_subplot(grid[:, 0])
ax1 = fig.add_subplot(grid[0, 1])
ax2 = fig.add_subplot(grid[1, 1])


kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.freqs')
kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
for column in range(kaldo_y.shape[1]):
    ax0.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='b', lw=2, zorder=10)

kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vels')
kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
for column in range(kaldo_y.shape[1]):
    ax1.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)

kaldo_y = np.loadtxt(f'{outfold}/{pathstring}.vnorms')
kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
for column in range(kaldo_y.shape[1]):
    ax2.scatter(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='r', lw=2, zorder=10)
plt.savefig(f'{outfold}/{pathstring}.disp.png')

plt.show()

# phonon = HarmonicWithQ(np.array([0.25, 0.25, 0.000]),
#                        phonons.forceconstants.second,
#                        distance_threshold=phonons.forceconstants.distance_threshold,
#                        storage='memory',
#                        is_nw=phonons.is_nw,
#                        is_unfolding=phonons.is_unfolding,
#                        is_nac=True, )
# print(phonon.frequency)
# exit()
#
# atoms = read('espresso_fcs/POSCAR', format='vasp')
# cell = atoms.cell
# lat = cell.get_bravais_lattice()
# path = cell.bandpath(pathstring, npoints=npoints)
# print('Unit cell detected: {} '.format(atoms))
# print('Special points on cell: ')
# print(lat.get_special_points())
# print('Path: {}'.format(path))
# np.savetxt('path.txt', np.hstack([path.kpts, np.ones((path.kpts.shape[0], 1))]))
# freqs = []
# for i,kpoint in enumerate(path.kpts):
#     phonon = HarmonicWithQ(kpoint,
#                            phonons.forceconstants.second,
#                            distance_threshold=phonons.forceconstants.distance_threshold,
#                            storage='memory',
#                            is_nw=phonons.is_nw,
#                            is_unfolding=phonons.is_unfolding,
#                            is_nac=True,)
#     print(kpoint, phonon.frequency.squeeze()[:3])
#     freqs.append(phonon.frequency.squeeze())

# print(np.array(freqs).squeeze().shape)
# np.savetxt('plots/7_7_1/kpts', path.kpts)
# np.savetxt('plots/7_7_1/dispersion', np.array(freqs))


# from matplotlib import pyplot as plt
# plt.style.use('/home/nwlundgren/spanners/configurations/nicholas.mplstyle')
# # Unit conversions
# # THz to cm-1
# fig = plt.figure(figsize=(16,12))
# ax = fig.add_subplot()
#
# colors = ['r', 'b',]
# kaldo_y = np.loadtxt('plots/7_7_1/dispersion')
# kaldo_x = np.linspace(0, 1, kaldo_y.shape[0])
# for column in range(kaldo_y.shape[1]):
#     ax.plot(kaldo_x/kaldo_x.max(), kaldo_y[:, column], color='b', lw=2, zorder=10)
# plt.show()
# #plot_dispersion(phonons, is_showing=True,
# #           manually_defined_path=path,) #folder=prefix+'/dispersion')
# #plot_dispersion(phonons, is_showing=True, n_k_points=150) #folder=prefix+'/dispersion')


'''
Code as written 7.11 - 19:28 in units for kaldo. Trying again with ShengBTE units
    def nac_velocities(self, direction, qpoint=None, gmax=14, alpha=1.0,):
        ///////////////////////////////
        Calculate the non-analytic correction to the dynamical matrix.

        Parameters
        ----------
        qpoint
        gmax
        alpha

        Returns
        -------
        correction_matrix
        ///////////////////////////////
        # Constants, and system information
        RyBr_to_eVA = units.Rydberg / (units.Bohr ** 2)  # Rydberg / Bohr^2 to eV/A^2
        eV_to_10Jmol = units.mol / (10 * units.J)
        e2 = 2.  # square of electron charge in A.U.
        gmax = 14  # maximum reciprocal vector
        alpha = 1.0  # Ewald parameter
        geg0 = 4 * alpha * gmax
        atoms = self.second.atoms
        natoms = len(atoms)
        omega_bohr = np.linalg.det(atoms.cell.array / units.Bohr) # Vol. in Bohr^3
        lattice_constant = atoms.cell[0, :].max() / units.Bohr
        positions_n = atoms.positions.copy() / lattice_constant  # Normalized positions
        distances_n = positions_n[:, None, :] - positions_n[None, :, :]  # distance in crystal coordinates
        reciprocal_n = np.round(atoms.cell.reciprocal(), 12)  # round to avoid accumulation of error
        reciprocal_n /= np.abs(reciprocal_n[0, 0])  # Normalized reciprocal cell
        correction_matrix = tf.zeros([3, 3, natoms, natoms], dtype=tf.complex64)
        prefactor = 4 * np.pi * e2 / omega_bohr

        sqrt_mass = np.sqrt(self.atoms.get_masses().repeat(3, axis=0))
        mass_prefactor = np.reciprocal(np.einsum('i,j->ij', sqrt_mass, sqrt_mass))

        # Charge information
        epsilon = atoms.info['dielectric']  # in e^2/Bohr
        zeff = atoms.get_array('charges')  # in e

        # Charge sum rules
        # Using the "simple" algorithm from QE, we enforce that the sum of
        # charges for each polarization (e.g. xy, or yy) is zero
        zeff -= zeff.mean(axis=0)

        # 1. Construct grid of reciprocal unit cells
        # a. Find the number of replicas to make
        n_greplicas = 2 + 2 * np.sqrt(geg0) / np.linalg.norm(reciprocal_n, axis=1)
        # b. If it's low-dimensional, don't replicate in reciprocal space along axes without replicas in real space
        n_greplicas[np.array(self.second.supercell) == 1] = 1
        # c. Generate the grid of replicas
        g_grid = Grid(n_greplicas.astype(int))
        g_replicas = g_grid.grid(is_wrapping=True)  # minimium distance replicas
        # d. Transform the raw indices, to coordinates in reciprocal space
        g_positions = np.einsum('ia,ab->ib', g_replicas, reciprocal_n)
        g_positions = g_positions + (qpoint @ reciprocal_n)

        # 2. Filter cells that don't meet our Ewald cutoff criteria
        # a. setup mask
        geg = np.einsum('ia,ab,ib->i', g_positions, epsilon, g_positions, dtype=np.float128)
        cells_to_include = (geg > 0) * (geg / (4 * alpha) < gmax)
        # b. apply mask
        geg = geg[cells_to_include]
        g_positions = g_positions[cells_to_include]
        g_replicas = g_replicas[cells_to_include]

        # 3. Calculate for each cell
        # a. exponential decay term based on distance in reciprocal space, and dielectric tensor
        decay = prefactor * np.exp(-1 * geg / (alpha * 4)) / geg
        # b. effective charges at each G-vector
        zag = np.einsum('nab,ia->inb', zeff, g_positions)

        # 4. Calculate the actual correction as a product of the effective charges, exponential decay term, and phase factor
        # the phase factor is based on the distance of the G-vector and atomic positions
        phase = np.exp(-1j * 2 * np.pi * np.einsum('ia,nma->inm', g_positions, distances_n))

        //////////////////////////////
        # All directions at once code
        # Terms 1 + 2
        zag_zeff = np.einsum('ina,mcb->inmabc', zag, zeff)
        zbg_zeff = np.transpose(zag_zeff, (0, 2, 1, 4, 3, 5))
        # Term 3 (imaginary)
        zag_zbg_rij = 1j * np.einsum('ina,imb,nmc->inmabc', zag, zag, distances_n)
        # Term 4 (negative)
        dgeg = np.einsum('ab,ib->ib', epsilon + epsilon.T, g_positions)
        zag_zbg_dgeg = -1 * np.einsum('ina,imb,ic,i->inmabc', zag, zag, dgeg, (1/(4*alpha) + 1/geg))

        # Combine terms!
        lr_correction = zag_zeff + zbg_zeff + zag_zbg_rij + zag_zbg_dgeg


        # Scale by exponential decay term
        lr_correction = np.einsum('i,inmabc->nmabc', decay, lr_correction)
        ///////////////////////////

        # Terms 1 + 2
        zag_zeff = np.einsum('ina,mb->inmab', zag, zeff[:, direction, :])
        zbg_zeff = np.transpose(zag_zeff, (0, 2, 1, 4, 3))
        # Term 3 (imaginary)
        zag_zbg_rij = 1j * np.einsum('ina,imb,nm->inmab', zag, zag, distances_n[:, :, direction])
        # Term 4 (negative)
        dgeg = np.einsum('ab,ib->ib', epsilon + epsilon.T, g_positions)[:, direction]
        zag_zbg_dgeg = -1 * np.einsum('ina,imb,i,i->inmab', zag, zag, dgeg, (1/(4*alpha) + 1/geg))

        # Combine terms!
        lr_correction = zag_zeff + zbg_zeff + zag_zbg_rij + zag_zbg_dgeg


        # Scale by exponential decay term
        lr_correction = np.einsum('i,inmab->nmab', decay, lr_correction)

        # Rotate, reshape, rescale, and, finally, return correction value
        correction_matrix = np.transpose(lr_correction, axes=(2, 0, 3, 1))
        correction_matrix = np.reshape(correction_matrix, (natoms * 3, natoms * 3))
        correction_matrix *= mass_prefactor # 1/sqrt(mass_i * mass_j)
        correction_matrix *= RyBr_to_eVA * eV_to_10Jmol # Rydberg / Bohr^2 to 10J/mol A^2
        return correction_matrix
        


'''
