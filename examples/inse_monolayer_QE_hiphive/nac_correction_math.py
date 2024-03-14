import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.grid import Grid
import ase.units as units
import tensorflow as tf

# Constants in Atomic Units
e2 = 2. # square of electron charge
gmax = 14 # maximum reciprocal vector
alpha = 1.0 # Ewald parameter
geg0 = 4 * alpha * gmax

k=5
nrep = 9
nrep_third = 5
supercell = np.array([nrep, nrep, 1])
kpts, kptfolder = [k, k, 1], '{}_{}_{}'.format(k,k,1)
third_supercell = np.array([nrep_third, nrep_third, 1])

forceconstant = ForceConstants.from_folder(
                       folder='espresso_fcs',
                       supercell=supercell,
                       only_second=True,
                       third_supercell=third_supercell,
                       is_acoustic_sum=True,
                       format='shengbte-qe')

atoms = forceconstant.atoms
natoms = len(atoms)

# Prepare cell and position data
cell_a = atoms.cell.copy() # in Angstrom
cell_b = atoms.cell.copy()
cell_b.array = cell_b.array / units.Bohr # in Bohr
alat_a = cell_a[0, 0] # in Angstrom
alat_b = cell_b[0, 0] # in Bohr
positions_a = atoms.positions.copy() # in Angstrom
positions_n = positions_a / cell_a[0, 0] # normalized!
distances_n = positions_n[:, None, :] - positions_n[None, :, :] # in crystal coordinates
reciprocal = cell_b.reciprocal() # "gcell" in qe
reciprocal_n = reciprocal / reciprocal[0, 0] # normalized!
omega_bohr = np.linalg.det(cell_b) # in Bohr^3

# Prefactor ("fac" in qe)
prefactor = 4 * np.pi * e2 / omega_bohr

# Charge information
epsilon = atoms.info['dielectric'] # in e^2/Bohr
zeff = atoms.get_array('charges') # in e

# Construct grid of reciprocal unit cells
# Find the number of replicas to make
n_greplicas = np.sqrt(geg0) / np.linalg.norm(reciprocal_n, axis=1) + 1
# Generate the grid of replicas
g_grid = Grid(n_greplicas.astype(int) * 2)
g_replicas = g_grid.grid(is_wrapping=True)
# Transform the raw indices, to coordinates in reciprocal space
g_positions = np.einsum('ij,jk->ik', g_replicas, reciprocal_n, dtype=np.float128)
# Calculate GEG (outer product scaled by epsilon)
# g_supercell_norms = np.linalg.norm(g_positions, axis=1)
geg = np.einsum('ij,jl,il->i', g_positions, epsilon, g_positions, dtype=np.float128)
cells_to_include = (geg > 0) * (geg/(4 * alpha) < gmax)

# Filter
#g_replicas_gamma = g_replicas[cells_to_include]
geg_gamma = geg[cells_to_include]
g_positions_gamma = g_positions[cells_to_include]

# Prefactors for the GEG
facgd = prefactor * np.exp( -1 * geg_gamma / (alpha * 4) ) / geg_gamma
zag = np.einsum('abi,jb->jai', zeff, g_positions_gamma)
cosine = np.cos( 2 * np.pi * np.einsum('ja,nma->jnm', g_positions_gamma, distances_n) )
long_range_vector = np.einsum('ijk,ika->ija', cosine, zag)

# Symmetrize the long range correction (Imposing Hermicity) which is the outer product of the effective charges
# scaled by the cosine term. Then take the average of M and M^T
long_range_correction = np.einsum('ijk,ijl->ijkl', long_range_vector, zag) / 2
long_range_correction += np.transpose(long_range_correction, (0, 1, 3, 2))

# Apply exponential factor and sum over G-vectors
# Arange axes of correction tensor to be compatible with the diagonals of the dynamical matrix
long_range_correction = np.einsum('i,ijkl->klj', facgd, long_range_correction)

# Fake dynamical matrix
dynamical_matrix = tf.zeros([3, 3, natoms, natoms], dtype=tf.complex64)
dynamical_matrix = tf.linalg.set_diag(dynamical_matrix, tf.linalg.diag_part(dynamical_matrix) - long_range_correction)

# Measure geg at q-points
kpoints = np.array([[0, 0, 0,], [0.5, 0.0, 0.0], [0.3, 0.3, 0.0], [0.4, 0.15, 0.0]]) @ reciprocal_n
g_plus_k = g_positions[:, None, :] + kpoints[None, :, :]
g_plus_k = g_plus_k.reshape(-1, 3) # List of G+K vectors
geg = np.einsum('ij,jl,il->i', g_plus_k, epsilon, g_plus_k)
cells_to_include = (geg > 0) * (geg/(4 * alpha) < gmax)

# Filter G+K vectors
geg = geg[cells_to_include]
g_plus_k = g_plus_k[cells_to_include]

# Prefactors for the GEG
facgd = prefactor * np.exp( -1 * geg / (alpha * 4) ) / geg
zag = np.einsum('abi,jb->jai', zeff, g_plus_k)
exponential = np.exp( -1j * 2 * np.pi * np.einsum('ja,nma->jnm', g_plus_k, distances_n) )

# Perform outer product on the effective charges at each G vector. Sum over the G-vectors (summing the first index)
# to get long range correction matrices for each atom pair. Finally, scale it by the exponential prefactor
long_range_correction = np.einsum('ija,ijk,ijb->ijkab', zag, exponential, zag)
long_range_correction = np.einsum('i,ijkab->abjk', facgd, long_range_correction)

dynamical_matrix += long_range_correction

'''

Final correction for atom 0,0
array([[ 2.8891337e+00+0.j,  1.5605538e-01+0.j, -1.3054897e-18+0.j],
       [ 1.5605538e-01+0.j,  3.1065879e+00+0.j, -1.7035558e-12+0.j],
       [-1.3054897e-18+0.j, -1.7035558e-12+0.j,  3.9052349e-02+0.j]],
       
Final correction for atom 1,3
array([[ 0.00305561-1.8416435e-03j,  0.00412143-3.5216290e-04j, -0.00100221+3.7023600e-04j],
       [ 0.00412143-3.5216290e-04j,  0.00310074+1.7178660e-03j, -0.00037889+5.7833642e-04j],
       [-0.00100221+3.7023600e-04j, -0.00037889+5.7833642e-04j, -0.00027778+4.2698375e-06j]]


#### TESTING GAMMA POINT CORRECTION ####
-4,0,0 + GAMMA

Our index: 26320
 After filter: 4424
 US    GEG 47.211837866666663
 THEM  GEG 47.211837866664993
                         ^
                      PERFECT

 US   GCELL POSITION  -4.        , -2.30940108,   0.
 THEM GCELL POSITION  -4.        , -2.309401076,  0.
                                             ^
                                             Perfect
US   EPSILON 2.2130549                 2.2130549,                1.1705941
THEM EPSILON 2.2130548999999999        2.2130548999999999        1.1705941000000000

US   facgd 8.4770120976283901873e-10
THEM facgd 8.4770119993643174E-010
                  ^
                  6 digits of agreement

Note: this correction is only applied to "diagonal" atoms (00, 11 etc)
                      DIRS|Atoms
                        a,b|i,j
US     dynamical_matrix[0,0|0,3] = 0.00000000e+00
THEM   dynamical_matrix[0,0|0,3] = 0.0000000000000000

US     dynamical_matrix[0,0|1,1] = 2.36764178e-07
THEM   dynamical_matrix[0,0|1,1] = 2.3676417632883645E-007
                                            ^
                                            9 digits

US     dynamical_matrix[0,0|1,2] = 1.44572741e-07
THEM   dynamical_matrix[0,0|1,2] = 1.4457274017368709E-007


#### TESTING FINITE-Q POINT CORRECTION ####
 -5,1,0 + 0.5,0,0

# Index of original point: 23876
# Index of original point + k: 23876,1 -> 95505
# Index for us: 16721 (after filtering)

THEM GEG      49.424892766666012
US   GEG      49.424892766666663285

them  fgd   4.6565962747579749E-010
us    fgd   4.6565963287377328188e-10
                    ^
                    8 digits

us   zag -10.8549666               -3.68236138,               0.
them zag -10.8549666               -3.6823613788511320        0.0000000000000000

                      a,b,i,j
us   dynamical_matrix(0,0,0,0) = 5.4868814186840845748e-08+0j
them dynamical_matrix(0,0,0,0) = 5.48688135507957436E-008
                                         ^
                                   Matches to 8 digits

us   dynamical_matrix(0,1,1,0) = 1.8613304831817361694e-08+0j
them dynamical_matrix(0,1,1,0) = 1.86133046160486444E-008


us   dynamical_matrix(0,1,1,1) = 6.31424466e-09
them dynamical_matrix(0,1,1,1) = 6.31424458283722623E-009


'''