import numpy as np
import ballistico.atoms_helper as ath
import ballistico.geometry_helper as ghl
from ballistico.PhononsAnharmonic import PhononsAnharmonic
from ballistico.MolecularSystem import MolecularSystem
from ballistico.PlotViewController import PlotViewController
from ballistico.ShengbteHelper import ShengbteHelper
from ballistico.interpolation_controller import interpolator
from ballistico.constants import hbar, k_b
from sparse import COO
import pandas as pd
import matplotlib.pyplot as plt
import spglib as spg

import ballistico.ConductivityController as ConductivityController
from scipy.interpolate import RegularGridInterpolator
import seaborn as sns
import matplotlib
# font = {'weight' : 'bold',
#         'size'   : 14}
# matplotlib.rc('font', **font)




folder = '/Users/giuseppe/Development/research-dev/PhononRelax/test-Si-54/'

def import_dynamical_matrix(replicas):
    dynamical_matrix_file = folder + 'Dyn.form'
    dynamical_matrix_frame = pd.read_csv(dynamical_matrix_file, header=None, delim_whitespace=True)
    dynamical_matrix_vector = dynamical_matrix_frame.values
    n_replicas = replicas[0] * replicas[1] * replicas[2]
    n_particles = int((dynamical_matrix_vector.size / (3. ** 2.)) ** (1. / 2.)/n_replicas)
    return dynamical_matrix_vector.reshape(n_replicas, n_particles, 3, n_replicas, n_particles, 3)


def import_third_order(ndim):
    third_order_frame = pd.read_csv (folder + 'THIRD', header=None, delim_whitespace=True)
    third_order = third_order_frame.values.T
    v3ijk = third_order[5:8].T
    n_particles = int (ndim / 3)
    coords = np.vstack ((third_order[0:5] - 1, 0 * np.ones ((third_order.shape[1]))))
    sparse_x = COO (coords, v3ijk[:, 0], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    coords = np.vstack ((third_order[0:5] - 1, 1 * np.ones ((third_order.shape[1]))))
    sparse_y = COO (coords, v3ijk[:, 1], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    coords = np.vstack ((third_order[0:5] - 1, 2 * np.ones ((third_order.shape[1]))))
    sparse_z = COO (coords, v3ijk[:, 2], shape=(n_particles, 3, n_particles, 3, n_particles, 3))
    sparse = sparse_x + sparse_y + sparse_z
    return sparse.reshape ((ndim, ndim, ndim))




if __name__ == "__main__":
    geometry = ath.from_filename ('examples/si-bulk.xyz')
    forcefield = ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"]
    replicas = np.array ([3, 3, 3])
    temperature = 300
    system = MolecularSystem (geometry, replicas=replicas, temperature=temperature, optimize=True, lammps_cmd=forcefield)
    k_mesh = np.array ([11, 11, 11])
    n_kpoints = np.prod(k_mesh)

    spacegroup = spg.get_spacegroup (geometry, symprec=1e-5)
    mapping, grid = spg.get_ir_reciprocal_mesh (k_mesh, geometry, is_shift=[0, 0, 0])
    # print ("Number of ir-kpoints: %d" % len (np.unique (mapping)))
    unique_points, degeneracy = np.unique (mapping, return_counts=True)
    
    phonons = PhononsAnharmonic (system, k_mesh)
    
    print(unique_points)
    
    
    phonons.calculate_second_all_grid()
    NKPOINTS_TO_PLOT = 100
    
    k_list, q, Q, point_names = ghl.create_k_and_symmetry_space (phonons.system, symmetry='fcc',
                                                                 n_k_points=NKPOINTS_TO_PLOT)
    freqs_plot = np.zeros ((k_list.shape[0], phonons.system.configuration.positions.shape[0] * 3))
    n_modes = system.configuration.positions.shape[0] * 3
    freqs_plot = np.zeros ((k_list.shape[0], n_modes))
    freqs = phonons._frequencies.reshape((k_mesh[0], k_mesh[1], k_mesh[2], n_modes))
    for mode in range (n_modes):
        freqs_plot[:, mode] = interpolator (k_list, freqs[:, :, :, mode])

    omega_e, dos_e = phonons.density_of_states (freqs)
    freqs_plot, _, _, velocities_plot = phonons.diagonalize_second_order_k (k_list)
    plot_vc = PlotViewController (system)
    plot_vc.plot_in_brillouin_zone (freqs_plot, 'fcc', n_k_points=NKPOINTS_TO_PLOT)
    plot_vc.plot_dos (omega_e, dos_e)
    plot_vc.show ()

    fig = plt.figure ()
    velocity_plot = velocities_plot[:, :, 0]
    plt.scatter (freqs_plot.flatten (), np.abs (velocity_plot.flatten ()), marker='.')
    plt.ylabel ('velocity $/ (m/s)$')
    plt.xlabel ('frequency $/ THz$')
    # plt.ylim(0,10)
    plt.show ()
    plt.close (fig)
    
    is_classical = False

    import time
    ts = time.time ()
    gamma_plus, gamma_minus, ps_plus, ps_minus = phonons.calculate_gamma(unique_points)
    print('time spent = ', time.time() - ts)

    plt.ylim([0,0.30])
    plt.scatter (phonons.frequencies.flatten (), gamma_plus.flatten ())
    plt.scatter (phonons.frequencies.flatten (), gamma_minus.flatten ())
    plt.show ()

    plt.scatter (phonons.frequencies.flatten (), ps_plus.flatten ())
    plt.scatter (phonons.frequencies.flatten (), ps_minus.flatten ())
    ps_plus[np.isnan(ps_plus)] = 0

    max_ps = np.array ([ps_plus.max (), ps_minus.max()]).max ()
    plt.ylim([0,max_ps])
    plt.show ()
    gamma = gamma_plus + gamma_minus
    tau_zero = np.empty_like(gamma)

    tau_zero[(gamma)!=0] = 1 / (gamma[gamma!=0])
    f_be = np.empty_like(phonons.frequencies)
    f_be[phonons.frequencies!=0] = 1. / (np.exp (hbar * phonons.frequencies[phonons.frequencies!=0] / (k_b * temperature)) - 1.)
    c_v = hbar ** 2 * f_be * (f_be + 1) * phonons.frequencies ** 2 / (k_b * temperature ** 2)
    cell = system.replicated_configuration.cell
    rlatticevec = np.linalg.inv (cell) * np.linalg.det (cell)
    volume = np.linalg.det (system.configuration.cell) / 1000.

    tau_zero[tau_zero == np.inf] = 0
    c_v[np.isnan(c_v)] = 0
    conductivity_per_mode = np.zeros ((3, 3))
    for index_k, (associated_index, gp) in enumerate (zip (mapping, grid)):
        print ((index_k, associated_index))
        for alpha in range (3):
            for beta in range (3):
                for mode in range(n_modes):
                    conductivity_per_mode[alpha, beta] += c_v[index_k, mode] * phonons.velocities[index_k, mode,beta] * tau_zero[associated_index, mode] * phonons.velocities[index_k,mode, alpha]

    conductivity_per_mode *= 1.E21 / (volume * n_kpoints)
    conductivity = conductivity_per_mode
    print(conductivity)

    n_modes = system.configuration.positions.shape[0] * 3
    freqs_plot = np.zeros ((k_list.shape[0], n_modes))

    for mode in range (n_modes):
        to_plot = ps_plus.reshape (
            (k_mesh[0], k_mesh[1], k_mesh[2], ps_plus.shape[1]))

        freqs_plot[:, mode] = interpolator (k_list, to_plot[:, :, :, mode])
    print('ps_plus', np.abs(ps_plus).sum())