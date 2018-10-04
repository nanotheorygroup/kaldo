import ballistico.geometry_helper as ghl
import ballistico.ase_helper as ash
from ballistico.PhononsAnharmonic import PhononsAnharmonic
from ballistico.MolecularSystem import MolecularSystem
from ballistico.PlotViewController import PlotViewController
from ballistico.interpolation_controller import interpolator
from ballistico.constants import hbar, k_b, evoverdlpoly

import ballistico.io_helper as ioh
from ballistico.atoms_helper import replicate_configuration

import matplotlib.pyplot as plt
import ase
import numpy as np

if __name__ == "__main__":
    
    
    geometry = ase.io.read ('examples/si-bulk.xyz')
    # geometry = ash.optimize(geometry)
    replicas = np.array ([3,3,3])
    n_replicas = np.prod(replicas)
    replicated_geometry, list_of_replicas, list_of_indices = replicate_configuration(geometry, replicas)

    # index_first_cell is going to be different than 0 only if we use a bigger third order grid \
    # and a different ordering than the one made by the helper
    index_first_cell = 0
    for replica_id in range (list_of_replicas.shape[0]):
        replica = list_of_replicas[replica_id]
        if (replica == np.zeros (3)).all ():
            index_first_cell = replica_id
            print (replica_id)
    
    
    
    
    
    
    # replicated_geometry = ash.optimize(replicated_geometry)
    ase.io.write ('examples/replicated_Si-2.xyz', replicated_geometry)


    temperature = 300
    system = MolecularSystem (configuration=geometry, replicas=replicas, temperature=temperature)
    dyn_mat = ioh.import_dynamical_matrix_dlpoly('Dyn.form', replicas)

    mass = np.sqrt (system.configuration.get_masses ())
    system.second_order = dyn_mat * mass[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis] * mass[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    n_particles = geometry.get_positions().shape[0]
    third_dlpoly = ioh.import_third_order_dlpoly('THIRD', geometry, replicas)

    system.third_order = third_dlpoly

    
    # try:
    #     system.second_order = np.load ('second.npy')
    # except IOError as err:
    #     print (err)
    #     system.second_order = ash.calculate_second (geometry, replicas)


    
    k_mesh = np.array ([5, 5, 5])
    n_kpoints = np.prod(k_mesh)
    phonons = PhononsAnharmonic (system, k_mesh)

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
    # freqs_plot, _, _, velocities_plot = phonons.diagonalize_second_order_k (k_list)
    plot_vc = PlotViewController (system)
    plot_vc.plot_in_brillouin_zone (freqs_plot, 'fcc', n_k_points=NKPOINTS_TO_PLOT)
    plot_vc.plot_dos (omega_e, dos_e)
    plot_vc.show ()

    is_classical = False

    # third_order = np.load ('third.npy')

    # try:
    #     system.third_order = np.load ('third.npy')
    # except IOError as err:
    #     print (err)
    #     system.third_order = ash.calculate_third (geometry, replicas)

    import time
    ts = time.time ()
    gamma_plus, gamma_minus, ps_plus, ps_minus = phonons.calculate_gamma()
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
    cell = replicated_geometry.cell
    rlatticevec = np.linalg.inv (cell) * np.linalg.det (cell)
    volume = np.linalg.det (system.configuration.cell) / 1000.

    tau_zero[tau_zero == np.inf] = 0
    c_v[np.isnan(c_v)] = 0
    conductivity_per_mode = np.zeros ((3, 3))
    for index_k in range(n_kpoints):
        for alpha in range (3):
            for beta in range (3):
                for mode in range(n_modes):
                    conductivity_per_mode[alpha, beta] += c_v[index_k, mode] * phonons.velocities[index_k, mode,beta] * tau_zero[index_k, mode] * phonons.velocities[index_k,mode, alpha]

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