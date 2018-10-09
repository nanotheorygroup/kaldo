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

def calculate_conductivity(system, gamma, k_mesh):
    n_kpoints = np.prod(k_mesh)
    tau_zero = np.empty_like (gamma)
    tau_zero[(gamma) != 0] = 1 / (gamma[gamma != 0])
    f_be = np.empty_like (phonons.frequencies)
    f_be[phonons.frequencies != 0] = 1. / (
                np.exp (hbar * phonons.frequencies[phonons.frequencies != 0] / (k_b * temperature)) - 1.)
    c_v = hbar ** 2 * f_be * (f_be + 1) * phonons.frequencies ** 2 / (k_b * temperature ** 2)
    volume = np.linalg.det (system.configuration.cell) / 1000.
    
    tau_zero[tau_zero == np.inf] = 0
    c_v[np.isnan (c_v)] = 0
    conductivity_per_mode = np.zeros ((3, 3))
    for index_k in range (n_kpoints):
        for alpha in range (3):
            for beta in range (3):
                for mode in range (n_modes):
                    conductivity_per_mode[alpha, beta] += c_v[index_k, mode] * phonons.velocities[index_k, mode, beta] * \
                                                          tau_zero[index_k, mode] * phonons.velocities[
                                                              index_k, mode, alpha]
    
    conductivity_per_mode *= 1.E21 / (volume * n_kpoints)
    conductivity = conductivity_per_mode
    return (conductivity)


NKPOINTS_TO_PLOT = 100

if __name__ == "__main__":
    
    # We start from a geometry
    geometry = ase.io.read ('examples/si-bulk.xyz')
    
    # and replicate it
    replicas = np.array ([3, 3, 3])
    n_replicas = np.prod(replicas)
    replicated_geometry, list_of_replicas = replicate_configuration(geometry, replicas)

    # then we store it
    ase.io.write ('CONFIG', replicated_geometry, format='dlp4')

    # we create our system
    temperature = 300
    system = MolecularSystem (configuration=geometry, replicas=replicas, temperature=temperature)

    # our phonon object built on the system
    k_mesh = np.array ([5, 5, 5])
    is_classical = False
    phonons = PhononsAnharmonic (system, k_mesh, is_classic=is_classical)

    # import the calculated second order
    system.second_order = ioh.import_second_dlpoly ('Dyn.form', geometry, replicas)

    # pick some k_points to plot
    k_list, q, Q, point_names = ghl.create_k_and_symmetry_space (phonons.system, symmetry='fcc',
                                                                 n_k_points=NKPOINTS_TO_PLOT)
    n_modes = system.configuration.positions.shape[0] * 3
    n_k_points_to_plot = k_list.shape[0]
    freqs_plot = np.zeros ((n_k_points_to_plot, n_modes))
    vel_to_plot = np.zeros ((n_k_points_to_plot, n_modes, 3))

    # Let's try some calculations
    for index_k in range(k_list.shape[0]):
        k_point = k_list[index_k]
        freqs_plot[index_k], _, _ ,vel_to_plot[index_k] = phonons.diagonalize_second_order_single_k (k_point)

    # Let's plot the energies
    plot_vc = PlotViewController (system)
    plot_vc.plot_in_brillouin_zone (freqs_plot, 'fcc', n_k_points=NKPOINTS_TO_PLOT)
    omega_e, dos_e = phonons.density_of_states (phonons.frequencies)
    plot_vc.plot_dos (omega_e, dos_e)
    plot_vc.show ()


    # Let's plot the velocities
    rms_velocity_to_plot = np.linalg.norm(vel_to_plot, axis=-1)
    plt.scatter(freqs_plot, rms_velocity_to_plot)
    plt.ylabel ("$v$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()


    # Import the calculated third
    system.third_order = ioh.import_third_order_dlpoly('THIRD', geometry, replicas)
    gamma_plus, gamma_minus, ps_plus, ps_minus = phonons.calculate_gamma()

    # Plot gamma
    plt.ylim([0,0.30])
    plt.scatter (phonons.frequencies.flatten (), gamma_plus.flatten ())
    plt.scatter (phonons.frequencies.flatten (), gamma_minus.flatten ())
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()

    # Plot phase space
    plt.scatter (phonons.frequencies.flatten (), ps_plus.flatten ())
    plt.scatter (phonons.frequencies.flatten (), ps_minus.flatten ())
    max_ps = np.array ([ps_plus.max (), ps_minus.max()]).max ()
    plt.ylim([0,max_ps])
    plt.gca ().yaxis.set_major_locator (plt.NullLocator ())
    plt.ylabel ("Phase Space", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()

    # Calculate conductivity
    print(calculate_conductivity(system, gamma_plus + gamma_minus, k_mesh))