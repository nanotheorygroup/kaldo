import numpy as np
import ase
import ase.io
import ballistico.geometry_helper as geometry_helper
import ballistico.atoms_helper as atoms_helper
import ballistico.io_helper as io_helper
from ballistico.Phonons import Phonons
from ballistico.ConductivityController import ConductivityController
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from ballistico.ShengbteHelper import ShengbteHelper
import logging
import sys

NKPOINTS_TO_PLOT = 100

if __name__ == "__main__":
    # We start from a geometry
    geometry = ase.io.read ('si-bulk.xyz')
    
    # and replicate it
    replicas = np.array ([3, 3, 3])
    n_replicas = np.prod(replicas)
    replicated_geometry, list_of_replicas = atoms_helper.replicate_configuration(geometry, replicas)
    
    # then we store it
    ase.io.write ('CONFIG', replicated_geometry, format='dlp4')

    # we create our system
    temperature = 300

    # import the calculated second order
    # Import the calculated third to calculate third order quantities

    # our Phonons object built on the system
    k_mesh = np.array ([3, 3, 3])
    is_classical = False
    second_order = io_helper.import_second_dlpoly (geometry, replicas)
    third_order = io_helper.import_third_order_dlpoly(geometry, replicas)
    phonons = Phonons (configuration=geometry,
                       replicas=replicas,
                       k_size=k_mesh,
                       second_order=second_order,
                       third_order=third_order,
                       is_classic=is_classical,
                       temperature=temperature)

    configuration = geometry

    # pick some k_points to plot
    k_list, q, Q, point_names = geometry_helper.create_k_and_symmetry_space (configuration, symmetry='fcc',
                                                                             n_k_points=NKPOINTS_TO_PLOT)
    n_modes = configuration.positions.shape[0] * 3
    n_k_points_to_plot = k_list.shape[0]
    freqs_to_plot = np.zeros ((n_k_points_to_plot, n_modes))
    vel_to_plot = np.zeros ((n_k_points_to_plot, n_modes, 3), dtype=np.complex)

    # Second order quantity calculated first
    for index_k in range(k_list.shape[0]):
        k_point = k_list[index_k]
        freqs_to_plot[index_k], _, _ , vel_to_plot[index_k] = phonons.diagonalize_second_order_single_k (k_point)

    fig = plt.figure ()
    grid = gridspec.GridSpec (ncols=4, nrows=3)

    # Crate a path in teh brillioun
    k_list, q, Q, point_names = geometry_helper.create_k_and_symmetry_space (configuration, symmetry='fcc', n_k_points=100)
    
    # Calculate density of state, this is currently a method (called with parentesis), soon will be an attribute
    omega_e, dos_e = phonons.density_of_states ()

    # Calculate the root mean square of the velocity
    rms_velocity_to_plot = np.linalg.norm(vel_to_plot, axis=-1)

    # Let's plot
    fig.add_subplot (grid[:, 0:3])
    plt.ylabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.xticks (Q, point_names, fontsize=16, fontweight='bold')
    plt.xlim (q[0], q[-1])
    plt.plot (q, freqs_to_plot, "-")
    plt.grid ()
    plt.ylim (freqs_to_plot.min (), freqs_to_plot.max () * 1.05)
    fig.add_subplot (grid[:, 3])
    plt.fill_betweenx (x1=0., x2=dos_e, y=omega_e, color='lightgrey', edgecolor='k')
    plt.plot (dos_e, omega_e, "-", color='black')
    plt.ylim (omega_e.min (), omega_e.max () * 1.05)
    plt.xticks ([], [])
    plt.grid ()
    plt.xlim (0, dos_e.max () * (1.2))
    fig.savefig ('omega.pdf')

    fig = plt.figure ()
    plt.scatter(freqs_to_plot[rms_velocity_to_plot > 0.001], rms_velocity_to_plot[rms_velocity_to_plot > 0.001])
    plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    fig.savefig ('velocity.pdf')
    
    # Calculate conductivity
    ConductivityController(phonons).calculate_conductivity(is_classical=is_classical)
    
    shl = ShengbteHelper(configuration, k_mesh, replicas, temperature)
    shl.second_order = second_order
    shl.third_order = third_order
    shl.save_second_order_matrix()
    shl.save_third_order_matrix()
    print(shl.run())
    frequency = shl.frequencies
    gamma = shl.decay_rates


    # Plot gamma
    fig = plt.figure ()
    plt.ylim([0,0.30])
    plt.scatter (phonons.frequencies.flatten (), (phonons.gamma[1] +  phonons.gamma[0]).flatten ())
    # plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    # plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    
    # fig.savefig ('gamma.pdf')

    # Plot gamma sheng
    # fig = plt.figure ()
    # plt.ylim([0,0.30])
    plt.scatter (frequency.flatten (), gamma.flatten ())
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    # fig.savefig ('gamma-sheng.pdf')
    print('done')
    plt.show()
