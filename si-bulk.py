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

np.set_printoptions(suppress=True)


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
    k_mesh = np.array ([5, 5, 5])
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

    
    # Calculate conductivity
    ConductivityController(phonons).calculate_conductivity(is_classical=is_classical)
    
    shl = ShengbteHelper(configuration, k_mesh, replicas, temperature)
    shl.second_order = second_order
    shl.third_order = third_order
    shl.save_second_order_matrix()
    shl.save_third_order_matrix()
    print(shl.run())
    frequency = shl.frequencies
    velocities = shl.velocities
    gamma = shl.decay_rates
    



    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), (np.linalg.norm(phonons.velocities,axis=-1)).flatten ())
    plt.scatter(frequency.flatten(), np.linalg.norm(velocities,axis=-1).flatten(), marker='.')
    plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show()
    # fig.savefig ('velocity.pdf')


    
    # Plot gamma
    fig = plt.figure ()
    plt.ylim ([0, 0.30])
    plt.scatter (phonons.frequencies.flatten (), np.sum(phonons.gamma, axis=0).flatten ())
    plt.scatter (frequency.flatten (), gamma.flatten (), marker='.')
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    print ('done')
    plt.show ()
    # fig.savefig ('gamma.pdf')

    
    print(shl.read_conductivity(converged=False))
