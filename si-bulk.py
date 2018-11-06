import numpy as np
import ase
import ase.io
import ballistico.geometry_helper as geometry_helper
import ballistico.atoms_helper as atoms_helper
import ballistico.io_helper as io_helper
from ase.phonons import Phonons
from ballistico.ballistico import Ballistico
from ballistico.conductivity_controller import ConductivityController
import matplotlib.pyplot as plt
from ballistico.shengbte import Shengbte

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # We start from a atoms
    atoms = ase.io.read ('si-bulk.xyz')
    
    # and replicate it
    supercell = np.array ([3, 3, 3])
    n_replicas = np.prod(supercell)
    replicated_geometry, _ = atoms_helper.replicate_configuration(atoms, supercell)
    
    # then we store it
    ase.io.write ('CONFIG', replicated_geometry, format='dlp4')

    # we create our system
    temperature = 300

    # our Phonons object built on the system
    kpts = np.array ([3, 3, 3])
    is_classical = False

    # import the calculated second order
    # Import the calculated third to calculate third order quantities
    second_order = io_helper.import_second_dlpoly (atoms, supercell)
    third_order = io_helper.import_third_order_dlpoly(atoms, supercell)
    
    ballistico_phonons = Ballistico (atoms=atoms,
                                     supercell=supercell,
                                     kpts=kpts,
                                     second_order=second_order,
                                     third_order=third_order,
                                     is_classic=is_classical,
                                     temperature=temperature)

    
    shengbte_phonons = Shengbte(atoms=atoms,
                                supercell=supercell,
                                kpts=kpts,
                                is_classic=is_classical,
                                temperature=temperature)
    
    shengbte_phonons.second_order = second_order
    shengbte_phonons.third_order = third_order
    print(shengbte_phonons.run())
    


    # Plot velocity
    fig = plt.figure ()
    plt.scatter (ballistico_phonons.frequencies.flatten (), (np.linalg.norm(ballistico_phonons.velocities, axis=-1)).flatten ())
    plt.scatter(shengbte_phonons.frequencies.flatten(), np.linalg.norm(shengbte_phonons.velocities,axis=-1).flatten(), marker='.')
    plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show()
    fig.savefig ('velocity.pdf')


    
    # Plot gamma
    fig = plt.figure ()
    plt.scatter (ballistico_phonons.frequencies.flatten (), ballistico_phonons.gamma.flatten ())
    plt.scatter (shengbte_phonons.frequencies.flatten (), shengbte_phonons.gamma.flatten (), marker='.')
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()
    fig.savefig ('gamma.pdf')


    # read the Shengbte conductivity
    print(shengbte_phonons.read_conductivity(converged=False))

    # Calculate conductivity
    ConductivityController (ballistico_phonons).calculate_conductivity (is_classical=is_classical)
    
    print('Calculation completed!')