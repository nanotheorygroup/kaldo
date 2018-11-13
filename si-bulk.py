import numpy as np
import ase
import ase.io
import ballistico.geometry_helper as geometry_helper
import ballistico.atoms_helper as atoms_helper
import ballistico.io_helper as io_helper
import ballistico.constants as constants
from ase.phonons import Phonons
from ballistico.ballistico_phonons import Ballistico_phonons
from ballistico.conductivity_controller import ConductivityController
import matplotlib.pyplot as plt
from ballistico.shengbte_phonons import Shengbte_phonons

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    # We start from a atoms
    atoms = ase.io.read ('si-bulk.xyz')
    
    # and replicate it
    supercell = np.array ([3, 3, 3])
    n_replicas = np.prod(supercell)
    replicated_geometry, _ = atoms_helper.replicate_atoms(atoms, supercell)
    
    # then we store it
    ase.io.write ('CONFIG', replicated_geometry, format='dlp4')

    # we create our system
    temperature = 300

    # our Phonons object built on the system
    kpts = np.array ([5, 5, 5])
    is_classic = False

    # import the calculated second order
    second_order = io_helper.import_second_dlpoly (atoms, supercell)

    # import the calculated third order
    third_order = io_helper.import_third_order_dlpoly(atoms, supercell)

    phonons = Shengbte_phonons(atoms=atoms,
                            supercell=supercell,
                            kpts=kpts,
                            temperature=temperature,
                            second_order=second_order,
                            third_order=third_order,
                            is_classic=is_classic)
    


    # Plot dos
    # fig = plt.figure ()
    # plt.scatter (phonons.dos[0].flatten (), phonons.dos[1].flatten ())
    # plt.ylabel ("dos", fontsize=16, fontweight='bold')
    # plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    # fig.savefig ('dos.pdf')

    # Plot velocity
    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), np.linalg.norm(phonons.velocities, axis=-1).flatten ())
    plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    fig.savefig ('velocities.pdf')

    # Plot gamma
    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), phonons.gamma.flatten ())
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    fig.savefig ('gamma.pdf')

    # Calculate conductivity
    ConductivityController (phonons).calculate_conductivity (is_classical=is_classic)
    print(phonons.read_conductivity(converged=False))
    
    

    phonons = Ballistico_phonons (atoms=atoms,
                                  supercell=supercell,
                                  kpts=kpts,
                                  is_classic=is_classic,
                                  temperature=temperature,
                                  second_order=second_order,
                                  third_order=third_order,
                                  # sigma_in=.1,
                                  is_persistency_enabled=False)
    ConductivityController (phonons).calculate_conductivity (is_classical=is_classic)