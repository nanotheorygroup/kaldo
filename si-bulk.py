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
    second_order = io_helper.import_second_dlpoly (atoms, supercell)

    # import the calculated third order
    third_order = io_helper.import_third_order_dlpoly(atoms, supercell)
    
    

    phonons = Shengbte(atoms=atoms,
                       supercell=supercell,
                       kpts=kpts,
                       is_classic=is_classical,
                       temperature=temperature)
    phonons.second_order = second_order
    phonons.third_order = third_order
    phonons.create_control_file ()

    print(phonons.run())
    # Plot velocity
    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), (np.linalg.norm (phonons.velocities, axis=-1)).flatten (),
                 label='ballistico')
    plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()

    # Plot gamma
    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), phonons.gamma.flatten (), label='ballistico')
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()

    # Calculate conductivity
    ConductivityController (phonons).calculate_conductivity (is_classical=is_classical)

    print(phonons.read_conductivity(converged=False))

    print('Calculation completed!')
    
    
    
    
    
    
    
    
    phonons = Ballistico (atoms=atoms,
                          supercell=supercell,
                          kpts=kpts,
                          second_order=second_order,
                          third_order=third_order,
                          is_classic=is_classical,
                          temperature=temperature)

    # Plot velocity
    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), (np.linalg.norm(phonons.velocities, axis=-1)).flatten (), label='ballistico')
    plt.ylabel ("$v_{rms}$ (10m/s)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()

    # Plot gamma
    fig = plt.figure ()
    plt.scatter (phonons.frequencies.flatten (), phonons.gamma.flatten (), label='ballistico')
    plt.ylabel ("$\gamma$ (Thz)", fontsize=16, fontweight='bold')
    plt.xlabel ("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.show ()


    # Calculate conductivity
    ConductivityController (phonons).calculate_conductivity (is_classical=is_classical)
    
    
    