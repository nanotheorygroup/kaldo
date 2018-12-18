import numpy as np
import ase
import ase.io
import ballistico.geometry_helper as geometry_helper
import ballistico.atoms_helper as atoms_helper
from ballistico.finite_difference import FiniteDifference
from ase.calculators.lammpslib import LAMMPSlib
from ballistico.ballistico_phonons import BallisticoPhonons as Phonons
from ballistico.conductivity_controller import ConductivityController
from ballistico.plotter import Plotter
import ballistico.io_helper as io_helper
from ase.build import bulk
from ase.calculators.espresso import Espresso
import matplotlib.pyplot as plt
import seaborn as sns
import ballistico.constants as constants

from ballistico.shengbte_phonons_controller import ShengbtePhononsController as Sheng

if __name__ == "__main__":
    # We start from a atoms
    atoms = ase.io.read('cubic-si.xyz')
    # atoms = bulk('Si', 'diamond', a=5.43)
    supercell = np.array([3, 3, 3])
    replicated_atoms = FiniteDifference(atoms, supercell).replicated_atoms
    ase.io.write('dlpoly_files/CONFIG', replicated_atoms)

    # and replicate it

    n_replicas = np.prod(supercell)

    temperature = 300
    # we create our system
    # temperature = 300

    # our Phonons object built on the system
    kpts = np.array([15, 15, 15])
    is_classic = True

    calculator = LAMMPSlib
    calculator_inputs = {'lmpcmds': ["pair_style tersoff", "pair_coeff * * forcefields/Si.tersoff Si"],
                         'log_file': 'log_lammps.out'}

    # calculator = Espresso
    # calculator_inputs = {'pseudopotentials':{'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'},
    #                 'tstress':True,
    #                 'tprnfor':True,
    #                 'input_data':
    #                      {'system': {'ecutwfc': 16.0},
    #                                            'electrons': {'conv_thr': 1e-8},
    #                                            'disk_io': 'low',
    #                                            'pseudo_dir': '/home/giuseppe/espresso/pseudo/'},
    #                 'koffset':(2, 2, 2),
    #                 'kpoints':(1, 1, 1)}

    # third_order_symmerty_inputs = {'NNEIGH': 4, 'SYMPREC': 1e-5}

    # Create a finite difference object
    finite_difference = FiniteDifference(atoms=atoms,
                                         supercell=supercell,
                                         is_persistency_enabled=False)
    replicated_atoms = finite_difference.replicated_atoms
    # ase.io.write('CONFIG', replicated_atoms, 'dlp4')

    finite_difference.second_order = io_helper.import_second_dlpoly(replicated_atoms)
    finite_difference.third_order = io_helper.import_third_order_dlpoly(replicated_atoms)
    sheng = Sheng(finite_difference=finite_difference, kpts=kpts, is_classic=is_classic,
          temperature=temperature, is_persistency_enabled=False)
    sns.set(color_codes=True)

    sheng_coeff = constants.thzovermev / (2 * np.pi)
    plt.scatter(sheng.frequencies, sheng.gamma * sheng_coeff, marker='x')
    plt.show()
    # Create a phonon object
    # phonons = Phonons(finite_difference=finite_difference, kpts=kpts, is_classic=is_classic,
                      # temperature=temperature, is_persistency_enabled=True, broadening_shape='gauss')
    plt.scatter(sheng.frequencies, sheng.gamma * sheng_coeff, marker='x')

    # Create a plot helper object
    # hbar = 6.35075751
    # mevoverdlpoly = 9.648538
    # coeff = hbar ** 2 * np.pi / 4. / mevoverdlpoly / 16 / np.pi ** 4

    # next line converts to meV > THz
    # shen_coeff = (2 * np.pi) * coeff

    hbar = 6.35075751
    mevoverdlpoly = 9.648538
    coeff = hbar ** 2 * np.pi / 4. / mevoverdlpoly / 16 / np.pi ** 4

    # next line converts to meV > THz
    coeff *= constants.mevoverthz
    # shen_coeff = (2 * np.pi) * coeff

    # plt.scatter(phonons.frequencies.flatten(), phonons.gamma.flatten() * coeff,
    #             label='width=%.3f THz' % width_thz)
    # plt.scatter(phonons.frequencies.flatten(), phonons.gamma.flatten() * coeff,
    #             marker='.')
    # plt.legend()
    # plt.ylim([0, 0.04])
    plt.xlim([0.1, 17.5])

    plt.ylabel("$\Gamma$ (meV)", fontsize=16, fontweight='bold')
    plt.xlabel("$\\nu$ (THz)", fontsize=16, fontweight='bold')
    plt.show()
    # call the method plot everything
    Plotter(sheng).plot_everything()
    sheng.save_csv_data()
    # calculate the conductivity creating a conductivity object and calling the
    # calculate_conductivity method
    # heat_capacity = sheng.c_v.mean()
    conductivity = ConductivityController(sheng).calculate_conductivity(is_classical=is_classic)[0, 0]
    #
    # plt.plot(temperatures, conductivities)
    #
    # plt.plot(temperatures, heat_capacities)
    # plt.show()
    print(conductivity)
