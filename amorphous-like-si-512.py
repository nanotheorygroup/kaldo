import ase
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.build import bulk

import ballistico.constants as constants
import ballistico.io_helper as io_helper
from ballistico.ballistico_phonons import BallisticoPhonons as Phonons
from ballistico.plotter import Plotter
from ballistico.finite_difference import FiniteDifference
import seaborn as sns

if __name__ == "__main__":
    is_classic = True
    # atoms = ase.io.read ('aSi.xyz', format='extxyz')
    atoms = ase.io.read('si-8.xyz', format='extxyz')

    atoms = FiniteDifference(atoms=atoms, supercell=(4, 4, 4)).replicated_atoms
    ase.io.write('dlpoly_files/CONFIG', atoms, format='dlp4')

    finite_difference = FiniteDifference(atoms)
    finite_difference.second_order = io_helper.import_second_dlpoly(atoms)
    finite_difference.third_order = io_helper.import_third_order_dlpoly(atoms)

    temperature = 300

    fig = plt.figure()
    sns.set(color_codes=True)

    # widths = [0.05, 0.1, 0.2, 0.5, 1]
    widths = [0.5, 1]
    for width in widths:
        width_thz = width / constants.terahertz

        # Create a phonon object
        phonons = Phonons(finite_difference=finite_difference,
                          is_classic=is_classic,
                          temperature=temperature,
                          sigma_in=width_thz,
                          broadening_shape='lorentz')

        hbar = 6.35075751
        mevoverdlpoly = 9.648538
        coeff = hbar ** 2 * np.pi / 4. / mevoverdlpoly / 16 / np.pi ** 4

        # next line converts to meV > THz
        coeff /= constants.terahertz
        # shen_coeff = (2 * np.pi) * coeff
        coeff = 1

        plt.scatter(phonons.frequencies.flatten(), phonons.gamma.flatten() * coeff,
                         label='width=%.3f THz' % width_thz)
        plt.legend()

    # plt.ylim([0, .06])
    plt.xlim([0, 20])
    plt.xlabel("$\\nu$ (THz)", fontsize=16, fontweight='bold')
    plt.ylabel("$\Gamma$ (THz)", fontsize=16, fontweight='bold')
    plt.show()

    fig.savefig('comparison_lorentz.pdf')
    # Plotter(phonons).plot_dos()
