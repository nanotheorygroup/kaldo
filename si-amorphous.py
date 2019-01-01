import ase
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ballistico.constants as constants
import ballistico.io_helper as io_helper
from ballistico.ballistico_phonons import BallisticoPhonons as Phonons
from ballistico.plotter import Plotter
from finite_difference.finite_difference import FiniteDifference
import seaborn as sns.

if __name__ == "__main__":
    is_classic = True
    # atoms = ase.io.read ('aSi.xyz', format='extxyz')
    atoms = ase.io.read('X.xyz', format='extxyz')
    ase.io.write('dlpoly_files/CONFIG', atoms, format='dlp4')
    temperature = 300
    second_order = io_helper.import_second_dlpoly(atoms)
    third_order = io_helper.import_third_order_dlpoly(atoms)

    finite_difference = FiniteDifference(atoms=atoms,
                                         second_order=second_order,
                                         third_order=third_order,
                                         is_persistency_enabled=True)
    fig = plt.figure()
    sns.set(color_codes=True)

    # widths = [0.05, 0.1, 0.2, 0.5, 1]
    widths = [0.05, 0.5]
    for width in widths:
        width_thz = width / constants.petahertz

        # Create a phonon object
        phonons = Phonons(finite_difference=finite_difference,
                          is_classic=is_classic,
                          temperature=temperature,
                          sigma_in=width_thz,
                          is_persistency_enabled=True,
                          broadening_shape='lorentz')

        hbar = 6.35075751
        mevoverdlpoly = 9.648538
        coeff = hbar ** 2 * np.pi / 4. / mevoverdlpoly / 16 / np.pi ** 4

        # next line converts to meV > THz
        # shen_coeff = (2 * np.pi) * coeff

        ax = sns.kdeplot(phonons.frequencies.flatten(), phonons.gamma.flatten() * coeff,
                         label='width=%.3f THz' % width_thz)
        plt.legend()

    plt.ylim([0, .06])
    plt.xlim([0, 20])
    plt.xlabel("$\\nu$ (THz)", fontsize=16, fontweight='bold')
    plt.ylabel("$\Gamma$ (THz)", fontsize=16, fontweight='bold')
    plt.show()

    fig.savefig('comparison_lorentz.pdf')
    Plotter(phonons).plot_dos()