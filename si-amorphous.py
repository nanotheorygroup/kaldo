import ase
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ballistico.constants as constants
import ballistico.io_helper as io_helper
from ballistico.ballistico_phonons import BallisticoPhonons as Phonons
from ballistico.plotter import Plotter
from ballistico.finite_difference import FiniteDifference


if __name__ == "__main__":
    is_classic = True
    # atoms = ase.io.read ('aSi.xyz', format='extxyz')
    atoms = ase.io.read ('CONFIG', format='dlp4')
    temperature = 300
    second_order = io_helper.import_second_dlpoly (atoms)
    third_order = io_helper.import_third_order_dlpoly (atoms)


    finite_difference = FiniteDifference(atoms=atoms,
                                         second_order=second_order,
                                         third_order=third_order,
                                         is_persistency_enabled=True)
    fig = plt.figure()

    for width in (0.02, 0.05, 0.10, 0.20):
        logging.info('sigma meV ', width)
        width_thz = width * constants.mevoverthz

        # Create a phonon object
        phonons = Phonons(finite_difference=finite_difference,
                          is_classic=is_classic,
                          temperature=temperature,
                          sigma_in=width_thz,
                          is_persistency_enabled=True)


        frequencies = phonons.frequencies.flatten()
        observable = phonons.gamma.flatten() #* 0.2418
        observable *= constants.davide_coeff / constants.gamma_coeff
        # print(observable)
        plt.scatter(frequencies[3:],
                    observable[3:], marker=".", label='gaussian, ' + str(width) + 'meV')

    plt.ylabel('$\Gamma$ (meV)', fontsize=16, fontweight='bold')
    plt.xlabel("$\\nu$ (Thz)", fontsize=16, fontweight='bold')
    plt.ylim([0,5])
    plt.legend()
    fig.savefig('comparison-bandwidths.pdf')
    plt.show()
