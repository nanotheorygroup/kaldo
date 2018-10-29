import ase
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ballistico.constants as constants
import ballistico.io_helper as io_helper
from ballistico.MolecularSystem import MolecularSystem
from ballistico.Phonons import Phonons


if __name__ == "__main__":
    is_classic = True
    geometry = ase.io.read ('reference.xyz')
    replicas = np.array ([1, 1, 1])
    system = MolecularSystem (configuration=geometry, replicas=replicas, temperature=300.)
    n_phonons = system.configuration.get_positions ().shape[0] * 3
    system.second_order = io_helper.import_second_dlpoly(geometry, replicas)
    logging.info('second order loaded')
    system.third_order = io_helper.import_third_order_dlpoly(geometry, replicas)
    logging.info('third order loaded')
    phonons = Phonons (system, is_classic=is_classic)
    energies = phonons.frequencies.squeeze ()
    plt.plot (energies)
    plt.ylabel ('$\\nu$/THz', fontsize='14', fontweight='bold')
    plt.xlabel ("phonon id", fontsize='14', fontweight='bold')
    plt.show ()
    width = 0.05 # mev
    logging.info('sigma meV ', width)
    width = width * constants.mevoverthz
    gamma = phonons.calculate_gamma(sigma_in=width)
    gamma_plus = gamma[1]
    gamma_minus = gamma[0]
    coeff = 1000 * constants.hbar / constants.charge_of_electron
    gamma_plus = coeff * gamma_plus
    gamma_minus = coeff * gamma_minus
    logging.info(gamma_plus, gamma_minus)
    gamma_to_plot = (gamma_plus + gamma_minus)
    plt.scatter (energies, gamma_to_plot.squeeze ())
    plt.xlabel ('$\\nu$/THz', fontsize='14', fontweight='bold')
    plt.ylabel ("$2\Gamma$/meV", fontsize='14', fontweight='bold')
    plt.ylim([0,5])
    plt.show ()
    logging.info('End of calculations')
