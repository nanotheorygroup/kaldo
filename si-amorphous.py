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

    width = 0.05 # mev
    logging.info('sigma meV ', width)
    width = width * constants.mevoverthz


    finite_difference = FiniteDifference(atoms=atoms,
                                         second_order=second_order,
                                         third_order=third_order,
                                         is_persistency_enabled=True)

    # Create a phonon object
    phonons = Phonons(finite_difference=finite_difference,
                      is_classic=is_classic,
                      temperature=temperature,
                      sigma_in=width,
                      is_persistency_enabled=True)

    # Create a plot helper object
    plotter = Plotter(phonons=phonons,
                      is_showing=True,
                      folder='plot/ballistico/',
                      is_persistency_enabled=True)

    # call the method plot everything
    plotter.plot_everything()