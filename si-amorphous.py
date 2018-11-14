import ase
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ballistico.constants as constants
import ballistico.io_helper as io_helper
from ballistico.ballistico_phonons import Ballistico_phonons
from ballistico.PlotViewController import PlotViewController


if __name__ == "__main__":
    is_classic = True
    atoms = ase.io.read ('reference.xyz')
    temperature = 300
    second_order = io_helper.import_second_dlpoly (atoms)
    third_order = io_helper.import_third_order_dlpoly (atoms)

    width = 0.05 # mev
    logging.info('sigma meV ', width)
    width = width * constants.mevoverthz
    
    phonons = Ballistico_phonons (atoms=atoms,
                                  is_classic=is_classic,
                                  temperature=temperature,
                                  second_order=second_order,
                                  third_order=third_order,
                                  sigma_in=width,
                                  is_persistency_enabled=True)
    


    PlotViewController(phonons).plot_c_v()
    PlotViewController(phonons).plot_velocities()
    PlotViewController(phonons).plot_gamma()
