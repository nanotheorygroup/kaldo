import numpy as np
import pandas as pd
from ballistico.Phonons import Phonons
from ballistico.MolecularSystem import MolecularSystem
import matplotlib.pyplot as plt
import ballistico.constants as constants
from sparse import COO
from ballistico.constants import evoverdlpoly
import ase
from ballistico.io_helper import import_second_dlpoly, import_third_order_dlpoly



def plot_file(filename, is_classic=False):
    if is_classic:
        column_id = 2
    else:
        column_id = 1
    data = pd.read_csv (filename, delim_whitespace=True, skiprows=[0, 1], header=None)
    data = np.array (data.values)
    freqs_to_plot = data[:, 0]
    gamma_to_plot = data[:, column_id]
    plt.scatter (freqs_to_plot, gamma_to_plot, marker='.', color='red')

    

if __name__ == "__main__":
    is_classic = True
    
    geometry = ase.io.read ('reference.xyz')
    
    replicas = np.array ([1, 1, 1])
    system = MolecularSystem (configuration=geometry, replicas=replicas, temperature=300.)
    n_phonons = system.configuration.get_positions ().shape[0] * 3
    
    system.second_order = import_second_dlpoly('Dyn.form', geometry, replicas)
    
    print ('second order loaded')
    system.third_order = import_third_order_dlpoly('THIRD', geometry, replicas)
    print ('third order loaded')
    
    phonons = Phonons (system, np.array ([1, 1, 1]), is_classic=is_classic)
    energies = phonons.frequencies.squeeze ()
    
    
    plt.plot (energies)
    
    plt.ylabel ('$\\nu$/THz', fontsize='14', fontweight='bold')
    plt.xlabel ("phonon id", fontsize='14', fontweight='bold')
    plt.show ()

    width = 0.05 # mev

    print ('sigma meV ', width)
    width = width * constants.mevoverthz
    
    gamma_plus, gamma_minus, ps_plus, ps_minus = phonons.calculate_gamma(sigma=width)

    in_ph = 3
    n_phonons = energies.shape[0]
    max_ph = 10
    gamma_plus, gamma_minus = phonons.calculate_gamma_amorphous(in_ph, max_ph, sigma=width)
    gamma_to_plot = (gamma_plus + gamma_minus)
    
    plt.scatter (energies[in_ph:max_ph], gamma_to_plot[in_ph:max_ph])
    
    plt.xlabel ('$\\nu$/THz', fontsize='14', fontweight='bold')
    plt.ylabel ("$2\Gamma$/meV", fontsize='14', fontweight='bold')
    plt.ylim([0,5])
    
    for i in range (1):
        plot_file ('Decay.' + str (i), is_classic=is_classic)
    
    plt.show ()
    print ('End of calculations')
