import numpy as np
from ase.build import bulk
import matplotlib.pyplot as plt

from ase.calculators.espresso import Espresso
import os

if __name__ == "__main__":

    os.environ['ASE_ESPRESSO_COMMAND'] = '/Users/giuse/bin/pw.x -in PREFIX.pwi > PREFIX.pwo'


    calculator = Espresso
    calculator_inputs = {'system': {'ecutwfc': 20.0},
                         'electrons': {'conv_thr': 1e-10},
                         'disk_io': 'low',
                         'pseudo_dir': '/Users/giuse/espresso/pseudo/'}
    pseudopotentials = {'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'}

    calc = Espresso(pseudopotentials=pseudopotentials,
                    tstress=True,
                    tprnfor=True,  # kwargs added to parameters
                    input_data=calculator_inputs,
                    koffset=(1, 1, 1),
                    kpoints=(2, 2, 2)
                    )

    lattice_constants = np.arange(5.516, 5.517, 0.0001)
    potential = np.zeros_like(lattice_constants)
    for i in range(lattice_constants.shape[0]):
        lattice_constant = lattice_constants[i]
        atoms = bulk ('Si', 'diamond', a=lattice_constant)
        atoms.set_calculator(calc)

        potential[i] = atoms.get_potential_energy()
    plt.plot(lattice_constants, potential)
    plt.show()