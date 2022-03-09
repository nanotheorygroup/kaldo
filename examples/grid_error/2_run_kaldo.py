import numpy as np
from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons

k =  15
folder='./'
temperature = 300.
kpts = (int(k), int(k), int(k))
supercell = (25, 12, 1)
forceconstants = ForceConstants.from_folder(folder=folder, supercell=supercell, format='lammps', only_second=False)

phonons = Phonons(forceconstants=forceconstants,
                  kpts=kpts,
                  is_classic=False,
                  temperature=temperature,
                  folder=folder,
                  storage='formatted')
