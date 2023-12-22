from kaldo.forceconstants import ForceConstants
from kaldo.phonons import Phonons
from kaldo.conductivity import Conductivity
import matplotlib.pyplot as plt
import kaldo.controllers.plotter as plotter

fold = 'ald'
k = 5
kpts = [k, k, k]
supercell_size = 5
supercell = [supercell_size, supercell_size, supercell_size]

temperature = 300

folder = 'input'

forceconstant = ForceConstants.from_folder(folder=folder,
                                           supercell=supercell,
                                           format='shengbte-qe',
                                           is_acoustic_sum=True)

atoms = forceconstant.second.atoms

phonons = Phonons(forceconstants=forceconstant,
                  kpts=kpts,
                  is_classic=False,
                  temperature=300,
                  folder=fold,
                  storage='memory',
                  is_unfolding=True)
plotter.plot_dispersion(phonons, n_k_points=50, with_velocity=True, is_showing=True)

plt.plot(phonons.frequency, phonons.bandwidth, '.')
plt.show()

print('RTA conductivity')
print(Conductivity(phonons=phonons, method='rta', storage='memory').conductivity.sum(axis=0))
