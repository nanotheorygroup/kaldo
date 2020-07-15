from kaldo.controllers import plotter
from kaldo.forceconstants import ForceConstants
from kaldo.conductivity import Conductivity
from kaldo.phonons import Phonons

fold = 'ald'
kpts = [5, 5, 5]
supercell = [5, 5, 5]
temperature = 300
folder = 'fc'

forceconstant = ForceConstants.from_folder(folder=folder,
                                           supercell=[5, 5, 5],
                                           format='shengbte-qe')



for k in [5]:
    kpts = [k, k, k]

    phonons = Phonons(forceconstants=forceconstant,
                      kpts=kpts,
                      is_classic=False,
                      temperature=300,
                      folder='ald',
                      is_tf_backend=True,
                      grid_type='C')


    plotter.plot_dispersion(phonons, n_k_points=100, with_velocity=False)

    print('Inverse conductivity W/m/K')
    print(Conductivity(phonons=phonons, method='inverse', storage='memory').conductivity.sum(axis=0))

