import numpy as np

# 1 thz = 0.00414 mev
thzovermev = 4.13566553853599
mevoverthz = 1. / thzovermev

bohroverangstrom = 0.529177
rydbergoverev = 13.6056980

evoverjoule = 1.60217662e-19  # coulombs
avogadro = 6.022140857e23

mass_factor = 1.8218779 * 6.022e-4

thzoverjoule = 6.626069570305e-22
kelvinoverjoule = 1.380648813e-23  # J / K


evoverdlpoly = evoverjoule * avogadro / 10

gamma_coeff = 1 / (2 * np.pi) * 1 / (4. * np.pi) ** 3 * avogadro ** 3 * thzoverjoule * evoverjoule * evoverjoule / 1000

davide_coeff = 1 / (2 * np.pi) * gamma_coeff * thzovermev  # result in mev
